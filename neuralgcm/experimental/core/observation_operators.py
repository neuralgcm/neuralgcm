# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines observation operator API and sample operators for NeuralGCM."""

import abc
import copy
import dataclasses
import functools
from typing import Sequence

from flax import nnx
import jax
import jax.numpy as jnp
from neuralgcm.experimental import coordax as cx
from neuralgcm.experimental import pytree_mappings
from neuralgcm.experimental import pytree_transforms
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import pytree_utils
from neuralgcm.experimental.core import standard_layers
from neuralgcm.experimental.core import typing


# pylint: disable=g-classes-have-attributes


class ObservationOperator(nnx.Module, abc.ABC):
  """Base class for observation operators."""

  @abc.abstractmethod
  def observe(
      self,
      inputs: dict[str, cx.Field],
      query: dict[str, cx.Field | cx.Coordinate],
  ) -> dict[str, cx.Field]:
    """Returns observations for `query`."""
    ...


@dataclasses.dataclass
class DataObservationOperator(ObservationOperator):
  """Operator that returns pre-computed fields for matching coordinate queries.

  This observation operator matches keys and coordinates in the pre-computed
  dictionary of `coordax.Field`s and the query to the observation operator. This
  operator requires that all `query` entries are of `coordax.Coordinate` type.

  Attributes:
    fields: A dictionary of `coordax.Field`s to return in the observation.
  """

  fields: dict[str, cx.Field]

  def observe(
      self,
      inputs: dict[str, cx.Field],
      query: dict[str, cx.Field | cx.Coordinate],
  ) -> dict[str, cx.Field]:
    """Returns observations for `query` matched against `self.fields`."""
    del inputs  # unused.
    observations = {}
    is_coordinate = lambda x: isinstance(x, cx.Coordinate)
    valid_keys = list(self.fields.keys())
    for k, v in query.items():
      if k not in valid_keys:
        raise ValueError(f'Query contains {k=} not in {valid_keys}')
      if not is_coordinate(v):
        raise ValueError(
            f'DataObservationOperator only supports coordinate queries, got {v}'
        )
      coord = cx.get_coordinate(self.fields[k])
      if v != coord:
        raise ValueError(f'Query ({k}, {v}) does not match field.{coord=}')
      observations[k] = self.fields[k]
    return observations


@dataclasses.dataclass
class ObservationOperatorWithRenaming(ObservationOperator):
  """Operator wrapper that converts between different naming conventions.

  Attributes:
    operator: Observation operator that performs computation.
    renaming_dict: A dictionary mapping new names to those used by `operator`.
  """

  operator: ObservationOperator
  renaming_dict: dict[str, str]

  def observe(
      self,
      inputs: dict[str, cx.Field],
      query: dict[str, cx.Field | cx.Coordinate],
  ) -> dict[str, cx.Field]:
    """Returns observations for `query` matched against `self.fields`."""
    renamed_query = {
        self.renaming_dict.get(k, k): v for k, v in query.items()
    }
    observation = self.operator.observe(inputs, renamed_query)
    inverse_renaming_dict = {v: k for k, v in self.renaming_dict.items()}
    return {inverse_renaming_dict.get(k, k): v for k, v in observation.items()}


@dataclasses.dataclass
class FixedLearnedObservationOperator(ObservationOperator):
  """Operator that computes fixed set of observations using state mapping."""

  coordinate_mapping: pytree_mappings.CoordsStateMapping

  def observe(
      self,
      inputs: dict[str, cx.Field],
      query: dict[str, cx.Field | cx.Coordinate],
  ) -> dict[str, cx.Field]:
    """Returns predicted observations matching `query`."""
    inputs = jax.tree.map(lambda x: x.data, inputs, is_leaf=cx.is_field)
    # TODO(dkochkov): make dummy axis expansion part of the mapping specs.
    maybe_add_dummy = lambda x: jnp.expand_dims(x, 0) if x.ndim == 2 else x
    inputs = jax.tree.map(maybe_add_dummy, inputs)
    predictions = self.coordinate_mapping(inputs)
    # TODO(dkochkov): move squeezing to the mapping.
    predictions = {
        k: jnp.squeeze(v, axis=0) if v.shape[0] == 1 else v
        for k, v in predictions.items()
    }
    is_coordinate = lambda x: isinstance(x, cx.Coordinate)
    prediction_keys = list(predictions.keys())
    observations = {}
    coord = self.coordinate_mapping.coords
    for k, v in query.items():
      if k not in prediction_keys:
        raise ValueError(f'Query contains {k=} not in {prediction_keys}')
      if not is_coordinate(v):
        raise ValueError(
            'FixedLearnedObservationOperator only supports coordinate queries,'
            f' got {v}'
        )
      if (coord == v) or (coord.horizontal == v):
        observations[k] = cx.wrap(predictions[k], v)
      else:
        raise ValueError(f'Query ({k}, {v}) is not compatible with {coord=}')
    return observations


class LearnedSparseScalarObservationFromNeighbors(nnx.Module):
  """Observation operator for scalar observations at sparse locations.

  This operator predicts scalar observations that are conditioned on the
  features of the nearest neighbor on the grid and displacement features derived
  from the relative location of the query point and the neighbor.

  The expected structure of the query processed by this operator is:
  ```
    operator_query = {
        'longitude': cx.Field,
        'latitude': cx.Field,
        'scalar_name_1': cx.Coordinate,
        'scalar_name_2': cx.Coordinate,
        ...
    }
  ```

  Args:
    scalar_names: names of the scalar fields predicted by this operator.
    grid: grid on which state features are computed.
    features_module: module that computes state features from the prognostics.
    displacement_features_module: module that computes features that represent
      the relative location of the query point and the neighbor on the `grid`.
    input_state_shapes: shapes of the input state.
    layer_factory: factory for instantiating a NN that will compute predictions.
    rngs: random number generator.
  """

  def __init__(
      self,
      scalar_names: Sequence[str],
      grid: coordinates.LonLatGrid,
      *,
      features_module: pytree_transforms.Transform,
      displacement_features_module: pytree_transforms.Transform = pytree_transforms.Identity(),
      input_state_shapes: typing.Pytree,
      layer_factory: standard_layers.UnaryLayerFactory,
      rngs: nnx.Rngs,
  ):
    neighbor_feature_shapes = features_module.output_shapes(input_state_shapes)
    f_axis = -3  # default column axis.
    neighbor_feature_size = sum(
        [x.shape[f_axis] for x in jax.tree.leaves(neighbor_feature_shapes)]
    )
    displacement_shapes = {
        'delta_lon': typing.ShapeFloatStruct([1]),
        'delta_lat': typing.ShapeFloatStruct([1]),
    }
    displacement_feature_shapes = displacement_features_module.output_shapes(
        displacement_shapes
    )
    displacement_feature_size = sum(
        [x.shape[0] for x in jax.tree.leaves(displacement_feature_shapes)]
    )
    input_size = neighbor_feature_size + displacement_feature_size
    output_size = len(scalar_names)
    self.net = layer_factory(input_size, output_size, rngs=rngs)
    self.scalar_names = scalar_names
    self.features_module = features_module
    self.displacement_features_module = displacement_features_module
    self.grid = grid

  def _lon_lat_neighbor_indices(
      self,
      longitudes: typing.Array,
      latitudes: typing.Array,
      lon: typing.Array,
      lat: typing.Array,
  ) -> tuple[typing.Array, typing.Array]:
    """Returns grid indices corresponding to the point closest to (lon, lat)."""
    longitudes, latitudes = jnp.deg2rad(longitudes), jnp.deg2rad(latitudes)
    subtract_lons = lambda a, b: jnp.mod(a - b + jnp.pi, 2 * jnp.pi) - jnp.pi
    subtract_lats = lambda a, b: a - b
    lon_deltas = subtract_lons(longitudes, jnp.deg2rad(lon))
    lat_deltas = subtract_lats(latitudes, jnp.deg2rad(lat))
    return jnp.argmin(jnp.abs(lon_deltas)), jnp.argmin(jnp.abs(lat_deltas))

  def observe(
      self,
      inputs: dict[str, cx.Field],
      query: dict[str, cx.Field | cx.Coordinate],
  ) -> dict[str, cx.Field]:
    inputs = copy.copy(inputs)  # ensure that inputs are not modified.
    query = copy.copy(query)  # ensure that query is not modified.
    all_features = self.features_module(inputs)
    grid = self.grid
    lon_query, lat_query = query.pop('longitude'), query.pop('latitude')
    sparse_coord = cx.get_coordinate(lon_query)
    assert sparse_coord == cx.get_coordinate(lat_query)  # should be the same.
    lon, lat = grid.fields['longitude'].data, grid.fields['latitude'].data
    get_indices_fn = functools.partial(self._lon_lat_neighbor_indices, lon, lat)
    lon_idx, lat_idx = cx.cmap(get_indices_fn)(lon_query, lat_query)
    delta_lon = (lon_query - cx.wrap_like(lon[lon_idx.data], lon_query)).data
    delta_lat = (lat_query - cx.wrap_like(lat[lat_idx.data], lat_query)).data

    def vmap_features(feature_module, inputs):
      return feature_module(inputs)

    mapped_features = nnx.vmap(vmap_features, in_axes=(None, 0))
    displacement_features = mapped_features(
        self.displacement_features_module,
        {
            'delta_lon': delta_lon[..., jnp.newaxis],
            'delta_lat': delta_lat[..., jnp.newaxis],
        },
    )

    def _select_features_at_lon_lat(array, lon_idx, lat_idx):
      # if lon_idx/lat_idx are batched, move then upfront keeping features last.
      return jnp.asarray(array)[..., lon_idx, lat_idx].T

    nearest_features = {
        k: _select_features_at_lon_lat(v, lon_idx.data, lat_idx.data)
        for k, v in all_features.items()
    }
    all_features = pytree_utils.pack_pytree(
        nearest_features | displacement_features, axis=-1
    )

    def vmap_fn(net, inputs):
      return net(inputs)

    mapped_scalar_net = nnx.vmap(vmap_fn, in_axes=(None, 0))
    all_scalars = mapped_scalar_net(self.net, all_features)
    output_shapes = {
        k: typing.ShapeFloatStruct(sparse_coord.shape + (1,))
        for k in self.scalar_names
    }
    predictions = pytree_utils.unpack_to_pytree(
        all_scalars, output_shapes, axis=1
    )
    prediction_keys = list(predictions.keys())
    observations = {}
    for k, v in query.items():
      if k not in prediction_keys:
        raise ValueError(f'Query contains {k=} not in {prediction_keys}')
      if v == sparse_coord:
        observations[k] = cx.wrap(jnp.squeeze(predictions[k], axis=1), v)
      else:
        raise ValueError(
            f'Query ({k}, {v}) is not compatible with {sparse_coord=}'
        )
    # TODO(dkochkov): Consider not returning field entries in operators.
    observations['longitude'] = lon_query
    observations['latitude'] = lat_query
    return observations
