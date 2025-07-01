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

import coordax as cx
from flax import nnx
import jax.numpy as jnp
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import learned_transforms
from neuralgcm.experimental.core import nnx_compat
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import towers
from neuralgcm.experimental.core import transforms
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
    renamed_query = {self.renaming_dict.get(k, k): v for k, v in query.items()}
    observation = self.operator.observe(inputs, renamed_query)
    inverse_renaming_dict = {v: k for k, v in self.renaming_dict.items()}
    return {inverse_renaming_dict.get(k, k): v for k, v in observation.items()}


@dataclasses.dataclass
class FixedLearnedObservationOperator(ObservationOperator):
  """Operator that computes fixed set of observations using state mapping."""

  coordinate_mapping: learned_transforms.ForwardTowerTransform

  def observe(
      self,
      inputs: dict[str, cx.Field],
      query: dict[str, cx.Field | cx.Coordinate],
  ) -> dict[str, cx.Field]:
    """Returns predicted observations matching `query`."""
    predictions = self.coordinate_mapping(inputs)
    return DataObservationOperator(predictions).observe(inputs, query)


@nnx_compat.dataclass
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
    input_shapes: shapes of the inputs.
    layer_factory: factory for instantiating a NN that will compute predictions.
    rngs: random number generator.
  """

  target_predictions: dict[str, cx.Coordinate]
  grid: coordinates.LonLatGrid
  grid_features: transforms.Transform
  tower: towers.ForwardTower
  prediction_transform: transforms.Transform = transforms.Identity()
  mesh: parallelism.Mesh = dataclasses.field(kw_only=True)

  @classmethod
  def build_using_factories(
      cls,
      input_shapes: dict[str, cx.Field],
      target_predictions: dict[str, cx.Coordinate],
      *,
      grid: coordinates.LonLatGrid,
      grid_features: transforms.Transform,
      tower_factory: towers.ForwardTowerFactory,
      prediction_transform: transforms.Transform = transforms.Identity(),
      mesh: parallelism.Mesh,
      rngs: nnx.Rngs,
  ):
    # TODO(dkochkov): Add check that target_predictions are at most 1D.
    grid_features_shapes = grid_features.output_shapes(input_shapes)
    loc_feature_sizes = {
        k: (
            {d: s for d, s in v.named_shape.items() if d not in grid.dims}
            if v.ndim > grid.ndim
            else {None: 1}
        )
        for k, v in grid_features_shapes.items()
    }
    input_size = sum(
        [v.popitem()[1] for v in loc_feature_sizes.values()], start=2
    )
    output_size = sum(
        [x.shape[0] if x.shape else 1 for x in target_predictions.values()]
    )
    tower = tower_factory(input_size, output_size, rngs=rngs)
    return cls(
        target_predictions=target_predictions,
        grid=grid,
        grid_features=grid_features,
        tower=tower,
        prediction_transform=prediction_transform,
        mesh=mesh,
    )

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
    grid_features = self.grid_features(inputs)
    lon_query, lat_query = query.pop('longitude'), query.pop('latitude')
    sparse_coord = cx.get_coordinate(lon_query)
    assert sparse_coord == cx.get_coordinate(lat_query)  # should be the same.
    grid = self.grid
    lon, lat = grid.fields['longitude'].data, grid.fields['latitude'].data
    get_indices_fn = self._lon_lat_neighbor_indices
    lon_idx, lat_idx = cx.cmap(get_indices_fn)(lon_query, lat_query, lon, lat)
    delta_lon = lon_query - cx.wrap_like(lon[lon_idx.data], lon_query)
    delta_lat = lat_query - cx.wrap_like(lat[lat_idx.data], lat_query)
    displacement_inputs = {
        'delta_lon': delta_lon,
        'delta_lat': delta_lat,
    }

    def _select_features_at_lon_lat(array, lon_idx, lat_idx):
      # if lon_idx/lat_idx are batched, move then upfront keeping features last.
      return jnp.asarray(array)[lon_idx, lat_idx]

    nearest_grid_features = {
        k: cx.cmap(_select_features_at_lon_lat)(v.untag(grid), lon_idx, lat_idx)
        for k, v in grid_features.items()
    }
    all_features = nearest_grid_features | displacement_inputs
    target_predictions = {
        k: cx.compose_coordinates(v, sparse_coord)
        for k, v in self.target_predictions.items()
    }
    observe_sparse_transform = learned_transforms.ForwardTowerTransform(
        targets=target_predictions,
        tower=self.tower,
        dims_to_align=(sparse_coord,),
        out_transform=self.prediction_transform,
        # feature_sharding_schema=self.feature_sharding_schema,  # need this?
        # result_sharding_schema=self.result_sharding_schema,
        mesh=self.mesh,
    )
    predictions = observe_sparse_transform(all_features)
    obs = DataObservationOperator(predictions).observe(inputs, query)
    # TODO(dkochkov): Consider not returning field entries in operators.
    return obs | {'longitude': lon_query, 'latitude': lat_query}


@dataclasses.dataclass
class MultiObservationOperator(ObservationOperator):
  """Operator that dispatches queries to multiple operators.

  Attributes:
    keys_to_operator: A dictionary mapping query keys to observation operators.
  """

  keys_to_operator: dict[tuple[str, ...], ObservationOperator]

  def observe(
      self,
      inputs: dict[str, cx.Field],
      query: dict[str, cx.Field | cx.Coordinate],
  ) -> dict[str, cx.Field]:
    outputs = {}
    supported_keys = set(sum(self.keys_to_operator.keys(), start=()))
    query_keys = set(query.keys())
    if not query_keys.issubset(supported_keys):
      raise ValueError(
          f'query keys {query_keys} are not a subset of supported keys'
          f' {supported_keys}'
      )
    for key_tuple, obs_op in self.keys_to_operator.items():
      sub_query = {}
      for key in key_tuple:
        if key in query:
          sub_query[key] = query[key]
      outputs |= obs_op.observe(inputs, sub_query)
    return outputs
