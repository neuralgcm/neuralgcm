# Copyright 2024 Google LLC
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

"""Transforms that specialize in generating features."""

from __future__ import annotations

from typing import Sequence

import coordax as cx
from flax import nnx
import jax.numpy as jnp
import jax_solar
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import dynamic_io
from neuralgcm.experimental.core import nnx_compat
from neuralgcm.experimental.core import orographies
from neuralgcm.experimental.core import random_processes
from neuralgcm.experimental.core import transforms
from neuralgcm.experimental.core import typing
from neuralgcm.experimental.core import units
import xarray


TransformParams = transforms.TransformParams


@nnx_compat.dataclass
class RadiationFeatures(transforms.TransformABC):
  """Returns incident radiation flux Field."""

  grid: coordinates.LonLatGrid

  @property
  def lon(self) -> typing.Array:
    # TODO(dkochkov): support field broadcasting to a coordinate.
    dummy = cx.wrap(jnp.zeros(self.grid.shape), self.grid)
    return self.grid.fields['longitude'].broadcast_like(dummy).data

  @property
  def lat(self) -> typing.Array:
    # TODO(dkochkov): support field broadcasting to a coordinate.
    dummy = cx.wrap(jnp.zeros(self.grid.shape), self.grid)
    return self.grid.fields['latitude'].broadcast_like(dummy).data

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    features = {}
    features['radiation'] = cx.wrap(
        jax_solar.normalized_radiation_flux(
            time=inputs['time'].data, longitude=self.lon, latitude=self.lat
        ),
        self.grid,
    )
    return features


@nnx_compat.dataclass
class LatitudeFeatures(transforms.TransformABC):
  """Returns cos and sin of latitude features."""

  grid: coordinates.LonLatGrid

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    del inputs  # unused.
    # TODO(dkochkov): support field broadcasting to a coordinate.
    dummy = cx.wrap(jnp.zeros(self.grid.shape), self.grid)
    latitudes = jnp.deg2rad(
        self.grid.fields['latitude'].broadcast_like(dummy).data
    )
    features = {
        'cos_latitude': cx.wrap(jnp.cos(latitudes), self.grid),
        'sin_latitude': cx.wrap(jnp.sin(latitudes), self.grid),
    }
    return features


@nnx_compat.dataclass
class RandomnessFeatures(transforms.TransformABC):
  """Returns values from a random process evaluated on a grid."""

  random_process: random_processes.RandomProcessModule
  grid: cx.Coordinate
  feature_name: str = 'randomness'

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    del inputs  # unused.
    return {
        self.feature_name: self.random_process.state_values(self.grid),
    }


@nnx_compat.dataclass
class DynamicInputFeatures(transforms.TransformABC):
  """Returns subset of dynamic input values."""

  keys: Sequence[str]
  dynamic_input_module: dynamic_io.DynamicInputSlice

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    data_features = self.dynamic_input_module(inputs['time'])
    return {k: data_features[k] for k in self.keys}


@nnx_compat.dataclass
class OrographyFeatures(transforms.TransformABC):
  """Returns elevation values in real space representing orography."""

  orography_module: orographies.ModalOrography | orographies.Orography

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    del inputs  # unused.
    # TODO(dkochkov): update orographies to return fields.
    if isinstance(self.orography_module, orographies.ModalOrography):
      grid = self.orography_module.ylm_transform.nodal_grid
    elif isinstance(self.orography_module, orographies.Orography):
      grid = self.orography_module.grid
    else:
      raise ValueError(
          'orography_module must be either ModalOrography or Orography, but'
          f' got {type(self.orography_module)}'
      )
    return {'orography': cx.wrap(self.orography_module.nodal_orography, grid)}


@nnx_compat.dataclass
class OrographyWithGradsFeatures(transforms.TransformABC):
  """Returns orography features and their gradients."""

  orography_module: orographies.ModalOrography
  compute_gradients_transform: transforms.ToModalWithFilteredGradients
  include_raw_orography: bool = True

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    del inputs  # unused.
    ylm_transform = self.orography_module.ylm_transform
    ylm_grid = ylm_transform.modal_grid
    grid = ylm_transform.nodal_grid
    modal_features = {
        'orography': cx.wrap(self.orography_module.modal_orography, ylm_grid)
    }
    modal_features = {
        typing.KeyWithCosLatFactor(k, 0): v for k, v in modal_features.items()
    }
    modal_gradient_features = self.compute_gradients_transform(modal_features)
    sh_grid = ylm_transform.dinosaur_grid
    sec_lat = 1 / sh_grid.cos_lat
    sec2_lat = sh_grid.sec2_lat
    lat_axis = grid.axes[1]
    sec_lat_scales = {
        0: 1,
        1: cx.wrap(sec_lat, lat_axis),
        2: cx.wrap(sec2_lat, lat_axis),
    }
    features = {}
    if self.include_raw_orography:
      all_modal_features = modal_gradient_features | modal_features
    else:
      all_modal_features = modal_gradient_features
    for k, v in all_modal_features.items():
      sec_lat_scale = sec_lat_scales[k.factor_order]
      features[k.name] = ylm_transform.to_nodal(v) * sec_lat_scale
    return features


class CoordFeatures(transforms.TransformABC):
  """Returns features that are static in space and time."""

  def __init__(
      self,
      coords: dict[str, cx.Coordinate],
      *,
      param_type: TransformParams | nnx.Param = TransformParams,
      initializer: nnx.Initializer = nnx.initializers.truncated_normal(),
      rngs: nnx.Rngs,
  ):
    self.coords = coords
    self.features = {
        k: param_type(cx.wrap(initializer(rngs.params(), c.shape), c))
        for k, c in coords.items()
    }

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    del inputs  # unused.
    return {k: v.value for k, v in self.features.items()}

  def update_features_from_data(
      self,
      dataset: xarray.Dataset,
      sim_units: units.SimUnits,
  ):
    """Updates `self.features` with data from dataset."""
    for key, feature_value in self.features.items():
      if key in dataset:
        da = dataset[key]
        data_units = units.parse_units(da.attrs['units'])
        da.data = sim_units.nondimensionalize(da.values * data_units)
        candidate = coordinates.field_from_xarray(da)
        candidate = candidate.order_as(self.coords[key])
        self.features[key] = type(feature_value)(candidate)
