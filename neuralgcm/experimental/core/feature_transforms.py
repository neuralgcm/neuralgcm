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
import jax_datetime as jdt
import jax_solar
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import diagnostics
from neuralgcm.experimental.core import dynamic_io
from neuralgcm.experimental.core import orographies
from neuralgcm.experimental.core import random_processes
from neuralgcm.experimental.core import transforms
from neuralgcm.experimental.core import typing
from neuralgcm.experimental.core import units
from neuralgcm.experimental.core import xarray_utils
import numpy as np
import xarray


TransformParams = transforms.TransformParams


@nnx.dataclass
class RadiationFeatures(transforms.PytreeTransformABC):
  """Returns incident radiation flux Field."""

  grid: coordinates.LonLatGrid

  @property
  def lon(self) -> typing.Array:
    return self.grid.fields['longitude'].broadcast_like(self.grid).data

  @property
  def lat(self) -> typing.Array:
    return self.grid.fields['latitude'].broadcast_like(self.grid).data

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    features = {}
    features['radiation'] = cx.field(
        jax_solar.normalized_radiation_flux(
            time=inputs['time'].data, longitude=self.lon, latitude=self.lat
        ),
        self.grid,
    )
    return features


@nnx.dataclass
class LatitudeFeatures(transforms.PytreeTransformABC):
  """Returns cos and sin of latitude features."""

  grid: coordinates.LonLatGrid

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    del inputs  # unused.
    latitudes = jnp.deg2rad(
        self.grid.fields['latitude'].broadcast_like(self.grid).data
    )
    features = {
        'cos_latitude': cx.field(jnp.cos(latitudes), self.grid),
        'sin_latitude': cx.field(jnp.sin(latitudes), self.grid),
    }
    return features


@nnx.dataclass
class DiagnosticValueFeatures(transforms.TransformABC):
  """Returns diagnostic values to be used as features."""

  diagnostic_module: diagnostics.DiagnosticModule

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    del inputs  # unused
    return self.diagnostic_module.diagnostic_values()


@nnx.dataclass
class RandomnessFeatures(transforms.TransformABC):
  """Returns values from a random process evaluated on a grid."""

  random_process: random_processes.RandomProcessModule
  coord: cx.Coordinate
  feature_name: str = 'randomness'

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    del inputs  # unused.
    return {
        self.feature_name: self.random_process.state_values(self.coord),
    }


@nnx.dataclass
class DynamicInputFeatures(transforms.TransformABC):
  """Returns subset of dynamic input values."""

  keys: Sequence[str]
  dynamic_input_module: dynamic_io.DynamicInputSlice
  time_offsets: dict[str, np.timedelta64] | None = None

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    if self.time_offsets is None:
      data_features = self.dynamic_input_module(inputs['time'])
      return {k: data_features[k] for k in self.keys}
    features = {}
    for suffix, offset in self.time_offsets.items():
      offset_time = inputs['time'] + cx.field(jdt.to_timedelta(offset))
      data_features = self.dynamic_input_module(offset_time)
      for k in self.keys:
        features[f'{k}_{suffix}'] = data_features[k]
    return features


@nnx.dataclass
class OrographyFeatures(transforms.TransformABC):
  """Returns elevation values in real space representing orography."""

  orography_module: orographies.ModalOrography | orographies.Orography

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    del inputs  # unused.
    return {'orography': self.orography_module.nodal_orography}


@nnx.dataclass
class OrographyWithGradsFeatures(transforms.PytreeTransformABC):
  """Returns orography features and their gradients."""

  orography_module: orographies.ModalOrography
  compute_gradients_transform: transforms.ToModalWithDerivatives
  include_raw_orography: bool = True

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    del inputs  # unused.
    ylm_map = self.orography_module.ylm_map
    grid = ylm_map.nodal_grid
    modal_features = {'orography': self.orography_module.modal_orography}
    modal_features = {
        typing.KeyWithCosLatFactor(k, 0): v for k, v in modal_features.items()
    }
    modal_gradient_features = self.compute_gradients_transform(modal_features)
    sh_grid = ylm_map.dinosaur_grid
    sec_lat = 1 / sh_grid.cos_lat
    sec2_lat = sh_grid.sec2_lat
    lat_axis = grid.axes[1]
    sec_lat_scales = {
        0: 1,
        1: cx.field(sec_lat, lat_axis),
        2: cx.field(sec2_lat, lat_axis),
    }
    features = {}
    if self.include_raw_orography:
      all_modal_features = modal_gradient_features | modal_features
    else:
      all_modal_features = modal_gradient_features
    for k, v in all_modal_features.items():
      sec_lat_scale = sec_lat_scales[k.factor_order]
      features[k.name] = ylm_map.to_nodal(v) * sec_lat_scale
    return features


class ParamFeatures(transforms.PytreeTransformABC):
  """Returns, potentially learnable, features on fixed coordinates."""

  def __init__(
      self,
      coords: dict[str, cx.Coordinate],
      *,
      param_type: TransformParams | nnx.Param = TransformParams,
      initializer: nnx.Initializer = nnx.initializers.truncated_normal(),
      rngs: nnx.Rngs,
  ):
    self.coords = coords
    self.features = nnx.data({
        k: param_type(initializer(rngs.params(), c.shape))
        for k, c in coords.items()
    })

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    del inputs  # unused.
    return {
        k: cx.field(v[...], self.coords[k]) for k, v in self.features.items()
    }

  def update_from_xarray(
      self,
      dataset: xarray.Dataset,
      **kwargs,
  ):
    """Updates `self.features` with data from dataset."""
    sim_units = kwargs['sim_units']
    for key, feature in self.features.items():
      if key in dataset:
        da = dataset[key]
        data_units = units.parse_units(da.attrs['units'])
        da = da.copy(data=sim_units.nondimensionalize(da.values * data_units))
        candidate = xarray_utils.field_from_xarray(da)
        feature.set_value(candidate.order_as(self.coords[key]).data)
