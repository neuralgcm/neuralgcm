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
"""Defines `forcing` modules that produce time-dependent focing values."""

from __future__ import annotations

import functools
import logging
from typing import Any, Optional, Union

from dinosaur import coordinate_systems
from dinosaur import scales
from dinosaur import typing
import gin
import haiku as hk
import jax
import jax.numpy as jnp
from neuralgcm.legacy import transforms
import numpy as np

tree_leaves = jax.tree_util.tree_leaves
tree_map = jax.tree_util.tree_map
units = scales.units

Pytree = typing.Pytree
ForcingData = typing.ForcingData
ForcingFn = typing.ForcingFn
Forcing = typing.Forcing
TransformModule = typing.TransformModule
Quantity = units.Quantity
QuantityOrStr = Union[str, scales.Quantity]


# _FORCING_ERRORS global will store errors obtained during a io_callback.
# The user can periodically call _check_errors to see if errors have accumulated
# TODO(langmore) Use a more universal mechanism (not just in forcings.py) to
# handle errors, if we like this, then make public.
_FORCING_ERRORS = []

# pylint: disable=logging-fstring-interpolation


class ForcingDataError(Exception):
  """To raise when an error is encountered with forcing data."""


@gin.register(denylist=['coords', 'dt', 'physics_specs', 'aux_features'])
class NoForcing(hk.Module):
  """Module that returns an empty Forcing object."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      time_axis: int = 0,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    del coords, dt, physics_specs, aux_features, time_axis

  def __call__(
      self,
      forcing_data: ForcingData,
      sim_time: float,
  ) -> Forcing:
    """Returns forcings at the specified sim_time."""
    del forcing_data, sim_time
    return {}


@gin.register(denylist=['coords', 'dt', 'physics_specs', 'aux_features'])
class DynamicDataForcing(hk.Module):
  """Modules that returns forcing values by querying time-varying data.

  Input to __call__ `sim_time` must match a value in forcing_data['sim_time']
  within dt_tolerance, or else it returns nan for all pytree values.
  """

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      inputs_to_units_mapping: dict[str, str],
      forcing_transform: TransformModule = transforms.IdentityTransform,
      time_axis: int = 0,
      data_time_step: float | QuantityOrStr | None = None,
      dt_tolerance: Union[float, QuantityOrStr] = '1 hour',
      # TODO(langmore) Remove checking once bug arising from http://cl/624039690
      # is fixed.
      check_sim_time_errors: bool = False,
      name: Optional[str] = None,
  ):
    logging.info(f'[NGCM] Initializing DynamicDataForcing with {dt_tolerance=}')
    # TODO(shoyer): remove data_time_step entirely, once we're sure that no
    # saved checkpoints that we care about will break.
    del data_time_step  # no longer used
    super().__init__(name=name)
    self.time_axis = time_axis
    self.nondim_transform_fn = transforms.NondimensionalizeTransform(
        coords,
        dt,
        physics_specs,
        aux_features,
        input_coords=None,
        inputs_to_units_mapping=inputs_to_units_mapping,
    )
    self.forcing_transform_fn = forcing_transform(
        coords, dt, physics_specs, aux_features
    )
    if isinstance(dt_tolerance, (str, scales.Quantity)):
      dt_tolerance = physics_specs.nondimensionalize(
          scales.Quantity(dt_tolerance)
      )
    self.dt_tolerance = dt_tolerance
    self._check_sim_time_errors = check_sim_time_errors

  def __call__(
      self,
      forcing_data: ForcingData,
      sim_time: float,
  ) -> Forcing:
    """Returns forcings at the specified sim_time."""
    forcing_data = self.nondim_transform_fn(forcing_data)

    times = forcing_data['sim_time']
    approx_index = jnp.interp(sim_time, times, jnp.arange(times.size))
    index = jnp.round(approx_index).astype('int32')

    # Slice leaf values by index
    field_index_fn = functools.partial(
        jax.lax.dynamic_index_in_dim,
        index=index,
        axis=self.time_axis,
        keepdims=False,
    )
    _assert_no_scalars(forcing_data)
    forcing = tree_map(field_index_fn, forcing_data)

    # Replace leaf values with nan if forcing['sim_time'] does not match
    # the requested sim_time value within dt_tolerance.
    abs_error = jnp.abs(forcing['sim_time'] - sim_time)
    is_valid = abs_error < self.dt_tolerance
    forcing = jax.tree_util.tree_map(
        lambda x: jnp.where(is_valid, x, jnp.nan), forcing
    )

    # Also add errors (if any) to _FORCING_ERRORS so _check_errors can be called
    # to raise.
    if self._check_sim_time_errors:
      jax.experimental.io_callback(
          _check_sim_time_close_to_forcing_sim_time,
          None,  # Returns None
          sim_time=sim_time,
          forcing_sim_time=forcing['sim_time'],
          tolerance=self.dt_tolerance,
      )
    return self.forcing_transform_fn(forcing)


@gin.register(denylist=['coords', 'dt', 'physics_specs', 'aux_features'])
class PersistenceDataForcing(hk.Module):
  """Modules that returns forcing using first time index of forcing_data."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      inputs_to_units_mapping: dict[str, str],
      forcing_transform: TransformModule = transforms.IdentityTransform,
      time_axis: int = 0,
      name: Optional[str] = None,
  ):
    logging.info('[NGCM] Initializing PersistenceDataForcing')
    super().__init__(name=name)
    self.time_axis = time_axis
    self.nondim_transform_fn = transforms.NondimensionalizeTransform(
        coords,
        dt,
        physics_specs,
        aux_features,
        input_coords=None,
        inputs_to_units_mapping=inputs_to_units_mapping,
    )
    self.forcing_transform_fn = forcing_transform(
        coords, dt, physics_specs, aux_features
    )

  def __call__(
      self,
      forcing_data: ForcingData,
      sim_time: float,
  ) -> Forcing:
    """Returns forcings from the first time index of sim_time."""
    del sim_time  # unused
    forcing_data = self.nondim_transform_fn(forcing_data)
    idx = 0

    # Slice leaf values by index
    field_index_fn = functools.partial(
        jax.lax.dynamic_index_in_dim,
        index=idx,
        axis=self.time_axis,
        keepdims=False,
    )
    _assert_no_scalars(forcing_data)
    forcing = tree_map(field_index_fn, forcing_data)
    return self.forcing_transform_fn(forcing)


@gin.register(denylist=['coords', 'dt', 'physics_specs', 'aux_features'])
class IncrementSSTForcingTransform(hk.Module):
  """Transform Forcing by uniformly incrementing sea surface temperature."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      temperature_change: Quantity,
      key: str = 'sea_surface_temperature',
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    del coords, dt, aux_features  # unused
    self.temperature_change = physics_specs.nondimensionalize(
        units.Quantity(temperature_change)
    )
    self.key = key

  def __call__(self, forcing: Forcing) -> Forcing:
    assert isinstance(forcing, dict)
    forcing = forcing.copy()
    forcing[self.key] = forcing[self.key] + self.temperature_change
    return forcing


def _assert_no_scalars(tree: Pytree):
  dims = tree_map(lambda x: len(jnp.shape(x)), tree)
  if not all(d > 0 for d in tree_leaves(dims)):
    raise ValueError(f'Scalar shapes encountered: {dims=}')


# TODO(langmore) Use a more universal mechanism (not just in forcings.py) to
# handle errors, if we like this, then make public.
def _check_sim_time_close_to_forcing_sim_time(
    sim_time: np.ndarray,
    forcing_sim_time: np.ndarray,
    tolerance: float,
) -> None:
  """Checks |sim_time - forcing_sim_time| < tolerance add to _FORCING_ERRORS."""
  abs_error = np.abs(forcing_sim_time - sim_time)
  if abs_error < tolerance:
    return
  err_msg = (
      f'{sim_time=} differed from {forcing_sim_time=} by {abs_error=} which is '
      f'> {tolerance=}'
  )
  _FORCING_ERRORS.append(err_msg)


# TODO(langmore) Use a more universal mechanism (not just in forcings.py) to
# handle errors, if we like this, then make public.
def _check_errors(  # pylint: disable=dangerous-default-value
    max_to_print: int = 4,
    err_list: list[str] = _FORCING_ERRORS,
) -> None:
  """Check err_list and raise ForcingDataError if nonempty."""
  n_err = len(err_list)
  if n_err:
    raise ForcingDataError(
        f'ForcingDataError found: {n_err} exceptions: '
        f'The first {min(n_err, max_to_print)} are: '
        f'{", ".join(err_list[:max_to_print])}'
    )
