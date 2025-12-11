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

"""Modules for interpolating between atmosphere specific coordinates."""

import dataclasses
from typing import Literal, Sequence
import coordax as cx
from flax import nnx
import jax
import jax.numpy as jnp
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import nnx_compat
from neuralgcm.experimental.core import typing
from neuralgcm.experimental.core import units


def _linear_interp_with_linear_extrap(x, xp, fp):
  """Linear interpolation with unlimited linear extrapolation at each end."""
  n = len(xp)
  i = jnp.arange(n)
  dx = xp[1:] - xp[:-1]
  delta = x - xp[:-1]
  w = delta / dx
  w_left = jnp.pad(1 - w, [(0, 1)])
  w_right = jnp.pad(w, [(1, 0)])
  u = jnp.searchsorted(xp, x, side='right', method='compare_all')
  u = jnp.clip(u, 1, n - 1)
  weights = w_left * (i == (u - 1)) + w_right * (i == u)
  return jnp.dot(weights, fp, precision='highest')


def _dot_interp(x, xp, fp):
  """Interpolate with a dot product instead of indexing."""
  n = len(xp)
  i = jnp.arange(n)
  dx = xp[1:] - xp[:-1]
  delta = x - xp[:-1]
  w = delta / dx
  w_left = jnp.pad(1 - w, [(0, 1)])
  w_right = jnp.pad(w, [(1, 0)])
  u = jnp.searchsorted(xp, x, side='right', method='compare_all')
  u = jnp.clip(u, 1, n - 1)
  weights = w_left * (i == (u - 1)) + w_right * (i == u)
  weights = jnp.where(x < xp[0], i == 0, weights)
  weights = jnp.where(x > xp[-1], i == (n - 1), weights)
  return jnp.dot(weights, fp, precision='highest')


def _interp(x, xp, fp):
  """Optimized version of jnp.interp."""
  return jax.lax.platform_dependent(
      x, xp, fp, tpu=_dot_interp, default=jnp.interp
  )


def _interval_overlap(
    source_bounds: jnp.ndarray, target_bounds: jnp.ndarray
) -> jnp.ndarray:
  """Calculate the interval overlap between grid cells."""
  # based on https://gist.github.com/shoyer/c0f1ddf409667650a076c058f9a17276
  upper = jnp.minimum(
      target_bounds[1:, jnp.newaxis], source_bounds[jnp.newaxis, 1:]
  )
  lower = jnp.maximum(
      target_bounds[:-1, jnp.newaxis], source_bounds[jnp.newaxis, :-1]
  )
  return jnp.maximum(upper - lower, 0)


def conservative_regrid_weights(
    source_bounds: jnp.ndarray, target_bounds: jnp.ndarray
) -> jnp.ndarray:
  """Create a weight matrix for conservative regridding on pressure levels."""
  weights = _interval_overlap(source_bounds, target_bounds)
  weights /= jnp.sum(weights, axis=1, keepdims=True)
  return weights


@nnx_compat.dataclass
class LinearOnPressure(nnx.Module):
  """Regridder that linearly interpolates between vertical levels using pressure.

  This module interpolates fields from either sigma or pressure coordinates to a
  target set of vertical levels (sigma or pressure). The interpolation is
  performed linearly with respect to pressure.

  Attributes:
    target_levels: The vertical levels to interpolate to.
    extrapolation: The extrapolation method to use. One of 'linear','constant'.
    include_surface_pressure_in_output: Whether to forward the surface pressure.
    sim_units: An optional `SimUnits` instance. If provided, inputs are
      interpreted as nondimensionalized and appropriate coordinate
      nondimensionalization is performed. If None, assumes that surface pressure
      is provided in Pascals.
    allow_no_levels: If True, fields without vertical levels are passed through.
      If False, an error is raised.
    supported_level_types: A sequence of supported vertical coordinate types.
  """

  target_levels: (
      coordinates.SigmaLevels
      | coordinates.PressureLevels
      | coordinates.HybridLevels
  )
  extrapolation: Literal['linear', 'constant'] = 'linear'
  include_surface_pressure_in_output: bool = False
  sim_units: units.SimUnits | None = None
  allow_no_levels: bool = False
  supported_level_types: Sequence[cx.Coordinate] = dataclasses.field(
      default=(
          coordinates.PressureLevels,
          coordinates.SigmaLevels,
          coordinates.HybridLevels,
      ),
      kw_only=True,
  )

  def interpolate_array(self, x, xp, fp):
    # x - target pressure; xp - data pressure level, fp - value at said level.
    if self.extrapolation == 'linear':
      return _linear_interp_with_linear_extrap(x, xp, fp)
    elif self.extrapolation == 'constant':
      return _interp(x, xp, fp)
    else:
      raise ValueError(f'Unknown extrapolation method "{self.extrapolation}".')

  def interpolate_f(self, f: cx.Field, surface_pressure: cx.Field) -> cx.Field:
    """Interpolate field `f` to `self.target_levels` levels."""
    canonical = cx.canonicalize_coordinates(f.coordinate)
    levels = [c for c in canonical if type(c) in self.supported_level_types]
    if not levels:
      if self.allow_no_levels:
        return f
      else:
        raise ValueError(
            'No vertical levels of supported type'
            f' "{(self.supported_level_types)}" found on {f=}.'
        )
    levels = set(levels)
    if len(levels) > 1:
      raise ValueError(
          'Multiple levels of supported type'
          f' "{self.supported_level_types}" found on {f=}.'
      )
    [level] = list(levels)
    if isinstance(level, coordinates.PressureLevels):
      pressure = level.fields['pressure'] * 100  # In Pascal.
      if self.sim_units is not None:
        assert isinstance(self.sim_units, units.SimUnits)
        pressure = cx.wrap(
            self.sim_units.nondimensionalize(pressure.data * typing.units.Pa),
            level,
        )
    elif isinstance(level, coordinates.SigmaLevels):
      if surface_pressure is None:
        raise ValueError(
            '`surface_pressure` must be provided when interpolating from sigma'
        )
      pressure = level.fields['sigma'] * surface_pressure
    elif isinstance(level, coordinates.HybridLevels):
      if surface_pressure is None:
        raise ValueError(
            '`surface_pressure` must be provided when interpolating from hybrid'
        )
      pressure = level.pressure_centers(surface_pressure, self.sim_units)
    else:
      raise ValueError(f'Unsupported level type {type(level)}.')

    target_levels = self.target_levels
    if isinstance(target_levels, coordinates.SigmaLevels):
      if surface_pressure is None:
        raise ValueError(
            'Missing `surface_pressure` needed when interpolating to sigma.'
        )
      desired = target_levels.fields['sigma'] * surface_pressure
    elif isinstance(target_levels, coordinates.HybridLevels):
      if surface_pressure is None:
        raise ValueError(
            'Missing `surface_pressure` needed when interpolating to hybrid.'
        )
      desired = target_levels.pressure_centers(surface_pressure, self.sim_units)
    elif isinstance(target_levels, coordinates.PressureLevels):
      desired = target_levels.fields['pressure'] * 100  # In Pascal.
      if self.sim_units is not None:
        assert isinstance(self.sim_units, units.SimUnits)
        desired = cx.wrap(
            self.sim_units.nondimensionalize(desired.data * typing.units.Pa),
            target_levels,
        )
    else:
      raise ValueError(f'Unsupported {type(target_levels)=}.')
    out_coord = cx.replace_axes_in_coordinate(
        f.coordinate, level, target_levels
    )
    # we specify out_axes to preserve the dimension order in the output.
    out_axes = {d: i for i, d in enumerate(out_coord.dims)}
    regrid_fn = cx.cmap(self.interpolate_array, out_axes)
    return regrid_fn(desired, pressure.untag(level), f.untag(level))

  def __call__(
      self,
      inputs: dict[str, cx.Field],
  ) -> dict[str, cx.Field]:
    inputs = inputs.copy()  # avoid mutating inputs.
    surface_pressure = inputs.pop('surface_pressure', None)
    out = {
        k: self.interpolate_f(v, surface_pressure) for k, v in inputs.items()
    }
    if self.include_surface_pressure_in_output and surface_pressure is not None:
      out['surface_pressure'] = surface_pressure
    return out


@nnx_compat.dataclass
class ConservativeOnPressure(nnx.Module):
  """Regridder that interpolates vertically to sigma levels via pressure.

  This module performs conservative regridding of fields from either sigma or
  pressure coordinates to a target set of sigma levels. The regridding is
  conservative with respect to pressure.

  Attributes:
    target_levels: The target sigma levels to regrid to.
    include_surface_pressure_in_output: Whether to forward the surface pressure.
    sim_units: An optional `SimUnits` instance. If provided, inputs are
      interpreted as nondimensionalized and appropriate coordinate
      nondimensionalization is performed. If None, assumes that surface pressure
      is provided in Pascals.
    allow_no_levels: If True, fields without vertical levels are passed through.
      If False, an error is raised.
    supported_level_types: A sequence of supported vertical coordinate types.
  """

  target_levels: coordinates.SigmaLevels
  include_surface_pressure_in_output: bool = False
  sim_units: units.SimUnits | None = None
  allow_no_levels: bool = False
  supported_level_types: Sequence[cx.Coordinate] = dataclasses.field(
      default=(coordinates.PressureLevels, coordinates.SigmaLevels),
      kw_only=True,
  )

  def regrid_array(self, target_pressure_bounds, source_pressure_bounds, fp):
    weights = conservative_regrid_weights(
        source_pressure_bounds, target_pressure_bounds
    )
    # using `nan_to_num` to handle cases where weights are NaN due to zero sum.
    weights = jnp.nan_to_num(weights)
    return jnp.einsum('ab,b->a', weights, fp, precision='float32')

  def interpolate_f(self, f: cx.Field, surface_pressure: cx.Field) -> cx.Field:
    """Interpolate field `f` to `self.sigma` levels."""
    canonical = cx.canonicalize_coordinates(f.coordinate)
    levels = [c for c in canonical if type(c) in self.supported_level_types]
    if not levels:
      if self.allow_no_levels:
        return f
      else:
        raise ValueError(
            'No vertical levels of supported type'
            f' f{(self.supported_level_types)} found on {f=}.'
        )
    levels = set(levels)
    if len(levels) > 1:
      raise ValueError(
          'Multiple levels of supported type'
          f' {self.supported_level_types} found on {f=}.'
      )
    [level] = list(levels)

    if isinstance(level, coordinates.PressureLevels):
      centers = level.fields['pressure'].data * 100  # In Pascal.
      if self.sim_units is not None:
        assert isinstance(self.sim_units, units.SimUnits)
        centers = self.sim_units.nondimensionalize(
            centers * typing.units.Pa
        )
      midpoints = (centers[:-1] + centers[1:]) / 2
      first = centers[0] - (centers[1] - centers[0]) / 2
      last = centers[-1] + (centers[-1] - centers[-2]) / 2
      boundaries = jnp.concatenate(
          [jnp.array([first]), midpoints, jnp.array([last])]
      )
      axis = cx.SizedAxis(f'{level.dims[0]}_boundaries', boundaries.shape[0])
      source_bounds = cx.wrap(boundaries, axis)
    elif isinstance(level, coordinates.SigmaLevels):
      axis = level.to_sigma_boundaries()
      source_bounds = level.fields['sigma_boundaries'] * surface_pressure
    else:
      raise ValueError(f'Unsupported level type {type(level)}.')

    sigma_b = self.target_levels.to_sigma_boundaries()
    target_bounds = sigma_b.fields['sigma_boundaries'] * surface_pressure

    out_axes = {
        d: i for i, d in enumerate(f.coordinate.dims) if d not in level.dims
    }
    regrid_fn = cx.cmap(self.regrid_array, out_axes=out_axes)
    result = regrid_fn(
        target_bounds.untag(sigma_b),
        source_bounds.untag(axis),
        f.untag(level),
    )
    return result.tag(self.target_levels)

  def __call__(
      self,
      inputs: dict[str, cx.Field],
  ) -> dict[str, cx.Field]:
    inputs = inputs.copy()  # avoid mutating inputs.
    if 'surface_pressure' not in inputs:
      raise ValueError(
          f'`surface_pressure` must be provided, got {inputs.keys()}.'
      )
    surface_pressure = inputs.pop('surface_pressure')
    out = {
        k: self.interpolate_f(v, surface_pressure) for k, v in inputs.items()
    }
    if self.include_surface_pressure_in_output:
      out['surface_pressure'] = surface_pressure
    return out


def get_surface_pressure(
    geopotential: cx.Field,
    geopotential_at_surface: cx.Field,
    sim_units: units.SimUnits | None = None,
) -> cx.Field:
  """Computes surface pressure from geopotential on pressure levels.

  Args:
    geopotential: geopotential Field with pressure level axis.
    geopotential_at_surface: geopotential at the surface.
    sim_units: optional object describing simulation units. If provided, the
      returned surface pressure will be nondimensionalized. Otherwise returns
      surface pressure in Pascals.

  Returns:
    Surface pressure field on a horizontal grid.
  """
  canonical = cx.canonicalize_coordinates(geopotential.coordinate)
  levels = [c for c in canonical if isinstance(c, coordinates.PressureLevels)]
  if len(levels) != 1:
    raise ValueError(
        f'geopotential must have exactly 1 pressure-like coord, got {levels}'
    )
  [levels] = levels
  pressure = levels.fields['pressure'] * 100  # In Pascal.
  if sim_units is not None:
    assert isinstance(sim_units, units.SimUnits)
    pressure = cx.wrap(
        sim_units.nondimensionalize(pressure.data * typing.units.Pa),
        levels,
    )

  relative_heights = geopotential_at_surface - geopotential

  # find pressure where at relative_height == 0.
  def find_intercept(relative_height, pressure):
    return _linear_interp_with_linear_extrap(0.0, relative_height, pressure)

  out_coord = cx.replace_axes_in_coordinate(
      geopotential.coordinate, levels, cx.Scalar()  # levels are reduced.
  )
  out_axes = {d: i for i, d in enumerate(out_coord.dims)}
  surface_pressure = cx.cmap(find_intercept, out_axes=out_axes)(
      relative_heights.untag(levels), pressure.untag(levels)
  )
  return surface_pressure
