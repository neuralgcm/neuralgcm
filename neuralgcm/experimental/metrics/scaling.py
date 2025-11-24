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

"""Defines classes that implement rescaling schemes for statistics."""

from __future__ import annotations

import abc
import dataclasses
import functools

import coordax as cx
import jax.numpy as jnp
from neuralgcm.experimental.core import coordinates
import numpy as np


@dataclasses.dataclass
class ScaleFactor(abc.ABC):
  """Abstract class for scaling statistics.

  Scalers are applied to statistics before they are weighted and aggregated.
  This allows for controlling the scale of metric entrees that may depend on
  the their coordinate values, typically associated with spatial or temporal
  coordinates. Whereas `weighting.Weighting` keeps track of the relative
  statistical weights, `Scaler`s allow transforming the statistics. This is
  particularly useful for computing loss terms.

  `scales` are computed based on the input `field`, `field_name` and, in cases
  when the required coordinates are not present on the `field`, scalar
  coordinate values from the `context`. The latter indicates processing of
  statistics along dimensions one slice at a time.
  """

  @abc.abstractmethod
  def scales(
      self,
      field: cx.Field,
      field_name: str | None = None,
      context: dict[str, cx.Field] | None = None,
  ) -> cx.Field:
    """Return scaling factor for a given field."""
    ...


@dataclasses.dataclass
class ConstantScaler(ScaleFactor):
  """ScaleFactor that returns scales equal to a user-provided constant.

  Attributes:
    constant: A `cx.Field` containing the scaling factor. Its coordinates should
      be alignable with the field being scaled.
    skip_missing: If True, inputs without a matching coordinates will return a
      scale of 1.0, otherwise an error is raised.
  """

  constant: cx.Field
  skip_missing: bool = True

  def scales(
      self,
      field: cx.Field,
      field_name: str | None = None,
      context: dict[str, cx.Field] | None = None,
  ) -> cx.Field:
    """Returns the user-provided constant field for scaling."""
    del field_name, context  # unused.
    if all(d in field.dims for d in self.constant.dims):
      return self.constant
    if self.skip_missing:
      return cx.wrap(1.0)
    else:
      raise ValueError(
          f'{field=} does not have all coordinates in {self.constant=}.'
      )


@dataclasses.dataclass
class PerVariableScaler(ScaleFactor):
  """ScaleFactor that returns scales from `scalers_by_name[field_name]`."""

  scalers_by_name: dict[str, ScaleFactor]
  default_scaler: ScaleFactor | None = None

  def scales(
      self,
      field: cx.Field,
      field_name: str | None = None,
      context: dict[str, cx.Field] | None = None,
  ) -> cx.Field | float:
    """Return scales for `field` computed by a scaler for `field_name`."""
    if field_name is None:
      raise ValueError('PerVariableScaler requires a `field_name`.')

    scaler = self.scalers_by_name.get(field_name)
    if scaler is not None:
      return scaler.scales(field, field_name, context)

    if self.default_scaler is not None:
      return self.default_scaler.scales(field, field_name, context)

    raise KeyError(
        f'"{field_name}" not found in scalers_by_name and no default_scaler is'
        ' set.'
    )


@dataclasses.dataclass
class GridAreaScaler(ScaleFactor):
  """ScaleFactor that returns scales proportional to the area of grid cells.

  This weighting works with both `LonLatGrid` and `SphericalHarmonicGrid`.

  For `LonLatGrid`, weights are approximated by cos(lat), which are proportional
  to the proper quadrature weights of Gaussian grids. This ensures that grid
  cells near the poles have smaller weights than those near the equator.

  For `SphericalHarmonicGrid`, the basis functions are orthonormal, so uniform
  weights (1.0) are returned.

  If skip_missing attribute is set to True, fields without a grid will return
  a weight of 1.0, otherwise an error is raised.
  """

  skip_missing: bool = True

  def scales(
      self,
      field: cx.Field,
      field_name: str | None = None,
      context: dict[str, cx.Field] | None = None,
  ) -> cx.Field:
    del context  # unused.
    lon_lat_dims = ('longitude', 'latitude')
    ylm_dims = ('longitude_wavenumber', 'total_wavenumber')
    if all(d in field.axes for d in lon_lat_dims):
      grid = cx.compose_coordinates(*[field.axes.get(d) for d in lon_lat_dims])
    elif all(d in field.axes for d in ylm_dims):
      grid = cx.compose_coordinates(*[field.axes.get(d) for d in ylm_dims])
    else:
      grid = None

    if isinstance(grid, coordinates.LonLatGrid):

      def get_weight(x):
        # Latitudes are in degrees, convert to radians for cosine.
        lat = jnp.deg2rad(x)
        pi_over_2 = jnp.array([np.pi / 2])
        lat_cell_bounds = jnp.concatenate(
            [-pi_over_2, (lat[:-1] + lat[1:]) / 2, pi_over_2]
        )
        upper = lat_cell_bounds[1:]
        lower = lat_cell_bounds[:-1]
        return jnp.sin(upper) - jnp.sin(lower)

      get_weight = cx.cmap(get_weight)
      lats = grid.fields['latitude']
      lat_ax = lats.coordinate
      weights = get_weight(grid.fields['latitude'].untag(lat_ax)).tag(lat_ax)
      weights = weights.broadcast_like(grid)
    elif isinstance(grid, coordinates.SphericalHarmonicGrid):
      # avoid counting padding towards overall weight by using mask.
      weights = grid.fields['mask'].astype(jnp.float32)
    else:
      if self.skip_missing:
        weights = cx.wrap(1.0)
      else:
        raise ValueError(f'No LonLatGrid or SphericalHarmonicGrid on {field=}')
    return weights


@dataclasses.dataclass
class PressureLevelAtmosphericMassScaler(ScaleFactor):
  """ScaleFactor that returns scales proportional to pressure level thickness.

  This scaling results in statistics that upon the sum would represent a
  discrete integral over pressure (or approximate mass intergral) of the
  corresponding quantity.
  """

  standard_pressure: float = 1013.25  # standard pressure in hPa.

  def scales(
      self,
      field: cx.Field,
      field_name: str | None = None,
      context: dict[str, cx.Field] | None = None,
  ) -> cx.Field:
    """Return weights extracted from the pressure level coordinate."""
    del field_name, context  # unused.
    if 'pressure' not in field.dims:
      return cx.wrap(1.0)

    pressure = field.axes['pressure']
    padded = np.concatenate([
        np.asarray([0.0]),
        pressure.centers,
        np.asarray([self.standard_pressure]),
    ])
    # thickness is estimated as 0.5 * |p_{k+1} - p_{k-1}|.
    thickness = (np.roll(padded, -1) - np.roll(padded, 1))[1:-1] / 2
    return cx.wrap(thickness, pressure)


@dataclasses.dataclass
class WavenumberScaler(ScaleFactor):
  """ScaleFactor that returns fixed scale equal to "number of ylm_modes" / 4pi.

  For fields with `SphericalHarmonicGrid` dimension, this scaler rescales the
  statistics by the number of spherical harmonic modes divided by 4pi. This
  rescaling modifies the statistics from computing per-mode mean that is
  resolution-dependent to spatial mean that is resolution-invariant.

  Attributes:
    skip_missing: If True, fields without a grid will return a scale of 1.0.
  """

  skip_missing: bool = True

  def scales(
      self,
      field: cx.Field,
      field_name: str | None = None,
      context: dict[str, cx.Field] | None = None,
  ) -> cx.Field:
    del field_name, context  # unused.
    ylm_dims = ('longitude_wavenumber', 'total_wavenumber')
    if all(d in field.axes for d in ylm_dims):
      grid = cx.compose_coordinates(*[field.axes.get(d) for d in ylm_dims])
    else:
      grid = None

    if isinstance(grid, coordinates.SphericalHarmonicGrid):
      return cx.wrap(grid.fields['mask'].data.sum() / (4 * np.pi))
    elif self.skip_missing:
      return cx.wrap(1.0)
    else:
      raise ValueError(f'No SphericalHarmonicGrid on {field=}')


@dataclasses.dataclass
class CoordinateMaskScaler(ScaleFactor):
  """ScaleFactor that that returns masked/unmasked scales based on masking.

  This scaler is parameterized by a `mask_coord`. For each dimension in
  `mask_coord.dims`, it checks for a matching coordinate in the input `field`
  or, if not present on the `field`, a scalar value in the `context`.

  If a matching coordinate is found on the `field`, it returns `masked_value`
  for each value in that coordinate that is present in the `mask_coord`, and
  `unmasked_value` otherwise.

  If no matching coordinate is found of the `field`, the `context` is searched
  for scalar value using the dimension name, indicating in-context processing
  of `field` slices. If found, it returns `masked_value` if the value from the
  context (context[dim_name]) is present in `mask_coord`, and `unmasked_value`
  otherwise.

  If coordinates for multiple dimensions are found, the resulting masks are
  multiplied.

  If `skip_missing` is True, dimensions from `mask_coord` not found in the
  `field` or `context` are ignored (effectively resulting in `unmasked_value`).
  Otherwise, a ValueError is raised.
  """

  mask_coord: cx.Coordinate
  masked_value: float = 0.0
  unmasked_value: float = 1.0
  skip_missing: bool = True

  def scales(
      self,
      field: cx.Field,
      field_name: str | None = None,
      context: dict[str, cx.Field] | None = None,
  ) -> cx.Field:
    """Computes scales based on coordinate values."""
    del field_name  # unused.
    all_masks = []
    for dim_name in self.mask_coord.dims:
      in_context = context and dim_name in context
      in_field = dim_name in field.axes

      if in_context and in_field:
        raise ValueError(
            f'Coordinate for {dim_name!r} found both in context and field.'
        )

      if not in_context and not in_field:
        if self.skip_missing:
          continue
        raise ValueError(
            f'Coordinate for {dim_name!r} not found on {field=} or in context.'
        )

      mask_values_field = self.mask_coord.fields[dim_name]
      if in_context:
        current_value = context[dim_name]
        if current_value.ndim != 0:
          raise ValueError(
              f'Expected scalar {dim_name!r} in context, got '
              f'{current_value.shape=}'
          )
        mask_values = mask_values_field.untag(dim_name)
        is_present = (current_value == mask_values).data.any()
        mask = cx.wrap(is_present)
        all_masks.append(mask)
      elif in_field:
        coord_from_field = field.axes[dim_name]
        field_values = coord_from_field.fields[dim_name]
        mask_values_data = mask_values_field.untag(dim_name)
        is_present_broadcasted = field_values == mask_values_data
        mask_for_dim = cx.cmap(lambda x: x.any())(is_present_broadcasted)
        all_masks.append(mask_for_dim)

    if not all_masks:
      return cx.wrap(self.unmasked_value)

    final_mask = functools.reduce(lambda x, y: x & y, all_masks)
    return cx.cmap(
        lambda x: jnp.where(
            x, self.masked_value, self.unmasked_value
        ).astype(jnp.float32)
    )(final_mask)


@dataclasses.dataclass
class LeadTimeScaler(ScaleFactor):
  """ScaleFactor that returns scales equal to 1/(std of a random walk spread).

  It computes scales that are inversely proportional to the anticipated standard
  deviation of errors at a given lead time, assuming random-walk-like error
  growth. The weights are derived from the `TimeDelta` coordinate of the input
  field or `context` when producing weights for statistics slices along the
  `timedelta` dimension.

  Attributes:
    base_squared_error_in_hours: Number of hours before assumed variance starts
      growing (almost) linearly.
    asymptotic_squared_error_in_hours: Number of hours before assumed variance
      slows its growth. Set to None (the default) if variance grows
      indefinitely.
    normalize_weights: Whether to normalizing scaling factors such that the
      square of all of the weights add up to 1.
    skip_missing: If True, fields without a matching coordinate will return a
      scale of 1.0, otherwise an error is raised.
    weights_power: Optional power to which to raise the weights. Can be used to
      scale statistics that grow faster with leadtime (e.g. SquaredError).
  """

  base_squared_error_in_hours: float
  asymptotic_squared_error_in_hours: float | None = None
  normalize_weights: bool = True
  skip_missing: bool = True
  weights_power: float | None = None

  def scales(
      self,
      field: cx.Field,
      field_name: str | None = None,
      context: dict[str, cx.Field] | None = None,
  ) -> cx.Field:
    """Computes scale factors for statistics."""
    del field_name  # unused.
    time_coord = field.axes.get('timedelta', None)
    scale_from_context = context and 'timedelta' in context
    if time_coord is None and not scale_from_context:
      if self.skip_missing:
        return cx.wrap(1.0)
      raise ValueError(
          f'TimeDelta coordinate not found on {field=} or in context'
      )

    if scale_from_context:
      time_field = context['timedelta']
      if time_field.ndim != 0:
        raise ValueError(
            f'Expected scalar timedelta in context, got {time_field.shape=}'
        )
      t = time_field.data / np.timedelta64(1, 'h')
    else:
      t = time_coord.deltas / np.timedelta64(1, 'h')

    if self.asymptotic_squared_error_in_hours is not None:
      t = t / (1 + t / self.asymptotic_squared_error_in_hours)
    # Variance is assumed to grow linearly with our transformed time `t`.
    # Weights are inverse of standard deviation.
    inv_variance = 1 / (1 + t / self.base_squared_error_in_hours)
    if self.normalize_weights:
      if not scale_from_context:
        inv_variance /= inv_variance.sum()
      else:
        raise ValueError(
            'Normalizing weights is not supported when scales are computed from'
            ' context.'
        )
    inv_variance_sqrt = jnp.sqrt(inv_variance)
    if self.weights_power is not None:
      inv_variance_sqrt = inv_variance_sqrt ** self.weights_power
    if scale_from_context:
      return cx.wrap(inv_variance_sqrt)
    return cx.wrap(inv_variance_sqrt, time_coord)
