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

"""Defines classes that implement weighting schemes for aggregation."""

from __future__ import annotations

import abc
import dataclasses
import functools

import coordax as cx
import jax.numpy as jnp
from neuralgcm.experimental.core import coordinates
import numpy as np


@dataclasses.dataclass
class Weighting(abc.ABC):
  """Abstract class for weighting."""

  @abc.abstractmethod
  def weights(
      self,
      field: cx.Field,
      field_name: str | None = None,
      context: dict[str, cx.Field] | None = None,
  ) -> cx.Field:
    """Return raw weights for a given field."""
    ...


@dataclasses.dataclass
class GridAreaWeighting(Weighting):
  """Returns weights proportional to the area of grid cells.

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

  def weights(
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
class ConstantWeighting(Weighting):
  """Applies weights specified by a constant Field.

  Attributes:
    constant: A `cx.Field` containing the weights. Its coordinates should
      be alignable with the field being weighted.
    skip_missing: If True, fields without a matching coordinate will return a
      weight of 1.0, otherwise an error is raised.
  """

  constant: cx.Field
  skip_missing: bool = True

  def weights(
      self,
      field: cx.Field,
      field_name: str | None = None,
      context: dict[str, cx.Field] | None = None,
  ) -> cx.Field:
    """Returns the user-provided weights field, optionally normalized."""
    del context  # unused.
    if all(d in field.dims for d in self.constant.dims):
      weights = self.constant
      return weights
    if self.skip_missing:
      return cx.wrap(1.0)
    else:
      raise ValueError(
          f'{field=} does not have all coordinates in {self.constant=}.'
      )


@dataclasses.dataclass
class PressureLevelAtmosphericMassWeighting(Weighting):
  """Returns weights proportional to thickness of pressure levels.

  This weighting approximates the mass of the atmosphere for a given pressure
  level assuming hydrostatic balance and standard atmospheric pressure.
  """

  standard_pressure: float = 1013.25  # standard pressure in hPa.

  def weights(
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
class ClipWeighting(Weighting):
  """Wrapper around a weighting class that clips weight values to a range."""

  weighting: Weighting
  min_val: float
  max_val: float

  def weights(
      self,
      field: cx.Field,
      field_name: str | None = None,
      context: dict[str, cx.Field] | None = None,
  ) -> cx.Field:
    """Return weights for a given field, clipped to the specified range."""
    weights = self.weighting.weights(field, field_name, context=context)
    clip = functools.partial(jnp.clip, min=self.min_val, max=self.max_val)
    clip = cx.cmap(clip)
    return clip(weights)


@dataclasses.dataclass
class PerVariableWeighting(Weighting):
  """Applies weightings from a dictionary to fields with matching names."""

  weightings_by_name: dict[str, Weighting]
  default_weighting: Weighting | None = None

  def weights(
      self,
      field: cx.Field,
      field_name: str | None = None,
      context: dict[str, cx.Field] | None = None,
  ) -> cx.Field:
    """Return weights for `field` computed by a weighting for `field_name`."""
    if field_name is None:
      raise ValueError('PerVariableWeighting requires a `field_name`.')

    weighting_instance = self.weightings_by_name.get(field_name)
    if weighting_instance is not None:
      return weighting_instance.weights(field, field_name, context)
    if self.default_weighting is not None:
      return self.default_weighting.weights(field, field_name, context)
    raise KeyError(
        f"'{field_name}' not found in weightings_by_name and no "
        'default_weighting is set.'
    )

  @classmethod
  def from_constants(
      cls,
      variable_weights: dict[str, float | cx.Field],
      default_weighting: Weighting | None = None,
  ) -> PerVariableWeighting:
    """Returns a PerVariableWeighting with ConstantWeightings."""
    weightings = {
        name: ConstantWeighting(constant=w if cx.is_field(w) else cx.wrap(w))
        for name, w in variable_weights.items()
    }
    return cls(
        weightings_by_name=weightings, default_weighting=default_weighting
    )


@dataclasses.dataclass
class CoordinateMaskWeighting(Weighting):
  """Applies a 0/1 mask based on coordinate values.

  This weighting is parameterized by a `mask_coord`. For each dimension in
  `mask_coord.dims`, it checks for a matching coordinate in the input `field`
  or, if not present on the `field`, a scalar value in the `context`.

  If a matching coordinate is found on the `field`, it returns a weight of 1.0
  for each value in that coordinate that is present in the `mask_coord`, and 0.0
  otherwise.

  If no matching coordinate is found of the `field`, the `context` is serched
  for scalar value using the dimension name, indicating in-context processing of
  `field` slices. If found, it returns a weight of 1.0 if the value from the
  context (context[dim_name]) is present in `mask_coord`, and 0.0 otherwise.

  If coordinates for multiple dimensions are found, the resulting weights are
  multiplied.

  If `skip_missing` is True, dimensions from `mask_coord` not found in the
  `field` or `context` are ignored (effectively a weight of 1.0). Otherwise,
  a ValueError is raised.
  """

  mask_coord: cx.Coordinate
  skip_missing: bool = True

  def weights(
      self,
      field: cx.Field,
      field_name: str | None = None,
      context: dict[str, cx.Field] | None = None,
  ) -> cx.Field:
    """Computes 0/1 weights based on coordinate values."""
    del field_name  # unused.
    all_scales = []
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
        scale = cx.wrap(is_present.astype(jnp.float32))
        all_scales.append(scale)
      elif in_field:
        coord_from_field = field.axes[dim_name]
        field_values = coord_from_field.fields[dim_name]
        mask_values_data = mask_values_field.untag(dim_name)
        is_present_broadcasted = field_values == mask_values_data
        scale_for_dim = cx.cmap(lambda x: x.any().astype(jnp.float32))(
            is_present_broadcasted
        )
        all_scales.append(scale_for_dim)

    if not all_scales:
      return cx.wrap(1.0)
    return functools.reduce(lambda x, y: x * y, all_scales)
