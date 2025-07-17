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
import coordax as cx
import jax.numpy as jnp
from neuralgcm.experimental.core import coordinates
import numpy as np


@dataclasses.dataclass
class Weighting(abc.ABC):
  """Abstract class for weighting."""

  @abc.abstractmethod
  def weights(
      self, field: cx.Field, field_name: str | None = None
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
      self, field: cx.Field, field_name: str | None = None
  ) -> cx.Field:
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
      lat_ax = cx.get_coordinate(lats)
      weights = get_weight(grid.fields['latitude'].untag(lat_ax)).tag(lat_ax)
      dummy = cx.wrap(np.ones(grid.shape), grid)
      weights = weights.broadcast_like(dummy)
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
class FieldWeighting(Weighting):
  """Applies weights specified by a given Field.

  Attributes:
    custom_weights: A `cx.Field` containing the weights. Its coordinates should
      be alignable with the field being weighted.
    skip_missing: If True, fields without a matching coordinate will return a
      weight of 1.0, otherwise an error is raised.
  """

  custom_weights: cx.Field
  skip_missing: bool = True

  def weights(
      self, field: cx.Field, field_name: str | None = None
  ) -> cx.Field:
    """Returns the user-provided weights field, optionally normalized."""
    if all(d in field.dims for d in self.custom_weights.dims):
      weights = self.custom_weights
      return weights
    if self.skip_missing:
      return cx.wrap(1.0)
    else:
      raise ValueError(
          f'{field=} does not have all coordinates in {self.custom_weights=}.'
      )


@dataclasses.dataclass
class TimeDeltaWeighting(Weighting):
  """Time scaling that assumes error grows like a random walk.

  This is analogous to `linear_transforms.TimeRescaling`. It computes weights
  that are inversely proportional to the anticipated standard deviation of
  errors at a given lead time, assuming random-walk-like error growth. The
  weights are derived from the `TimeDelta` coordinate of the input field.

  Attributes:
    base_squared_error_in_hours: Number of hours before assumed variance starts
      growing (almost) linearly.
    asymptotic_squared_error_in_hours: Number of hours before assumed variance
      slows its growth. Set to None (the default) if variance grows
      indefinitely.
    skip_missing: If True, fields without a matching coordinate will return a
      weight of 1.0, otherwise an error is raised.
  """

  base_squared_error_in_hours: float
  asymptotic_squared_error_in_hours: float | None = None
  skip_missing: bool = True

  def weights(
      self, field: cx.Field, field_name: str | None = None
  ) -> cx.Field:
    """Computes weights based on the TimeDelta coordinate of the field."""
    time_coord = field.axes.get('timedelta', None)

    if time_coord is None and self.skip_missing:
      return cx.wrap(1.0)
    if not isinstance(time_coord, coordinates.TimeDelta):
      raise ValueError(f'TimeDelta coordinate not found on {field=}')
    # deltas are in timedelta64[s]. Convert to hours.
    t = time_coord.deltas / np.timedelta64(1, 'h')
    if self.asymptotic_squared_error_in_hours is not None:
      t = t / (1 + t / self.asymptotic_squared_error_in_hours)

    # Variance is assumed to grow linearly with our transformed time `t`.
    # Weights are inverse of standard deviation.
    inv_variance = 1 / (1 + t / self.base_squared_error_in_hours)
    weights_data = np.sqrt(inv_variance)
    return cx.wrap(weights_data, time_coord)


@dataclasses.dataclass
class PerVariableWeighting(Weighting):
  """Applies weights from a dictionary on a per-variable basis."""

  variable_weights: dict[str, float | int | cx.Field]

  def weights(
      self, field: cx.Field, field_name: str | None = None
  ) -> cx.Field:
    """Return weights for a given field, looked up by field_name."""
    if field_name is None:
      raise ValueError('PerVariableWeighting requires a `field_name`.')
    weight = self.variable_weights[field_name]
    if isinstance(weight, (int, float)):
      return cx.wrap(float(weight))
    return weight
