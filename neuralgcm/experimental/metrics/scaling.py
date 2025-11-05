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
from typing import Literal

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
  ) -> cx.Field | float:
    """Return scaling factor for a given field."""
    ...


@dataclasses.dataclass
class ConstantScaler(ScaleFactor):
  """Applies a constant scaling factor specified by a Field.

  Attributes:
    scale: A `cx.Field` containing the scaling factor. Its coordinates should be
      alignable with the field being scaled.
    skip_missing: If True, inputs without a matching coordinates will return a
      scale of 1.0, otherwise an error is raised.
  """

  scale: cx.Field
  skip_missing: bool = False

  def scales(
      self,
      field: cx.Field,
      field_name: str | None = None,
      context: dict[str, cx.Field] | None = None,
  ) -> cx.Field | float:
    """Returns the user-provided weights field, optionally normalized."""
    del field_name, context  # unused.
    if all(d in field.dims for d in self.scale.dims):
      return self.scale
    if self.skip_missing:
      return 1.0
    else:
      raise ValueError(
          f'{field=} does not have all coordinates in {self.scale=}.'
      )


@dataclasses.dataclass
class PerVariableScaler(ScaleFactor):
  """Applies scalers from a dictionary to fields with matching names."""

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
        f"'{field_name}' not found in scalers_by_name and no default_scaler is"
        ' set.'
    )


@dataclasses.dataclass
class WavenumberScaler(ScaleFactor):
  """Scales modal statistics by the number of modes / 4pi.

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
  ) -> cx.Field | float:
    del field_name, context  # unused.
    ylm_dims = ('longitude_wavenumber', 'total_wavenumber')
    if all(d in field.axes for d in ylm_dims):
      grid = cx.compose_coordinates(*[field.axes.get(d) for d in ylm_dims])
    else:
      grid = None

    if isinstance(grid, coordinates.SphericalHarmonicGrid):
      return grid.fields['mask'].data.sum() / (4 * np.pi)
    elif self.skip_missing:
      return 1.0
    else:
      raise ValueError(f'No SphericalHarmonicGrid on {field=}')


@dataclasses.dataclass
class MaskedScaler(ScaleFactor):
  """Masks a specified number of elements from the start or end of a dimension."""

  coord_name: str
  n_to_mask: int
  mask_position: Literal['start', 'end'] = 'start'
  masked_value: float = 0.0
  unmasked_value: float = 1.0

  def scales(
      self,
      field: cx.Field,
      field_name: str | None = None,
      context: dict[str, cx.Field] | None = None,
  ) -> cx.Field | float:
    del field_name, context  # unused.
    coord = field.axes.get(self.coord_name)
    if coord is None:
      raise ValueError(f"Dimension '{self.coord_name}' not found in field.")

    scales = np.full(coord.shape, self.unmasked_value, dtype=np.float32)
    if self.mask_position == 'start':
      scales[: self.n_to_mask] = self.masked_value
    elif self.mask_position == 'end':
      scales[-self.n_to_mask :] = self.masked_value
    else:
      raise ValueError("`mask_position` must be either 'start' or 'end'.")

    return cx.wrap(scales, coord)


@dataclasses.dataclass
class LeadTimeScaler(ScaleFactor):
  """Scales statistics by the inverse of the std of a random walk spread.

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
  ) -> cx.Field | float:
    """Computes scale factors for statistics."""
    del field_name  # unused.
    time_coord = field.axes.get('timedelta', None)
    scale_from_context = context and 'timedelta' in context
    if time_coord is None and not scale_from_context:
      if self.skip_missing:
        return 1.0
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
