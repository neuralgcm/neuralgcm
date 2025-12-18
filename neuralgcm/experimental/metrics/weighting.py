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
from neuralgcm.experimental.metrics import scaling


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
class WeightingFromScaler(Weighting, scaling.ScaleFactor, abc.ABC):
  """A Weighting class that derives its weights from a ScaleFactor.

  This class acts as a bridge, allowing any `scaling.ScaleFactor` to be used
  as a `weighting.Weighting`. Subclasses should inherit from this class and a
  concrete `ScaleFactor` implementation. The `weights` method is automatically
  implemented by calling the `scales` method from the `ScaleFactor` API.
  """

  def weights(
      self,
      field: cx.Field,
      field_name: str | None = None,
      context: dict[str, cx.Field] | None = None,
  ) -> cx.Field:
    """Returns weights computed by the underlying `scales` method."""
    return self.scales(field, field_name, context)


@dataclasses.dataclass
class GridAreaWeighting(WeightingFromScaler, scaling.GridAreaScaler):
  """Weighting that returns weights proportional to the area of grid cells.

  This weighting works with both `LonLatGrid` and `SphericalHarmonicGrid`.

  For `LonLatGrid`, weights are approximated by cos(lat), which are proportional
  to the proper quadrature weights of Gaussian grids. This ensures that grid
  cells near the poles have smaller weights than those near the equator.

  For `SphericalHarmonicGrid`, the basis functions are orthonormal, so uniform
  weights (1.0) are returned.

  If skip_missing attribute is set to True, fields without a grid will return
  a weight of 1.0, otherwise an error is raised.
  """


@dataclasses.dataclass
class ConstantWeighting(WeightingFromScaler, scaling.ConstantScaler):
  """Weighting that returns user-provided constant weights.

  Attributes:
    constant: A `cx.Field` containing the weights. Its coordinates should be
      alignable with the field being weighted.
    skip_missing: If True, fields without a matching coordinate will return a
      weight of 1.0, otherwise an error is raised.
  """


@dataclasses.dataclass
class PressureLevelAtmosphericMassWeighting(
    WeightingFromScaler, scaling.PressureLevelAtmosphericMassScaler
):
  """Weighting that returns weights proportional to the pressure lvl thickness.

  This weighting approximates the mass of the atmosphere for a given pressure
  level assuming hydrostatic balance and standard atmospheric pressure.
  """


@dataclasses.dataclass
class LeadTimeWeighting(WeightingFromScaler, scaling.LeadTimeScaler):
  """Weighting that returns weights equal to 1 / (std of a random walk spread).

  The weights are derived from the `TimeDelta` coordinate of the input
  field or `context` when producing weights for statistics slices along the
  `timedelta` dimension. See `scaling.LeadTimeScaler` for details on the
  computation.
  """


@dataclasses.dataclass
class ClipWeighting(Weighting):
  """Weighting that wraps a `weighting` and clips weight values to a range."""

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
  """Weighting that returns weights from `weightings_by_name[field_name]`."""

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
        name: ConstantWeighting(constant=w if cx.is_field(w) else cx.field(w))
        for name, w in variable_weights.items()
    }
    return cls(
        weightings_by_name=weightings, default_weighting=default_weighting
    )


@dataclasses.dataclass
class CoordinateMaskWeighting(
    WeightingFromScaler, scaling.CoordinateMaskScaler
):
  """Weighting that returns masked/unmasked weights based on masking.

  This weighting is parameterized by a `mask_coord`. For each dimension in
  `mask_coord.dims`, it checks for a matching coordinate in the input `field`
  or, if not present on the `field`, a scalar value in the `context`.

  If a matching coordinate is found on the `field`, it returns `masked_value`
  for each entry in that coordinate that is present in the `mask_coord`, and
  `unmasked_value` otherwise.

  If no matching coordinate is found of the `field`, the `context` is serched
  for scalar value using the dimension name, indicating in-context processing of
  `field` slices. If found, it returns a weight of `masked_value` if the value
  from the context (context[dim_name]) is present in `mask_coord`, and
  `unmasked_value` otherwise.

  If coordinates for multiple dimensions are found, the resulting weights are
  multiplied.

  If `skip_missing` is True, dimensions from `mask_coord` not found in the
  `field` or `context` are ignored (effectively a weight of 1.0). Otherwise,
  a ValueError is raised.
  """
