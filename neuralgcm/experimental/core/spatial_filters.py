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

"""Modules that implement spatial filters."""

import abc
from typing import Sequence

from dinosaur import filtering
from flax import nnx
from neuralgcm.experimental.core import nnx_compat
from neuralgcm.experimental.core import spherical_transforms
from neuralgcm.experimental.core import typing
from neuralgcm.experimental.core import units
import numpy as np


class SpatialFilter(nnx.Module, abc.ABC):
  """Base class for spatial filters."""

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    """Returns filtered ``inputs``."""


class ModalSpatialFilter(SpatialFilter):
  """Base class for filters."""

  @abc.abstractmethod
  def filter_modal(self, inputs: typing.Pytree) -> typing.Pytree:
    """Returns filtered modal ``inputs``."""


@nnx_compat.dataclass
class ExponentialModalFilter(ModalSpatialFilter):
  """Modal filter that removes high frequency components."""

  ylm_transform: spherical_transforms.SphericalHarmonicsTransform
  attenuation: float = 16.0
  order: int = 18
  cutoff: float = 0.0

  def filter_modal(self, inputs: typing.Pytree) -> typing.Pytree:
    filter_fn = filtering.exponential_filter(
        grid=self.ylm_transform.dinosaur_grid,
        attenuation=self.attenuation,
        order=self.order,
        cutoff=self.cutoff,
    )
    return filter_fn(inputs)

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    return self.ylm_transform.dinosaur_grid.to_nodal(
        self.filter_modal(self.ylm_transform.dinosaur_grid.to_modal(inputs))
    )

  @classmethod
  def from_timescale(
      cls,
      ylm_transform: spherical_transforms.SphericalHarmonicsTransform,
      dt: float | typing.Quantity | typing.Numeric,
      timescale: float | typing.Quantity | typing.Numeric,
      order: int = 18,
      cutoff: float = 0.0,
      *,
      sim_units: units.SimUnits,
  ):
    """Returns a filter with the given timescale."""
    if isinstance(dt, np.timedelta64):
      dt = units.nondimensionalize_timedelta64(dt, sim_units)
    else:
      dt = units.maybe_nondimensionalize(dt, sim_units)
    if isinstance(timescale, np.timedelta64):
      timescale = units.nondimensionalize_timedelta64(timescale, sim_units)
    else:
      timescale = units.maybe_nondimensionalize(timescale, sim_units)
    return cls(
        ylm_transform=ylm_transform,
        attenuation=(dt / timescale),
        order=order,
        cutoff=cutoff,
    )


@nnx_compat.dataclass
class SequentialModalFilter(ModalSpatialFilter):
  """Modal filter that applies multiple filters sequentially."""

  filters: Sequence[ModalSpatialFilter]
  ylm_transform: spherical_transforms.SphericalHarmonicsTransform

  def filter_modal(self, inputs: typing.Pytree) -> typing.Pytree:
    for modal_filter in self.filters:
      inputs = modal_filter.filter_modal(inputs)
    return inputs

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    modal_inputs = self.ylm_transform.dinosaur_grid.to_modal(inputs)
    modal_outputs = self.filter_modal(modal_inputs)
    return self.ylm_transform.dinosaur_grid.to_nodal(modal_outputs)
