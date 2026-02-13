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

"""Modules that implement step filters."""

import abc

import coordax as cx
from flax import nnx
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import nnx_compat


class StepFilter(nnx.Module, abc.ABC):
  """Base class for spatial filters."""

  @abc.abstractmethod
  def __call__(
      self, state: dict[str, cx.Field], next_state: dict[str, cx.Field]
  ) -> dict[str, cx.Field]:
    """Returns filtered ``inputs``."""

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(pytree=False, **kwargs)


class NoFilter(StepFilter):
  """Filter that does nothing."""

  def __call__(
      self, state: dict[str, cx.Field], next_state: dict[str, cx.Field]
  ) -> dict[str, cx.Field]:
    del state
    return next_state


@nnx_compat.dataclass
class ModalFixedGlobalMeanFilter(StepFilter):
  """Filter that removes the change in the global mean of certain keys."""

  ylm_grid: coordinates.SphericalHarmonicGrid
  keys: tuple[str, ...]

  def __post_init__(self):
    # TODO(dkochkov): consider adding `.sel` on Field to simplify this.
    if self.ylm_grid.fields['total_wavenumber'].data[0] != 0:
      raise ValueError(
          'ModalFixedGlobalMeanFilter assumes total wavenumber to start with'
          f' 0, but got {self.ylm_grid.fields["total_wavenumber"].data}'
      )

  def __call__(
      self, state: dict[str, cx.Field], next_state: dict[str, cx.Field]
  ) -> dict[str, cx.Field]:
    ylm_grid = self.ylm_grid
    for key in self.keys:
      field = state[key]
      global_mean = cx.cpmap(lambda x: x[:, 0])(field.untag(ylm_grid))
      next_field = next_state[key].untag(ylm_grid)
      set_mean_fn = cx.cpmap(lambda x, mean: x.at[:, 0].set(mean))
      next_state[key] = set_mean_fn(next_field, global_mean).tag(ylm_grid)

    return next_state
