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

from flax import nnx
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import pytree_utils
from neuralgcm.experimental.core import typing


class StepFilter(nnx.Module, abc.ABC):
  """Base class for spatial filters."""

  def __call__(
      self, state: typing.Pytree, next_state: typing.Pytree
  ) -> typing.Pytree:
    """Returns filtered ``inputs``."""


class NoFilter(StepFilter):
  """Filter that does nothing."""

  def __call__(
      self, state: typing.Pytree, next_state: typing.Pytree
  ) -> typing.Pytree:
    del state
    return next_state


class ModalFixedGlobalMeanFilter(StepFilter):
  """Filter that removes the change in the global mean of certain keys."""

  def __init__(
      self,
      keys: tuple[str, ...] = ('log_surface_pressure',),
  ):
    self.keys = keys

  def __call__(
      self, state: typing.Pytree, next_state: typing.Pytree
  ) -> typing.Pytree:
    state_dict, _ = pytree_utils.as_dict(state)
    next_state_dict, from_dict_fn = pytree_utils.as_dict(next_state)
    # TODO(dkochkov): implementation below is dangerous, as it assumes a
    # specific layout of wavenumbers in arrays. Use grid from `state` once we
    # pass it as dict of cx.Field objects.
    for key in self.keys:
      global_mean = state_dict[key][..., 0]  # assuming modal
      next_state_dict[key] = next_state_dict[key].at[..., 0].set(global_mean)

    outputs = from_dict_fn(next_state_dict)
    return outputs
