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

"""Module-based API for calculating diagnostics of NeuralGCM models."""

from typing import Protocol

import coordax as cx
from flax import nnx
import jax.numpy as jnp
from neuralgcm.experimental.core import nnx_compat
from neuralgcm.experimental.core import typing


Diagnostic = typing.Diagnostic


@nnx_compat.dataclass
class DiagnosticModule(nnx.Module):
  """Base API for diagnostic modules."""

  def diagnostic_values(self) -> typing.Pytree:
    """Returns formatted diagnostics computed from the internal module state."""
    raise NotImplementedError(f'`diagnostic_values` on {self.__name__=}.')

  def reset_diagnostic_state(self):
    """Resets the internal diagnostic state."""
    raise NotImplementedError(f'`reset_diagnostic_state` on {self.__name__=}.')

  def __call__(self, *args, **kwargs) -> None:
    """Updates the internal module state from the inputs."""
    raise NotImplementedError(f'`__call__` on {self.__name__=}.')

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(pytree=False, **kwargs)


class Extract(Protocol):
  """Protocol for diagnostic methods that extract values from a method call."""

  def __call__(
      self,
      result: typing.Pytree,
      *args,
      **kwargs,
  ) -> dict[str, cx.Field]:
    """Extracts diagnostic fields from the callback method result and args."""


@nnx_compat.dataclass
class CumulativeDiagnostic(DiagnosticModule):
  """Diagnostic that tracks cumulative value of a dictionary of fields."""

  extract: Extract
  extract_coords: dict[str, cx.Coordinate]

  def __post_init__(self):
    self.cumulatives = {
        k: Diagnostic(cx.wrap(jnp.zeros(c.shape), c))
        for k, c in self.extract_coords.items()
    }

  def reset_diagnostic_state(self):
    """Resets the internal diagnostic state."""
    for k, v in self.cumulatives.items():
      v = v.value  # get the underlying Field.
      self.cumulatives[k].value = cx.wrap_like(jnp.zeros(v.shape), v)

  def diagnostic_values(self) -> typing.Pytree:
    return {k: v.value for k, v in self.cumulatives.items()}

  def __call__(self, inputs, *args, **kwargs):
    diagnostics = self.extract(inputs, *args, **kwargs)
    for k, v in diagnostics.items():
      self.cumulatives[k].value += v


@nnx_compat.dataclass
class InstantDiagnostic(DiagnosticModule):
  """Diagnostic that tracks instant value of a dictionary of fields."""

  extract: Extract
  extract_coords: dict[str, cx.Coordinate]

  def __post_init__(self):
    self.instants = {
        k: Diagnostic(cx.wrap(jnp.zeros(c.shape), c))
        for k, c in self.extract_coords.items()
    }

  def reset_diagnostic_state(self):
    """Resets the internal diagnostic state."""
    for k, v in self.instants.items():
      v = v.value  # get the underlying Field.
      self.instants[k].value = cx.wrap_like(jnp.zeros(v.shape), v)

  def diagnostic_values(self) -> typing.Pytree:
    return {k: v.value for k, v in self.instants.items()}

  def __call__(self, inputs, *args, **kwargs):
    diagnostics = self.extract(inputs, *args, **kwargs)
    for k, v in diagnostics.items():
      self.instants[k].value = v


@nnx_compat.dataclass
class IntervalDiagnostic(DiagnosticModule):
  """Diagnostic that tracks interval value of a dictionary of fields.

  This diagnostic keeps track of several lagged cumulative values to output
  values accumulated over intervals specified by `interval_axis`. It requires
  an explicit call to `next_interval` to increment the interval index, allowing
  it to be called at a user-defined frequency. This implementation does not
  avoid potential loss of numerical precision due to cumulative sum.

  Attributes:
    extract: callable that computes diagnostic values.
    extract_coords: coordinates for each of the diagnostic fields.
    interval_axis: coordinate axis for the interval dimension.
    include_instant: whether to include the instant value of the diagnostic
      fields.
  """

  extract: Extract
  extract_coords: dict[str, cx.Coordinate]
  interval_axis: cx.Coordinate
  include_instant: bool = False
  # TODO(jianingfang): see if there a better way to handle include_instant.

  def __post_init__(self):
    if self.interval_axis.ndim != 1:
      raise ValueError(
          f'interval_axis must be a 1D coordinate, but got {self.interval_axis}'
      )
    i_ax = self.interval_axis
    self.lagged_cumulatives = {
        k: Diagnostic(cx.wrap(jnp.zeros(i_ax.shape + c.shape), i_ax, c))
        for k, c in self.extract_coords.items()
    }
    self.cumulative = {
        k: Diagnostic(cx.wrap(jnp.zeros(c.shape), c))
        for k, c in self.extract_coords.items()
    }
    if self.include_instant:
      self.instants = {
          k: Diagnostic(cx.wrap(jnp.zeros(c.shape), c))
          for k, c in self.extract_coords.items()
      }

  def reset_diagnostic_state(self):
    """Resets the internal diagnostic state."""
    zeros_like = lambda v: cx.wrap_like(jnp.zeros(v.value.shape), v.value)
    for k in self.extract_coords:
      self.lagged_cumulatives[k].value = zeros_like(self.lagged_cumulatives[k])
      self.cumulative[k].value = zeros_like(self.cumulative[k])
      if self.include_instant:
        self.instants[k].value = zeros_like(self.instants[k])

  def next_interval(self, inputs, *args, **kwargs):
    del inputs, args, kwargs  # unused.
    update_lagged = lambda intervals, cumulative: jnp.concat([
        jnp.roll(intervals, -1, axis=0)[:-1],
        cumulative[jnp.newaxis],
    ])
    for k, v in self.lagged_cumulatives.items():
      state_coord, i_ax = self.extract_coords[k], self.interval_axis
      current = v.value.untag(i_ax, state_coord)
      cumulative = self.cumulative[k].value.untag(state_coord)
      updated = cx.cpmap(update_lagged)(current, cumulative)
      self.lagged_cumulatives[k].value = updated.tag(i_ax, state_coord)

  def diagnostic_values(self) -> typing.Pytree:
    values = {}
    for k, v in self.lagged_cumulatives.items():
      cumulative = self.cumulative[k].value
      lagged = v.value.untag(self.interval_axis)
      oldest_lagged = cx.cmap(lambda x: x[0], cumulative.named_axes)(lagged)
      values[k] = cumulative - oldest_lagged
    if self.include_instant:
      for k, v in self.instants.items():
        values[k + '_instant'] = v.value
    return values

  def __call__(self, inputs, *args, **kwargs):
    diagnostics = self.extract(inputs, *args, **kwargs)
    for k, v in diagnostics.items():
      self.cumulative[k].value += v
      if self.include_instant:
        self.instants[k].value = v
