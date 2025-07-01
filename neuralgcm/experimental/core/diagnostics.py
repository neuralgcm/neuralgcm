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
import jax_datetime as jdt
from neuralgcm.experimental.core import nnx_compat
from neuralgcm.experimental.core import typing


class DiagnosticValue(nnx.Intermediate):
  """Variable type in which diagnostic values are stored."""


@nnx_compat.dataclass
class DiagnosticModule(nnx.Module):
  """Base API for diagnostic modules."""

  def format_diagnostics(self, time: jdt.Datetime) -> typing.Pytree:
    """Returns formatted diagnostics computed from the internal module state."""
    raise NotImplementedError(f'`format_diagnostics` on {self.__name__=}.')

  def reset_diagnostic_state(self):
    """Resets the internal diagnostic state."""
    raise NotImplementedError(f'`reset_diagnostic_state` on {self.__name__=}.')

  def __call__(self, *args, **kwargs) -> None:
    """Updates the internal module state from the inputs."""
    raise NotImplementedError(f'`__call__` on {self.__name__=}.')


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
        k: DiagnosticValue(jnp.zeros(v.shape))
        for k, v in self.extract_coords.items()
    }

  def reset_diagnostic_state(self):
    """Resets the internal diagnostic state."""
    for k, v in self.cumulatives.items():
      self.cumulatives[k].value = jnp.zeros_like(v)

  def format_diagnostics(self, time: jdt.Datetime) -> typing.Pytree:
    # TODO(dkochkov): remove time arg, it is no longer used.
    return {
        k: cx.wrap(v.value, self.extract_coords[k])
        for k, v in self.cumulatives.items()
    }

  def __call__(self, inputs, *args, **kwargs):
    diagnostic_values = self.extract(inputs, *args, **kwargs)
    for k, v in diagnostic_values.items():
      # TODO(dkochkov): consider storing values as Field type.
      self.cumulatives[k].value += v.data


@nnx_compat.dataclass
class InstantDiagnostic(DiagnosticModule):
  """Diagnostic that tracks instant value of a dictionary of fields."""

  extract: Extract
  extract_coords: dict[str, cx.Coordinate]

  def __post_init__(self):
    self.instants = {
        k: DiagnosticValue(jnp.zeros(v.shape))
        for k, v in self.extract_coords.items()
    }

  def reset_diagnostic_state(self):
    """Resets the internal diagnostic state."""
    for k, v in self.instants.items():
      self.instants[k].value = jnp.zeros_like(v)

  def format_diagnostics(self, time: jdt.Datetime) -> typing.Pytree:
    # TODO(dkochkov): remove time arg, it is no longer used.
    return {
        k: cx.wrap(v.value, self.extract_coords[k])
        for k, v in self.instants.items()
    }

  def __call__(self, inputs, *args, **kwargs):
    diagnostic_values = self.extract(inputs, *args, **kwargs)
    for k, v in diagnostic_values.items():
      # TODO(dkochkov): consider storing values as Field type.
      self.instants[k].value = v.data


@nnx_compat.dataclass
class IntervalDiagnostic(DiagnosticModule):
  """Diagnostic that tracks interval value of a dictionary of fields.

  This diagnostic keeps track of several lagged cumulative values to output
  values accumulated over intervals of length `interval_length`. It requires
  an explicit call to `next_interval` to increment the interval index, allowing
  it to be called at a user-defined frequency. This implementation does not
  avoid potential loss of numerical precision due to cumulative sum.

  Attributes:
    extract: callable that computes diagnostic values.
    extract_coords: coordinates for each of the diagnostic fields.
    interval_length: length of the interval to track.
  """

  extract: Extract
  extract_coords: dict[str, cx.Coordinate]
  interval_length: int

  def __post_init__(self):
    self.interval_values = {
        k: DiagnosticValue(jnp.zeros(((self.interval_length,) + v.shape)))
        for k, v in self.extract_coords.items()
    }
    self.cumulative = {
        k: DiagnosticValue(jnp.zeros(v.shape))
        for k, v in self.extract_coords.items()
    }

  def reset_diagnostic_state(self):
    """Resets the internal diagnostic state."""
    for k in self.extract_coords:
      self.interval_values[k].value = jnp.zeros_like(self.interval_values[k])
      self.cumulative[k].value = jnp.zeros_like(self.cumulative[k])

  def next_interval(self, inputs, *args, **kwargs):
    del inputs, args, kwargs
    for k, v in self.interval_values.items():
      self.interval_values[k].value = jnp.concat([
          jnp.roll(v.value, -1, axis=0)[:-1],
          self.cumulative[k].value[jnp.newaxis],
      ])

  def format_diagnostics(self, time: jdt.Datetime) -> typing.Pytree:
    # TODO(dkochkov): remove time arg, it is no longer used.
    return {
        k: cx.wrap(
            self.cumulative[k].value - v.value[0], self.extract_coords[k]
        )
        for k, v in self.interval_values.items()
    }

  def __call__(self, inputs, *args, **kwargs):
    diagnostic_values = self.extract(inputs, *args, **kwargs)
    for k, v in diagnostic_values.items():
      # TODO(dkochkov): consider storing values as Field type.
      self.cumulative[k].value += v.data
