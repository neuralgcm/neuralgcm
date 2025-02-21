# Copyright 2024 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module-based API for calculating diagnostics of NeuralGCM models."""

from flax import nnx
import jax.numpy as jnp
from neuralgcm.experimental import coordax as cx
from neuralgcm.experimental import nnx_compat
from neuralgcm.experimental import typing
import neuralgcm.experimental.jax_datetime as jdt


class DiagnosticValue(nnx.Intermediate):
  """Variable type in which diagnostic values are stored."""

  ...


@nnx_compat.dataclass
class DiagnosticModule(nnx.Module):
  """Base API for diagnostic modules."""

  def format_diagnostics(self, time: jdt.Datetime) -> typing.Pytree:
    """Returns formatted diagnostics computed from the internal module state."""
    raise NotImplementedError(f'`format_diagnostics` on {self.__name__=}.')

  def __call__(self, *args, **kwargs) -> None:
    """Updates the internal module state from the inputs."""
    raise NotImplementedError(f'`__call__` on {self.__name__=}.')


# TODO(dkochkov) Generalize these to work on arbitrary pytrees.


class CumulativeDiagnostic(DiagnosticModule):
  """Diagnostic module that tracks cumulative value of an array stream."""

  def __init__(
      self,
      diagnostic_name: str,
      diagnostic_coords: cx.Coordinate,
      extract_fn=lambda x, *args, **kwargs: x,
  ):
    self.diagnostic_name = diagnostic_name
    self.coords = diagnostic_coords
    self.extract_fn = extract_fn
    self.cumulative = DiagnosticValue(jnp.zeros(self.coords.shape))

  def format_diagnostics(self, time: jdt.Datetime) -> typing.Pytree:
    return {
        self.diagnostic_name: cx.wrap(self.cumulative.value, self.coords),
        f'{self.diagnostic_name}_time': cx.wrap(time),
    }

  def __call__(self, inputs, *args, **kwargs):
    self.cumulative.value = self.cumulative.value + self.extract_fn(
        inputs, *args, **kwargs
    )


class InstantDiagnostic(DiagnosticModule):
  """Diagnostic module that tracks instant value of an array stream."""

  def __init__(
      self,
      diagnostic_name: str,
      diagnostic_coords: cx.Coordinate,
      extract_fn=lambda x, *args, **kwargs: x,
  ):
    self.diagnostic_name = diagnostic_name
    self.coords = diagnostic_coords
    self.extract_fn = extract_fn
    self.instant = DiagnosticValue(jnp.zeros(self.coords.shape))

  def format_diagnostics(self, time: jdt.Datetime) -> typing.Pytree:
    return {
        self.diagnostic_name: cx.wrap(self.instant.value, self.coords),
        f'{self.diagnostic_name}_time': cx.wrap(time),
    }

  def __call__(self, inputs, *args, **kwargs):
    self.instant.value = self.extract_fn(inputs, *args, **kwargs)


class IntervalDiagnostic(DiagnosticModule):
  """Diagnostic module that tracks interval cumulant of an array."""

  # TODO(dkochkov) verify this implementation and add tests.
  def __init__(
      self,
      diagnostic_name: str,
      interval_length: int,
      diagnostic_coords: cx.Coordinate,
      extract_fn=lambda x, *args, **kwargs: x,
  ):
    self.diagnostic_name = diagnostic_name
    self.coords = diagnostic_coords
    self.interval_length = interval_length
    self.extract_fn = extract_fn
    self.cumulative = DiagnosticValue(jnp.zeros(self.coords.shape))
    interval_shape = (self.interval_length + 1,) + tuple(self.coords.shape)
    self.interval_values = DiagnosticValue(jnp.zeros(interval_shape))

  def next_interval(self, inputs):
    del inputs
    interval_values = self.interval_values.value
    c = self.cumulative.value
    interval_values = jnp.concat(
        [jnp.roll(interval_values, -1)[:-1], c[jnp.newaxis]]
    )
    self.interval_values.value = interval_values

  def format_diagnostics(self, time: jdt.Datetime) -> typing.Pytree:
    interval_values = self.interval_values.value
    return {
        self.diagnostic_name: cx.wrap(
            self.cumulative.value - interval_values[0], self.coords
        ),
        f'{self.diagnostic_name}_time': cx.wrap(time),
    }

  def __call__(self, inputs, *args, **kwargs):
    self.cumulative.value = self.cumulative.value + self.extract_fn(
        inputs, *args, **kwargs
    )
