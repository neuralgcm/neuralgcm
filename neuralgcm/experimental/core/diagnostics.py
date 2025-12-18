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

import dataclasses
from typing import Protocol

import coordax as cx
from flax import nnx
import jax
import jax.numpy as jnp
import jax_datetime as jdt
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import nnx_compat
from neuralgcm.experimental.core import typing
import numpy as np


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
      v = v.get_value()  # get the underlying Field.
      self.cumulatives[k].set_value(cx.wrap_like(jnp.zeros(v.shape), v))

  def diagnostic_values(self) -> typing.Pytree:
    return {k: v.get_value() for k, v in self.cumulatives.items()}

  def __call__(self, inputs, *args, **kwargs):
    diagnostics = self.extract(inputs, *args, **kwargs)
    for k, v in diagnostics.items():
      self.cumulatives[k].set_value(v + self.cumulatives[k].get_value())


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
      v = v.get_value()  # get the underlying Field.
      self.instants[k].set_value(cx.wrap_like(jnp.zeros(v.shape), v))

  def diagnostic_values(self) -> typing.Pytree:
    return {k: v.get_value() for k, v in self.instants.items()}

  def __call__(self, inputs, *args, **kwargs):
    diagnostics = self.extract(inputs, *args, **kwargs)
    for k, v in diagnostics.items():
      self.instants[k].set_value(v)


@nnx_compat.dataclass
class IntervalDiagnostic(DiagnosticModule):
  """A diagnostic that tracks interval-accumulated values of fields.

  This diagnostic enables tracking of values accumulated over time `interval`,
  with sub-intervals of duration `resolution`. To provide the
  temporal context for accumulation this class requires an explicit call to
  `advance_diagnostic_clock` with time increments smaller than `resolution`.

  A call to `diagnostic_values` returns the accumulated values over the last
  completed interval, not including potential accumulation since the last
  sub-interval update. It includes `timedelta_since_sub_interval` to indicate
  the time since the last sub-interval update.

  Examples::

    # 6-hour cumulative precipitation, with output resolution of up to 2 hours.
    six_hr = np.timedelta64(6, 'h')
    two_hr = np.timedelta64(2, 'h')
    IntervalDiagnostic(get_precip, precip_coords, six_hr, resolution=two_hr)

    # weekly average temperature, with output resolution of up to 1 day.
    seven_days = np.timedelta64(7, 'D')
    one_day = np.timedelta64(1, 'D')
    IntervalDiagnostic(get_temp, temp_coords, seven_days, resolution=one_day)

  Attributes:
    extract: A callable that computes diagnostic values.
    extract_coords: Coordinates for each of the diagnostic fields keyed by name.
    interval: The total time interval over which diagnostics are accumulated.
    resolution: The duration of sub-intervals over which values are accumulated.
    default_timedelta: Time increment to use in `advance_diagnostic_clock` if
      explicit `timedelta` is not provided in the inputs. If specified,
      `resolution` must be a multiple of `default_timedelta`.
    include_instant: Whether to include additional instantaneous diagnostics.
    dt_mod_freq: Time since the last interval update.
    since_last_update: The accumulated values since the last sub-interval.
    per_period: The accumulated values for each `resolution` sub-interval.
    interval_axis: The coordinate axis for the interval dimension.
  """

  extract: Extract
  extract_coords: dict[str, cx.Coordinate]
  interval: np.timedelta64
  resolution: np.timedelta64
  default_timedelta: np.timedelta64 | None = None
  include_instant: bool = False
  dt_mod_freq: typing.Diagnostic = dataclasses.field(init=False)
  since_last_update: dict[str, typing.Diagnostic] = dataclasses.field(
      init=False
  )
  interval_axis: coordinates.TimeDelta = dataclasses.field(init=False)
  per_period: dict[str, typing.Diagnostic] = dataclasses.field(init=False)

  def __post_init__(self):
    if self.interval % self.resolution != np.timedelta64(0):
      raise ValueError(
          f'interval({self.interval}) must be a multiple of '
          f'resolution({self.resolution}).'
      )
    periods = self.interval // self.resolution
    float_seconds = self.resolution / np.timedelta64(1, 's')
    if float_seconds != np.floor(float_seconds):
      raise ValueError(
          'resolution must be an integer number of seconds, but '
          f'resolution({self.resolution}) has {float_seconds} seconds.'
      )
    self.dt_mod_freq = typing.Diagnostic(cx.wrap(jdt.Timedelta()))
    self.since_last_update = {
        k: Diagnostic(cx.wrap(jnp.zeros(c.shape), c))
        for k, c in self.extract_coords.items()
    }
    interval_deltas = np.arange(-periods, 0) * self.resolution
    self.interval_axis = coordinates.TimeDelta(interval_deltas)
    with_intrvl = lambda c: cx.compose_coordinates(self.interval_axis, c)
    self.per_period = {
        k: Diagnostic(cx.wrap(jnp.zeros(with_intrvl(c).shape), with_intrvl(c)))
        for k, c in self.extract_coords.items()
    }
    if self.include_instant:
      self.instants = {
          k: Diagnostic(cx.wrap(jnp.zeros(c.shape), c))
          for k, c in self.extract_coords.items()
      }

  def reset_diagnostic_state(self):
    """Resets the internal diagnostic state."""
    self.dt_mod_freq.set_value(cx.wrap(jdt.Timedelta()))
    zeros_like = lambda v: cx.wrap_like(jnp.zeros_like(v.get_value().data), v.get_value())
    for k in self.extract_coords:
      self.since_last_update[k].set_value(zeros_like(self.since_last_update[k]))
      self.per_period[k].set_value(zeros_like(self.per_period[k]))
      if self.include_instant:
        self.instants[k].set_value(zeros_like(self.instants[k]))

  def advance_diagnostic_clock(self, inputs, *args, **kwargs):
    """Advances the internal clock and updates interval accumulations."""
    del args, kwargs  # unused.
    timedelta = inputs.get('timedelta')
    if timedelta is None:
      if self.default_timedelta is None:
        raise ValueError(
            'Missing both `timedelta` in `inputs` and `default_timedelta`'
            f'got {inputs.keys()=}'
        )
      timedelta = jdt.to_timedelta(self.default_timedelta)
    else:
      if self.default_timedelta is not None:
        raise ValueError(
            'Specifying both `timedelta` in `inputs` and `default_timedelta`'
            f'is error-prone and not supported {inputs.keys()=}'
        )
    if not isinstance(timedelta, jdt.Timedelta):
      raise ValueError(f'timedelta must be Timedelta, got {type(timedelta)}')

    dt = self.dt_mod_freq.get_value() + timedelta
    is_update_step = (dt >= self.resolution).data
    recenter_timedelta = lambda t: t - self.resolution
    keep_timedelta = lambda t: t
    self.dt_mod_freq.set_value(jax.lax.cond(
        is_update_step, recenter_timedelta, keep_timedelta, dt
    ))

    for k in self.extract_coords:
      per_period = self.per_period[k].get_value()
      since_last = self.since_last_update[k].get_value()

      def _update_per_period(per_period, since_last):
        updated = jnp.concatenate([per_period[1:], since_last[None]])
        return jnp.where(is_update_step, updated, per_period)

      i_ax = self.interval_axis
      per_period = per_period.untag(i_ax)
      update_per_period = cx.cmap(_update_per_period, per_period.named_axes)
      updated_per_period = update_per_period(per_period, since_last).tag(i_ax)
      self.per_period[k].set_value(updated_per_period)
      self.since_last_update[k].set_value(jax.lax.cond(
          is_update_step, cx.wrap(0.0).broadcast_like, lambda x: x, since_last
      ))

  def diagnostic_values(self) -> typing.Pytree:
    """Returns formatted diagnostics computed from the internal module state."""
    values = {}
    for k, v in self.per_period.items():
      sum_fn = lambda x: jnp.sum(x, axis=0) if x.ndim > 0 else x
      sum_over_intervals = cx.cmap(sum_fn)(v.get_value().untag(self.interval_axis))
      values[k] = sum_over_intervals
    if self.include_instant:
      for k, v in self.instants.items():
        values[k + '_instant'] = v.get_value()
    values['timedelta_since_sub_interval'] = self.dt_mod_freq.get_value()
    return values

  def __call__(self, inputs, *args, **kwargs):
    """Updates the internal module state from the inputs."""
    diagnostics = self.extract(inputs, *args, **kwargs)
    for k, v in diagnostics.items():
      if k in self.extract_coords:
        self.since_last_update[k].set_value(self.since_last_update[k].get_value() + v)
        if self.include_instant:
          self.instants[k].set_value(v)
