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

import abc
from typing import Any, Protocol

import coordax as cx
from flax import nnx
import jax
import jax.numpy as jnp
import jax_datetime as jdt
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import typing
import numpy as np


Diagnostic = typing.Diagnostic


class DiagnosticModule(nnx.Module, abc.ABC):
  """Base API for diagnostic modules."""

  @abc.abstractmethod
  def diagnostic_values(self) -> typing.Pytree:
    """Returns formatted diagnostics computed from the internal module state."""

  @abc.abstractmethod
  def reset_diagnostic_state(self):
    """Resets the internal diagnostic state."""

  @abc.abstractmethod
  def __call__(self, *args, **kwargs) -> None:
    """Updates the internal module state from the inputs."""

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(pytree=False, **kwargs)


class TemporalDiagnosticModule(DiagnosticModule):
  """Base class for diagnostics that track time."""

  @abc.abstractmethod
  def advance_clock(self, inputs, *args, **kwargs):
    """Updates the internal state to reflect a time increment."""


class Extract(Protocol):
  """Protocol for diagnostic methods that extract values from a method call."""

  def __call__(
      self,
      result: typing.Pytree,
      *args,
      **kwargs,
  ) -> dict[str, cx.Field]:
    """Extracts diagnostic fields from the callback method result and args."""


def _check_and_get_periods(
    interval: np.timedelta64, resolution: np.timedelta64, name: str = 'interval'
) -> int:
  """Checks interval bounds and returns the number of periods."""
  if interval % resolution != np.timedelta64(0):
    raise ValueError(
        f'{name}({interval}) must be a multiple of resolution({resolution}).'
    )
  periods = interval // resolution
  float_seconds = resolution / np.timedelta64(1, 's')
  if float_seconds != np.floor(float_seconds):
    raise ValueError(
        'resolution must be an integer number of seconds, but '
        f'resolution({resolution}) has {float_seconds} seconds.'
    )
  return int(periods)


def _get_timedelta(
    inputs: dict[str, Any],
    default_timedelta: np.timedelta64 | None,
) -> jdt.Timedelta:
  """Extracts timedelta from inputs or defaults."""
  timedelta = inputs.get('timedelta')
  if timedelta is None:
    if default_timedelta is None:
      raise ValueError(
          'Missing both `timedelta` in `inputs` and `default_timedelta` '
          f'got {inputs.keys()=}'
      )
    timedelta = jdt.to_timedelta(default_timedelta)
  else:
    if default_timedelta is not None:
      raise ValueError(
          'Specifying both `timedelta` in `inputs` and `default_timedelta` '
          f'is error-prone and not supported {inputs.keys()=}'
      )
  if not isinstance(timedelta, jdt.Timedelta):
    raise ValueError(f'timedelta must be Timedelta, got {type(timedelta)}')
  return timedelta


def _update_clock(
    dt: jdt.Timedelta, resolution: np.timedelta64
) -> tuple[jdt.Timedelta, jax.Array]:
  """Returns updated clock and a boolean indicating if resolution was reached."""
  is_update_step = (dt >= resolution).data  # pytype: disable=attribute-error
  recenter_timedelta = lambda t: t - resolution
  keep_timedelta = lambda t: t
  new_dt = jax.lax.cond(is_update_step, recenter_timedelta, keep_timedelta, dt)
  return new_dt, is_update_step


@nnx.dataclass
class CumulativeDiagnostic(DiagnosticModule):
  """Diagnostic that tracks cumulative value of a dictionary of fields."""

  extract: Extract = nnx.data()
  extract_coords: dict[str, cx.Coordinate]

  def __post_init__(self):
    self.cumulatives = {
        k: Diagnostic(cx.field(jnp.zeros(c.shape), c))
        for k, c in self.extract_coords.items()
    }

  def reset_diagnostic_state(self):
    """Resets the internal diagnostic state."""
    for k, v in self.cumulatives.items():
      v = v.get_value()  # get the underlying Field.
      self.cumulatives[k].set_value(cx.field(jnp.zeros(v.shape), v.coordinate))

  def diagnostic_values(self) -> typing.Pytree:
    return {k: v.get_value() for k, v in self.cumulatives.items()}

  def __call__(self, inputs, *args, **kwargs):
    diagnostics = self.extract(inputs, *args, **kwargs)
    for k, v in diagnostics.items():
      self.cumulatives[k].set_value(v + self.cumulatives[k].get_value())


@nnx.dataclass
class InstantDiagnostic(DiagnosticModule):
  """Diagnostic that tracks instant value of a dictionary of fields."""

  extract: Extract = nnx.data()
  extract_coords: dict[str, cx.Coordinate]

  def __post_init__(self):
    self.instants = {
        k: Diagnostic(cx.field(jnp.zeros(c.shape), c))
        for k, c in self.extract_coords.items()
    }

  def reset_diagnostic_state(self):
    """Resets the internal diagnostic state."""
    for k, v in self.instants.items():
      v = v.get_value()  # get the underlying Field.
      self.instants[k].set_value(cx.field(jnp.zeros(v.shape), v.coordinate))

  def diagnostic_values(self) -> typing.Pytree:
    return {k: v.get_value() for k, v in self.instants.items()}

  def __call__(self, inputs, *args, **kwargs):
    diagnostics = self.extract(inputs, *args, **kwargs)
    for k, v in diagnostics.items():
      self.instants[k].set_value(v)


@nnx.dataclass
class IntervalDiagnostic(TemporalDiagnosticModule):
  """A diagnostic that tracks interval-accumulated values of fields.

  This diagnostic enables tracking of values accumulated over time `interval`,
  with sub-intervals of duration `resolution`. To provide the
  temporal context for accumulation this class requires an explicit call to
  `advance_clock` with time increments smaller than `resolution`.

  A call to `diagnostic_values` returns the accumulated values over the last
  completed interval, not including potential accumulation since the last
  sub-interval update. It includes `timedelta_since_sub_interval` to indicate
  the time since the last sub-interval update.

  Examples::

    # 6-hour cumulative precipitation, with output resolution of up to 2 hours.
    six_hr = np.timedelta64(6, 'h')
    two_hr = np.timedelta64(2, 'h')
    IntervalDiagnostic(get_precip, precip_coords, six_hr, resolution=two_hr)

    # weekly accumulated temperature, with output resolution of up to 1 day.
    # Note: To obtain a weekly mean, you must explicitly divide the result
    # by the number of accumulation steps (7 in this case) downstream.
    seven_days = np.timedelta64(7, 'D')
    one_day = np.timedelta64(1, 'D')
    IntervalDiagnostic(get_temp, temp_coords, seven_days, resolution=one_day)

  Attributes:
    extract: A callable that computes diagnostic values.
    extract_coords: Coordinates for each of the diagnostic fields keyed by name.
    interval: The total time interval or a dictionary of time intervals over
      which to compute accumulated diagnostics.
    resolution: The duration of sub-intervals over which values are accumulated.
    default_timedelta: Time increment to use in `advance_clock` if
      explicit `timedelta` is not provided in the inputs. If specified,
      `resolution` must be a multiple of `default_timedelta`.
    include_instant: Whether to include additional instantaneous diagnostics.
    include_dt_offset: Whether to include an additional diagnostic for the time
      since the last sub-interval update.
    dt_mod_freq: Time since the last interval update.
    since_last_update: The accumulated values since the last sub-interval.
    per_period: The accumulated values for each `resolution` sub-interval.
    interval_axis: The coordinate axis for the interval dimension.
  """

  extract: Extract = nnx.data()
  extract_coords: dict[str, cx.Coordinate]
  interval: np.timedelta64 | dict[str, np.timedelta64]
  resolution: np.timedelta64
  default_timedelta: np.timedelta64 | None = None
  include_instant: bool = False
  include_dt_offset: bool = False
  dt_mod_freq: typing.Diagnostic = nnx.data(init=False)
  since_last_update: dict[str, typing.Diagnostic] = nnx.data(
      init=False
  )
  interval_axis: coordinates.TimeDelta = nnx.static(init=False)
  per_period: dict[str, typing.Diagnostic] = nnx.data(init=False)
  periods: int = nnx.static(init=False)

  def __post_init__(self):
    if isinstance(self.interval, np.timedelta64):
      self._intervals = {'': self.interval}
    else:
      self._intervals = self.interval

    max_interval = np.timedelta64(0, 's')
    for name, interval in self._intervals.items():
      _check_and_get_periods(interval, self.resolution, f'interval[{name}]')
      max_interval = max(max_interval, interval)

    self.periods = _check_and_get_periods(
        max_interval, self.resolution, 'max_interval'
    )

    self.dt_mod_freq = typing.Diagnostic(cx.field(jdt.Timedelta()))
    self.since_last_update = {
        k: Diagnostic(cx.field(jnp.zeros(c.shape), c))
        for k, c in self.extract_coords.items()
    }
    interval_deltas = np.arange(-self.periods, 0) * self.resolution
    self.interval_axis = coordinates.TimeDelta(interval_deltas)
    with_intrvl = lambda c: cx.coords.compose(self.interval_axis, c)
    self.per_period = {
        k: Diagnostic(cx.field(jnp.zeros(with_intrvl(c).shape), with_intrvl(c)))
        for k, c in self.extract_coords.items()
    }
    if self.include_instant:
      self.instants = {
          k: Diagnostic(cx.field(jnp.zeros(c.shape), c))
          for k, c in self.extract_coords.items()
      }

  def reset_diagnostic_state(self):
    """Resets the internal diagnostic state."""
    self.dt_mod_freq.set_value(cx.field(jdt.Timedelta()))
    zeros_like = lambda v: cx.field(
        jnp.zeros_like(v.get_value().data), v.get_value().coordinate
    )
    for k in self.extract_coords:
      self.since_last_update[k].set_value(zeros_like(self.since_last_update[k]))
      self.per_period[k].set_value(zeros_like(self.per_period[k]))
      if self.include_instant:
        self.instants[k].set_value(zeros_like(self.instants[k]))

  def advance_clock(self, inputs, *args, **kwargs):
    """Advances the internal clock and updates interval accumulations."""
    del args, kwargs  # unused.
    timedelta = _get_timedelta(inputs, self.default_timedelta)
    dt = self.dt_mod_freq.get_value() + timedelta
    new_dt, is_update_step = _update_clock(dt, self.resolution)
    self.dt_mod_freq.set_value(new_dt)

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
      self.since_last_update[k].set_value(
          jax.lax.cond(
              is_update_step,
              cx.field(0.0).broadcast_like,
              lambda x: x,
              since_last,
          )
      )

  def diagnostic_values(self) -> typing.Pytree:
    """Returns formatted diagnostics computed from the internal module state."""
    values = {}
    periods = self.periods
    for name, interval in self._intervals.items():
      target_period = interval // self.resolution
      index = periods - int(target_period)
      for k, v in self.per_period.items():
        # pylint: disable=cell-var-from-loop
        sum_fn = cx.cmap(lambda x: jnp.sum(x[index:], axis=0))
        # pylint: enable=cell-var-from-loop
        sum_over_intervals = sum_fn(v.get_value().untag(self.interval_axis))
        out_key = f'{k}_{name}' if name else k
        values[out_key] = sum_over_intervals
    if self.include_instant:
      for k, v in self.instants.items():
        values[k + '_instant'] = v.get_value()
    if self.include_dt_offset:
      values['timedelta_since_sub_interval'] = self.dt_mod_freq.get_value()
    return values

  def __call__(self, inputs, *args, **kwargs):
    """Updates the internal module state from the inputs."""
    diagnostics = self.extract(inputs, *args, **kwargs)
    for k, v in diagnostics.items():
      if k in self.extract_coords:
        self.since_last_update[k].set_value(
            self.since_last_update[k].get_value() + v
        )
        if self.include_instant:
          self.instants[k].set_value(v)


@nnx.dataclass
class TimeOffsetDiagnostic(TemporalDiagnosticModule):
  """A diagnostic that tracks field values at given backwards-looking offset.

  This diagnostic enables retrieval of field values saved at previous simulation
  steps. A call to ``diagnostic_values`` returns values with teporal `offset`
  that are saved on the module's internal state. To provide the temporal context
  for the value updates, this module requires an explicit call to
  `advance_clock` method.

  Attributes:
    extract: A callable that computes diagnostic values.
    extract_coords: Coordinates for each of the diagnostic fields keyed by name.
    offset: Timedelta or a dict of timedeltas to provide diagnostics for. If
      specified as a dict, the output keys include a suffix from the dict keys.
    resolution: The duration of sub-intervals at which values are stored. Must
      evenly divide `offset` and `default_timedelta` if specified.
    default_timedelta: Time increment to use in `advance_clock` if
      explicit 'timedelta' of type jdt.Timedelta is not provided in the inputs.
    include_dt_offset: Whether to include an additional diagnostic for the time
      since the last sub-interval update, useful for debugging.
    dt_mod_freq: Time since the last interval update.
    latest_update: The most recent values from `__call__`.
    past_values: The stored values for each `resolution` step into the past.
    offset_axis: The coordinate axis for the offset dimension.
  """

  extract: Extract = nnx.data()
  extract_coords: dict[str, cx.Coordinate] = nnx.static()
  offset: np.timedelta64 | dict[str, np.timedelta64] = nnx.static()
  resolution: np.timedelta64 = nnx.static()
  default_timedelta: np.timedelta64 | None = nnx.static(default=None)
  include_dt_offset: bool = nnx.static(default=False)
  dt_mod_freq: typing.Diagnostic = nnx.data(init=False)
  latest_update: dict[str, typing.Diagnostic] = nnx.data(init=False)
  offset_axis: coordinates.TimeDelta = nnx.static(init=False)
  past_values: dict[str, typing.Diagnostic] = nnx.data(init=False)
  periods: int = nnx.static(init=False)

  def __post_init__(self):
    if isinstance(self.offset, np.timedelta64):
      self._offsets = {'': self.offset}
    else:
      self._offsets = self.offset

    max_offset = np.timedelta64(0, 's')
    for name, offset in self._offsets.items():
      _check_and_get_periods(offset, self.resolution, f'offset[{name}]')
      max_offset = max(max_offset, offset)
    self.periods = _check_and_get_periods(
        max_offset, self.resolution, 'max_offset'
    )
    self.dt_mod_freq = typing.Diagnostic(cx.field(jdt.Timedelta()))
    self.latest_update = {
        k: Diagnostic(cx.field(jnp.zeros(c.shape), c))
        for k, c in self.extract_coords.items()
    }
    offset_deltas = np.arange(-self.periods, 1) * self.resolution
    self.offset_axis = coordinates.TimeDelta(offset_deltas)
    with_offset = lambda c: cx.coords.compose(self.offset_axis, c)
    self.past_values = {
        k: Diagnostic(cx.field(jnp.zeros(with_offset(c).shape), with_offset(c)))
        for k, c in self.extract_coords.items()
    }

  def reset_diagnostic_state(self):
    """Resets the internal diagnostic state."""
    self.dt_mod_freq.set_value(cx.field(jdt.Timedelta()))
    zeros_like = lambda v: cx.field(
        jnp.zeros_like(v.get_value().data), v.get_value().coordinate
    )
    for k in self.extract_coords:
      self.latest_update[k].set_value(zeros_like(self.latest_update[k]))
      self.past_values[k].set_value(zeros_like(self.past_values[k]))

  def advance_clock(self, inputs, *args, **kwargs):
    """Advances the internal clock and updates past values."""
    del args, kwargs  # unused.
    timedelta = _get_timedelta(inputs, self.default_timedelta)
    dt = self.dt_mod_freq.get_value() + timedelta
    new_dt, is_update_step = _update_clock(dt, self.resolution)
    self.dt_mod_freq.set_value(new_dt)

    for k in self.extract_coords:
      past = self.past_values[k].get_value()
      latest = self.latest_update[k].get_value()

      def _update_past(past, latest):
        updated = jnp.concatenate([past[1:], latest[None]])
        return jnp.where(is_update_step, updated, past)

      past = past.untag(self.offset_axis)
      update_past = cx.cmap(_update_past, past.named_axes)
      updated_past = update_past(past, latest).tag(self.offset_axis)
      self.past_values[k].set_value(updated_past)

  def diagnostic_values(self) -> typing.Pytree:
    """Returns formatted diagnostics computed from the internal module state."""
    values = {}
    periods = self.periods
    for key_suffix, offset in self._offsets.items():
      target_period = offset // self.resolution
      index = periods - int(target_period)
      for k, v in self.past_values.items():
        field = v.get_value()
        # pylint: disable=cell-var-from-loop
        selected = cx.cmap(lambda x: x[index])(field.untag(self.offset_axis))
        # pylint: enable=cell-var-from-loop
        out_key = f'{k}_{key_suffix}' if key_suffix else k
        values[out_key] = selected

    if self.include_dt_offset:
      values['timedelta_since_sub_interval'] = self.dt_mod_freq.get_value()
    return values

  def __call__(self, inputs, *args, **kwargs):
    """Updates the internal module state from the inputs."""
    diagnostics = self.extract(inputs, *args, **kwargs)
    for k, v in diagnostics.items():
      if k in self.extract_coords:
        self.latest_update[k].set_value(v)


@nnx.dataclass
class ExtractTransformedOutputs(nnx.Module):
  """Extract module that applies `transform` to the diagnosed module outputs."""

  transform: typing.Transform = nnx.data()

  def __call__(self, result, *args, **kwargs) -> dict[str, cx.Field]:
    del args, kwargs  # unused.
    return self.transform(result)


@nnx.dataclass
class ExtractFixedQueryObservations(nnx.Module):
  """Extract module that evaluates observation operator with a fixed query.

  This module extracts values by calling `observation_operator` with a fixed
  `query`. The prognostics data is extracted from the arguments passed to the
  `__call__` method, based on `prognostics_arg_key`. An optional `transform` can
  be applied to the results of the observation.

  Attributes:
    observation_operator: An ObservationOperator module to be evaluated.
    query: A fixed query for the `observation_operator` to generate outputs.
    prognostics_arg_key: The key or index used to extract the prognostics
      dictionary from the arguments passed to `__call__`. Prognostics are passed
      to the observation operator together with the `query`.
    transform: An optional function to apply to the outputs returned by the
      `observation_operator`.
  """

  observation_operator: typing.ObservationOperator = nnx.data()
  query: dict[str, cx.Coordinate | cx.Field] = nnx.data()
  prognostics_arg_key: str | int = 'prognostics'
  transform: typing.Transform | None = None

  def _extract_prognostics(self, *args, **kwargs) -> dict[str, cx.Field]:
    if isinstance(self.prognostics_arg_key, int):
      prognostics = args[self.prognostics_arg_key]
    else:
      prognostics = kwargs.get(self.prognostics_arg_key)
    if not isinstance(prognostics, dict):
      raise ValueError(
          f'Prognostics must be a dictionary, got {type(prognostics)=} instead.'
      )
    return prognostics

  def __call__(self, result, *args, **kwargs) -> dict[str, cx.Field]:
    del result  # unused.
    prognostics = self._extract_prognostics(*args, **kwargs)
    extracted = self.observation_operator.observe(prognostics, query=self.query)
    if self.transform is not None:
      extracted = self.transform(extracted)
    return extracted
