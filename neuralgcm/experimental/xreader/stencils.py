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
"""Utilities for sampling data on grids with stencils."""

import dataclasses
import re
from typing import Generic, Literal, TypeVar

import numpy as np


T = TypeVar('T')


def _divide_evenly(
    x: np.typing.ArrayLike, y: np.typing.ArrayLike
) -> np.ndarray:
  """Divide x by y, raising an error if the result is not an integer."""
  x = np.asarray(x)
  y = np.asarray(y)
  q = np.around(x / y).astype(int)
  if np.issubdtype(x.dtype, np.timedelta64):
    # requires an exact match
    uneven = q * y != x
  else:
    epsilon = 1e-6
    uneven = abs(q * y - x) > epsilon
  if np.any(uneven):
    raise ValueError(f'{y} must evenly divide {x}')
  return q


Closed = Literal['left', 'right', 'both', 'neither']

_INCLUDE_START = {'left', 'both'}
_INCLUDE_STOP = {'right', 'both'}


@dataclasses.dataclass
class Stencil(Generic[T]):
  """Stencil of points to applying for sampling data defined on a grid.

  Example usage:

  >>> stencil = Stencil(start=-2, stop=2, step=0.5, closed='both')
  >>> stencil.points
  array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])

  Attributes:
    start: The start of the stencil, in data coordinates.
    stop: The end of the stencil, in data coordinates.
    step: The step between samples, in data coordinates.
    closed: Whether the start and/or stop points are included in the stencil.
      Valid options are 'left', 'right', 'both', and 'neither'.
  """

  start: T
  stop: T
  step: T
  closed: Closed = dataclasses.field(default='left', kw_only=True)

  def __post_init__(self):
    if self.closed not in {'left', 'right', 'both', 'neither'}:
      raise ValueError(f'invalid value for closed: {self.closed!r}')
    if not self.stop > self.start:
      raise ValueError(
          f'stop must be greater than start: {self.stop} vs {self.start}'
      )

  @property
  def includes_start(self) -> bool:
    """Does this stencil include the start point?"""
    return self.closed in _INCLUDE_START

  @property
  def includes_stop(self) -> bool:
    """Does this stencil include the stop point?"""
    return self.closed in _INCLUDE_STOP

  @property
  def points(self) -> np.ndarray:
    """Returns the points at which the stencil is defined."""
    num = _divide_evenly(self.stop - self.start, self.step)
    result = self.start + self.step * np.arange(num + 1)
    if not self.includes_start:
      result = result[1:]
    if not self.includes_stop:
      result = result[:-1]
    return result


def _normalize_time_unit(unit: str):
  if unit in {'D', 'day', 'days'}:
    return 'D'
  elif unit in {'h', 'hr', 'hour', 'hours'}:
    return 'h'
  elif unit in {'m', 'min', 'minute', 'minutes'}:
    return 'm'
  elif unit in {'s', 'sec', 'second', 'seconds'}:
    return 's'
  else:
    raise ValueError(f'unsupported time unit: {unit!r}')


def _parse_timedelta_string(value: str) -> np.timedelta64:
  match = re.match(r'([+-]?\d+) ?([a-zA-Z]+)', value)
  if not match:
    raise ValueError(f'invalid time delta string: {value}')
  value = int(match.group(1))
  unit = _normalize_time_unit(match.group(2))
  return np.timedelta64(value, unit)


class TimeStencil(Stencil[np.timedelta64]):
  """Stencil with specified by np.timedelta64.

  Example usage:

  >>> TimeStencil(start='-9h', stop='3h', step='1h', closed='both')
  TimeStencil(start='-9 hours', stop='3 hours', step='1 hours', closed='both')
  """

  def __init__(self, start: str, stop: str, step: str, closed: Closed = 'left'):
    super().__init__(
        _parse_timedelta_string(start),
        _parse_timedelta_string(stop),
        _parse_timedelta_string(step),
        closed=closed,
    )

  def __repr__(self):
    return (
        f"TimeStencil(start='{self.start}', stop='{self.stop}',"
        f" step='{self.step}', closed='{self.closed}')"
    )


def build_sampling_slices(
    source_points: np.typing.ArrayLike,
    sample_origins: np.typing.ArrayLike,
    stencil: Stencil,
) -> list[slice]:
  """Create slice objects for sampling with a Stencil.

  Args:
    source_points: data coordinates at which the source data is defined.
    sample_origins: data coordinate corresponding to origin of sample stencils.
    stencil: stencil objects that define the shape of a sample.

  Returns:
    A list of slice objects that can be used to sample Stencils from the
    source data, corresponding to each of the given locations.
  """
  source_points = np.asarray(source_points)
  sample_origins = np.asarray(sample_origins)

  if source_points.ndim != 1:
    raise ValueError(f'source_points must be 1D, got {source_points.shape=}')

  if sample_origins.ndim != 1:
    raise ValueError(f'sample_origins must be 1D, got {sample_origins.shape=}')

  source_steps = np.diff(source_points)
  source_step = source_steps[0]
  if np.any(source_steps != source_step):
    raise ValueError(f'source_points must have constant step: {source_points=}')

  start_points = sample_origins + stencil.start
  starts = _divide_evenly(start_points - source_points[0], source_step)
  if np.any(starts < 0):
    raise ValueError(
        f'first stencil point must be non-negative: {start_points=}'
    )
  if not stencil.includes_start:
    starts += 1

  stop_points = sample_origins + stencil.stop
  stops = _divide_evenly(stop_points - source_points[0], source_step)
  if np.any(stops >= source_points.size):
    raise ValueError(
        'last stencil point must be less than the number of source points:'
        f' {stop_points=}, {source_points=}'
    )
  if stencil.includes_stop:
    stops += 1

  stride = _divide_evenly(stencil.step, source_step).item()

  return [
      slice(start, stop, stride)
      for start, stop in zip(starts.tolist(), stops.tolist())
  ]
