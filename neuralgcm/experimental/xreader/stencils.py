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


@dataclasses.dataclass(frozen=True)
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
    if self.start == self.stop:
      if self.step != self.stop - self.start:  # check step is zero.
        raise ValueError(
            'For single value stencil ``step`` must equal zero: '
            f'{self.step=} vs {self.stop - self.start=}'
        )
      if self.closed != 'both':
        raise ValueError(
            'For single value stencil ``closed`` must be "both": '
            f'{self.closed=}'
        )
    elif self.start > self.stop:
      raise ValueError(
          f'start must not be greater than stop: {self.start} vs {self.stop}'
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
    num = _divide_evenly(self.stop - self.start, self.step) if self.step else 0
    result = self.start + self.step * np.arange(num + 1)
    if not self.includes_start and self.step:
      result = result[1:]
    if not self.includes_stop and self.step:
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


def _to_timedelta64(value: str | np.timedelta64) -> np.timedelta64:
  if isinstance(value, np.timedelta64):
    return value
  match = re.match(r'([+-]?\d+) ?([a-zA-Z]+)', value)
  if not match:
    raise ValueError(f'invalid time delta string: {value}')
  value = int(match.group(1))
  unit = _normalize_time_unit(match.group(2))
  return np.timedelta64(value, unit)


class TimeStencil(Stencil[np.timedelta64]):
  """Stencil specified by np.timedelta64.

  Example usage:

  >>> TimeStencil(start='-9h', stop='3h', step='1h', closed='both')
  TimeStencil(start='-9 hours', stop='3 hours', step='1 hours', closed='both')
  """

  def __init__(
      self,
      start: str | np.timedelta64,
      stop: str | np.timedelta64,
      step: str | np.timedelta64,
      closed: Closed = 'left',
  ):
    super().__init__(
        _to_timedelta64(start),
        _to_timedelta64(stop),
        _to_timedelta64(step),
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
    source_points: data coordinates at which the source data is defined. These
      must be evenly spaced.
    sample_origins: data coordinates corresponding to origin of sample stencils.
      These must be in order and match items from source_points. It is
      permisissable for sample_origins to contain arbitrary gaps.
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
  if not np.all(source_steps > 0):
    raise ValueError(f'source_points must be sorted: {source_points=}')

  source_step = source_steps[0]
  if np.any(source_steps != source_step):
    raise ValueError(f'source_points must have constant step: {source_points=}')

  if not np.all(np.diff(sample_origins) > 0):
    raise ValueError(f'sample_origins must be sorted: {sample_origins=}')

  start_points = sample_origins + stencil.start
  starts = _divide_evenly(start_points - source_points[0], source_step)
  if sample_origins[0] + stencil.points[0] < source_points[0]:
    raise ValueError(
        'all points in the stencil centered on the first sample_origin must be '
        'at or after the first source point: '
        f'{sample_origins[0] + stencil.points} vs {source_points[0]}'
    )
  if not stencil.includes_start:
    starts += 1

  stop_points = sample_origins + stencil.stop
  stops = _divide_evenly(stop_points - source_points[0], source_step)
  if sample_origins[-1] + stencil.points[-1] > source_points[-1]:
    raise ValueError(
        'all points in the stencil centered on the last sample_origin must be'
        ' at or before the last source point:'
        f' {sample_origins[-1] + stencil.points} vs {source_points[-1]}'
    )
  if stencil.includes_stop:
    # Add 1 since the slice end should be past the desired stop point
    stops += 1

  # Ensure that the stride is at least 1 to form valid slices.
  stride = max(_divide_evenly(stencil.step, source_step).item(), 1)

  return [
      slice(start, stop, stride)
      for start, stop in zip(starts.tolist(), stops.tolist())
  ]


def valid_origin_points(
    source_points: np.typing.ArrayLike, stencil: Stencil
) -> np.ndarray:
  """Get all possible sample origins for a given stencil.

  This trims the ends of the returned sample origin points to ensure that all
  samples fit within the range defined by source_points.

  Args:
    source_points: data coordinates at which the source data is defined. These
      must be evenly spaced.
    stencil: stencil objects that define the shape of a sample.

  Returns:
    An array of sample origins that meet the constraints set by
    source_points and stencil.
  """
  source_points = np.asarray(source_points)

  min_origin = source_points[0] - stencil.start
  if not stencil.includes_start:
    min_origin -= stencil.step

  max_origin = source_points[-1] - stencil.stop
  if not stencil.includes_stop:
    max_origin += stencil.step

  valid_origins = source_points[
      (source_points >= min_origin) & (source_points <= max_origin)
  ]
  return valid_origins
