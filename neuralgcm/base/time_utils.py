# Copyright 2026 Google LLC
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

"""Timedelta slicing and step ratio utilities for NeuralGCM models."""

import coordax as cx
import numpy as np
from terrax.core import coordinates


def exact_timedelta_ratio(
    numerator: np.timedelta64,
    denominator: np.timedelta64,
    error_message: str,
) -> int:
  """Returns `numerator / denominator`, which must be a positive integer."""
  ratio, remainder = divmod(numerator, denominator)
  if ratio < 1 or remainder:
    raise ValueError(error_message)
  return int(ratio)


def _timedelta_indices(field: cx.Field, deltas: np.ndarray) -> np.ndarray:
  """Returns indices of `deltas` along the `timedelta` axis of `field`."""
  timedelta = cx.coords.extract(field.coordinate, coordinates.TimeDelta)
  matches = timedelta.deltas[:, np.newaxis] == deltas
  if missing := [str(d) for d in deltas[~matches.any(axis=0)]]:
    raise ValueError(
        f'Inputs must include timedelta={", ".join(missing)}, got '
        f'{timedelta.deltas=} in {field=}'
    )
  return matches.argmax(axis=0)


def select_zero_timedelta(field: cx.Field) -> cx.Field:
  """Returns `field` at timedelta=0, or `field` as is if it has no timedelta."""
  if 'timedelta' not in field.dims:
    return field
  [idx] = _timedelta_indices(field, np.array([0], dtype='timedelta64[s]'))
  return cx.cmap(lambda x: x[idx])(field.untag('timedelta'))


def select_timedeltas(
    field: cx.Field, timedelta: coordinates.TimeDelta
) -> cx.Field:
  """Returns `field` at values of `timedelta`, which must be in its axis."""
  idx = _timedelta_indices(field, timedelta.deltas)
  return cx.cmap(lambda x: x[idx])(field.untag('timedelta')).tag(timedelta)
