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
"""Simple timing utilities for measuring performance."""

from __future__ import annotations

from collections.abc import Callable
import math
import time
from typing import Generic, TypeVar

import jax


T = TypeVar('T')


class Timed(Generic[T]):
  """Function wrapps for timing evaluations."""

  def __init__(self, func: Callable[..., T], block_until_ready: bool = True):
    """Constructor.

    Args:
      func: function to wrap.
      block_until_ready: whether to use jax.block_until_ready() wait until
        JAX computations are complete when timing the function. This is a no-op
        if the function does not return any JAX arrays.
    """
    self._func = func
    self._timer = Timer()
    self._block_until_ready = block_until_ready

  @property
  def timer(self) -> Timer:
    return self._timer

  def __call__(self, *args, **kwargs) -> T:
    with self.timer:
      result = self._func(*args, **kwargs)
      if self._block_until_ready:
        result = jax.block_until_ready(result)
      return result


class Timer:
  """A minimal object for measuring elapsed time per step."""

  def __init__(
      self,
      counter: Callable[[], float] = time.perf_counter,
  ):
    """Constructor.

    Args:
      counter: function that returns a monotonically increasing time. By
        default, uses time.perf_counter to report the current time in seconds.
    """
    self._counter = counter
    self._start = None
    self._last = math.nan
    self._total = 0.0

  @property
  def last(self) -> float:
    return self._last

  @property
  def total(self) -> float:
    return self._total

  def running(self) -> bool:
    return self._start is not None

  def reset_total(self):
    self._total = 0.0

  def begin_step(self):
    if self.running():
      raise RuntimeError('Timer is already timing a step')
    self._start = self._counter()

  def finish_step(self):
    if not self.running():
      raise RuntimeError('Timer is not running a step')
    elapsed = self._counter() - self._start
    self._last = elapsed
    self._total += elapsed
    self._start = None

  def __enter__(self):
    self.begin_step()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.finish_step()
