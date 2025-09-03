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
"""Utilities for streaming execution in a separate thread."""

import concurrent.futures
from typing import Generic, Callable, TypeVar

from neuralgcm.experimental.inference import timing


class EmptyStreamError(Exception):
  pass


class FullStreamError(Exception):
  pass


T = TypeVar('T')


class SingleTaskExecutor(Generic[T]):
  """Executes an asynchronous function, repeatedly, using one thread.

  A SingleTaskExecutor object executes a single task at a time, and needs to
  be explicitly reset after every calculation with wait() or get().
  """

  def __init__(self, func: Callable[..., T], block_until_ready: bool = True):
    self._timed = timing.Timed(func, block_until_ready=block_until_ready)
    self._executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix=getattr(func, '__name__', '')
    )
    self._future = None

  @property
  def timer(self) -> timing.Timer:
    return self._timed.timer

  def running(self) -> bool:
    return self._future is not None and self._future.running()

  def get(self) -> T:
    """Fetch the latest calculation result."""
    if self._future is None:
      raise EmptyStreamError('cannot get() without a task to fetch')
    try:
      result = self._future.result()
    finally:
      self._future = None
    return result

  def wait(self):
    """Wait until any pending calculation is complete, discarding the result."""
    if self._future is not None:
      self.get()

  def submit(self, *args, **kwargs):
    """Start calculating func(*args, **kwargs)."""
    if self._future is not None:
      raise FullStreamError('cannot submit() with an unfetched task')
    self._future = self._executor.submit(self._timed, *args, **kwargs)
