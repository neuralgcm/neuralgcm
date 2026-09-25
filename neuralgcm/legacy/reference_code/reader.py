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
# pylint: disable=line-too-long
# pyformat: disable
"""Xarray based readers for feeding time-series into tf.data."""
# pyformat: enable
from __future__ import annotations
from collections import abc
import concurrent.futures
import dataclasses
import logging
import math
import random
from typing import Callable, Optional, TypeVar

import numpy as np
import tensorflow as tf
import xarray


# pylint: disable=logging-fstring-interpolation


def _xarray_bytes_per_element(
    source: xarray.Dataset, exclude_dims: set[str]
) -> int:
  bytes_per_element = 0
  for variable in source.values():
    items_per_element = math.prod(
        size for dim, size in variable.sizes.items() if dim not in exclude_dims
    )
    bytes_per_element += variable.dtype.itemsize * items_per_element
  return bytes_per_element


def _calculate_block_size(
    source: xarray.Dataset,
    block_dims: list[str],
    bytes_per_request: float,
    min_elements_per_request: int = 1,
) -> int:
  """Calculate the size of blocks to read simultaneously from disk."""
  bytes_per_element = _xarray_bytes_per_element(source, set(block_dims))
  elements_per_request = round(bytes_per_request / bytes_per_element)
  max_elements = math.prod(source.sizes[dim] for dim in block_dims)
  elements_per_request = min(
      max(elements_per_request, min_elements_per_request), max_elements
  )
  return elements_per_request


def _iterate_windowed_block_slices(
    sample_size: int,
    total_size: int,
    block_size: int,
    stride_between_samples: int = 1,
    output_window_stride: int = 1,
    first_sample_offset: int = 0,
) -> abc.Iterator[slice]:
  """Yields slices for every block needed to generate windowed samples.

  Args:
    sample_size: size of each sample.
    total_size: total size of the dimension being sampled along.
    block_size: desired size of blocks to read from disk.
    stride_between_samples: shift between starts of sampled windows.
    output_window_stride: shift between samples within a window.
    first_sample_offset: offset of the first sample.

  Yields:
    Slice objects with integer bounds for each block.
  """
  assert stride_between_samples >= 1
  assert output_window_stride >= 1
  assert first_sample_offset >= 0

  sample_input_size = (
      range(0, sample_size * output_window_stride, output_window_stride)[-1] + 1
  )
  assert 0 < sample_input_size <= block_size <= total_size

  sample_stop = 0  # unused

  # first block
  block_start = first_sample_offset
  block_stop = first_sample_offset + block_size

  # iterate through all slices, in order
  for start in range(
      first_sample_offset,
      total_size - sample_input_size + 1,
      stride_between_samples,
  ):
    prev_sample_stop = sample_stop
    sample_stop = start + sample_input_size

    if sample_stop > block_stop:
      # yield previous block
      assert prev_sample_stop > 0
      yield slice(block_start, prev_sample_stop)

      # begin new block
      block_start = start
      block_stop = start + block_size

  if sample_stop > block_start:
    # yield the final block
    yield slice(block_start, sample_stop)


def _drop_static_vars(dataset: xarray.Dataset) -> xarray.Dataset:
  """Drop fields that are static and do not vary with time."""
  vars_to_drop = [k for k, v in dataset.items() if 'time' not in v.dims[0]]  # pytype: disable=unsupported-operands
  return dataset.drop_vars(vars_to_drop)


NestedTensors = TypeVar('NestedTensors', tf.Tensor, dict[str, tf.Tensor])


@tf.function(jit_compile=True, autograph=False)
def rolling_window_tensors(
    inputs: NestedTensors, /, size: int, shift: int = 1, stride: int = 1
) -> NestedTensors:
  """Calculate a tensor of rolling windows.

  Example usage:

    >>> rolling_window_tensors(tf.range(10), size=6, shift=2)
    <tf.Tensor: shape=(3, 6), dtype=int32, numpy=
    array([[0, 1, 2, 3, 4, 5],
          [2, 3, 4, 5, 6, 7],
          [4, 5, 6, 7, 8, 9]], dtype=int32)>

    >>> rolling_window_tensors(tf.range(10), size=4, stride=2)
    <tf.Tensor: shape=(4, 4), dtype=int32, numpy=
    array([[0, 2, 4, 6],
          [1, 3, 5, 7],
          [2, 4, 6, 8],
          [3, 5, 7, 9]], dtype=int32)>

  Args:
    inputs: nested data structure with tf.Tensor values of shape [T, ...].
    size: size of the time dimension in rolling window samples.
    shift: shift between subsequent window samples along time.
    stride: shift within a window along time.

  Returns:
    Nested tensors of shape [S, W, ...] sampled from inputs, where S is the
    number of samples and W is the window size.
  """

  def calculate_windows(tensor):
    shifts = tf.range(0, tf.shape(tensor)[0] - stride * (size - 1), shift)
    indices = tf.range(0, size * stride, stride)
    samples = tf.vectorized_map(
        lambda shift: tf.gather(tensor, shift + indices), shifts
    )
    samples = tf.ensure_shape(samples, [None, size] + tensor.shape[1:])
    return samples

  return tf.nest.map_structure(calculate_windows, inputs)


class Sampler:
  """Base class for sampling from blocks."""

  def list_block_slices(self, block_size: int, total_size: int) -> list[slice]:
    """Returns a list of slices bounding blocks to sample from."""
    raise NotImplementedError

  def sample_block(self, data: NestedTensors) -> NestedTensors:
    """Returns sample tensors from block tensors."""
    raise NotImplementedError

  @property
  def example_size(self) -> int:
    """Size of each example."""
    raise NotImplementedError

  def examples_per_block(self, block_size: int) -> int:
    """Number of examples per block."""
    raise NotImplementedError


@dataclasses.dataclass
class Splitter(Sampler):
  """Split samples along the first axis."""


  def list_block_slices(self, block_size: int, total_size: int) -> list[slice]:
    return [
        slice(start, min(start + block_size, total_size))
        for start in range(0, total_size, block_size)
    ]

  def sample_block(self, data: NestedTensors) -> NestedTensors:
    # Insert a dummy dimension for time-series length, which is always one.
    return tf.nest.map_structure(lambda x: x[:, tf.newaxis, ...], data)

  @property
  def example_size(self) -> int:
    return 1

  def examples_per_block(self, block_size: int) -> int:
    return block_size


@dataclasses.dataclass
class Windower(Sampler):
  """Sample rolling windows along the first axis.

  Attributes:
    window_size: size of output windows.
    stride_between_windows: offset between starting sequential windows.
    output_window_stride: separation between between observations within a
      window.
    first_window_offset: offset of starting the first window.
  """

  window_size: int
  stride_between_windows: int
  output_window_stride: int = 1
  first_window_offset: int = 0

  def list_block_slices(self, block_size: int, total_size: int) -> list[slice]:
    return list(
        _iterate_windowed_block_slices(
            sample_size=self.window_size,
            block_size=block_size,
            total_size=total_size,
            stride_between_samples=self.stride_between_windows,
            output_window_stride=self.output_window_stride,
            first_sample_offset=self.first_window_offset,
        )
    )

  def sample_block(self, data: NestedTensors) -> NestedTensors:
    # NOTE(shoyer): It is tempting to try to use tf.data.Dataset.window instead
    # for sampling windows, but that method does something different: it
    # calculates windows over Dataset elements, rather than calculating windows
    # within each Dataset element.
    return rolling_window_tensors(
        data,
        size=self.window_size,
        shift=self.stride_between_windows,
        stride=self.output_window_stride,
    )

  @property
  def example_size(self) -> int:
    return self.window_size

  def examples_per_block(self, block_size: int) -> int:
    stop = block_size - self.output_window_stride * (self.window_size - 1)
    return len(range(0, stop, self.stride_between_windows))


@dataclasses.dataclass
class WindowerAtOffsets(Sampler):
  """Sample rolling windows along the first axis at specified offsets."""

  window_size: int
  window_offsets: list[int]
  output_window_stride: int = 1

  def list_block_slices(self, block_size: int, total_size: int) -> list[slice]:
    stride = self.output_window_stride
    # This suffices for now because generally we cache evaluation data.
    sample_input_size = range(0, self.window_size * stride, stride)[-1] + 1
    assert 0 < sample_input_size <= block_size <= total_size
    slices = []
    for start in self.window_offsets:
      stop = start + sample_input_size
      if stop > total_size:
        raise ValueError(
            f'offset at {start} needs data through {stop=}, which is beyond'
            f' {total_size=}'
        )
      slices.append(slice(start, stop))
    return slices

  def sample_block(self, data: NestedTensors) -> NestedTensors:
    def strided_sample(tensor):
      # Insert a dummy batch/sample dimension.
      return tf.ensure_shape(
          tensor[tf.newaxis, :: self.output_window_stride],
          [1, self.window_size] + tensor.shape[1:],
      )

    return tf.nest.map_structure(strided_sample, data)

  @property
  def example_size(self) -> int:
    return self.window_size

  def examples_per_block(self, block_size: int) -> int:
    del block_size  # unused
    return 1


class Selector:
  """Base class for block selection."""

  def select(self, blocks: list[slice]) -> list[slice]:
    """Select a subset of blocks for sampling."""
    raise NotImplementedError


class CompleteSelector(Selector):

  def select(self, blocks: list[slice]) -> list[slice]:
    return blocks


@dataclasses.dataclass
class ShardSelector(Selector):
  shard_index: int
  shard_count: int

  def select(self, blocks: list[slice]) -> list[slice]:
    return [
        block
        for i, block in enumerate(blocks)
        if i % self.shard_count == self.shard_index
    ]


@dataclasses.dataclass
class ShuffleSelector(Selector):
  seed: int = 0
  reshuffle_each_iteration: bool = True

  def select(self, blocks: list[slice]) -> list[slice]:
    rng = random.Random(self.seed)
    if self.reshuffle_each_iteration:
      self.seed = rng.randrange(2**63)
    return rng.sample(blocks, k=len(blocks))


@dataclasses.dataclass
class ComposedSelector(Selector):
  components: list[Selector]

  def select(self, blocks: list[slice]) -> list[slice]:
    for component in self.components:
      blocks = component.select(blocks)
    return blocks


@dataclasses.dataclass
class CustomSelector(Selector):
  select: Callable[[list[slice]], list[slice]]


def _thread_pool_loader(max_workers: int = 100):
  """Dataset loader using a large thread pool for concurrency."""
  # We use a separate thread for reading each data variable in each block.
  executor = concurrent.futures.ThreadPoolExecutor(max_workers)

  def load(dataset: xarray.Dataset) -> xarray.Dataset:
    arrays = executor.map(lambda var: var.values, dataset.values())
    return dataset.copy(data={k: v for k, v in zip(dataset, arrays)})

  return load


class _Reader:
  """Class for reading an xarray.Dataset."""

  def __init__(
      self,
      source: xarray.Dataset,
      sampler: Sampler,
      block_selector: Selector = CompleteSelector(),
      *,
      sample_dim: str = 'time',
      block_size_in_bytes: float = 1e8,
      parallel_block_reads: int = tf.data.AUTOTUNE,
      parallel_samples: int = tf.data.AUTOTUNE,
      dataset_loader: Optional[
          Callable[[xarray.Dataset], xarray.Dataset]
      ] = None,
  ):
    if dataset_loader is None:
      # In principle, it could make sense to support passing alternative
      # loaders, such as xarray_tensorstore.read() or a dask loader that calls
      # .compute(). We don't yet have any use cases where this seems to make a
      # difference, though. (The thread pool loader works as well as
      # xarray_tensorstore.read.)
      dataset_loader = _thread_pool_loader()

    if sample_dim not in source.dims:
      raise ValueError(
          'source does not include variables with a'
          f' {sample_dim!r} dimension:\n{source}'
      )
    source = _drop_static_vars(source)
    source = source.transpose(sample_dim, ...)

    block_size = _calculate_block_size(
        source,
        block_dims=[sample_dim],
        bytes_per_request=block_size_in_bytes,
        min_elements_per_request=sampler.example_size,
    )

    block_slices = sampler.list_block_slices(
        block_size, source.sizes[sample_dim]
    )

    bytes_per_element = _xarray_bytes_per_element(source, {sample_dim})
    bytes_per_example = sampler.example_size * bytes_per_element
    examples_per_block = sampler.examples_per_block(block_size)
    sample_bytes_per_block = bytes_per_example * examples_per_block
    expansion = sample_bytes_per_block / block_size_in_bytes
    logging.info(
        f'picked block_size={block_size}, corresponding to {len(block_slices)} '
        f'blocks with examples_per_block={examples_per_block}, based on '
        f'sampler={sampler} and {block_size_in_bytes=:g}. '
        f'{sample_bytes_per_block=:g} is a {expansion:1.2f}x expansion.'
    )

    self.source = source
    self.sampler = sampler
    self.block_selector = block_selector
    self.parallel_block_reads = parallel_block_reads
    self.parallel_samples = parallel_samples
    self.dataset_loader = dataset_loader

    self.block_size = block_size
    self.block_slices = block_slices
    self.bytes_per_example = bytes_per_example
    self.examples_per_block = examples_per_block

  def read(self) -> tf.data.Dataset:
    """Read this dataset into a tf.data.Dataset."""

    def generate_blocks():
      for block in self.block_selector.select(self.block_slices):
        yield (block.start, block.stop)

    def np_read_block(start: np.ndarray, stop: np.ndarray) -> list[np.ndarray]:
      selection = self.source.isel(time=slice(start, stop))
      loaded = self.dataset_loader(selection)
      arrays = [x.values for x in loaded.values()]
      return arrays

    def tf_read_block(start: tf.Tensor, stop: tf.Tensor):
      dtypes = [v.dtype for v in self.source.values()]
      shapes = [(None,) + v.shape[1:] for v in self.source.values()]
      tensors = tf.numpy_function(np_read_block, [start, stop], dtypes)
      for tensor, shape in zip(tensors, shapes):
        tensor.set_shape(shape)
      return dict(zip(self.source.keys(), tensors))

    data = tf.data.Dataset.from_generator(
        generate_blocks, output_signature=2 * (tf.TensorSpec((), tf.int64),)
    )
    data = data.map(tf_read_block, num_parallel_calls=self.parallel_block_reads)
    data = data.map(
        self.sampler.sample_block, num_parallel_calls=self.parallel_samples
    )

    data = data.unbatch()

    return data


def read_timeseries(
    source: xarray.Dataset,
    sampler: Sampler,
    block_selector: Selector = CompleteSelector(),
    *,
    sample_dim: str = 'time',
    block_size_in_bytes: float = 1e8,
    parallel_block_reads: int = tf.data.AUTOTUNE,
    parallel_samples: int = tf.data.AUTOTUNE,
) -> tf.data.Dataset:
  """Read a time-series xarray.Dataset into a tf.data.Dataset of windows.

  See go/whirl-zarr-reader for a detailed description of the design.

  Args:
    source: lazy xarray.Dataset, e.g., opened from a Zarr file with
      `open_zarr(..., chunks=None)`. All data variables with a 'time' dimension
      will be sampled. Note: setting `chunks=None` to avoid using Dask is
      preferred for optimal performance.
    sampler: specification of what time-series samples of this dataset should
      look like. Currently the only supported sampler is Windower.
    block_selector: selector called at each pass through the source dataset,
      indicating the blocks to read in order. The returned blocks should be a
      subset of passed in blocks.
    sample_dim: name of the dimension to sample along.
    block_size_in_bytes: number of bytes to use for each reading a "block" of
      data from the source data. Larger block sizes are more efficient.
    parallel_block_reads: number of blocks to read in parallel.
    parallel_samples: number of threads to use for generating samples from
      blocks.

  Returns:
    tf.data.Dataset where each element is a dict of arrays.
  """
  return _Reader(
      source=source,
      sampler=sampler,
      block_selector=block_selector,
      sample_dim=sample_dim,
      block_size_in_bytes=block_size_in_bytes,
      parallel_block_reads=parallel_block_reads,
      parallel_samples=parallel_samples,
  ).read()


def read_shuffled_shard(
    source: xarray.Dataset,
    sampler: Sampler,
    *,
    sample_dim: str = 'time',
    block_size_in_bytes: float = 1e8,
    buffer_size_in_bytes: float = 1e10,
    min_buffer_blocks: float = 10,
    parallel_block_reads: int = tf.data.AUTOTUNE,
    parallel_samples: int = tf.data.AUTOTUNE,
    shard_index: Optional[int] = None,
    shard_count: Optional[int] = None,
    seed: int = 0,
    reshuffle_each_iteration: bool = True,
) -> tf.data.Dataset:
  """Read a time-series with samples in randomly shuffled order.

  Args:
    source: lazy xarray.Dataset, e.g., opened from a Zarr file with
      `open_zarr(..., chunks=None)`. All data variables with a 'time' dimension
      will be sampled. Note: setting `chunks=None` to avoid using Dask is
      preferred for optimal performance.
    sampler: specification of what time-series samples of this dataset should
      look like.
    sample_dim: name of the dimension to sample along.
    block_size_in_bytes: number of bytes to use for each reading a "block" of
      data from the source data. Larger block sizes are more efficient.
    buffer_size_in_bytes: number of bytes to use in the shuffle buffer.
    min_buffer_blocks: minimum number of blocks that must be represented in the
      shuffle buffer, if more than one sample is taken from each block.
      Typically this should be at least as large as the batch size.
    parallel_block_reads: number of blocks to read in parallel.
    parallel_samples: number of threads to use for generating samples from
      blocks.
    shard_index: integer index for this shard of the data, in the range `[0,
      shard_count)`. In a multi-host JAX training setup, this should equal
      `jax.process_index()`.
    shard_count: total number of data shards. In a multi-host JAX training
      setup, this should equal `jax.process_count()`.
    seed: seed to use for random number generation.
    reshuffle_each_iteration: whether to use a new shuffle order for elements
      after each iteration through `source` or not.

  Returns:
    tf.data.Dataset where each element is a dict of arrays.
  """
  if shard_index is None and shard_count is None:
    shard_index = 0
    shard_count = 1

  if shard_index is None or shard_count is None:
    raise ValueError('must set both or neither of shard_index and shard_count')

  selector = ComposedSelector([
      ShardSelector(shard_index, shard_count),
      ShuffleSelector(seed, reshuffle_each_iteration),
  ])

  def _make_reader(block_size_in_bytes):
    reader = _Reader(
        source=source,
        sampler=sampler,
        sample_dim=sample_dim,
        block_selector=selector,
        block_size_in_bytes=block_size_in_bytes,
        parallel_block_reads=parallel_block_reads,
        parallel_samples=parallel_samples,
    )
    buffer_size = int(buffer_size_in_bytes / reader.bytes_per_example)
    logging.info(
        f'picked shuffle buffer size of {buffer_size} based on '
        f'{buffer_size_in_bytes=:g}'
    )
    return reader, buffer_size

  reader, buffer_size = _make_reader(block_size_in_bytes)

  if buffer_size:
    examples_per_block = reader.examples_per_block
    buffer_blocks = buffer_size / examples_per_block
    if examples_per_block > 1 and buffer_blocks < min_buffer_blocks:
      block_size_in_bytes = reader.bytes_per_example
      logging.warning(
          'insufficient diversity in proposed shuffle buffer: '
          f'{examples_per_block=} and {buffer_size=} means that on average '
          f'only {buffer_blocks:g} blocks will be represented in the shuffle '
          f'buffer, which is less than {min_buffer_blocks=}. Falling back to '
          f'one example per block ({block_size_in_bytes=:g}).'
      )
      reader, buffer_size = _make_reader(block_size_in_bytes)
      assert reader.examples_per_block == 1

  data = reader.read()

  if buffer_size:
    # for testing, disable the shuffle buffer if it has size zero (the shuffle
    # method does not support size zero buffers)
    data = data.shuffle(buffer_size, seed, reshuffle_each_iteration)

  return data
