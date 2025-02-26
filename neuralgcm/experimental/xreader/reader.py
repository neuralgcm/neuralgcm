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
"""Xarray data reader for NeuralGCM training."""

from __future__ import annotations

import abc
import concurrent.futures
import dataclasses
import functools
import logging
import math
from typing import Any, Callable, TYPE_CHECKING, TypeVar

import grain.python as grain
import jax
import numpy as np
import xarray

if TYPE_CHECKING:
  # pylint: disable=g-import-not-at-top,g-bad-import-order
  import neuralgcm.experimental.jax_datetime as jdt

# pylint: disable=logging-fstring-interpolation


# Old versions of Grain (namely, 0.2.3) didn't expose WindowShuffleIterDataset
# as a public API.
if not hasattr(grain.experimental, 'WindowShuffleIterDataset'):
  # pylint: disable=g-import-not-at-top,g-bad-import-order
  from grain._src.python.dataset.transformations import shuffle  # pylint: disable=protected-access

  grain.experimental.WindowShuffleIterDataset = shuffle.WindowShuffleIterDataset


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
    sample_dims: list[str],
    bytes_per_request: float,
    min_elements_per_request: int = 1,
) -> int:
  """Calculate the size of blocks to read simultaneously from disk."""
  bytes_per_element = _xarray_bytes_per_element(source, set(sample_dims))
  elements_per_request = round(bytes_per_request / bytes_per_element)
  max_elements = math.prod(source.sizes[dim] for dim in sample_dims)
  elements_per_request = min(
      max(elements_per_request, min_elements_per_request), max_elements
  )
  return elements_per_request


def _drop_static_vars(
    dataset: xarray.Dataset, sample_dim: str
) -> xarray.Dataset:
  """Drop fields that are static and do not vary across the sample dimension."""
  vars_to_drop = [k for k, v in dataset.items() if sample_dim not in v.dims]
  return dataset.drop_vars(vars_to_drop)


PyTree = Any


class Sampler(abc.ABC):
  """Base class for sampling from blocks.

  Attributes:
    example_size: size of each example along the sampled dimension.
  """

  example_size: int

  @abc.abstractmethod
  def get_block_slices(self, block_size: int, total_size: int) -> list[slice]:
    """Returns a list of slices bounding blocks to sample from.

    Args:
      block_size: desired size of blocks to read from disk.
      total_size: total size of the dimension being sampled along.

    Returns:
      Slices for each block to sample from.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def sample_block(self, block: PyTree) -> list[PyTree]:
    """Returns sample pytrees from block pytrees.

    Args:
      block: pytree of arrays to sample from.

    Returns:
      List of pytrees of arrays sampled from this block.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def examples_per_block(self, block_size: int) -> int:
    """Number of examples per block."""
    raise NotImplementedError


def _leading_axis_size(block: PyTree) -> int:
  return jax.tree.leaves(block)[0].shape[0]


@dataclasses.dataclass(init=False)
class Windower(Sampler):
  """Sample rolling windows along the first axis.

  Attributes:
    example_size: size of output windows.
    output_window_stride: separation between between observations within a
      window.
    stride_between_windows: offset between starting sequential windows. By
      default, matches output_window_stride.
    first_window_offset: offset of starting the first window.
  """

  example_size: int
  output_window_stride: int
  stride_between_windows: int
  first_window_offset: int

  def __init__(
      self,
      example_size: int,
      *,
      output_window_stride: int = 1,
      stride_between_windows: int | None = None,
      first_window_offset: int = 0,
  ):
    if stride_between_windows is None:
      stride_between_windows = output_window_stride
    self.example_size = example_size
    self.output_window_stride = output_window_stride
    self.stride_between_windows = stride_between_windows
    self.first_window_offset = first_window_offset

  def get_block_slices(self, block_size: int, total_size: int) -> list[slice]:
    example_size = self.example_size
    output_window_stride = self.output_window_stride
    stride_between_windows = self.stride_between_windows
    first_sample_offset = self.first_window_offset

    assert stride_between_windows >= 1
    assert output_window_stride >= 1
    assert first_sample_offset >= 0

    sample_input_size = (
        range(0, example_size * output_window_stride, output_window_stride)[-1]
        + 1
    )
    assert 0 < sample_input_size <= block_size <= total_size

    sample_stop = 0  # unused

    # first block
    block_start = first_sample_offset
    block_stop = first_sample_offset + block_size

    slices = []

    # iterate through all slices, in order
    for start in range(
        first_sample_offset,
        total_size - sample_input_size + 1,
        stride_between_windows,
    ):
      prev_sample_stop = sample_stop
      sample_stop = start + sample_input_size

      if sample_stop > block_stop:
        # append the previous block
        assert prev_sample_stop > 0
        slices.append(slice(block_start, prev_sample_stop))

        # begin new block
        block_start = start
        block_stop = start + block_size

    if sample_stop > block_start:
      # append the final block
      slices.append(slice(block_start, sample_stop))

    return slices

  def _all_slice_starts(self, block_size: int) -> range:
    stop = block_size - self.output_window_stride * (self.example_size - 1)
    return range(0, stop, self.stride_between_windows)

  def sample_block(self, block: PyTree) -> list[PyTree]:
    block_size = _leading_axis_size(block)
    starts = self._all_slice_starts(block_size)
    size = self.example_size * self.output_window_stride
    stride = self.output_window_stride
    return [
        jax.tree.map(lambda x: x[start : start + size : stride], block)  # pylint: disable=cell-var-from-loop
        for start in starts
    ]

  def examples_per_block(self, block_size: int) -> int:
    return len(self._all_slice_starts(block_size))


@dataclasses.dataclass
class WindowerAtOffsets(Sampler):
  """Sample rolling windows along the first axis at specified offsets."""

  example_size: int
  window_offsets: list[int]
  output_window_stride: int = 1

  def get_block_slices(self, block_size: int, total_size: int) -> list[slice]:
    stride = self.output_window_stride
    # TODO(shoyer): optimize this Sampler to read more than 1 example per block.
    # This suffices for now because generally we cache evaluation data.
    sample_input_size = range(0, self.example_size * stride, stride)[-1] + 1
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

  def sample_block(self, block: PyTree) -> list[PyTree]:
    return [jax.tree.map(lambda x: x[:: self.output_window_stride], block)]

  def examples_per_block(self, block_size: int) -> int:
    del block_size  # unused
    return 1


def _thread_pool_loader(max_workers: int = 100):
  """Dataset loader using a large thread pool for concurrency."""
  # We use a separate thread for reading each data variable in each block. This
  # should suffice for maximum concurrency when reading from Zarr, as long as
  # the Zarr store uses internal concurrency for loading each variable.
  executor = concurrent.futures.ThreadPoolExecutor(max_workers)

  def load(dataset: xarray.Dataset) -> xarray.Dataset:
    arrays = executor.map(lambda var: var.values, dataset.values())
    return dataset.copy(data={k: v for k, v in zip(dataset, arrays)})

  return load


@functools.cache
def _array_converter(dtype_kind: str) -> Callable[[np.ndarray], PyTree]:
  # NumPy dtype kinds corresponding to datetime64 and timedelta64:
  # https://numpy.org/doc/stable/reference/generated/numpy.dtype.kind.html
  if dtype_kind in 'mM':
    import neuralgcm.experimental.jax_datetime as jdt  # pylint: disable=g-import-not-at-top

    to_jdt = {'M': jdt.to_datetime, 'm': jdt.to_timedelta}[dtype_kind]
    return to_jdt
  else:
    return lambda x: x


def to_cpu_array(data: np.ndarray) -> np.ndarray | jdt.Datetime | jdt.Timedelta:
  converter = _array_converter(data.dtype.kind)
  return converter(data)


T = TypeVar('T')


def _leaves_to_dict(names, arrays):
  assert len(arrays) == len(names)
  return dict(zip(names, arrays))


def _default_leaf_wrapper(
    source: xarray.Dataset,
) -> Callable[[list[T]], dict[str, T]]:

  names = list(source.keys())
  return functools.partial(_leaves_to_dict, names)


@dataclasses.dataclass(repr=False, eq=False)
class _XarrayBlockSource(grain.RandomAccessDataSource[T]):
  """Grain data source for reading blocks from an xarray.Dataset."""

  source: xarray.Dataset
  blocks: list[slice]
  leaf_wrapper: Callable[[list[Any]], T]

  @functools.cached_property
  def loader(self) -> Callable[[xarray.Dataset], xarray.Dataset]:
    # In principle, it could make sense to support passing alternative
    # loaders, such as xarray_tensorstore.read() or a dask loader that calls
    # .compute(). We don't yet have any use cases where this seems to make a
    # difference, though. (The thread pool loader works as well as
    # xarray_tensorstore.read.)
    return _thread_pool_loader()

  def __getitem__(self, index):
    selection = self.source.isel(time=self.blocks[index])
    loaded = self.loader(selection)
    arrays = [to_cpu_array(x.values) for x in loaded.values()]
    return self.leaf_wrapper(arrays)

  def __len__(self):
    return len(self.blocks)

  def __getstate__(self):
    # Cannot pickle the loader because it includes a thread pool.
    return {
        'source': self.source,
        'blocks': self.blocks,
        'leaf_wrapper': self.leaf_wrapper,
    }


def _prepare_source(source: xarray.Dataset, sample_dim: str) -> xarray.Dataset:
  """Prepare an xarray.Dataset for reading."""
  # TODO(shoyer): support multiple sample dimensions
  if sample_dim not in source.dims:
    raise ValueError(
        'source does not include variables with a'
        f' {sample_dim!r} dimension:\n{source}'
    )
  if sample_dim in source.indexes:
    source = source.reset_index(sample_dim)
  if sample_dim in source.coords:
    source = source.reset_coords(sample_dim)
  source = _drop_static_vars(source, sample_dim)
  source = source.transpose(sample_dim, ...)
  return source


@dataclasses.dataclass
class _UnbatchTransform(grain.experimental.FlatMapTransform):
  max_fan_out: int

  def flat_map(self, element):
    return element


class _Reader:
  """Shared logic for loading an xarray.Dataset."""

  # TODO(shoyer): explore this as a public API?

  def __init__(
      self,
      source: xarray.Dataset,
      sampler: Sampler,
      *,
      sample_dim: str = 'time',
      block_size_in_bytes: float = 1e8,
      leaf_wrapper: Callable[[list[np.ndarray]], PyTree] | None = None,
  ):
    source = _prepare_source(source, sample_dim)

    if leaf_wrapper is None:
      leaf_wrapper = _default_leaf_wrapper(source)

    block_size = _calculate_block_size(
        source,
        sample_dims=[sample_dim],
        bytes_per_request=block_size_in_bytes,
        min_elements_per_request=sampler.example_size,
    )
    block_slices = sampler.get_block_slices(
        block_size=block_size, total_size=source.sizes[sample_dim]
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
    self.block_size = block_size
    self.bytes_per_block = bytes_per_example * examples_per_block
    self.bytes_per_example = bytes_per_example
    self.examples_per_block = examples_per_block

    self.block_source = _XarrayBlockSource(
        source=source,
        blocks=block_slices,
        leaf_wrapper=leaf_wrapper,
    )


def read_timeseries(
    source: xarray.Dataset,
    sampler: Sampler,
    *,
    sample_dim: str = 'time',
    num_epochs: int | None = 1,
    leaf_wrapper: Callable[[list[Any]], PyTree] | None = None,
    block_size_in_bytes: float = 1e8,
    read_options: grain.ReadOptions | None = None,
) -> grain.IterDataset:
  """Read a time-series xarray.Dataset into an iterator of samples.

  Args:
    source: lazy xarray.Dataset, e.g., opened from a Zarr file with
      `open_zarr(..., chunks=None)`. All data variables with a 'time' dimension
      will be sampled. Note: setting `chunks=None` to avoid using Dask is
      preferred for optimal performance.
    sampler: specification of what time-series samples of this dataset should
      look like. Currently the only supported sampler is Windower.
    sample_dim: name of the dimension to sample along.
    num_epochs: number of epoch for which to read the dataset, or `None` to read
      indefinitely.
    leaf_wrapper: function that wraps the leaves of the source dataset into a
      PyTree.
    block_size_in_bytes: number of bytes to use for each reading a "block" of
      data from the source data. Larger block sizes are more efficient.
    read_options: options to use for reading the Grain dataset. Buffer size here
      indicates the number of blocks to prefetch.

  Returns:
    grain.IterDataset where each element is a pytree of arrays.
  """
  if read_options is None:
    read_options = grain.ReadOptions(num_threads=16, prefetch_buffer_size=32)

  reader = _Reader(
      source=source,
      sampler=sampler,
      sample_dim=sample_dim,
      block_size_in_bytes=block_size_in_bytes,
      leaf_wrapper=leaf_wrapper,
  )
  # Our strategy of emitting multiple examples per block means that we need to
  # drop into Grain's lower-level Dataset API.
  # We use IterDataset instead of MapDataset because the flat map transform on
  # MapDataset will re-read the entire block for each example, rather than
  # fetching it all at once.
  dataset = grain.MapDataset.source(reader.block_source)
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.map(sampler.sample_block)
  dataset = dataset.to_iter_dataset(read_options=read_options)
  dataset = grain.experimental.FlatMapIterDataset(
      dataset, _UnbatchTransform(reader.examples_per_block)
  )
  return dataset


def read_shuffled_shard(
    source: xarray.Dataset,
    sampler: Sampler,
    *,
    sample_dim: str = 'time',
    num_epochs: int | None = 1,
    leaf_wrapper: Callable[[list[Any]], PyTree] | None = None,
    block_size_in_bytes: float = 1e8,
    buffer_size_in_bytes: float = 1e10,
    min_buffer_blocks: float = 10,
    read_options: grain.ReadOptions | None = None,
    shard_index: int | None = None,
    shard_count: int | None = None,
    seed: int = 0,
) -> grain.IterDataset:
  """Read a time-series into an iterator of samples in randomly shuffled order.

  Args:
    source: lazy xarray.Dataset, e.g., opened from a Zarr file with
      `open_zarr(..., chunks=None)`. All data variables with a 'time' dimension
      will be sampled. Note: setting `chunks=None` to avoid using Dask is
      preferred for optimal performance.
    sampler: specification of what time-series samples of this dataset should
      look like.
    sample_dim: name of the dimension to sample along.
    num_epochs: number of epoch for which to read the dataset, or `None` to read
      indefinitely.
    leaf_wrapper: function that wraps the leaves of the source dataset into a
      PyTree.
    block_size_in_bytes: number of bytes to use for each reading a "block" of
      data from the source data. Larger block sizes are more efficient.
    buffer_size_in_bytes: number of bytes to use in the shuffle window.
    min_buffer_blocks: minimum number of blocks that must be represented in the
      shuffle buffer, if more than one sample is taken from each block.
      Typically this should be at least as large as the batch size.
    read_options: options to use for reading the Grain dataset. Buffer size here
      indicates the number of blocks to prefetch.
    shard_index: integer index for this shard of the data, in the range `[0,
      shard_count)`. In a multi-host JAX training setup, this should equal
      `jax.process_index()`.
    shard_count: total number of data shards. In a multi-host JAX training
      setup, this should equal `jax.process_count()`.
    seed: seed to use for random number generation.

  Returns:
    grain.IterDataset where each element is a pytree of arrays.
  """
  if shard_index is None and shard_count is None:
    shard_index = 0
    shard_count = 1

  if shard_index is None or shard_count is None:
    raise ValueError('must set both or neither of shard_index and shard_count')

  def _make_reader(block_size_in_bytes):
    reader = _Reader(
        source=source,
        sampler=sampler,
        sample_dim=sample_dim,
        block_size_in_bytes=block_size_in_bytes,
        leaf_wrapper=leaf_wrapper,
    )
    shuffle_window = round(buffer_size_in_bytes / reader.bytes_per_example)
    logging.info(
        f'picked shuffle window size of {shuffle_window} based on '
        f'{buffer_size_in_bytes=:g}'
    )
    return reader, shuffle_window

  reader, shuffle_window = _make_reader(block_size_in_bytes)

  if shuffle_window:
    examples_per_block = reader.examples_per_block
    buffer_blocks = shuffle_window / examples_per_block
    if examples_per_block > 1 and buffer_blocks < min_buffer_blocks:
      block_size_in_bytes = reader.bytes_per_example
      logging.warning(
          'insufficient diversity in proposed shuffle buffer: '
          f'{examples_per_block=} and {shuffle_window=} means that on average '
          f'only {buffer_blocks:g} blocks will be represented in the shuffle '
          f'buffer, which is less than {min_buffer_blocks=}. Falling back to '
          f'one example per block ({block_size_in_bytes=:g}).'
      )
      reader, shuffle_window = _make_reader(block_size_in_bytes)
      assert reader.examples_per_block == 1

  if read_options is None:
    buffer_blocks = 2 * round(buffer_size_in_bytes / reader.bytes_per_block)
    read_options = grain.ReadOptions(
        num_threads=16, prefetch_buffer_size=buffer_blocks
    )

  rng = np.random.default_rng(seed)
  global_seed = rng.integers(2**32)
  window_seed = rng.integers(2**32)

  dataset = grain.MapDataset.source(reader.block_source)
  dataset = dataset[shard_index::shard_count]
  dataset = dataset.shuffle(global_seed)
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.to_iter_dataset(read_options=read_options)
  dataset = dataset.map(sampler.sample_block)
  dataset = grain.experimental.FlatMapIterDataset(
      dataset, _UnbatchTransform(reader.examples_per_block)
  )
  if shuffle_window:
    dataset = grain.experimental.WindowShuffleIterDataset(
        dataset,
        window_size=shuffle_window,
        seed=window_seed,
    )

  return dataset
