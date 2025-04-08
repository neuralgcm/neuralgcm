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

import concurrent.futures
import dataclasses
import functools
from typing import Any, Callable, TypeVar

import grain.python as grain
import jax
from neuralgcm.experimental.xreader import stencils
import numpy as np
import xarray

# pylint: disable=logging-fstring-interpolation


# Old versions of Grain (namely, 0.2.3) didn't expose WindowShuffleIterDataset
# as a public API.
if not hasattr(grain.experimental, 'WindowShuffleIterDataset'):
  # pylint: disable=g-import-not-at-top,g-bad-import-order
  from grain._src.python.dataset.transformations import shuffle  # pylint: disable=protected-access

  grain.experimental.WindowShuffleIterDataset = shuffle.WindowShuffleIterDataset


T = TypeVar('T')


def _is_xarray_dataset(x: Any) -> bool:
  return isinstance(x, xarray.Dataset)


def _example_nbytes(
    source: xarray.Dataset | PyTree,
    stencil: stencils.Stencil | PyTree,
    sample_dim: str,
) -> int:
  """Calculate the number of elements in a single sample."""
  if not isinstance(source, xarray.Dataset):
    nbytes = jax.tree.map(
        functools.partial(_example_nbytes, sample_dim=sample_dim),
        source,
        stencil,
        is_leaf=_is_xarray_dataset,
    )
    return sum(jax.tree.leaves(nbytes))

  elements_per_sample = stencil.points.size
  bytes_per_element = source.isel({sample_dim: slice(1)}).nbytes
  return elements_per_sample * bytes_per_element


def _drop_static_vars(
    dataset: xarray.Dataset, sample_dim: str
) -> xarray.Dataset:
  """Drop fields that are static and do not vary across the sample dimension."""
  vars_to_drop = [k for k, v in dataset.items() if sample_dim not in v.dims]
  return dataset.drop_vars(vars_to_drop)


PyTree = Any


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


@dataclasses.dataclass(repr=False, eq=False)
class _XarraySliceSource(grain.RandomAccessDataSource[T]):
  """Grain data source for reading slices from an xarray.Dataset."""

  source: xarray.Dataset
  slices: list[slice]
  sample_dim: str

  @functools.cached_property
  def loader(self) -> Callable[[xarray.Dataset], xarray.Dataset]:
    # In principle, it could make sense to support passing alternative
    # loaders, such as xarray_tensorstore.read() or a dask loader that calls
    # .compute(). We don't yet have any use cases where this seems to make a
    # difference, though. (The thread pool loader works as well as
    # xarray_tensorstore.read.)
    return _thread_pool_loader()

  def __getitem__(self, index):
    selection = self.source.isel({self.sample_dim: self.slices[index]})
    return self.loader(selection)

  def __len__(self):
    return len(self.slices)

  def __getstate__(self):
    # Cannot pickle the loader because it includes a thread pool.
    return {
        'source': self.source,
        'slices': self.slices,
        'sample_dim': self.sample_dim,
    }


@dataclasses.dataclass(eq=False)
class _BlockResult:
  block: PyTree
  sub_slices: list[slice]


@dataclasses.dataclass(repr=False, eq=False)
class _XarrayBlockSource(grain.RandomAccessDataSource):
  """Grain data source for reading blocks from an xarray.Dataset."""

  source: xarray.Dataset
  groups: list[tuple[slice, list[slice]]]
  stencil: stencils.Stencil
  sample_dim: str

  @functools.cached_property
  def loader(self) -> Callable[[xarray.Dataset], xarray.Dataset]:
    # In principle, it could make sense to support passing alternative
    # loaders, such as xarray_tensorstore.read() or a dask loader that calls
    # .compute(). We don't yet have any use cases where this seems to make a
    # difference, though. (The thread pool loader works as well as
    # xarray_tensorstore.read.)
    return _thread_pool_loader()

  def __getitem__(self, index: int) -> _BlockResult:
    block_slice, sub_slices = self.groups[index]
    selection = self.source.isel({self.sample_dim: block_slice})
    loaded = self.loader(selection)
    return _BlockResult(loaded, sub_slices)

  def __len__(self) -> int:
    return len(self.groups)

  def __getstate__(self):
    # Cannot pickle the loader because it includes a thread pool.
    return {
        'source': self.source,
        'groups': self.groups,
        'sample_dim': self.sample_dim,
    }


@dataclasses.dataclass(eq=False)
class _PytreeSource(grain.RandomAccessDataSource):
  """Grain data source that reads from a dict of sources."""

  sources: PyTree

  def __len__(self) -> int:
    lengths = {len(source) for source in jax.tree.leaves(self.sources)}
    if len(lengths) != 1:
      raise ValueError(
          f'all sources must have the same length, but got lengths: {lengths}'
      )
    [length] = lengths
    return length

  def __getitem__(self, index: int) -> dict[str, Any]:
    return jax.tree.map(lambda x: x[index], self.sources)


def _prepare_xarray_source(
    source: xarray.Dataset | PyTree, sample_dim: str
) -> xarray.Dataset | PyTree:
  """Prepare a pytree of xarray.Datasets for reading."""

  def _prepare_dataset(source: xarray.Dataset) -> xarray.Dataset:
    # TODO(shoyer): support multiple sample dimensions
    if sample_dim not in source.dims:
      raise ValueError(
          'source does not include variables with a'
          f' {sample_dim!r} dimension:\n{source}'
      )
    source = _drop_static_vars(source, sample_dim)
    source = source.transpose(sample_dim, ...)
    return source

  return jax.tree.map(_prepare_dataset, source, is_leaf=_is_xarray_dataset)


@dataclasses.dataclass
class _UnbatchTransform(grain.experimental.FlatMapTransform):
  max_fan_out: int

  def flat_map(self, element):
    return element


def evaluation_iterator(
    source: xarray.Dataset | PyTree,
    stencil: stencils.Stencil | PyTree,
    sample_origins: np.ndarray,
    *,
    sample_dim: str = 'time',
    read_options: grain.ReadOptions | None = None,
) -> grain.IterDataset:
  """Read a time-series xarray.Dataset into an iterator of samples.

  Args:
    source: lazy xarray.Dataset or JAX pytree of lazy datasets, e.g., opened
      from a Zarr file with `xarray.open_zarr(..., chunks=None)`. All data
      variables along `sample_dim` will be sampled. Note: setting `chunks=None`
      to avoid using Dask is preferred for optimal performance.
    stencil: specification of what time-series samples of this dataset should
      look like. Should either be a single Stencil or a pytree of Stencils
      matching the structure of `source`.
    sample_origins: points at which to sample the time-series.
    sample_dim: name of the dimension to sample along.
    read_options: options to use for reading the Grain dataset. Buffer size here
      indicates the number of blocks to prefetch.

  Returns:
    grain.IterDataset where each element is a sampled xarray.Dataset.
  """
  if read_options is None:
    read_options = grain.ReadOptions(num_threads=16, prefetch_buffer_size=32)

  if isinstance(stencil, stencils.Stencil):
    stencil = jax.tree.map(
        lambda _: stencil, source, is_leaf=_is_xarray_dataset
    )

  def _build_grain_source(
      source: xarray.Dataset, stencil: stencils.Stencil
  ) -> _XarraySliceSource:

    source = _prepare_xarray_source(source, sample_dim)
    source_points = source[sample_dim].values
    slices = stencils.build_sampling_slices(
        source_points=source_points,
        sample_origins=sample_origins,
        stencil=stencil,
    )
    return _XarraySliceSource(source, slices, sample_dim)

  grain_source = _PytreeSource(
      jax.tree.map(
          _build_grain_source, source, stencil, is_leaf=_is_xarray_dataset
      )
  )
  dataset = grain.MapDataset.source(grain_source)
  dataset = dataset.to_iter_dataset(read_options=read_options)
  return dataset


def group_slices(
    slices: list[slice], group_size: int
) -> list[tuple[slice, list[slice]]]:
  """Partition a list of slices into groups for reading in parallel."""
  groups = []
  for i in range(0, len(slices), group_size):
    slices_to_consolidate = slices[i : i + group_size]

    block_start = slices_to_consolidate[0].start
    block_stop = slices_to_consolidate[-1].stop
    block = slice(block_start, block_stop)

    sub_slices = [
        slice(s.start - block_start, s.stop - block_start)
        for s in slices_to_consolidate
    ]
    groups.append((block, sub_slices))

  return groups


def _split_block(sample_dim: str, block_tree: PyTree) -> list[PyTree]:
  """Apply sub_slices to each block in a pytree of _BlockResult."""
  block_leaves, outer_treedef = jax.tree.flatten(
      block_tree, is_leaf=lambda x: isinstance(x, _BlockResult)
  )
  unflatten = functools.partial(jax.tree.unflatten, outer_treedef)
  flat_outputs = []
  for block in block_leaves:
    flat_outputs.append([
        jax.tree.map(lambda x: x.isel({sample_dim: s}), block.block)  # pylint: disable=cell-var-from-loop
        for s in block.sub_slices
    ])
  out = [unflatten(zipped) for zipped in zip(*flat_outputs)]
  return out


def training_iterator(
    source: xarray.Dataset | PyTree,
    stencil: stencils.Stencil | PyTree,
    sample_origins: np.ndarray,
    *,
    sample_dim: str = 'time',
    num_epochs: int | None = 1,
    buffer_size_in_bytes: float = 1e10,
    buffer_diversity: int = 10,
    read_options: grain.ReadOptions | None = None,
    shard_index: int | None = None,
    shard_count: int | None = None,
    seed: int = 0,
) -> grain.IterDataset:
  """Read a time-series into an iterator of samples in randomly shuffled order.

  Args:
    source: lazy xarray.Dataset or JAX pytree of lazy datasets, e.g., opened
      from a Zarr file with `xarray.open_zarr(..., chunks=None)`. All data
      variables along `sample_dim` will be sampled. Note: setting `chunks=None`
      to avoid using Dask is preferred for optimal performance.
    stencil: specification of what time-series samples of this dataset should
      look like. Should either be a single Stencil or a pytree of Stencils
      matching the structure of `source`.
    sample_origins: points at which to sample the time-series.
    sample_dim: name of the dimension to sample along.
    num_epochs: number of epoch for which to read the dataset, or `None` to read
      indefinitely.
    buffer_size_in_bytes: number of bytes to store in the shuffle buffer.
    buffer_diversity: number of blocks that should be represented in the shuffle
      buffer, if more than one sample is taken from each block. Typically this
      should be at least as large as the batch size.
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

  if isinstance(stencil, stencils.Stencil):
    stencil = jax.tree.map(
        lambda _: stencil, source, is_leaf=_is_xarray_dataset
    )

  source = _prepare_xarray_source(source, sample_dim)
  bytes_per_example = _example_nbytes(source, stencil, sample_dim)
  shuffle_window = round(buffer_size_in_bytes / bytes_per_example)
  group_size = round(
      (buffer_size_in_bytes / buffer_diversity) / bytes_per_example
  )
  group_size = max(group_size, 1)

  def _build_grain_source(
      source: xarray.Dataset, stencil: stencils.Stencil
  ) -> _XarrayBlockSource:
    source_points = source[sample_dim].values
    slices = stencils.build_sampling_slices(
        source_points=source_points,
        sample_origins=sample_origins,
        stencil=stencil,
    )
    grouped_slices = group_slices(slices, group_size)
    return _XarrayBlockSource(
        source, grouped_slices, stencil, sample_dim
    )

  grain_source = _PytreeSource(
      jax.tree.map(
          _build_grain_source, source, stencil, is_leaf=_is_xarray_dataset
      )
  )

  if read_options is None:
    # We already have multiple threads inside the source, so we don't need to
    # use Grain's internal concurrency.
    read_options = grain.ReadOptions(num_threads=1, prefetch_buffer_size=2)

  rng = np.random.default_rng(seed)
  global_seed = rng.integers(2**32)
  window_seed = rng.integers(2**32)

  # Our strategy of emitting multiple examples per block means that we need to
  # drop into Grain's lower-level Dataset API.
  # We use IterDataset instead of MapDataset because the flat map transform on
  # MapDataset will re-read the entire block for each example, rather than
  # fetching it all at once.
  dataset = grain.MapDataset.source(grain_source)
  dataset = dataset[shard_index::shard_count]
  dataset = dataset.shuffle(global_seed)
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.to_iter_dataset(read_options=read_options)
  dataset = dataset.map(functools.partial(_split_block, sample_dim))
  dataset = grain.experimental.FlatMapIterDataset(
      dataset, _UnbatchTransform(group_size)
  )
  if shuffle_window:
    dataset = grain.experimental.WindowShuffleIterDataset(
        dataset,
        window_size=shuffle_window,
        seed=window_seed,
    )

  return dataset