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

import pickle

from absl.testing import absltest
from absl.testing import parameterized
import chex
from neuralgcm.experimental import coordax
import neuralgcm.experimental.jax_datetime as jdt
from neuralgcm.experimental.xreader import reader
import numpy as np
import xarray


class ReaderTest(parameterized.TestCase):

  def test_calculate_block_size(self):
    source = xarray.Dataset({'a': ('x', np.arange(1000))})
    block_size = reader._calculate_block_size(
        source, sample_dims=['x'], bytes_per_request=8 * 100
    )
    self.assertEqual(block_size, 100)

    block_size = reader._calculate_block_size(
        source, sample_dims=['x'], bytes_per_request=1
    )
    self.assertEqual(block_size, 1)

    source = xarray.Dataset(
        {'a': ('x', np.arange(1000)), 'b': (('x', 'y'), np.zeros((1000, 9)))}
    )
    block_size = reader._calculate_block_size(
        source, sample_dims=['x'], bytes_per_request=8 * 100
    )
    self.assertEqual(block_size, 10)

  @parameterized.parameters(
      {
          'example_size': 5,
          'total_size': 5,
          'block_size': 5,
          'expected': [slice(0, 5)],
      },
      {
          'example_size': 5,
          'total_size': 10,
          'block_size': 10,
          'expected': [slice(0, 10)],
      },
      {
          'example_size': 5,
          'total_size': 8,
          'block_size': 5,
          'expected': [slice(0, 5), slice(1, 6), slice(2, 7), slice(3, 8)],
      },
      {
          'example_size': 5,
          'total_size': 10,
          'block_size': 5,
          'stride_between_windows': 2,
          'expected': [slice(0, 5), slice(2, 7), slice(4, 9)],
      },
      {
          'example_size': 5,
          'total_size': 8,
          'block_size': 5,
          'first_window_offset': 1,
          'expected': [slice(1, 6), slice(2, 7), slice(3, 8)],
      },
      {
          'example_size': 5,
          'total_size': 8,
          'block_size': 5,
          'stride_between_windows': 2,
          'first_window_offset': 1,
          'expected': [slice(1, 6), slice(3, 8)],
      },
      {
          'example_size': 3,
          'total_size': 10,
          'block_size': 6,
          'expected': [slice(0, 6), slice(4, 10)],
      },
      {
          'example_size': 3,
          'total_size': 10,
          'block_size': 6,
          'stride_between_windows': 2,
          # [0 1 2] [2 3 4] | [4 5 6] [6 7 8]
          'expected': [slice(0, 5), slice(4, 9)],
      },
      {
          'example_size': 3,
          'total_size': 10,
          'block_size': 6,
          'output_window_stride': 2,
          'stride_between_windows': 1,
          # [0 2 4] [1 3 5] | [2 4 6] [3 5 7] | [4 6 8] [5 7 9]
          'expected': [slice(0, 6), slice(2, 8), slice(4, 10)],
      },
      {
          'example_size': 2,
          'total_size': 10,
          'block_size': 6,
          'output_window_stride': 2,
          'stride_between_windows': 2,
          # [0 2] [2 4] | [4 6] [6 8]
          'expected': [slice(0, 5), slice(4, 9)],
      },
  )
  def test_windower_get_block_slices(
      self, block_size, total_size, expected, **kwargs
  ):
    actual = reader.Windower(**kwargs).get_block_slices(block_size, total_size)
    self.assertEqual(actual, expected)

  @parameterized.parameters(
      {
          'inputs': np.arange(5),
          'example_size': 5,
          'stride_between_windows': 1,
          'expected': [np.arange(5)],
      },
      {
          'inputs': np.arange(5),
          'example_size': 3,
          'stride_between_windows': 1,
          'output_window_stride': 2,
          'expected': [np.array([0, 2, 4])],
      },
      {
          'inputs': np.arange(10),
          'example_size': 6,
          'stride_between_windows': 2,
          'expected': [
              np.array([0, 1, 2, 3, 4, 5]),
              np.array([2, 3, 4, 5, 6, 7]),
              np.array([4, 5, 6, 7, 8, 9]),
          ],
      },
      {
          'inputs': np.arange(10),
          'example_size': 4,
          'stride_between_windows': 1,
          'output_window_stride': 2,
          'expected': [
              np.array([0, 2, 4, 6]),
              np.array([1, 3, 5, 7]),
              np.array([2, 4, 6, 8]),
              np.array([3, 5, 7, 9]),
          ],
      },
      {
          'inputs': np.arange(9),
          'example_size': 3,
          'stride_between_windows': 2,
          'output_window_stride': 2,
          'expected': [
              np.array([0, 2, 4]),
              np.array([2, 4, 6]),
              np.array([4, 6, 8]),
          ],
      },
      {
          'inputs': np.arange(10),
          'example_size': 3,
          'stride_between_windows': 1,
          'output_window_stride': 3,
          'expected': [
              np.array([0, 3, 6]),
              np.array([1, 4, 7]),
              np.array([2, 5, 8]),
              np.array([3, 6, 9]),
          ],
      },
      {
          'inputs': {
              'x': np.array([1, 2, 3]),
              'y': np.array([[1, 2], [3, 4], [5, 6]]),
          },
          'example_size': 2,
          'stride_between_windows': 1,
          'expected': [
              {'x': np.array([1, 2]), 'y': np.array([[1, 2], [3, 4]])},
              {'x': np.array([2, 3]), 'y': np.array([[3, 4], [5, 6]])},
          ],
      },
  )
  def test_windower_sample_block(self, inputs, expected, **kwargs):
    actual = reader.Windower(**kwargs).sample_block(inputs)
    chex.assert_trees_all_equal(actual, expected)

  def test_read_timeseries_windower_at_offsets(self):
    source = xarray.Dataset({'x': ('time', np.arange(10))})
    sampler = reader.WindowerAtOffsets(
        example_size=3, window_offsets=[0, 1, 5], output_window_stride=2
    )
    data = reader.read_timeseries(source, sampler)
    expected = [
        {'x': np.array([0, 2, 4])},
        {'x': np.array([1, 3, 5])},
        {'x': np.array([5, 7, 9])},
    ]
    actual = [item for item in data]
    chex.assert_trees_all_equal(actual, expected)

    with self.assertRaisesRegex(
        ValueError,
        'offset at 1 needs data through stop=6, which is beyond total_size=5',
    ):
      reader.read_timeseries(source.head(5), sampler)

  def test_read_timeseries_basic_windower(self):
    source = xarray.Dataset({'x': ('time', np.arange(5, dtype=np.int64))})
    sampler = reader.Windower(example_size=3, stride_between_windows=1)
    data = reader.read_timeseries(source, sampler)
    actual = [item for item in data]
    expected = [
        {'x': np.array([0, 1, 2])},
        {'x': np.array([1, 2, 3])},
        {'x': np.array([2, 3, 4])},
    ]
    chex.assert_trees_all_equal(actual, expected)

  def test_read_timeseries_stride_between_windows(self):
    source = xarray.Dataset({'x': ('time', np.arange(50, dtype=np.int64))})
    sampler = reader.Windower(
        example_size=15,
        stride_between_windows=10,
        first_window_offset=5,
    )
    data = reader.read_timeseries(source, sampler)
    actual = [item for item in data]
    expected = [
        {'x': np.arange(5, 20)},
        {'x': np.arange(15, 30)},
        {'x': np.arange(25, 40)},
        {'x': np.arange(35, 50)},
    ]
    chex.assert_trees_all_equal(actual, expected)

  def test_read_timeseries_output_window_stride(self):
    source = xarray.Dataset({'x': ('time', np.arange(70, dtype=np.int64))})
    sampler = reader.Windower(
        example_size=5,
        stride_between_windows=20,
        output_window_stride=10,
    )
    data = reader.read_timeseries(source, sampler)

    actual = [item for item in data]
    expected = [
        {'x': np.array([0, 10, 20, 30, 40])},
        {'x': np.array([20, 30, 40, 50, 60])},
    ]
    chex.assert_trees_all_equal(actual, expected)

  @parameterized.parameters(
      # Each sample has 10 * 10 * 8 bytes.
      {'block_size_in_bytes': 10 * 10 * 8},
      {'block_size_in_bytes': 10 * 10 * 8 * 7},
      {'block_size_in_bytes': 10 * 10 * 8 * 10},
      {'block_size_in_bytes': 1e9},
  )
  def test_read_timeseries_complete(self, block_size_in_bytes):
    rs = np.random.RandomState(0)
    source = xarray.Dataset({
        'foo': ('time', rs.randn(500).astype(np.float32)),
        'bar': (('time', 'x', 'y'), rs.randn(500, 3, 3)),
    })
    sampler = reader.Windower(example_size=10, stride_between_windows=10)
    data = reader.read_timeseries(
        source, sampler, block_size_in_bytes=block_size_in_bytes
    )
    actual = [item for item in data]
    actual_foo = np.concatenate([x['foo'] for x in actual])
    actual_bar = np.concatenate([x['bar'] for x in actual])
    np.testing.assert_equal(actual_foo, source['foo'].values)
    np.testing.assert_equal(actual_bar, source['bar'].values)

  def test_read_shuffled_shard_basic(self):
    source = xarray.Dataset({'x': ('time', np.arange(100))})
    sampler = reader.Windower(example_size=5, stride_between_windows=5)
    data = reader.read_shuffled_shard(
        source,
        sampler,
        block_size_in_bytes=10 * 8,
        buffer_size_in_bytes=100 * 8,
        seed=0,
    )
    expected = np.arange(100)
    actual = np.sort(np.concatenate([x['x'] for x in data]))
    np.testing.assert_equal(actual, expected)

  def test_read_shuffled_shard_warns(self):
    source = xarray.Dataset({'x': ('time', np.arange(100))})
    sampler = reader.Windower(example_size=5, stride_between_windows=5)
    with self.assertLogs(level='WARNING') as logs_context:
      data = reader.read_shuffled_shard(
          source,
          sampler,
          block_size_in_bytes=20 * 8,
          buffer_size_in_bytes=25 * 8,
          min_buffer_blocks=2,
      )
    self.assertIn('insufficient diversity', logs_context.output[0])

    expected = np.arange(100)
    actual = np.sort(np.concatenate([x['x'] for x in data]))
    np.testing.assert_equal(actual, expected)

  @parameterized.named_parameters(
      {
          'testcase_name': 'only_window_shuffle',
          'block_size_in_bytes': 100 * 8,
          'buffer_size_in_bytes': 100 * 8,
      },
      {
          'testcase_name': 'only_global_shuffle',
          'block_size_in_bytes': 10 * 8,
          'buffer_size_in_bytes': 0,
      },
  )
  def test_read_shuffled_shard_shuffling(self, **kwargs):
    source = xarray.Dataset({'x': ('time', np.arange(100, dtype=np.int64))})
    sampler = reader.Windower(example_size=10, stride_between_windows=10)
    expected = np.arange(100)

    # same values repeated twice, but in a different order
    data = reader.read_shuffled_shard(source, sampler, num_epochs=2, **kwargs)
    actual_2x = np.concatenate([x['x'] for x in data])
    np.testing.assert_equal(np.sort(actual_2x[:100]), expected)
    np.testing.assert_equal(np.sort(actual_2x[100:]), expected)
    self.assertFalse((actual_2x[:100] == actual_2x[100:]).all())

  def test_read_shuffled_shard_pickle(self):
    source = xarray.Dataset({'x': ('time', np.arange(100))})
    sampler = reader.Windower(example_size=5, stride_between_windows=5)
    data = reader.read_shuffled_shard(source, sampler)
    restored = pickle.loads(pickle.dumps(data))
    expected = np.arange(100)
    actual = np.sort(np.concatenate([x['x'] for x in restored]))
    np.testing.assert_equal(actual, expected)

  def test_read_timeseries_with_datetime(self):
    times = np.arange(
        np.datetime64('2000-01-01'),
        np.datetime64('2000-01-04'),
        np.timedelta64(1, 'D'),
    )
    source = xarray.Dataset(coords={'time': ('time', times)})
    sampler = reader.Windower(example_size=1)
    data = reader.read_timeseries(source, sampler)
    actual = [item for item in data]
    expected = [
        {'time': jdt.to_datetime(times[0])},
        {'time': jdt.to_datetime(times[1])},
        {'time': jdt.to_datetime(times[2])},
    ]
    chex.assert_trees_all_equal(actual, expected)

  def test_read_timeseries_with_timedelta(self):
    times = np.arange(
        np.timedelta64(1, 'D'),
        np.timedelta64(4, 'D'),
        np.timedelta64(1, 'D'),
    )
    source = xarray.Dataset(coords={'time': ('time', times)})
    sampler = reader.Windower(example_size=1)
    data = reader.read_timeseries(source, sampler)
    actual = [item for item in data]
    expected = [
        {'time': jdt.to_timedelta(times[0])},
        {'time': jdt.to_timedelta(times[1])},
        {'time': jdt.to_timedelta(times[2])},
    ]
    chex.assert_trees_all_equal(actual, expected)

  def test_coordax_unflattener(self):
    source = xarray.Dataset({
        'foo': (('time', 'x'), np.array([[1, 2], [3, 4], [5, 6]])),
    })
    sampler = reader.Windower(example_size=2)
    unflattener = reader.CoordaxUnflattener()
    data = reader.read_timeseries(source, sampler, unflattener=unflattener)
    actual = [item for item in data]
    expected = [
        {'foo': coordax.Field(np.array([[1, 2], [3, 4]]), dims=(None, 'x'))},
        {'foo': coordax.Field(np.array([[3, 4], [5, 6]]), dims=(None, 'x'))},
    ]
    chex.assert_trees_all_equal(actual, expected)


if __name__ == '__main__':
  absltest.main()
