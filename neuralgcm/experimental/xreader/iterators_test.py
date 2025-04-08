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
import jax
from neuralgcm.experimental import xreader
import numpy as np
import xarray


def assert_xarray_trees_identical(actual, expected):
  """Asserts that two pytrees of xarray.Dataset are identical."""
  actual_leaves, actual_treedef = jax.tree.flatten(
      actual, is_leaf=lambda x: isinstance(x, xarray.Dataset)
  )
  expected_leaves, expected_treedef = jax.tree.flatten(
      expected, is_leaf=lambda x: isinstance(x, xarray.Dataset)
  )
  if actual_treedef != expected_treedef:
    raise ValueError(
        f'Trees have different structures: {actual_treedef} vs'
        f' {expected_treedef}'
    )
  for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves):
    xarray.testing.assert_identical(actual_leaf, expected_leaf)


class IteratorsTest(parameterized.TestCase):

  def test_evaluation_iterator(self):
    source = xarray.Dataset(
        {'x': ('time', np.arange(10))}, coords={'time': np.arange(10)}
    )
    stencil = xreader.Stencil(start=-2, stop=4, step=2)
    sample_origins = np.array([2, 3, 7])
    data = xreader.evaluation_iterator(source, stencil, sample_origins)
    expected = [
        xarray.Dataset(
            {'x': ('time', np.array([0, 2, 4]))},
            coords={'time': np.array([0, 2, 4])},
        ),
        xarray.Dataset(
            {'x': ('time', np.array([1, 3, 5]))},
            coords={'time': np.array([1, 3, 5])},
        ),
        xarray.Dataset(
            {'x': ('time', np.array([5, 7, 9]))},
            coords={'time': np.array([5, 7, 9])},
        ),
    ]
    actual = [item for item in data]
    assert_xarray_trees_identical(actual, expected)

  def test_evaluation_iterator_pytree(self):
    source = {
        'x': xarray.Dataset(coords={'time': np.arange(-1, 10)}),
        'y': xarray.Dataset(coords={'time': np.arange(0, 10, 2)}),
    }
    stencil = {
        'x': xreader.Stencil(start=-1, stop=1, step=1, closed='both'),
        'y': xreader.Stencil(start=0, stop=2, step=2, closed='both'),
    }
    sample_origins = np.array([0, 2, 4])
    data = xreader.evaluation_iterator(source, stencil, sample_origins)
    expected = [
        {
            'x': xarray.Dataset(coords={'time': np.array([-1, 0, 1])}),
            'y': xarray.Dataset(coords={'time': np.array([0, 2])}),
        },
        {
            'x': xarray.Dataset(coords={'time': np.array([1, 2, 3])}),
            'y': xarray.Dataset(coords={'time': np.array([2, 4])}),
        },
        {
            'x': xarray.Dataset(coords={'time': np.array([3, 4, 5])}),
            'y': xarray.Dataset(coords={'time': np.array([4, 6])}),
        },
    ]
    actual = [item for item in data]
    assert_xarray_trees_identical(actual, expected)

  def test_evaluation_iterator_non_pytree_stencil(self):
    source = {
        'foo': xarray.Dataset(coords={'time': np.arange(100)}),
        'bar': xarray.Dataset(coords={'time': np.arange(100)}),
    }
    stencil = xreader.Stencil(start=0, stop=1, step=1)
    sample_origins = np.array([0, 10])
    data = xreader.evaluation_iterator(source, stencil, sample_origins)
    expected = [
        {
            'foo': xarray.Dataset(coords={'time': np.array([0])}),
            'bar': xarray.Dataset(coords={'time': np.array([0])}),
        },
        {
            'foo': xarray.Dataset(coords={'time': np.array([10])}),
            'bar': xarray.Dataset(coords={'time': np.array([10])}),
        },
    ]
    actual = [item for item in data]
    assert_xarray_trees_identical(actual, expected)

  def test_training_iterator_basic(self):
    source = xarray.Dataset(
        {'x': ('time', -np.arange(100))}, coords={'time': np.arange(100)}
    )
    stencil = xreader.Stencil(start=-2, stop=3, step=1)
    sample_origins = np.array([10, 20, 30])
    data = xreader.training_iterator(
        source,
        stencil,
        sample_origins,
        buffer_size_in_bytes=100 * 8,
        seed=0,
    )
    expected = [
        xarray.Dataset(
            {'x': ('time', np.array([-8, -9, -10, -11, -12]))},
            coords={'time': np.array([8, 9, 10, 11, 12])},
        ),
        xarray.Dataset(
            {'x': ('time', np.array([-18, -19, -20, -21, -22]))},
            coords={'time': np.array([18, 19, 20, 21, 22])},
        ),
        xarray.Dataset(
            {'x': ('time', np.array([-28, -29, -30, -31, -32]))},
            coords={'time': np.array([28, 29, 30, 31, 32])},
        ),
    ]
    actual = sorted([item for item in data], key=lambda x: x['time'][0].item())
    assert_xarray_trees_identical(actual, expected)

  @parameterized.named_parameters(
      {'testcase_name': 'only_window_shuffle', 'buffer_size_in_bytes': 100 * 8},
      {'testcase_name': 'only_global_shuffle', 'buffer_size_in_bytes': 0},
  )
  def test_training_iterator_shuffling(self, **kwargs):
    source = xarray.Dataset(coords={'time': np.arange(100)})
    stencil = xreader.Stencil(start=-2, stop=3, step=1)
    sample_origins = np.arange(2, 100, 5)
    # same values repeated twice, but in a different order
    data = xreader.training_iterator(
        source, stencil, sample_origins, num_epochs=2, **kwargs
    )
    actual_2x = np.concatenate([x['time'].data for x in data])
    expected = np.arange(100)
    np.testing.assert_array_equal(np.sort(actual_2x[:100]), expected)
    np.testing.assert_array_equal(np.sort(actual_2x[100:]), expected)
    self.assertFalse((actual_2x[:100] == actual_2x[100:]).all())

  def test_training_iterator_pickle(self):
    source = xarray.Dataset(coords={'time': np.arange(100)})
    stencil = xreader.Stencil(start=-2, stop=3, step=1)
    sample_origins = np.arange(2, 100, 5)
    data = xreader.training_iterator(source, stencil, sample_origins)
    restored = pickle.loads(pickle.dumps(data))
    expected = np.arange(100)
    actual = np.sort(np.concatenate([x['time'].data for x in restored]))
    np.testing.assert_array_equal(actual, expected)

  def test_training_iterator_nested(self):
    source = {
        'foo': xarray.Dataset(coords={'time': np.arange(100)}),
        'bar': xarray.Dataset(
            {'baz': ('time', np.arange(0, 1000, 10))},
            coords={'time': np.arange(100)},
        ),
    }
    stencil = {
        'foo': xreader.Stencil(start=-2, stop=3, step=1),
        'bar': xreader.Stencil(start=0, stop=1, step=1),
    }
    sample_origins = np.array([10, 20, 30])
    data = xreader.training_iterator(source, stencil, sample_origins)
    expected = [
        {
            'foo': xarray.Dataset(
                coords={'time': np.array([8, 9, 10, 11, 12])}
            ),
            'bar': xarray.Dataset(
                {'baz': ('time', np.array([100]))},
                coords={'time': np.array([10])},
            ),
        },
        {
            'foo': xarray.Dataset(
                coords={'time': np.array([18, 19, 20, 21, 22])}
            ),
            'bar': xarray.Dataset(
                {'baz': ('time', np.array([200]))},
                coords={'time': np.array([20])},
            ),
        },
        {
            'foo': xarray.Dataset(
                coords={'time': np.array([28, 29, 30, 31, 32])}
            ),
            'bar': xarray.Dataset(
                {'baz': ('time', np.array([300]))},
                coords={'time': np.array([30])},
            ),
        },
    ]
    actual = sorted(
        [item for item in data],
        key=lambda x: x['bar']['time'].item(),
    )
    assert_xarray_trees_identical(actual, expected)

  def test_training_iterator_non_pytree_stencil(self):
    source = {
        'foo': xarray.Dataset(coords={'time': np.arange(100)}),
        'bar': xarray.Dataset(coords={'time': np.arange(100)}),
    }
    stencil = xreader.Stencil(start=0, stop=1, step=1)
    sample_origins = np.array([0, 10])
    data = xreader.training_iterator(source, stencil, sample_origins)
    expected = [
        {
            'foo': xarray.Dataset(coords={'time': np.array([0])}),
            'bar': xarray.Dataset(coords={'time': np.array([0])}),
        },
        {
            'foo': xarray.Dataset(coords={'time': np.array([10])}),
            'bar': xarray.Dataset(coords={'time': np.array([10])}),
        },
    ]
    actual = sorted(
        [item for item in data],
        key=lambda x: x['bar']['time'].item(),
    )
    assert_xarray_trees_identical(actual, expected)

  def test_iterator_with_datetime(self):
    times = np.arange(
        np.datetime64('2000-01-01'),
        np.datetime64('2000-01-04'),
        np.timedelta64(1, 'D'),
    )
    source = xarray.Dataset(coords={'time': ('time', times)})
    stencil = xreader.TimeStencil(start='0h', stop='24h', step='24h')
    sample_origins = times
    data = xreader.evaluation_iterator(source, stencil, sample_origins)
    actual = [item for item in data]
    expected = [
        xarray.Dataset(coords={'time': times[0:1]}),
        xarray.Dataset(coords={'time': times[1:2]}),
        xarray.Dataset(coords={'time': times[2:3]}),
    ]
    assert_xarray_trees_identical(actual, expected)


if __name__ == '__main__':
  absltest.main()
