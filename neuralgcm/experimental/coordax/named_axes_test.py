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
import dataclasses
import math
import re
import textwrap

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from neuralgcm.experimental import jax_datetime as jdt
from neuralgcm.experimental.coordax import named_axes
from neuralgcm.experimental.coordax import ndarrays
import numpy as np
import treescope


def assert_named_array_equal(
    actual: named_axes.NamedArray,
    expected: named_axes.NamedArray,
) -> None:
  """Asserts that a NamedArray has the expected data and dims."""
  np.testing.assert_array_equal(actual.data, expected.data)
  assert actual.dims == expected.dims, (expected.dims, actual.dims)


def assert_pytree_equal(actual, expected) -> None:
  """Asserts that two pytrees are equal."""
  actual_path_vals, actual_treedef = jax.tree.flatten_with_path(actual)
  expected_path_vals, expected_treedef = jax.tree.flatten_with_path(expected)
  if actual_treedef != expected_treedef:
    raise AssertionError(
        f'actual treedef: {actual_treedef}\n'
        f'expected treedef: {expected_treedef}'
    )
  for (k, actual_value), (_, expected_value) in zip(
      actual_path_vals, expected_path_vals
  ):
    np.testing.assert_array_equal(actual_value, expected_value, err_msg=k)


class NamedAxesTest(parameterized.TestCase):

  def test_named_array(self):
    data = np.arange(10).reshape((2, 5))
    array = named_axes.NamedArray(data, ('x', None))
    self.assertEqual(array.dims, ('x', None))
    np.testing.assert_array_equal(array.data, data)
    self.assertEqual(array.ndim, 2)
    self.assertEqual(array.shape, (2, 5))
    self.assertEqual(array.positional_shape, (5,))
    self.assertEqual(array.named_shape, {'x': 2})
    self.assertEqual(
        repr(array),
        textwrap.dedent("""\
            NamedArray(
                data=array([[0, 1, 2, 3, 4],
                            [5, 6, 7, 8, 9]]),
                dims=('x', None),
            )"""),
    )

  def test_constructor_array_types(self):
    with self.subTest('py-scalar'):
      array = named_axes.NamedArray(1.0, ())
      self.assertIsInstance(array.data, np.ndarray)

    with self.subTest('np-scalar'):
      array = named_axes.NamedArray(np.float32(1.0), ())
      self.assertIsInstance(array.data, np.ndarray)

    with self.subTest('numpy'):
      data = np.arange(10)
      array = named_axes.NamedArray(data, ('x',))
      self.assertIsInstance(array.data, np.ndarray)

    with self.subTest('jax'):
      data = jnp.arange(10)
      array = named_axes.NamedArray(data, ('x',))
      self.assertIsInstance(array.data, jnp.ndarray)

  def test_constructor_datetime(self):
    dt = jdt.to_timedelta(1, 'day')

    array = named_axes.NamedArray(dt, ())
    self.assertIsInstance(array.data, jdt.Timedelta)

    array = named_axes.NamedArray(dt.to_timedelta64(), ())
    self.assertIsInstance(array.data, jdt.Timedelta)

    array = named_axes.NamedArray(np.timedelta64(dt.to_timedelta64()), ())
    self.assertIsInstance(array.data, jdt.Timedelta)

  def test_constructor_error(self):
    with self.assertRaisesWithLiteralMatch(
        TypeError,
        'data must be a np.ndarray, jax.Array or a duck-typed array registered'
        ' with coordax.register_ndarray(), got dict: {}',
    ):
      named_axes.NamedArray({})
    with self.assertRaisesRegex(
        ValueError, re.escape(r'data.ndim=2 != len(dims)=1')
    ):
      named_axes.NamedArray(np.zeros((2, 5)), ('x',))
    with self.assertRaisesRegex(
        ValueError, re.escape(r'dimension names may not be repeated')
    ):
      named_axes.NamedArray(np.zeros((2, 5)), ('x', 'x'))

  def test_constructor_no_dims(self):
    data = np.arange(10).reshape((2, 5))
    expected = named_axes.NamedArray(data, (None, None))
    actual = named_axes.NamedArray(data)
    assert_named_array_equal(actual, expected)

  def test_tree_map_same_dims(self):
    data = np.arange(10).reshape((2, 5))
    array = named_axes.NamedArray(data, ('x', 'y'))
    actual = jax.tree.map(lambda x: x, array)
    assert_named_array_equal(actual, array)

  def test_tree_map_cannot_trim(self):
    data = np.arange(10).reshape((2, 5))
    array = named_axes.NamedArray(data, ('x', 'y'))
    with self.assertRaisesRegex(
        ValueError,
        re.escape(
            'cannot trim named dimensions when unflattening to a NamedArray:'
            " ('x',)."
        ),
    ):
      jax.tree.map(lambda x: x[0, :], array)

  def test_tree_map_wrong_dim_size(self):
    data = np.arange(10).reshape((2, 5))
    array = named_axes.NamedArray(data, ('x', 'y'))
    with self.assertRaisesRegex(
        ValueError,
        re.escape(
            'named shape mismatch when unflattening to a NamedArray: '
            "{'x': 2, 'y': 3} != {'x': 2, 'y': 5}."
        ),
    ):
      jax.tree.map(lambda x: x[:, :3], array)

  def test_tree_map_new_dim(self):
    data = np.arange(10).reshape((2, 5))
    array = named_axes.NamedArray(data, ('x', 'y'))
    expected = named_axes.NamedArray(data[np.newaxis, ...], (None, 'x', 'y'))
    actual = jax.tree.map(lambda x: x[np.newaxis, ...], array)
    assert_named_array_equal(actual, expected)

  def test_tree_map_trim_dim(self):
    data = np.arange(10).reshape((2, 5))
    array = named_axes.NamedArray(data, (None, 'y'))
    expected = named_axes.NamedArray(data[0, ...], ('y',))
    actual = jax.tree.map(lambda x: x[0, ...], array)
    assert_named_array_equal(actual, expected)

  def test_tree_map_replace_with_non_array_same_treedef(self):
    data = np.arange(10).reshape((2, 5))
    array = named_axes.NamedArray(data, ('x', 'y'))
    expected_treedef = jax.tree.structure(array)
    expected_leaf = object()
    object_array = jax.tree.map(lambda x: expected_leaf, array)
    [actual_leaf], actual_treedef = jax.tree.flatten(object_array)
    self.assertIs(actual_leaf, expected_leaf)
    self.assertEqual(actual_treedef, expected_treedef)

  def test_jit(self):
    data = np.arange(10).reshape((2, 5))
    array = named_axes.NamedArray(data, ('x', 'y'))
    actual = jax.jit(lambda x: x)(array)
    assert_named_array_equal(actual, array)

    lowered = jax.jit(lambda x: x).lower(array)
    compiled = lowered.compile()
    actual = compiled(array)
    assert_named_array_equal(actual, array)

  def test_jit_constructor(self):
    data = np.arange(10).reshape((2, 5))
    expected = named_axes.NamedArray(data)
    self.assertIsInstance(expected.data, np.ndarray)
    actual = jax.jit(named_axes.NamedArray)(data)
    self.assertIsInstance(actual.data, jnp.ndarray)
    assert_named_array_equal(actual, expected)

  def test_grad(self):
    data = np.arange(6.0).reshape((2, 3))
    array = named_axes.NamedArray(data, ('x', 'y'))
    expected = named_axes.NamedArray(jnp.ones_like(data), ('x', 'y'))
    actual = jax.grad(lambda x: x.data.sum())(array)
    assert_named_array_equal(actual, expected)

  def test_vmap(self):
    data = np.arange(10).reshape((2, 5))
    array = named_axes.NamedArray(data, (None, 'y'))

    def identity_with_checks(x):
      self.assertEqual(x.dims, ('y',))
      return x

    actual = jax.vmap(identity_with_checks)(array)
    assert_named_array_equal(actual, array)

    array = named_axes.NamedArray(data, ('x', 'y'))
    with self.assertRaisesRegex(
        ValueError,
        re.escape(
            'If you are using vmap or scan, the first dimension must be'
            ' unnamed.'
        ),
    ):
      jax.vmap(lambda x: x)(array)

  def test_vmap_constructor(self):
    data = np.arange(3)
    array = jax.vmap(named_axes.NamedArray)(data)
    actual = named_axes.NamedArray(data)
    assert_named_array_equal(actual, array)

  def test_scan(self):
    data = np.arange(10).reshape((2, 5))
    array = named_axes.NamedArray(data, (None, 'y'))
    _, actual = jax.lax.scan(lambda _, x: (None, x), init=None, xs=array)
    assert_named_array_equal(actual, array)

  def test_tag_valid(self):
    data = np.arange(10).reshape((2, 5))

    array = named_axes.NamedArray(data, (None, 'y'))
    expected = array
    actual = array.tag(None)
    assert_named_array_equal(actual, expected)

    array = named_axes.NamedArray(data, (None, 'y'))
    expected = named_axes.NamedArray(data, ('x', 'y'))
    actual = array.tag('x')
    assert_named_array_equal(actual, expected)

    array = named_axes.NamedArray(data, (None, None))
    expected = named_axes.NamedArray(data, ('x', 'y'))
    actual = array.tag('x', 'y')
    assert_named_array_equal(actual, expected)

  def test_tag_ellipsis(self):
    data = np.arange(10).reshape((2, 5))

    array = named_axes.NamedArray(data, (None, 'y'))
    expected = named_axes.NamedArray(data, ('x', 'y'))
    actual = array.tag('x', ...)
    assert_named_array_equal(actual, expected)

    array = named_axes.NamedArray(data, (None, None))
    expected = named_axes.NamedArray(data, ('x', 'y'))
    actual = array.tag('x', ..., 'y')
    assert_named_array_equal(actual, expected)
    actual = array.tag(..., 'x', 'y')
    assert_named_array_equal(actual, expected)
    actual = array.tag('x', 'y', ...)
    assert_named_array_equal(actual, expected)

    array = named_axes.NamedArray(data, ('x', None))
    expected = named_axes.NamedArray(data, ('x', 'y'))
    actual = array.tag(..., 'y')
    assert_named_array_equal(actual, expected)

  def test_tag_errors(self):
    data = np.arange(10).reshape((2, 5))

    array = named_axes.NamedArray(data, (None, 'y'))
    with self.assertRaisesRegex(
        ValueError,
        re.escape(
            'there must be exactly as many dimensions given to `tag` as there'
            ' are positional axes in the array, but got () for '
            '1 positional axis.'
        ),
    ):
      array.tag()

    array = named_axes.NamedArray(data, (None, None))
    with self.assertRaisesRegex(
        ValueError,
        re.escape(
            'there must be exactly as many dimensions given to `tag` as there'
            " are positional axes in the array, but got ('x',) for "
            '2 positional axes.'
        ),
    ):
      array.tag('x')

    with self.assertRaisesRegex(
        TypeError,
        re.escape('dimension names must be strings, ... or None: (1, 2)'),
    ):
      array.tag(1, 2)

    with self.assertRaisesRegex(
        ValueError,
        re.escape(
            'dimension names contain multiple ellipses (...): '
            '(Ellipsis, Ellipsis)'
        ),
    ):
      array.tag(..., ...)

    with self.assertRaisesRegex(
        ValueError,
        re.escape(
            'too many dimensions supplied to `tag` for the 2 positional axes: '
            "('x', 'y', 'z', Ellipsis)"
        ),
    ):
      array.tag('x', 'y', 'z', ...)

  def test_untag_valid(self):
    data = np.arange(10).reshape((2, 5))
    array = named_axes.NamedArray(data, ('x', 'y'))

    expected = named_axes.NamedArray(data, (None, 'y'))
    actual = array.untag('x')
    assert_named_array_equal(actual, expected)

    expected = named_axes.NamedArray(data, ('x', None))
    actual = array.untag('y')
    assert_named_array_equal(actual, expected)

    expected = named_axes.NamedArray(data, (None, None))
    actual = array.untag('x', 'y')
    assert_named_array_equal(actual, expected)

  def test_untag_invalid(self):
    data = np.arange(10).reshape((2, 5))
    partially_named_array = named_axes.NamedArray(data, (None, 'y'))
    fully_named_array = named_axes.NamedArray(data, ('x', 'y'))

    with self.assertRaisesRegex(
        ValueError,
        re.escape(
            '`untag` cannot be used to introduce positional axes for a'
            ' NamedArray that already has positional axes. Please assign names'
            ' to the existing positional axes first using `tag`.'
        ),
    ):
      partially_named_array.untag('y')

    with self.assertRaisesRegex(
        ValueError,
        re.escape(
            "cannot untag ('invalid',) because they are not a subset of the"
            " current named dimensions ('x', 'y')"
        ),
    ):
      fully_named_array.untag('invalid')

    with self.assertRaisesRegex(
        ValueError,
        re.escape(
            "cannot untag ('y', 'x') because they do not appear in the order of"
            " the current named dimensions ('x', 'y')"
        ),
    ):
      fully_named_array.untag('y', 'x')

  def test_order_as(self):
    data = np.arange(10).reshape((2, 5))
    array = named_axes.NamedArray(data, ('x', 'y'))

    actual = array.order_as('x', 'y')
    assert_named_array_equal(actual, array)

    actual = array.order_as('x', ...)
    assert_named_array_equal(actual, array)

    actual = array.order_as(..., 'y')
    assert_named_array_equal(actual, array)

    actual = array.order_as(...)
    assert_named_array_equal(actual, array)

    expected = named_axes.NamedArray(data.T, ('y', 'x'))
    actual = array.order_as('y', 'x')
    assert_named_array_equal(actual, expected)

    expected = named_axes.NamedArray(data.T, ('y', 'x'))
    actual = array.order_as('y', ...)
    assert_named_array_equal(actual, expected)

    expected = named_axes.NamedArray(data.T, ('y', 'x'))
    actual = array.order_as(..., 'x')
    assert_named_array_equal(actual, expected)

  def test_order_as_unnamed_dims(self):
    data = np.arange(10).reshape((2, 5))
    array = named_axes.NamedArray(data, ('x', None))
    with self.assertRaisesRegex(
        ValueError,
        re.escape(
            'cannot reorder the dimensions of an array with unnamed '
            "dimensions: ('x', None)"
        ),
    ):
      array.order_as('x', ...)

  def test_order_as_repeated_ellipsis(self):
    data = np.arange(10).reshape((2, 5))
    array = named_axes.NamedArray(data, ('x', 'y'))
    with self.assertRaisesRegex(
        ValueError,
        re.escape(
            "dimension names contain multiple ellipses (...): (Ellipsis, 'x',"
            ' Ellipsis)'
        ),
    ):
      array.order_as(..., 'x', ...)

  def test_order_as_within_vmap(self):
    data = np.arange(10).reshape((1, 2, 5))
    array = named_axes.NamedArray(data, (None, 'x', 'y'))
    expected = named_axes.NamedArray(data.mT, (None, 'y', 'x'))
    actual = jax.vmap(lambda x: x.order_as('y', 'x'))(array)
    assert_named_array_equal(actual, expected)

  @parameterized.named_parameters(
      dict(testcase_name='numpy', xp=np),
      dict(testcase_name='jax.numpy', xp=jnp),
  )
  def test_nmap_identity(self, xp):
    data = xp.arange(2 * 3 * 4).reshape((2, 3, 4))

    def identity_assert_ndim(ndim):
      def f(x):
        self.assertEqual(x.ndim, ndim)
        return x

      return f

    array = named_axes.NamedArray(data, (None, None, None))
    actual = named_axes.nmap(identity_assert_ndim(ndim=3))(array)
    assert_named_array_equal(actual, array)

    array = named_axes.NamedArray(data, ('x', 'y', 'z'))
    actual = named_axes.nmap(identity_assert_ndim(ndim=0))(array)
    assert_named_array_equal(actual, array)

    array = named_axes.NamedArray(data, ('x', 'y', None))
    expected = array.tag('z').order_as('z', ...).untag('z')
    actual = named_axes.nmap(identity_assert_ndim(ndim=1))(array)
    assert_named_array_equal(actual, expected)

  def test_nmap_scalar_only(self):
    expected = named_axes.NamedArray(3, ())
    actual = named_axes.nmap(jnp.add)(1, 2)
    assert_named_array_equal(actual, expected)

  def test_nmap_namedarray_and_scalar(self):
    data = np.arange(2 * 3 * 4).reshape((2, 3, 4))
    array = named_axes.NamedArray(data, ('x', 'y', 'z'))
    expected = named_axes.NamedArray(data + 1, ('x', 'y', 'z'))

    actual = named_axes.nmap(jnp.add)(array, 1)
    assert_named_array_equal(actual, expected)

    actual = named_axes.nmap(jnp.add)(1, array)
    assert_named_array_equal(actual, expected)

  def test_nmap_two_named_arrays(self):
    data = np.arange(2 * 3 * 4).reshape((2, 3, 4))

    array = named_axes.NamedArray(data, ('x', 'y', 'z'))
    expected = named_axes.NamedArray(data * 2, ('x', 'y', 'z'))
    actual = named_axes.nmap(jnp.add)(array, array)
    assert_named_array_equal(actual, expected)

    actual = named_axes.nmap(jnp.add)(array, array.order_as('z', 'y', 'x'))
    assert_named_array_equal(actual, expected)

    array1 = named_axes.NamedArray(data, ('x', 'y', 'z'))
    array2 = named_axes.NamedArray(100 * data[0, :, 0], ('y',))
    expected = named_axes.NamedArray(
        data + 100 * data[:1, :, :1], ('x', 'y', 'z')
    )
    actual = named_axes.nmap(jnp.add)(array1, array2)
    assert_named_array_equal(actual, expected)

    expected = expected.order_as('y', ...).untag('y')
    actual = named_axes.nmap(jnp.add)(array1.untag('y'), array2.untag('y'))
    assert_named_array_equal(actual, expected)

    array1 = named_axes.NamedArray(data[:, 0, 0], dims=('x',))
    array2 = named_axes.NamedArray(100 * data[0, :, 0], dims=('y',))
    expected = named_axes.NamedArray(
        data[:, :1, 0] + 100 * data[:1, :, 0], ('x', 'y')
    )
    actual = named_axes.nmap(jnp.add)(array1, array2)
    assert_named_array_equal(actual, expected)

    actual = named_axes.nmap(jnp.add)(array2, array1)
    assert_named_array_equal(actual, expected.order_as('y', 'x'))

  def test_nmap_axis_name(self):
    data = np.arange(2 * 3).reshape((2, 3))
    array = named_axes.NamedArray(data, ('x', 'y'))
    expected = named_axes.NamedArray(
        data - data.sum(axis=1, keepdims=True), ('x', 'y')
    )
    actual = named_axes.nmap(lambda x: x - x.sum(axis='y'))(array)
    assert_named_array_equal(actual, expected)

  def test_nmap_inconsistent_named_shape(self):

    def accepts_anything(*unused_args, **unused_kwargs):
      return 1

    array1 = named_axes.NamedArray(np.zeros((2, 3)), ('x', 'y'))
    array2 = named_axes.NamedArray(np.zeros((4,)), 'y')

    with self.assertRaisesRegex(
        ValueError,
        re.escape(
            'Inconsistent sizes in a call to nmap(<NAME>) '
            "for dimensions ['y']:"
            "\n  args[0].named_shape == {'x': 2, 'y': 3}"
            "\n  args[1].named_shape == {'y': 4}"
        ).replace('NAME', '.+'),
    ):
      named_axes.nmap(accepts_anything)(array1, array2)

    array1 = named_axes.NamedArray(np.zeros((2, 3)), ('x', 'y'))
    array2 = named_axes.NamedArray(np.zeros((4, 5)), ('y', 'x'))

    with self.assertRaisesRegex(
        ValueError,
        re.escape(
            'Inconsistent sizes in a call to nmap(<NAME>) '
            "for dimensions ['x', 'y']:"
            "\n  kwargs['bar'][0].named_shape == {'y': 4, 'x': 5}"
            "\n  kwargs['foo'].named_shape == {'x': 2, 'y': 3}"
        ).replace('NAME', '.+'),
    ):
      named_axes.nmap(accepts_anything)(foo=array1, bar=[array2])

  def test_nmap_out_axes_reorder(self):
    data = np.arange(2 * 3 * 4).reshape((2, 3, 4))
    array = named_axes.NamedArray(data, ('x', 'y', 'z'))

    expected = array.order_as('y', 'x', 'z')
    out_axes = {'y': 0, 'x': 1, 'z': 2}
    actual = named_axes.nmap(lambda x: x, out_axes=out_axes)(array)
    assert_named_array_equal(actual, expected)

    expected = array.order_as('y', 'x', 'z')
    out_axes = {'y': -3, 'x': -2, 'z': -1}
    actual = named_axes.nmap(lambda x: x, out_axes=out_axes)(array)
    assert_named_array_equal(actual, expected)

    expected = array.order_as('z', 'x', 'y')
    out_axes = {'z': 0, 'x': 1, 'y': 2}
    actual = named_axes.nmap(lambda x: x, out_axes=out_axes)(array)
    assert_named_array_equal(actual, expected)

  def test_nmap_out_axes_new_dim(self):
    data = np.arange(2 * 3 * 4).reshape((2, 3, 4))
    array = named_axes.NamedArray(data, ('x', 'y', 'z'))

    expected = named_axes.NamedArray(
        data[jnp.newaxis, ...], (None, 'x', 'y', 'z')
    )
    out_axes = {'x': 1, 'y': 2, 'z': 3}
    actual = named_axes.nmap(lambda x: x[jnp.newaxis], out_axes=out_axes)(array)
    assert_named_array_equal(actual, expected)

    expected = named_axes.NamedArray(
        data[:, jnp.newaxis, ...], ('x', None, 'y', 'z')
    )
    out_axes = {'x': 0, 'y': 2, 'z': 3}
    actual = named_axes.nmap(lambda x: x[jnp.newaxis], out_axes=out_axes)(array)
    assert_named_array_equal(actual, expected)

    expected = named_axes.NamedArray(
        data[..., jnp.newaxis], ('x', 'y', 'z', None)
    )
    out_axes = {'x': 0, 'y': 1, 'z': 2}
    actual = named_axes.nmap(lambda x: x[jnp.newaxis], out_axes=out_axes)(array)
    assert_named_array_equal(actual, expected)

    expected = named_axes.NamedArray(
        data[:, jnp.newaxis, ...], ('x', None, 'y', 'z')
    )
    out_axes = {'x': -4, 'y': -2, 'z': -1}
    actual = named_axes.nmap(lambda x: x[jnp.newaxis], out_axes=out_axes)(array)
    assert_named_array_equal(actual, expected)

    expected = named_axes.NamedArray(
        data.mT[..., jnp.newaxis], ('x', 'z', 'y', None)
    )
    out_axes = {'x': -4, 'z': -3, 'y': -2}
    actual = named_axes.nmap(lambda x: x[jnp.newaxis], out_axes=out_axes)(array)
    assert_named_array_equal(actual, expected)

  def test_nmap_out_binary(self):
    data1 = np.arange(2 * 3).reshape((2, 3))
    data2 = 10 * np.arange(3 * 2).reshape((3, 2))
    array1 = named_axes.NamedArray(data1, ('x', 'y'))
    array2 = named_axes.NamedArray(data2, ('y', 'x'))

    expected1 = array1
    expected2 = array2.order_as('x', 'y')
    actual1, actual2 = named_axes.nmap(lambda x, y: (x, y))(array1, array2)
    assert_named_array_equal(actual1, expected1)
    assert_named_array_equal(actual2, expected2)

    expected1 = array1
    expected2 = array2.order_as('x', 'y')
    actual1, actual2 = named_axes.nmap(
        lambda x, y: (x, y), out_axes={'x': 0, 'y': 1}
    )(array1, array2)
    assert_named_array_equal(actual1, expected1)
    assert_named_array_equal(actual2, expected2)

    expected1 = array1.order_as('y', 'x')
    expected2 = array2
    actual1, actual2 = named_axes.nmap(
        lambda x, y: (x, y), out_axes={'x': 1, 'y': 0}
    )(array1, array2)
    assert_named_array_equal(actual1, expected1)
    assert_named_array_equal(actual2, expected2)

  def test_nmap_with_custom_vmap(self):

    @dataclasses.dataclass
    class NonPytree:
      a: jnp.ndarray

    def custom_vmap(fun, in_axes, out_axes, **kwargs):
      def mapped_fun(*args) -> jax.Array:
        leaves, argdef = jax.tree.flatten(args)
        leaves = [x.a if isinstance(x, NonPytree) else x for x in leaves]
        args = jax.tree.unflatten(argdef, leaves)
        return jax.vmap(fun, in_axes, out_axes, **kwargs)(*args)
      return mapped_fun

    def foo(x, y):
      assert y.ndim == 1  # will only run under vmap on 2d inputs.
      return x + y

    y = named_axes.NamedArray(jnp.arange(5 * 7).reshape((5, 7)), (None, 'i'))
    array_x = jnp.arange(5)[::-1]
    expected = named_axes.nmap(foo)(array_x, y)
    actual = named_axes.nmap(foo, vmap=custom_vmap)(NonPytree(a=array_x), y)
    assert_named_array_equal(actual, expected)

  def test_nmap_invalid_out_axes(self):
    data = np.arange(2 * 3).reshape((2, 3))
    array = named_axes.NamedArray(data, ('x', 'y'))

    with self.assertRaisesRegex(
        ValueError,
        re.escape(
            "out_axes keys ['x'] must must match the named "
            "dimensions ['x', 'y']"
        ),
    ):
      named_axes.nmap(lambda x: x, out_axes={'x': 0})(array)

    with self.assertRaisesRegex(
        ValueError,
        re.escape(
            'out_axes must be either all positive or all negative, but got '
            "{'x': 0, 'y': -1}"
        ),
    ):
      named_axes.nmap(lambda x: x, out_axes={'x': 0, 'y': -1})(array)

    with self.assertRaisesRegex(
        ValueError,
        re.escape(
            "out_axes must all have unique values, but got {'x': 0, 'y': 0}"
        ),
    ):
      named_axes.nmap(lambda x: x, out_axes={'x': 0, 'y': 0})(array)

  def test_vectorized_methods(self):
    data = np.arange(2 * 3 * 4).reshape((2, 3, 4))
    dims = (None, 'y', 'z')
    array = named_axes.NamedArray(data, dims)

    expected = named_axes.NamedArray(-data, dims)
    actual = -array
    assert_named_array_equal(actual, expected)

    expected = named_axes.NamedArray(data - 1, dims)
    actual = array - 1
    assert_named_array_equal(actual, expected)

    expected = named_axes.NamedArray(1 - data, dims)
    actual = 1 - array
    assert_named_array_equal(actual, expected)

    expected = named_axes.NamedArray(data, dims)
    actual = (array - 1j * array).real
    assert_named_array_equal(actual, expected)

    expected = named_axes.NamedArray(data + 1j * data, dims)
    actual = (array - 1j * array).conj()
    assert_named_array_equal(actual, expected)

  def test_scalar_conversion(self):
    array = named_axes.NamedArray(1, dims=())
    expected = 1
    actual = int(array)
    self.assertIsInstance(actual, int)
    self.assertEqual(expected, actual)

  def test_tag_and_untag_function(self):
    data = np.arange(2 * 3).reshape((2, 3))
    untagged = named_axes.NamedArray(data, (None, None))
    tagged = named_axes.NamedArray(data, ('x', 'y'))
    untagged_tree = {'a': untagged, 'b': untagged}
    tagged_tree = {'a': tagged, 'b': tagged}

    actual = named_axes.tag(untagged_tree, 'x', 'y')
    assert_named_array_equal(actual['a'], tagged_tree['a'])
    assert_named_array_equal(actual['b'], tagged_tree['b'])

    actual = named_axes.untag(tagged_tree, 'x', 'y')
    assert_named_array_equal(actual['a'], untagged_tree['a'])
    assert_named_array_equal(actual['b'], untagged_tree['b'])

  def test_array_renders_without_error(self):
    data = np.arange(19 * 23).reshape((19, 23))
    array = named_axes.NamedArray(data, ('x', 'y'))

    with self.subTest('explicit_unmasked'):
      res = treescope.render_array(array)
      self.assertTrue(hasattr(res, '_repr_html_'))

    with self.subTest('explicit_masked'):
      res = treescope.render_array(array, valid_mask=array > 100)
      self.assertTrue(hasattr(res, '_repr_html_'))

    with self.subTest('explicit_masked_truncated'):
      res = treescope.render_array(
          array, valid_mask=array > 100, truncate=True, maximum_size=100
      )
      self.assertTrue(hasattr(res, '_repr_html_'))

    with self.subTest('automatic'):
      with treescope.active_autovisualizer.set_scoped(
          treescope.ArrayAutovisualizer()
      ):
        res = treescope.render_to_html(
            array, ignore_exceptions=False, compressed=False
        )
        self.assertIsInstance(res, str)
        self.assertIn('arrayviz', res)

  def test_named_array_info_(self):
    data = np.arange(19 * 23).reshape((19, 23))
    array = named_axes.NamedArray(data, ('x', 'y'))
    adapter = treescope.type_registries.lookup_ndarray_adapter(array)
    with self.subTest('adapter_lookup'):
      self.assertIsNotNone(adapter)
    with self.subTest('named_positional_axis_info'):
      self.assertEqual(
          adapter.get_axis_info_for_array_data(array),
          (
              treescope.ndarray_adapters.NamedPositionalAxisInfo(0, 'x', 19),
              treescope.ndarray_adapters.NamedPositionalAxisInfo(1, 'y', 23),
          ),
      )
    with self.subTest('named_and_positional_axis_info'):
      array = named_axes.NamedArray(data, (None, 'x'))
      self.assertEqual(
          adapter.get_axis_info_for_array_data(array),
          (
              treescope.ndarray_adapters.PositionalAxisInfo(0, 19),
              treescope.ndarray_adapters.NamedPositionalAxisInfo(1, 'x', 23),
          ),
      )

  def test_duckarray(self):

    @jax.tree_util.register_dataclass
    @dataclasses.dataclass
    class Duck(ndarrays.NDArray):
      a: jnp.ndarray
      b: jnp.ndarray

      @property
      def shape(self) -> tuple[int, ...]:
        return self.a.shape

      @property
      def size(self) -> int:
        return math.prod(self.a.shape)

      @property
      def ndim(self) -> int:
        return len(self.a.shape)

      def transpose(self, axes: tuple[int, ...]):
        return Duck(self.a.transpose(axes), self.b.transpose(axes))

      def __getitem__(self, value):
        return Duck(self.a[value], self.b[value])

      def __add__(self, other: int):
        # intentionally implement something funny, that is not equal to
        # jax.tree.map(jnp.add, self, other)
        return Duck(self.a * other, self.b * other)

    duck = Duck(a=jnp.array([1, 2]), b=jnp.array([3, 4]))
    array = named_axes.NamedArray(duck)
    np.testing.assert_array_equal(array.data.a, jnp.array([1, 2]))
    np.testing.assert_array_equal(array.data.b, jnp.array([3, 4]))
    self.assertEqual(array.shape, (2,))
    self.assertIsNone(array.dtype)

    def is_duck_identity(x):
      self.assertIsInstance(x, Duck)
      return x

    result = named_axes.nmap(is_duck_identity)(array)
    chex.assert_trees_all_equal(result, array)

    expected = named_axes.NamedArray(duck + 2)
    actual = array + 2
    chex.assert_trees_all_equal(actual, expected)

    actual = jax.jit(named_axes.nmap(lambda x: x + 2))(array)
    chex.assert_trees_all_equal(actual, expected)

    array_2d = jax.tree.map(lambda x: x[jnp.newaxis, :], array).tag('x', 'y')
    expected = named_axes.NamedArray(
        Duck(a=jnp.array([[1], [2]]), b=jnp.array([[3], [4]])), dims=('y', 'x')
    )
    actual = array_2d.order_as('y', 'x')
    chex.assert_trees_all_equal(actual, expected)

    array_3d = jax.tree.map(lambda x: x[jnp.newaxis, ...], array_2d)
    actual = jax.vmap(lambda x: x.order_as('y', 'x'))(array_3d).tag('z')
    expected = named_axes.NamedArray(
        Duck(a=jnp.array([[[1], [2]]]), b=jnp.array([[[3], [4]]])),
        dims=('z', 'y', 'x'),
    )
    chex.assert_trees_all_equal(actual, expected)


if __name__ == '__main__':
  jax.config.update('jax_traceback_filtering', 'off')
  absltest.main()
