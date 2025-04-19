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

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from neuralgcm.experimental.core import pytree_utils
from neuralgcm.experimental.core import typing
import numpy as np
import tree_math


@tree_math.struct
class TreeMathStruct:
  array_attr: typing.Array
  float_attr: float
  dict_attr: dict[str, float]


class PytreeUtilsTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          pytree={
              'a': np.arange(8).reshape((2, 2, 2)),
              'b': np.ones((2, 2, 2)),
          },
          axis=0,
      ),
      dict(
          pytree=(np.arange(16).reshape((2, 4, 2)), np.zeros((2, 2, 2))),
          axis=1,
      ),
      dict(
          pytree=(np.arange(4).reshape((4, 1)), np.zeros((4, 5))),
          axis=-1,
      ),
  )
  def test_tree_pack_unpack_roundtrip(self, pytree, axis):
    """Tests that packing -> unpacking a pytree preserves the data."""
    pytree_shapes = pytree_utils.shape_structure(pytree)
    packed = pytree_utils.pack_pytree(pytree, axis=axis)
    unpacked = pytree_utils.unpack_to_pytree(packed, pytree_shapes, axis=axis)
    chex.assert_trees_all_close(pytree, jax.device_get(unpacked))

  @parameterized.parameters(
      dict(
          pytree={
              'a': np.arange(8).reshape((1, 2, 4)),
              'b': np.ones((2, 4)),
          },
          ndim=3,
          axis=0,
      ),
      dict(
          pytree=(np.arange(16), np.zeros((16, 7))),
          ndim=2,
          axis=1,
      ),
      dict(
          pytree=(np.arange(4).reshape((4, 1)), np.zeros((4, 5))),
          ndim=2,
          axis=-1,
      ),
  )
  def test_expand_to_ndim_squeeze_to_shapes_roundtrip(self, pytree, ndim, axis):
    """Tests that expand_to_ndim --> squeeze_to_shapes preserves the data."""
    shapes = pytree_utils.shape_structure(pytree)
    expanded = pytree_utils.expand_to_ndim(pytree, ndim=ndim, axis=axis)
    squeezed = pytree_utils.squeeze_to_shapes(expanded, shapes, axis=axis)
    chex.assert_trees_all_close(pytree, jax.device_get(squeezed))

  def test_expand_to_ndim_works_with_shape_dtypes(self):
    inputs = {
        'a': typing.ShapeFloatStruct((12, 32, 64)),
        'b': typing.ShapeFloatStruct((32, 64)),
    }
    expanded = pytree_utils.expand_to_ndim(inputs, ndim=3, axis=0)
    expected = {
        'a': typing.ShapeFloatStruct((12, 32, 64)),
        'b': typing.ShapeFloatStruct((1, 32, 64)),
    }
    chex.assert_trees_all_equal(expanded, expected)

  @parameterized.parameters(
      dict(
          pytree={
              'a': np.array([1, 2, 3]),
              'b': np.array([4, 5, 6]),
              'c': {'d': np.array([7, 8, 9])},
          },
          axis=0,
          stacked_expected=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
      ),
      dict(
          pytree={
              'a': np.array([1, 2, 3]),
              'b': np.array([4, 5, 6]),
              'c': {'d': np.array([7, 8, 9])},
          },
          axis=1,
          stacked_expected=np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]]),
      ),
      dict(
          pytree={
              'a': np.ones((1, 2, 4)),
              'b': np.ones((1, 2, 4)),
              'c': {'d': np.ones((1, 2, 4))},
          },
          axis=-2,
          stacked_expected=np.ones((1, 2, 3, 4)),
      ),
  )
  def test_tree_stack_and_unstack(self, pytree, stacked_expected, axis):
    """Tests that tree stacking/roundtrip returns expected values."""
    stacked = pytree_utils.stack_pytree(pytree, axis)
    with self.subTest('stack_pytree'):
      chex.assert_trees_all_close(jax.device_get(stacked), stacked_expected)
    with self.subTest('unstack_pytree'):
      unstack_shapes = pytree_utils.shape_structure(pytree)
      unstacked = pytree_utils.unstack_to_pytree(stacked, unstack_shapes, axis)
      chex.assert_trees_all_close(pytree, jax.device_get(unstacked))
    with self.subTest('jit'):

      def roundtrip(x):
        stacked = pytree_utils.stack_pytree(x, axis)
        unstack_shapes = pytree_utils.shape_structure(x)
        return pytree_utils.unstack_to_pytree(stacked, unstack_shapes, axis)

      roundtrip_fn = jax.jit(roundtrip)
      chex.assert_trees_all_close(pytree, jax.device_get(roundtrip_fn(pytree)))

  @parameterized.parameters(
      dict(pytree=(np.zeros((6, 3)), np.ones((6, 2, 2))), idx=3, axis=0),
      dict(pytree=(np.zeros((3, 8)), np.ones((6, 8))), idx=3, axis=1),
      dict(pytree={'a': np.zeros((3, 9)), 'b': np.ones((6, 9))}, idx=3, axis=1),
      dict(pytree=np.zeros((13, 5)), idx=6, axis=0),
      dict(pytree=(np.zeros(9), (np.ones((9, 1)), np.ones(9))), idx=6, axis=0),
  )
  def test_split_and_concat(self, pytree, idx, axis):
    """Tests that split_along_axis, concat_along_axis return expected shapes."""
    split_a, split_b = pytree_utils.split_along_axis(pytree, idx, axis, False)
    with self.subTest('split_shape'):
      self.assertEqual(jax.tree.leaves(split_a)[0].shape[axis], idx)

    reconstruction = pytree_utils.concat_along_axis([split_a, split_b], axis)
    with self.subTest('split_concat_roundtrip'):
      chex.assert_trees_all_close(reconstruction, pytree)

    same_ndims = len(set(a.ndim for a in jax.tree.leaves(reconstruction))) == 1
    if not same_ndims:
      with self.subTest('raises_when_wrong_ndims'):
        with self.assertRaisesRegex(ValueError, 'arrays in `inputs` expected'):
          _, _ = pytree_utils.split_along_axis(pytree, idx, axis, True)

    with self.subTest('multiple_concat_shape'):
      arrays = [split_a, split_a, split_b, split_b]
      double_concat = pytree_utils.concat_along_axis(arrays, axis)
      actual_shape = jax.tree.leaves(double_concat)[0].shape[axis]
      expected_shape = jax.tree.leaves(pytree)[0].shape[axis] * 2
      self.assertEqual(actual_shape, expected_shape)

  def test_pytree_cache(self):
    """Tests that tree_cache works as expected."""
    eval_count = 0

    @pytree_utils.tree_cache
    def cached_func(unused_arg):
      nonlocal eval_count
      eval_count += 1
      return eval_count

    args = {'a': 1, 'b': np.arange(3)}
    result = cached_func(args)
    self.assertEqual(result, 1)  # function called once.

    result = cached_func(args)
    self.assertEqual(result, 1)  # still returns 1, since called with same args.

    result = cached_func('something else')
    self.assertEqual(result, 2)

  @parameterized.named_parameters(
      dict(
          testcase_name='nested',
          example={'a': np.arange(8), 'b': {'c': np.ones((2, 2))}, 'd': 3.14},
      ),
      dict(
          testcase_name='double_nesting',
          example={'a': {'b': {'c': 1}}, 'd': 2},
      ),
      dict(
          testcase_name='nesting_same_subelements',
          example={'a': {'b': 4.12}, 'b': 3.14},
      ),
      dict(
          testcase_name='contains_empty_subdict',
          example={'a': {'b': 4.12}, 'd': {}},
      ),
      dict(
          testcase_name='contains_nested_empty_subdict',
          example={'a': {'b': 4.12}, 'd': {'x': {}}},
      ),
      dict(
          testcase_name='raises_on_sep_in_name',
          example={'a': {'b&c': 4.12}, 'b': 3.14},
          should_raise=True,
      ),
  )
  def test_dict_flatten_unflatten_roundtrip(self, example, should_raise=False):
    """Tests that flatten_dcit -> unflatten_dict acts as identity."""
    if should_raise:
      with self.assertRaises(ValueError):
        pytree_utils.unflatten_dict(*pytree_utils.flatten_dict(example))
    else:
      actual = pytree_utils.unflatten_dict(*pytree_utils.flatten_dict(example))
      chex.assert_trees_all_close(actual, example)

  @parameterized.named_parameters(
      dict(
          testcase_name='replace_root_and_nested_value',
          inputs={'a': np.arange(8), 'b': {'c': np.ones((2, 2))}, 'd': 3.14},
          replace_dict={'a': 2.73, 'b': {'c': np.zeros((2, 2))}},
          default=np.ones(1),
          expected={'a': 2.73, 'b': {'c': np.zeros((2, 2))}, 'd': np.ones(1)},
      ),
      dict(
          testcase_name='double_nesting',
          inputs={'a': {'b': {'c': 1}}, 'd': 2},
          replace_dict={'a': {'b': {'c': 2.73}}},
          default=1.0,
          expected={'a': {'b': {'c': 2.73}}, 'd': 1.0},
      ),
      dict(
          testcase_name='empty_subdict',
          inputs={'a': 1, 'b': {}, 'c': 3},
          replace_dict={'a': 2, 'c': 5},
          default=1.0,
          expected={'a': 2, 'b': {}, 'c': 5},
      ),
      dict(
          testcase_name='nested_empty_subdict',
          inputs={'a': 1, 'b': {}, 'c': 3, 'd': {'x': {}}},
          replace_dict={'a': 2, 'c': 5},
          default=1.0,
          expected={'a': 2, 'b': {}, 'c': 5, 'd': {'x': {}}},
      ),
      dict(
          testcase_name='raises_on_unused_replace_values',
          inputs={'a': 1.2, 'b': 2.5},
          replace_dict={'a': 2.1, 'spc_humidity': -9000},
          default=1.0,
          expected={'a': {'b': {'c': 2.73}}, 'd': 1.0},
          should_raise=True,
      ),
  )
  def test_replace_with_matching_or_default(
      self,
      inputs,
      replace_dict,
      default,
      expected,
      should_raise=False,
  ):
    """Tests that replace_with_matching_or_default works as expected."""
    if should_raise:
      with self.assertRaises(ValueError):
        pytree_utils.replace_with_matching_or_default(
            inputs, replace_dict, default
        )
    else:
      actual = pytree_utils.replace_with_matching_or_default(
          inputs, replace_dict, default
      )
      chex.assert_trees_all_close(actual, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name='sqrt_1d_square_rest',
          condition_fn=lambda x: jnp.asarray(x).ndim == 1,
          f=lambda x: x**0.5,
          g=lambda x: x**2,
          pytree={'a': np.arange(8), 'b': 5, 'c': 2 * np.eye(2)},
          expected={'a': np.arange(8) ** 0.5, 'b': 5**2, 'c': 2**2 * np.eye(2)},
      ),
  )
  def test_tree_map_where(self, condition_fn, f, g, pytree, expected):
    """Tests that tree_map_where works as expected."""
    with self.subTest('jax_backed'):
      actual = pytree_utils.tree_map_where(condition_fn, f, g, pytree)
      chex.assert_trees_all_close(actual, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name='sqrt_1d_square_rest',
          f=lambda x: x**0.5,
          scalar_fn=lambda x: x**2,
          pytree={'a': np.arange(8), 'b': 5, 'c': 2 * np.eye(2)},
          expected={
              'a': np.arange(8) ** 0.5,
              'b': 5**2,
              'c': 2**0.5 * np.eye(2),
          },
      ),
      dict(
          testcase_name='resclae_non_time',
          f=lambda x: x / 2,
          scalar_fn=lambda x: x,
          pytree={'a': np.ones(8), 'b': np.eye(2), 'time': 3.14},
          expected={'a': np.ones(8) / 2, 'b': np.eye(2) / 2, 'time': 3.14},
      ),
  )
  def test_tree_map_over_nonscalars(self, f, scalar_fn, pytree, expected):
    """Tests that tree_map_over_nonscalars works as expected."""
    assert_is_jax_array = lambda x: self.assertIsInstance(x, jax.Array)
    assert_not_jax_array = lambda x: self.assertNotIsInstance(x, jax.Array)
    with self.subTest('jax_backed'):
      actual = pytree_utils.tree_map_over_nonscalars(
          f=f, x=pytree, scalar_fn=scalar_fn
      )  # jax is the default backend.
      jax.tree_util.tree_map(assert_is_jax_array, actual)
      chex.assert_trees_all_close(actual, expected)
    with self.subTest('numpy_backed'):
      actual = pytree_utils.tree_map_over_nonscalars(
          f=f, x=pytree, scalar_fn=scalar_fn, backend='numpy'
      )
      jax.tree_util.tree_map(assert_not_jax_array, actual)
      chex.assert_trees_all_close(actual, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name='split_two_along_axis_0',
          pytree={'a': np.stack([np.ones(4), np.zeros(4)]), 'b': np.arange(2)},
          axis=0,
          expected=(
              {'a': np.ones(4), 'b': np.asarray(0)},
              {'a': np.zeros(4), 'b': np.asarray(1)},
          ),
      ),
      dict(
          testcase_name='split_three_along_axis_1',
          pytree={'a': np.eye(3), 'b': np.arange(3)[np.newaxis, :]},
          axis=1,
          expected=(
              {'a': np.eye(3)[:, 0], 'b': np.asarray([0])},
              {'a': np.eye(3)[:, 1], 'b': np.asarray([1])},
              {'a': np.eye(3)[:, 2], 'b': np.asarray([2])},
          ),
      ),
  )
  def test_split_axis(self, pytree, axis, expected):
    """Tests that split_axis works as expected."""
    actual = pytree_utils.split_axis(pytree, axis=axis)
    chex.assert_trees_all_close(actual, expected)
    chex.assert_trees_all_equal_shapes(actual, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name='dict_input',
          inputs={'a': np.ones((2, 2, 2)), 'b': np.arange(2), 'c': 2},
      ),
      dict(
          testcase_name='nested_dict_input',
          inputs={'a': np.ones((2, 2, 2)), 'b': np.arange(2), 'c': {'d': 4}},
      ),
      dict(
          testcase_name='tree_math_struct_input',
          inputs=TreeMathStruct(np.ones(10), 1.54, {'a': 0.5, 'b': 0.25}),
      ),
  )
  def test_asdict_forward_and_roundtrip(self, inputs):
    dict_repr, from_dict_fn = pytree_utils.as_dict(inputs)
    with self.subTest('forward'):
      self.assertIsInstance(dict_repr, dict)
    with self.subTest('round_trip'):
      reconstructed = from_dict_fn(dict_repr)
      chex.assert_trees_all_close(reconstructed, inputs)


if __name__ == '__main__':
  absltest.main()
