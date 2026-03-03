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
