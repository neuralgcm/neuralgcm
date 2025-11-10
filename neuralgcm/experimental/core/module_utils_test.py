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
"""Tests that normalization modules work as expected."""

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import chex
import coordax as cx
from flax import nnx
import jax
from jax import config  # pylint: disable=g-importing-member
import jax.numpy as jnp
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import module_utils
from neuralgcm.experimental.core import nnx_compat
from neuralgcm.experimental.core import typing
import numpy as np


class MockModule(nnx.Module):
  """Mock module for testing ensure_unchanged_state_structure."""

  def __init__(self):
    nx: int = 5
    x = cx.SizedAxis('x', nx)
    dt = coordinates.TimeDelta(np.array([np.timedelta64(1, 'h')]))
    self.x = x
    self.u = typing.Prognostic(cx.wrap(jnp.zeros(nx), x))
    self.u_trajectory = typing.DynamicInput(cx.wrap(jnp.ones((1, nx)), dt, x))
    self.u_sum = typing.Diagnostic(cx.wrap(jnp.zeros(())))

  @module_utils.ensure_unchanged_state_structure
  def always_preserves_state(self, u: cx.Field):
    self.u_sum.value += jnp.sum(u.data)

  @module_utils.ensure_unchanged_state_structure
  def preserves_state_if_same_coords(self, u: cx.Field):
    self.u.value += u

  @module_utils.ensure_unchanged_state_structure(excluded_dims=['timedelta'])
  def allowed_to_change_only_along_time(self, u: cx.Field):
    self.u_trajectory.value = u


class ModuleUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='1d_input_same_coords',
          u_input=cx.wrap(np.ones(5), cx.SizedAxis('x', 5)),
          raises_if_same_coords=False,
          raises_on_timedelta_change=True,
      ),
      dict(
          testcase_name='1d_input_diff_coords',
          u_input=cx.wrap(np.ones(5), cx.SizedAxis('NotX', 5)),
          raises_if_same_coords=True,
          raises_on_timedelta_change=True,
      ),
      dict(
          testcase_name='2d_input_with_timedelta',
          u_input=cx.wrap(
              np.ones((2, 5)),
              coordinates.TimeDelta(np.arange(2) * np.timedelta64(1, 'D')),
              cx.SizedAxis('x', 5),
          ),
          raises_if_same_coords=True,
          raises_on_timedelta_change=False,
      ),
      dict(
          testcase_name='2d_input',
          u_input=cx.wrap(
              np.ones((2, 5)),
              cx.SizedAxis('y', 2),
              cx.SizedAxis('x', 5),
          ),
          raises_if_same_coords=True,
          raises_on_timedelta_change=True,
      ),
  )
  def test_ensure_unchanged_state_structure(
      self,
      u_input,
      raises_if_same_coords,
      raises_on_timedelta_change,
  ):
    module = MockModule()
    module.always_preserves_state(u_input)
    if raises_if_same_coords:
      with self.assertRaises(ValueError):
        module.preserves_state_if_same_coords(u_input)
    else:
      module.preserves_state_if_same_coords(u_input)
    if raises_on_timedelta_change:
      with self.assertRaises(ValueError):
        module.allowed_to_change_only_along_time(u_input)
    else:
      module.allowed_to_change_only_along_time(u_input)

  def test_vectorize_module_subset_of_variables(self):
    module = MockModule()
    u_traj_coords = module.u_trajectory.coordinate
    b = cx.SizedAxis('batch', 3)
    vectorization_specs = {typing.Prognostic: b}
    module_utils.vectorize_module(module, vectorization_specs)
    self.assertEqual(module.u.coordinate, cx.compose_coordinates(b, module.x))
    self.assertEqual(module.u_trajectory.coordinate, u_traj_coords)
    self.assertEqual(module.u_sum.coordinate, cx.Scalar())

  def test_vectorize_module_twice(self):
    module = MockModule()
    u_coord = module.u.coordinate
    u_traj_coord = module.u_trajectory.coordinate
    b, e = cx.SizedAxis('batch', 3), cx.SizedAxis('ensemble', 2)
    vectorization_specs_b = {typing.SimulationVariable: b}
    module_utils.vectorize_module(module, vectorization_specs_b)
    self.assertEqual(module.u.coordinate, cx.compose_coordinates(b, u_coord))
    self.assertEqual(module.u_sum.coordinate, b)
    self.assertEqual(module.u_trajectory.coordinate, u_traj_coord)
    vectorization_specs_e = {typing.SimulationVariable: e}
    module_utils.vectorize_module(module, vectorization_specs_e)
    self.assertEqual(module.u.coordinate, cx.compose_coordinates(e, b, u_coord))
    self.assertEqual(module.u_sum.coordinate, cx.compose_coordinates(e, b))
    self.assertEqual(module.u_trajectory.coordinate, u_traj_coord)

  def test_vectorize_module_different_specs_for_types(self):
    module = MockModule()
    u_coord = module.u.coordinate
    u_traj_coords = module.u_trajectory.coordinate
    b, e = cx.SizedAxis('batch', 3), cx.SizedAxis('ensemble', 2)
    vectorization_specs = {
        typing.Prognostic: b,
        typing.Diagnostic: e,
    }
    module_utils.vectorize_module(module, vectorization_specs)
    self.assertEqual(module.u.coordinate, cx.compose_coordinates(b, u_coord))
    self.assertEqual(module.u_trajectory.coordinate, u_traj_coords)
    self.assertEqual(module.u_sum.coordinate, e)

  def test_vectorize_2d_coordinate(self):
    module = MockModule()
    u_coord = module.u.coordinate
    u_traj_coords = module.u_trajectory.coordinate
    b, e = cx.SizedAxis('batch', 3), cx.SizedAxis('ensemble', 2)
    vectorization_specs = {typing.Prognostic: cx.compose_coordinates(e, b)}
    module_utils.vectorize_module(module, vectorization_specs)
    self.assertEqual(module.u.coordinate, cx.compose_coordinates(e, b, u_coord))
    self.assertEqual(module.u_trajectory.coordinate, u_traj_coords)
    self.assertEqual(module.u_sum.coordinate, cx.Scalar())

  def test_vectorize_composite_filter(self):
    module = MockModule()
    u_coord = module.u.coordinate
    u_traj_coords = module.u_trajectory.coordinate
    b = cx.SizedAxis('batch', 3)
    vectorization_specs = {(typing.Prognostic, typing.DynamicInput): b}
    module_utils.vectorize_module(module, vectorization_specs)
    self.assertEqual(module.u_sum.coordinate, cx.Scalar())
    self.assertEqual(module.u.coordinate, cx.compose_coordinates(b, u_coord))
    self.assertEqual(
        module.u_trajectory.coordinate, cx.compose_coordinates(b, u_traj_coords)
    )

  def test_tag_untag_module_state(self):
    module = MockModule()
    u_coord = module.u.coordinate
    u_traj_coord = module.u_trajectory.coordinate
    b, e = cx.SizedAxis('batch', 3), cx.SizedAxis('ensemble', 2)
    vectorization_specs = {
        typing.Prognostic: b,
        typing.Diagnostic: e,
        typing.DynamicInput: cx.compose_coordinates(b, e),
    }
    module_utils.vectorize_module(module, vectorization_specs)
    with self.subTest('original_state'):
      self.assertEqual(module.u_sum.coordinate, e)
      self.assertEqual(module.u.coordinate, cx.compose_coordinates(b, u_coord))
      self.assertEqual(
          module.u_trajectory.coordinate,
          cx.compose_coordinates(b, e, u_traj_coord),
      )

    # Untag batch axis
    pos_b, pos_e = cx.DummyAxis(None, 3), cx.DummyAxis(None, 2)
    with self.subTest('untag_batch_axis'):
      module_utils.untag_module_state(module, b, vectorization_specs)
      self.assertEqual(module.u_sum.coordinate, e)
      self.assertEqual(
          module.u.coordinate, cx.compose_coordinates(pos_b, u_coord)
      )
      self.assertEqual(
          module.u_trajectory.coordinate,
          cx.compose_coordinates(pos_b, e, u_traj_coord),  # None, e, ...
      )
      module_utils.tag_module_state(module, b, vectorization_specs)  # retag.

    with self.subTest('untag_ensemble_axis'):
      module_utils.untag_module_state(module, e, vectorization_specs)
      self.assertEqual(module.u_sum.coordinate, pos_e)
      self.assertEqual(
          module.u.coordinate, cx.compose_coordinates(b, u_coord)  # b, ...
      )
      self.assertEqual(
          module.u_trajectory.coordinate,
          cx.compose_coordinates(b, pos_e, u_traj_coord),  # b, None, ...;
      )
      module_utils.tag_module_state(module, e, vectorization_specs)  # retag.

    with self.subTest('axis_not_in_vectorized_axes'):
      new_b = cx.LabeledAxis('batch', np.linspace(0, 1, 3))
      with self.assertRaisesRegex(ValueError, 'that are not present in'):
        module_utils.untag_module_state(module, new_b, vectorization_specs)

      module_utils.untag_module_state(module, b, vectorization_specs)
      with self.assertRaisesRegex(ValueError, 'that are not present in'):
        module_utils.tag_module_state(module, new_b, vectorization_specs)

  @parameterized.named_parameters(
      dict(
          testcase_name='disjoint_filters',
          specs_list=[
              {typing.Prognostic: cx.SizedAxis('batch', 3)},
              {typing.Diagnostic: cx.SizedAxis('ensemble', 2)},
          ],
      ),
      dict(
          testcase_name='overlapping_filters',
          specs_list=[
              {typing.Prognostic: cx.SizedAxis('batch', 3)},
              {typing.Prognostic: cx.SizedAxis('ensemble', 2)},
          ],
      ),
      dict(
          testcase_name='overlapping_filters_with_addition',
          specs_list=[
              {typing.Prognostic: cx.SizedAxis('batch', 3)},
              {
                  typing.Prognostic: cx.SizedAxis('ensemble', 2),
                  typing.Diagnostic: cx.SizedAxis('ensemble', 2),
              },
          ],
      ),
      dict(
          testcase_name='filters_with_mixed_vectorization',
          specs_list=[
              {
                  typing.Prognostic: cx.compose_coordinates(
                      cx.SizedAxis('batch', 3), cx.SizedAxis('forcing', 4)
                  ),
                  typing.DynamicInput: cx.SizedAxis('forcing', 4),
              },
              {
                  typing.Prognostic: cx.SizedAxis('ensemble', 2),
                  typing.DynamicInput: cx.SizedAxis('ensemble', 2),
              },
          ],
      ),
      dict(
          testcase_name='with_ellipsis',
          specs_list=[
              {typing.Prognostic: cx.SizedAxis('batch', 3)},
              {...: cx.SizedAxis('ensemble', 2)},
          ],
      ),
  )
  def test_merge_vectorized_axes_tracks_vectorization(self, specs_list):
    module = MockModule()
    orig_module = MockModule()
    vector_axes = {}
    for spec in specs_list:
      module_utils.vectorize_module(module, spec)
      vector_axes = module_utils.merge_vectorized_axes(spec, vector_axes)
      self._check_vectorized_axes_align(vector_axes, module, orig_module)

  def _check_vectorized_axes_align(
      self, vector_axes, vectorized_module, original_module
  ):
    original_states = nnx.state(original_module, *vector_axes.keys())
    vector_states = nnx.state(vectorized_module, *vector_axes.keys())
    if len(vector_axes) == 1:
      vector_states = [vector_states]
      original_states = [original_states]
    state_and_axes = zip(original_states, vector_states, vector_axes.values())
    for original_state, vector_state, axes in state_and_axes:
      original_leaves = jax.tree.leaves(original_state, is_leaf=cx.is_field)
      vector_leaves = jax.tree.leaves(vector_state, is_leaf=cx.is_field)
      for v_field, field in zip(vector_leaves, original_leaves):
        self.assertEqual(
            cx.compose_coordinates(axes, field.coordinate),
            v_field.coordinate,
        )

  def test_merge_vectorized_axes_with_overlap_and_difference(self):
    b = cx.SizedAxis('batch', 3)
    e = cx.SizedAxis('ensemble', 2)
    spec1 = {typing.Prognostic: b, typing.DynamicInput: b}
    spec2 = {typing.Prognostic: e}
    actual = module_utils.merge_vectorized_axes(spec1, spec2)
    expected = {
        typing.Prognostic: cx.compose_coordinates(b, e),
        typing.DynamicInput: b,
    }
    self.assertEqual(actual, expected)

  def test_merge_vectorized_axes_with_ellipsis(self):
    b = cx.SizedAxis('batch', 3)
    e = cx.SizedAxis('ensemble', 2)
    spec1 = {typing.Prognostic: b, typing.DynamicInput: b}
    spec2 = {typing.DynamicInput: cx.Scalar(), ...: e}
    actual = module_utils.merge_vectorized_axes(spec1, spec2)
    expected = {
        typing.Prognostic: cx.compose_coordinates(b, e),
        typing.DynamicInput: b,
        ...: e,
    }
    self.assertEqual(actual, expected)

  def test_merge_vectorized_axes_raises_error(self):
    b = cx.SizedAxis('batch', 3)
    e = cx.SizedAxis('ensemble', 2)
    spec1 = {typing.Prognostic: b}
    spec2 = {(typing.Prognostic, typing.Diagnostic): e}
    with self.assertRaisesRegex(ValueError, 'potentially overlapping filters'):
      module_utils.merge_vectorized_axes(spec1, spec2)


class StateInAxesUtilTest(parameterized.TestCase):
  """Tests in_axes utility functions."""

  def test_state_in_axes_for_coord(self):
    b, e = cx.SizedAxis('batch', 3), cx.SizedAxis('ensemble', 2)
    x = cx.SizedAxis('x', 5)
    vectorized_axes = {
        typing.Prognostic: cx.compose_coordinates(b, x),
        typing.Diagnostic: e,
        typing.DynamicInput: cx.compose_coordinates(b, e, x),
    }
    with self.subTest('map_over_b'):
      actual = module_utils.state_in_axes_for_coord(vectorized_axes, b)
      expected = nnx.StateAxes({
          typing.Prognostic: 0,
          typing.Diagnostic: None,
          typing.DynamicInput: 0,
      })
      chex.assert_trees_all_equal(actual, expected)
    with self.subTest('map_over_e'):
      actual = module_utils.state_in_axes_for_coord(vectorized_axes, e)
      expected = nnx.StateAxes({
          typing.Prognostic: None,
          typing.Diagnostic: 0,
          typing.DynamicInput: 1,
      })
      chex.assert_trees_all_equal(actual, expected)
    with self.subTest('map_over_x'):
      actual = module_utils.state_in_axes_for_coord(vectorized_axes, x)
      expected = nnx.StateAxes({
          typing.Prognostic: 1,
          typing.Diagnostic: None,
          typing.DynamicInput: 2,
      })
      chex.assert_trees_all_equal(actual, expected)

  def test_state_in_axes_for_coord_with_nesting(self):
    b, e = cx.SizedAxis('batch', 3), cx.SizedAxis('ensemble', 2)
    x = cx.SizedAxis('x', 5)
    vectorized_axes = {
        typing.Prognostic: cx.compose_coordinates(b, x),
        typing.Diagnostic: e,
        typing.DynamicInput: cx.compose_coordinates(b, e, x),
    }
    actual_outer, actual_inner = module_utils.state_in_axes_for_coord(
        vectorized_axes, [b, e]
    )
    expected_outer = nnx.StateAxes({
        typing.Prognostic: 0,
        typing.Diagnostic: None,
        typing.DynamicInput: 0,
    })
    expected_inner = nnx.StateAxes({
        typing.Prognostic: None,
        typing.Diagnostic: 0,
        typing.DynamicInput: 0,
    })
    self.assertEqual(set(actual_outer.keys()), set(expected_outer.keys()))
    self.assertEqual(set(actual_inner.keys()), set(expected_inner.keys()))
    for k, v in actual_outer.items():
      self.assertEqual(v, expected_outer[k])
    for k, v in actual_inner.items():
      self.assertEqual(v, expected_inner[k])


class VectorizeModuleFnTest(parameterized.TestCase):
  """Tests vectorize_module_fn transformation."""

  def test_vectorize_fn_no_args_state_update(self):
    b = cx.SizedAxis('batch', 3)
    module = MockModule()
    vector_axes = {typing.Diagnostic: b}
    module_utils.vectorize_module(module, vector_axes)
    # u_sum is Diagnostic, starts as 0, vectorized to [0,0,0] over b.

    def update_module_no_arg(module):
      module.always_preserves_state(cx.wrap(1.0))

    v_fn = module_utils.vectorize_module_fn(
        update_module_no_arg, vector_axes, b
    )
    v_fn(module)
    expected = cx.wrap(jnp.ones(3), b)
    chex.assert_trees_all_close(module.u_sum.value, expected)

  def test_vectorize_fn_with_arg_state_update(self):
    b = cx.SizedAxis('batch', 3)
    module = MockModule()
    x = module.x
    vector_axes = {typing.Prognostic: b}
    module_utils.vectorize_module(module, vector_axes)
    # u is Prognostic, starts as zeros(5,), vectorized to zeros(3,5) over b, x.

    def update_module_with_arg(module, u_field):
      module.preserves_state_if_same_coords(u_field)

    arg = cx.wrap(jnp.arange(b.size), b) * cx.wrap(jnp.ones(x.size), x)
    v_fn = module_utils.vectorize_module_fn(
        update_module_with_arg, vector_axes, b
    )
    v_fn(module, arg)
    expected = arg
    chex.assert_trees_all_close(module.u.value, expected)

  def test_vectorize_fn_with_return_value(self):
    b = cx.SizedAxis('batch', 3)
    module = MockModule()
    vector_axes = {typing.Prognostic: b}
    module_utils.vectorize_module(module, vector_axes)

    def get_u_plus_arg(module, arg_field):
      return module.u.value + arg_field

    arg = cx.wrap(jnp.ones((b.size, module.x.size)), b, module.x)
    v_fn = module_utils.vectorize_module_fn(get_u_plus_arg, vector_axes, b)
    result = v_fn(module, arg)
    expected = module.u.value + arg
    chex.assert_trees_all_close(result, expected)

  def test_vectorize_fn_nested_axes_with_state_update(self):
    b = cx.SizedAxis('batch', 3)
    e = cx.SizedAxis('ensemble', 2)
    module = MockModule()
    x = module.x
    vector_axes = {typing.Prognostic: cx.compose_coordinates(e, b)}
    module_utils.vectorize_module(module, vector_axes)
    # u is Prognostic, shape (2,3,5), coords (e,b,x)

    def update_module_with_arg(module, u_field):
      module.preserves_state_if_same_coords(u_field)

    arg_e = cx.wrap(jnp.arange(e.size), e)
    arg_b = cx.wrap(jnp.arange(b.size), b)
    arg_x = cx.wrap(jnp.ones(x.size), x)
    arg = arg_e * arg_b * arg_x  # shape (2,3,5), coords (e,b,x)

    v_fn = module_utils.vectorize_module_fn(
        update_module_with_arg, vector_axes, [e, b]
    )
    v_fn(module, arg)
    expected = arg
    chex.assert_trees_all_close(module.u.value, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name='one_level',
          in_axes=(nnx.StateAxes({'a': 0, 'b': 1}),),
          expected_nested=(nnx.StateAxes({'a': 0, 'b': 1}),),
      ),
      dict(
          testcase_name='two_levels_no_shift',
          in_axes=(
              nnx.StateAxes({'a': 1, 'b': 1}),
              nnx.StateAxes({'a': 0, 'b': 0}),
          ),
          expected_nested=(
              nnx.StateAxes({'a': 1, 'b': 1}),
              nnx.StateAxes({'a': 0, 'b': 0}),
          ),
      ),
      dict(
          testcase_name='two_levels_with_shift',
          in_axes=(
              nnx.StateAxes({typing.Prognostic: 0, typing.Diagnostic: 1}),
              nnx.StateAxes({typing.Prognostic: 1, typing.Diagnostic: 0}),
          ),
          expected_nested=(
              nnx.StateAxes({typing.Prognostic: 0, typing.Diagnostic: 1}),
              nnx.StateAxes({typing.Prognostic: 0, typing.Diagnostic: 0}),
          ),
      ),
      dict(
          testcase_name='three_levels_with_shifts_and_none',
          in_axes=(
              nnx.StateAxes({'a': 1, nnx.Param: None, 'c': 2}),
              nnx.StateAxes({'a': 0, nnx.Param: 1, 'c': 0}),
              nnx.StateAxes({'a': 2, nnx.Param: 0, 'c': 1}),
          ),
          expected_nested=(
              nnx.StateAxes({'a': 1, nnx.Param: None, 'c': 2}),
              nnx.StateAxes({'a': 0, nnx.Param: 1, 'c': 0}),
              nnx.StateAxes({'a': 0, nnx.Param: 0, 'c': 0}),
          ),
      ),
  )
  def test_nest_state_in_axes(self, in_axes, expected_nested):
    nested_in_axes = module_utils.nest_state_in_axes(*in_axes)
    for actual, expected in zip(nested_in_axes, expected_nested):
      self.assertEqual(set(actual.keys()), set(expected.keys()))
      for k, v in actual.items():
        self.assertEqual(v, expected[k])

  def test_vectorize_module_fn_with_args(self):

    @nnx_compat.dataclass
    class ScaleDemoModule(nnx.Module):
      param: nnx.Param = dataclasses.field(
          default_factory=lambda: nnx.Param(cx.wrap(2.0))
      )

      def __call__(self, x: cx.Field) -> cx.Field:
        return x * self.param.value

    e, b = cx.SizedAxis('e', 2), cx.SizedAxis('b', 4)
    eb = cx.compose_coordinates(e, b)

    # Vectorize module over `e` and simple `fn` over `eb`.
    module = ScaleDemoModule()
    vector_axes = {nnx.Param: e}  # vectorize module for dimension `e`.
    module_utils.vectorize_module(module, vector_axes)
    module.param.value = cx.wrap(1 + np.arange(e.size), e)  # update scales.
    fn = lambda module, x: module(x)
    eb_vectorized_fn = module_utils.vectorize_module_fn(
        fn, vector_axes, eb, allow_non_vector_axes=True
    )

    # Apply vectorized function to inputs with different axes order.
    x = cx.wrap(np.arange(e.size * b.size).reshape((e.size, b.size)), eb)
    x_reordered = x.order_as(b, e)
    expected_data = x.data * (1 + np.arange(e.size))[:, np.newaxis]

    with self.subTest('in_vectorization_order'):
      result_x = eb_vectorized_fn(module, x)
      self.assertEqual(result_x.coordinate, eb)
      np.testing.assert_allclose(result_x.data, expected_data)

    with self.subTest('in_different_from_vectorization_order'):
      result_x_reordered = eb_vectorized_fn(module, x_reordered)
      self.assertEqual(result_x_reordered.coordinate, eb)
      np.testing.assert_allclose(result_x_reordered.data, expected_data)


if __name__ == '__main__':
  config.update('jax_traceback_filtering', 'off')
  absltest.main()
