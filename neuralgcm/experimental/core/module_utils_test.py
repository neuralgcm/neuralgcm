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

from absl.testing import absltest
from absl.testing import parameterized
import coordax as cx
from flax import nnx
from jax import config  # pylint: disable=g-importing-member
import jax.numpy as jnp
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import module_utils
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
      with self.assertRaisesRegex(ValueError, 'not present anywhere in'):
        module_utils.untag_module_state(module, new_b, vectorization_specs)

      module_utils.untag_module_state(module, b, vectorization_specs)
      with self.assertRaisesRegex(ValueError, 'not present anywhere in'):
        module_utils.tag_module_state(module, new_b, vectorization_specs)


if __name__ == '__main__':
  config.update('jax_traceback_filtering', 'off')
  absltest.main()
