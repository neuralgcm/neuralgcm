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
import numpy as np


class MockModule(nnx.Module):
  """Mock module for testing ensure_unchanged_state_structure."""

  def __init__(self):
    x_size: int = 5
    x = cx.SizedAxis('x', x_size)
    dt = coordinates.TimeDelta(np.array([np.timedelta64(1, 'h')]))
    self.x = x
    self.u = nnx.Param(cx.wrap(jnp.zeros(x_size), x))
    self.u_trajectory = nnx.Variable(cx.wrap(jnp.ones((1, x_size)), dt, x))
    self.u_sum = nnx.Intermediate(jnp.zeros(()))

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


if __name__ == '__main__':
  config.update('jax_traceback_filtering', 'off')
  absltest.main()
