# Copyright 2026 Google LLC
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
import coordax as cx
import jax.numpy as jnp
from neuralgcm.experimental.couplers import generic
import numpy as np


class GenericCouplersTest(absltest.TestCase):
  """Tests generic couplers `__call__` and `update_fields` methods."""

  def test_instant_data_coupler(self):
    x = cx.LabeledAxis('x', np.arange(5))
    field_coords = {'u': x}
    var_to_data_keys = {'u': 'my_source_data'}
    coupler = generic.InstantDataCoupler(field_coords, var_to_data_keys)

    with self.subTest('init_values'):
      initial_state = coupler(None)
      self.assertIn('u', initial_state)
      np.testing.assert_array_equal(
          initial_state['u'].data, np.full(x.shape, np.nan)
      )

    with self.subTest('updated_values'):
      new_data = jnp.ones(x.shape)
      update_data = {'my_source_data': {'u': cx.field(new_data, x)}}
      coupler.update_fields(update_data)
      updated_state = coupler(None)
      np.testing.assert_array_equal(updated_state['u'].data, new_data)

    with self.subTest('raises_on_missing_fields'):
      with self.assertRaisesRegex(
          ValueError, 'missing Fields for "my_source_data"'
      ):
        coupler.update_fields({})


if __name__ == '__main__':
  absltest.main()
