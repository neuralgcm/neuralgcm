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
import jax_datetime as jdt
from neuralgcm.experimental.couplers import generic
import numpy as np


class GenericCouplersTest(absltest.TestCase):
  """Tests generic couplers `__call__` and `update_fields` methods."""

  def test_instant_data_coupler(self):
    x = cx.LabeledAxis('x', np.arange(5))
    field_coords = {'u': x}
    var_to_data_keys = {'u': 'src'}
    coupler = generic.InstantDataCoupler(field_coords, var_to_data_keys)

    with self.subTest('init_values'):
      initial_state = coupler(None)
      self.assertIn('u', initial_state)
      np.testing.assert_array_equal(
          initial_state['u'].data, np.full(x.shape, np.nan)
      )

    with self.subTest('updated_values'):
      new_data = jnp.ones(x.shape)
      update_data = {'src': {'u': cx.field(new_data, x)}}
      coupler.update_fields(update_data)
      updated_state = coupler(None)
      np.testing.assert_array_equal(updated_state['u'].data, new_data)

    with self.subTest('raises_on_missing_fields'):
      with self.assertRaisesRegex(
          ValueError, 'missing Fields for "src"'
      ):
        coupler.update_fields({})

  def test_multi_time_data_coupler(self):
    x = cx.LabeledAxis('x', np.arange(5))
    field_coords = {'u': x}
    var_to_data_keys = {'u': 'src'}
    coupler = generic.MultiTimeDataCoupler(
        field_coords, var_to_data_keys, time_data_key='src', n_time_slices=3
    )

    with self.subTest('init_values'):
      self.assertIn('u', coupler.coupling_fields)
      self.assertEqual(coupler.coupling_fields['u'].shape, (3, 5))

    with self.subTest('updated_values'):
      t0 = jdt.to_datetime('2020-01-01T00:00:00')
      for i in range(3):
        new_data = jnp.full(x.shape, i)
        update_data = {
            'src': {
                'u': cx.field(new_data, x),
                'time': cx.field(t0 + jdt.to_timedelta(i, 'h')),
            }
        }
        coupler.update_fields(update_data)

      for i in range(3):
        query_time = t0 + jdt.to_timedelta(i + 0.2 * i * np.timedelta64(1, 'h'))
        actual = coupler(cx.field(query_time))
        expected = cx.field(jnp.full(x.shape, i), x)
        cx.testing.assert_fields_equal(actual['u'], expected)


if __name__ == '__main__':
  absltest.main()
