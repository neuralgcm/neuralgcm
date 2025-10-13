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
import coordax as cx
import jax
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import data_specs
import numpy as np


class DataSpecsTest(parameterized.TestCase):

  def test_construct_query_with_coord_and_field_specs(self):
    x, y = cx.LabeledAxis('x', np.arange(3)), cx.LabeledAxis('y', np.arange(4))
    output_data_specs = {
        'src1': {
            'var_coord': data_specs.OutputDataSpec(x),
            'var_field': data_specs.OutputDataSpec(y, is_field=True),
        }
    }
    field_y = cx.wrap(np.ones(4), y)
    data = {'src1': {'var_field': field_y}}
    query = data_specs.construct_query(data, output_data_specs)
    self.assertIs(query['src1']['var_coord'], x)
    self.assertIs(query['src1']['var_field'], field_y)

  def test_are_valid_input_specs_exact_match_check(self):
    grid = coordinates.LonLatGrid.T21()
    input_data_specs = {'src1': {'var': data_specs.InputDataSpec(grid)}}
    with self.subTest('expected_true'):
      data_specs_ok = {'src1': {'var': grid}}
      self.assertTrue(
          data_specs.are_valid_input_specs(data_specs_ok, input_data_specs)
      )
    with self.subTest('expected_false'):
      data_specs_fail = {'src1': {'var': coordinates.LonLatGrid.T42()}}
      self.assertFalse(
          data_specs.are_valid_input_specs(data_specs_fail, input_data_specs)
      )

  def test_are_valid_input_specs_subset_check(self):
    dt = np.timedelta64(1, 'h')
    timedelta = coordinates.TimeDelta(np.arange(2) * dt)
    dim_match_rules = {'timedelta': 'subset'}
    input_data_specs = {
        'i': {'time': data_specs.InputDataSpec(timedelta, dim_match_rules)}
    }
    with self.subTest('expected_true'):
      data_specs_ok = {'i': {'time': coordinates.TimeDelta(np.arange(3) * dt)}}
      self.assertTrue(
          data_specs.are_valid_input_specs(data_specs_ok, input_data_specs)
      )
    with self.subTest('expected_false'):
      data_specs_fail = {
          'i': {'time': coordinates.TimeDelta(np.arange(1) * dt)}
      }
      self.assertFalse(
          data_specs.are_valid_input_specs(data_specs_fail, input_data_specs)
      )

  def test_are_valid_input_specs_presence_check(self):
    dt = np.timedelta64(1, 'h')
    timedelta = coordinates.TimeDelta(np.arange(2) * dt)
    x = cx.LabeledAxis('x', np.arange(3))
    tdelta_and_x = cx.compose_coordinates(timedelta, x)
    dim_match_rules = {'timedelta': 'present'}
    input_data_specs = {
        'i': {'var': data_specs.InputDataSpec(tdelta_and_x, dim_match_rules)}
    }
    with self.subTest('expected_true'):
      single_tdelta = coordinates.TimeDelta(dt[None])
      data_specs_ok = {'i': {'var': cx.compose_coordinates(single_tdelta, x)}}
      self.assertTrue(
          data_specs.are_valid_input_specs(data_specs_ok, input_data_specs)
      )
    with self.subTest('expected_false_missing_timedelta'):
      data_specs_fail = {'i': {'var': x}}
      self.assertFalse(
          data_specs.are_valid_input_specs(data_specs_fail, input_data_specs)
      )
    with self.subTest('expected_false_wrong_timedelta_type'):
      not_timedelta_coord = cx.LabeledAxis('timedelta', np.arange(2))
      data_specs_fail = {
          'i': {'var': cx.compose_coordinates(not_timedelta_coord, x)}
      }
      self.assertFalse(
          data_specs.are_valid_input_specs(data_specs_fail, input_data_specs)
      )

  def test_are_valid_input_specs_optional_var_missing(self):
    x = cx.LabeledAxis('x', np.arange(3))
    input_data_specs = {
        'src1': {'var_x': data_specs.InputDataSpec(x, optional=True)}
    }
    data_specs_ok = {'src1': {}}
    self.assertTrue(
        data_specs.are_valid_input_specs(data_specs_ok, input_data_specs)
    )
    input_data_specs['src1']['var_x'] = data_specs.InputDataSpec(x)
    self.assertFalse(
        data_specs.are_valid_input_specs(data_specs_ok, input_data_specs)
    )

  def test_are_valid_inputs(self):
    x = cx.LabeledAxis('x', np.arange(3))
    input_data_specs = {'src1': {'var_x': data_specs.InputDataSpec(x)}}
    field_ok = cx.wrap(np.ones(3), x)
    field_fail = cx.wrap(np.ones(4), cx.LabeledAxis('x', np.arange(4)))
    inputs_ok = {'src1': {'var_x': field_ok}}
    inputs_fail = {'src1': {'var_x': field_fail}}
    self.assertTrue(data_specs.are_valid_inputs(inputs_ok, input_data_specs))
    self.assertFalse(data_specs.are_valid_inputs(inputs_fail, input_data_specs))


class InputDataSpecsTest(parameterized.TestCase):

  def test_post_init_fills_dim_match_rules(self):
    """Tests that missing dim_match_rules default to 'exact'."""
    x = cx.LabeledAxis('x', np.arange(3))
    y = cx.LabeledAxis('y', np.arange(4))
    coord = cx.compose_coordinates(x, y)
    spec = data_specs.InputDataSpec(coord, dim_match_rules={'x': 'subset'})
    self.assertEqual(spec.dim_match_rules, {'x': 'subset', 'y': 'exact'})

  def test_post_init_raises_if_extra_dims_in_rules(self):
    """Tests that error is raised if dim_match_rules has extra dims."""
    x = cx.LabeledAxis('x', np.arange(3))
    with self.assertRaisesRegex(ValueError, 'incompatible sets of dimensions'):
      data_specs.InputDataSpec(x, dim_match_rules={'x': 'exact', 'y': 'exact'})

  def test_exact_match_coord_property(self):
    """Tests that exact_match_coord returns coord with 'exact' dims."""
    x = cx.LabeledAxis('x', np.arange(3))
    y = cx.LabeledAxis('y', np.arange(4))
    z = cx.LabeledAxis('z', np.arange(5))
    coord = cx.compose_coordinates(x, y, z)
    spec = data_specs.InputDataSpec(
        coord, dim_match_rules={'x': 'subset', 'y': 'exact', 'z': 'present'}
    )
    self.assertIs(spec.exact_match_coord, y)

  def test_with_any_timedelta_constructor(self):
    x = cx.LabeledAxis('x', np.arange(3))
    spec = data_specs.InputDataSpec.with_any_timedelta(x)
    self.assertEqual(
        spec.dim_match_rules, {'x': 'exact', 'timedelta': 'present'}
    )
    self.assertEqual(spec.coord.dims, ('timedelta', 'x'))

  def test_with_given_timedelta_constructor(self):
    x = cx.LabeledAxis('x', np.arange(3))
    timedeltas = np.arange(4) * np.timedelta64(1, 'h')
    spec = data_specs.InputDataSpec.with_given_timedelta(x, timedeltas)
    self.assertEqual(
        spec.dim_match_rules, {'x': 'exact', 'timedelta': 'subset'}
    )
    self.assertEqual(spec.coord.dims, ('timedelta', 'x'))
    expected_td_coord = coordinates.TimeDelta(timedeltas)
    td_axis_in_spec = spec.coord.axes[0]
    self.assertEqual(td_axis_in_spec, expected_td_coord)


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
