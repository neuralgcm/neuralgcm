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

from absl.testing import absltest
from absl.testing import parameterized
import chex
import coordax as cx
import jax
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import data_specs
import numpy as np


class ValidateInputsTest(parameterized.TestCase):
  """Tests that validate_inputs works as expected."""

  def test_validate_inputs_with_coordinate(self):
    t = coordinates.TimeDelta(np.arange(3) * np.timedelta64(1, 'h'))
    x = cx.LabeledAxis('x', np.linspace(0, np.pi, num=4))
    y = cx.LabeledAxis('y', np.linspace(0, np.e, num=5))
    tx = cx.compose_coordinates(t, x)
    ty = cx.compose_coordinates(t, y)
    rng = np.random.RandomState(42)
    inputs = {
        'data_key': {
            'u': cx.wrap(rng.randn(*tx.shape), tx),
            'v': cx.wrap(rng.randn(*ty.shape), ty),
        }
    }
    inputs_spec = {'data_key': {'u': tx, 'v': ty}}
    data_specs.validate_inputs(inputs, inputs_spec)

  def test_validate_inputs_with_exact_coord_spec(self):
    t = coordinates.TimeDelta(np.arange(3) * np.timedelta64(1, 'h'))
    x = cx.LabeledAxis('x', np.linspace(0, np.pi, num=4))
    tx = cx.compose_coordinates(t, x)
    rng = np.random.RandomState(42)
    inputs = {'data_key': {'u': cx.wrap(rng.randn(*tx.shape), tx)}}
    inputs_spec = {'data_key': {'u': data_specs.CoordSpec(tx)}}
    data_specs.validate_inputs(inputs, inputs_spec)

  def test_validate_inputs_with_present_coord_spec(self):
    x = cx.LabeledAxis('x', np.linspace(0, np.pi, num=4))
    spec_coord = data_specs.CoordSpec.with_any_timedelta(x)
    inputs_spec = {'data_key': {'u': spec_coord}}
    rng = np.random.RandomState(42)

    t = coordinates.TimeDelta(np.arange(10) * np.timedelta64(1, 'h'))
    tx = cx.compose_coordinates(t, x)
    inputs = {'data_key': {'u': cx.wrap(rng.randn(*tx.shape), tx)}}
    data_specs.validate_inputs(inputs, inputs_spec)

    with self.subTest('invalid_type'):
      t_wrong_type = cx.LabeledAxis('timedelta', np.arange(10))
      tx_wrong = cx.compose_coordinates(t_wrong_type, x)
      inputs_wrong = {
          'data_key': {'u': cx.wrap(rng.randn(*tx_wrong.shape), tx_wrong)}
      }
      with self.assertRaisesRegex(
          ValueError, 'is not present or not of the expected type'
      ):
        data_specs.validate_inputs(inputs_wrong, inputs_spec)

  def test_validate_inputs_with_subset_coord_spec(self):
    x = cx.LabeledAxis('x', np.linspace(0, np.pi, num=4))
    subset_timedelta = np.arange(5) * np.timedelta64(1, 'h')
    spec_coord = data_specs.CoordSpec.with_given_timedelta(
        x, subset_timedelta
    )
    inputs_spec = {'data_key': {'u': spec_coord}}
    rng = np.random.RandomState(42)

    t = coordinates.TimeDelta(np.arange(10) * np.timedelta64(1, 'h'))
    tx = cx.compose_coordinates(t, x)
    inputs = {'data_key': {'u': cx.wrap(rng.randn(*tx.shape), tx)}}
    data_specs.validate_inputs(inputs, inputs_spec)

    with self.subTest('not_a_subset'):
      t_wrong = coordinates.TimeDelta(
          (np.arange(10) + 5) * np.timedelta64(1, 'h')
      )
      tx_wrong = cx.compose_coordinates(t_wrong, x)
      inputs_wrong = {
          'data_key': {'u': cx.wrap(rng.randn(*tx_wrong.shape), tx_wrong)}
      }
      with self.assertRaisesRegex(
          ValueError, 'does not contain all required ticks'
      ):
        data_specs.validate_inputs(inputs_wrong, inputs_spec)

  def test_validate_inputs_with_optional_spec(self):
    x = cx.LabeledAxis('x', np.linspace(0, np.pi, num=4))
    y = cx.LabeledAxis('y', np.linspace(0, np.e, num=5))
    rng = np.random.RandomState(42)
    inputs_spec = {
        'data_key': {
            'u': x,
            'v': data_specs.OptionalSpec(spec=y),
        }
    }
    inputs = {'data_key': {'u': cx.wrap(rng.randn(*x.shape), x)}}
    data_specs.validate_inputs(inputs, inputs_spec)

    inputs_with_uv = {
        'data_key': {
            'u': cx.wrap(rng.randn(*x.shape), x),
            'v': cx.wrap(rng.randn(*y.shape), y),
        }
    }
    data_specs.validate_inputs(inputs_with_uv, inputs_spec)

    with self.subTest('optional_present_but_invalid'):
      inputs_wrong_v = {
          'data_key': {
              'u': cx.wrap(rng.randn(*x.shape), x),
              'v': cx.wrap(rng.randn(*x.shape), x),  # wrong coord.
          }
      }
      with self.assertRaisesRegex(
          ValueError, 'Coordinates .* have different dimensions'
      ):
        data_specs.validate_inputs(inputs_wrong_v, inputs_spec)

    with self.subTest('non_optional_is_missing'):
      inputs_no_u = {'data_key': {'v': cx.wrap(rng.randn(*y.shape), y)}}
      with self.assertRaisesWithLiteralMatch(
          ValueError, 'Missing non-optional variables "u"'
      ):
        data_specs.validate_inputs(inputs_no_u, inputs_spec)

  def test_validate_inputs_with_incompatible_coordinate_raises(self):
    x = cx.LabeledAxis('x', np.linspace(0, np.pi, num=4))
    wrong_x = cx.LabeledAxis('x', np.linspace(0, np.e, num=4))
    rng = np.random.RandomState(42)
    inputs = {'data_key': {'u': cx.wrap(rng.randn(*x.shape), x)}}
    inputs_spec = {'data_key': {'u': wrong_x}}
    with self.assertRaisesRegex(
        ValueError, 'Coordinate axis .* for dimension x does not match'
    ):
      data_specs.validate_inputs(inputs, inputs_spec)

  def test_validate_inputs_with_incompatible_coord_spec_raises(self):
    x = cx.LabeledAxis('x', np.linspace(0, np.pi, num=4))
    y = cx.LabeledAxis('y', np.linspace(0, np.e, num=5))
    rng = np.random.RandomState(42)
    inputs = {'data_key': {'u': cx.wrap(rng.randn(*x.shape), x)}}
    inputs_spec = {'data_key': {'u': data_specs.CoordSpec(y)}}
    with self.assertRaisesRegex(
        ValueError, 'Coordinates .* have different dimensions'
    ):
      data_specs.validate_inputs(inputs, inputs_spec)


class ConstructQueryTest(parameterized.TestCase):
  """Tests that construct_query works as expected."""

  def test_construct_query_with_coordinates(self):
    x = cx.LabeledAxis('x', np.linspace(0, np.pi, num=4))
    y = cx.LabeledAxis('y', np.linspace(0, np.e, num=5))
    queries_spec = {'data': {'u': x, 'v': y}}

    queries = data_specs.construct_query({}, queries_spec)
    chex.assert_trees_all_equal(queries, {'data': {'u': x, 'v': y}})

  def test_construct_query_with_coord_spec(self):
    x = cx.LabeledAxis('x', np.linspace(0, np.pi, num=4))
    y = cx.LabeledAxis('y', np.linspace(0, np.e, num=5))
    queries_spec = {'data': {'u': data_specs.CoordSpec(x), 'v': y}}
    queries = data_specs.construct_query({}, queries_spec)
    chex.assert_trees_all_equal(queries, {'data': {'u': x, 'v': y}})

  def test_construct_query_with_field_in_query_spec(self):
    x = cx.LabeledAxis('x', np.linspace(0, np.pi, num=4))
    y = cx.LabeledAxis('y', np.linspace(0, np.e, num=5))
    rng = np.random.RandomState(42)
    u = cx.wrap(rng.randn(*x.shape), x)
    v = cx.wrap(rng.randn(*y.shape), y)
    inputs = {'data': {'u': u, 'v': v}}
    queries_spec = {
        'data': {
            'u': data_specs.FieldInQuerySpec(spec=x),
            'v': y,
        }
    }
    queries = data_specs.construct_query(inputs, queries_spec)
    chex.assert_trees_all_equal(queries, {'data': {'u': u, 'v': y}})


class CoordSpecTest(parameterized.TestCase):
  """Tests CoordSpec custom constructors and post-init behavior."""

  def test_post_init_fills_dim_match_rules(self):
    """Tests that missing dim_match_rules default to 'exact'."""
    x = cx.LabeledAxis('x', np.arange(3))
    y = cx.LabeledAxis('y', np.arange(4))
    coord = cx.compose_coordinates(x, y)
    spec = data_specs.CoordSpec(
        coord, dim_match_rules={'x': data_specs.DimMatchRules.SUBSET}
    )
    expected_rules = {
        'x': data_specs.DimMatchRules.SUBSET,
        'y': data_specs.DimMatchRules.EXACT,
    }
    self.assertEqual(spec.dim_match_rules, expected_rules)

  def test_post_init_raises_if_extra_dims_in_rules(self):
    """Tests that error is raised if dim_match_rules has extra dims."""
    x = cx.LabeledAxis('x', np.arange(3))
    with self.assertRaisesRegex(ValueError, 'incompatible sets of dimensions'):
      data_specs.CoordSpec(
          x,
          dim_match_rules={
              'x': data_specs.DimMatchRules.EXACT,
              'y': data_specs.DimMatchRules.EXACT,
          },
      )

  def test_with_any_timedelta_constructor(self):
    x = cx.LabeledAxis('x', np.arange(3))
    spec = data_specs.CoordSpec.with_any_timedelta(x)
    self.assertEqual(
        spec.dim_match_rules,
        {
            'x': data_specs.DimMatchRules.EXACT,
            'timedelta': data_specs.DimMatchRules.PRESENT,
        },
    )
    self.assertEqual(spec.coord.dims, ('timedelta', 'x'))

  def test_with_given_timedelta_constructor(self):
    x = cx.LabeledAxis('x', np.arange(3))
    timedeltas = np.arange(4) * np.timedelta64(1, 'h')
    spec = data_specs.CoordSpec.with_given_timedelta(x, timedeltas)
    self.assertEqual(
        spec.dim_match_rules,
        {
            'x': data_specs.DimMatchRules.EXACT,
            'timedelta': data_specs.DimMatchRules.SUBSET,
        },
    )
    self.assertEqual(spec.coord.dims, ('timedelta', 'x'))
    expected_td_coord = coordinates.TimeDelta(timedeltas)
    td_axis_in_spec = spec.coord.axes[0]
    self.assertEqual(td_axis_in_spec, expected_td_coord)


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
