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
"""Tests for observation operator API and implementations."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import chex
import coordax as cx
from flax import nnx
from jax import config  # pylint: disable=g-importing-member
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import learned_transforms
from neuralgcm.experimental.core import observation_operators
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import pytree_utils
from neuralgcm.experimental.core import standard_layers
from neuralgcm.experimental.core import towers
from neuralgcm.experimental.core import transforms
import numpy as np


class DataObservationOperatorsTest(parameterized.TestCase):
  """Tests DataObservationOperator implementation."""

  def test_returns_only_queried_fields(self):
    fields = {
        'a': cx.field(np.ones(7), cx.LabeledAxis('x', np.arange(7))),
        'b': cx.field(np.arange(11), 'z'),
    }
    operator = observation_operators.DataObservationOperator(fields)
    query = {'a': cx.LabeledAxis('x', np.arange(7))}
    actual = operator.observe(inputs={}, query=query)
    expected = {
        'a': cx.field(np.ones(7), cx.LabeledAxis('x', np.arange(7))),
    }
    chex.assert_trees_all_equal(actual, expected)

  def test_subset_along_coord(self):
    pressure_coord = cx.LabeledAxis('pressure', [10, 20, 30, 40, 50])
    fields = {
        'a': cx.field(np.arange(5), pressure_coord),
    }
    operator = observation_operators.DataObservationOperator(fields)
    query_pressure_coord = cx.LabeledAxis('pressure', [20, 40])
    query = {'a': query_pressure_coord}
    actual = operator.observe(inputs={}, query=query)
    expected_field = cx.field(np.array([1, 3]), query_pressure_coord)
    expected = {'a': expected_field}
    chex.assert_trees_all_equal(actual, expected)

  def test_composite_custom_coords_subset(self):
    grid = coordinates.LonLatGrid.T21()
    pressure = coordinates.PressureLevels.with_era5_levels()
    coord = cx.coords.compose(pressure, grid)
    fields = {'a': cx.field(np.ones(coord.shape), coord)}
    operator = observation_operators.DataObservationOperator(fields)
    query_pressure = coordinates.PressureLevels.with_13_era5_levels()
    query_coord = cx.coords.compose(query_pressure, grid)
    query = {'a': query_coord}
    actual = operator.observe(inputs={}, query=query)
    expected_field = cx.field(np.ones(query_coord.shape), query_coord)
    expected = {'a': expected_field}
    chex.assert_trees_all_equal(actual, expected)

  def test_invalid_subset_along_coord(self):
    pressure_coord = cx.LabeledAxis('pressure', [10, 20, 30, 40, 50])
    fields = {
        'a': cx.field(np.arange(5), pressure_coord),
    }
    operator = observation_operators.DataObservationOperator(fields)
    query_pressure_coord = cx.LabeledAxis('pressure', [25, 40])
    query = {'a': query_pressure_coord}
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "query coordinate for 'a' is not a valid slice of"
        f' field:\n{query_pressure_coord}\nvs\n{pressure_coord}',
    ):
      operator.observe(inputs={}, query=query)

  def test_raises_on_missing_field(self):
    fields = {'a': cx.field(np.ones(7), cx.LabeledAxis('x', np.arange(7)))}
    operator = observation_operators.DataObservationOperator(fields)
    query = {'d': cx.LabeledAxis('x', np.arange(7))}
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "query contains k='d' not in ['a']",
    ):
      operator.observe(inputs={}, query=query)

  def test_raises_on_non_matching_coordinate(self):
    coord = cx.LabeledAxis('x', np.arange(7))
    fields = {'a': cx.field(np.ones(7), coord)}
    operator = observation_operators.DataObservationOperator(fields)
    q_coord = cx.LabeledAxis('x', np.linspace(0, 1, 7))
    query = {'a': q_coord}
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "query coordinate for 'a' is not a valid slice of"
        f' field:\n{q_coord}\nvs\n{coord}',
    ):
      operator.observe(inputs={}, query=query)

  def test_raises_on_field_in_query(self):
    coord = cx.LabeledAxis('rel_x', np.arange(7))
    fields = {
        'a': cx.field(np.ones(7), coord),
        'x': cx.field(np.linspace(0, np.pi, 7), coord),
    }
    operator = observation_operators.DataObservationOperator(fields)
    query = {'a': coord, 'x': fields['x'] + 10.0}
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'DataObservationOperator only supports coordinate queries, got'
        f' {query["x"]}',
    ):
      operator.observe(inputs={}, query=query)


class TransformObservationOperatorTest(parameterized.TestCase):
  """Tests TransformObservationOperator with learned transform."""

  def setUp(self):
    super().setUp()
    lon_lat_grid = coordinates.LonLatGrid.T21()
    sigma_levels = coordinates.SigmaLevels.equidistant(4)
    self.lon_lat_grid = lon_lat_grid
    self.sigma_levels = sigma_levels

    input_names = ('u', 'v', 't')
    full_coord = cx.coords.compose(sigma_levels, lon_lat_grid)
    self.inputs = {
        k: cx.field(np.ones(full_coord.shape), full_coord) for k in input_names
    }
    feature_module = transforms.Select(input_names)
    net_factory = functools.partial(
        standard_layers.Mlp.uniform, hidden_size=6, hidden_layers=2
    )
    tower_factory = functools.partial(
        towers.ForwardTower.build_using_factories,
        inputs_in_dims=('d',),
        out_dims=('d',),
        neural_net_factory=net_factory,
    )
    self.observation_transform = (
        learned_transforms.ForwardTowerTransform.build_using_factories(
            input_shapes=pytree_utils.shape_structure(self.inputs),
            target_split_axes={
                'turbulence_index': sigma_levels,
                'evap_rate': cx.Scalar(),
            },
            tower_factory=tower_factory,
            concat_dims=(sigma_levels.dims[0],),
            inputs_transform=feature_module,
            mesh=parallelism.Mesh(None),
            rngs=nnx.Rngs(0),
        )
    )

  def test_predictions_have_correct_coordinates(self):
    operator = observation_operators.TransformObservationOperator(
        self.observation_transform
    )
    full_coord = cx.coords.compose(self.sigma_levels, self.lon_lat_grid)
    query = {'turbulence_index': full_coord, 'evap_rate': self.lon_lat_grid}
    actual = operator.observe(inputs=self.inputs, query=query)
    self.assertSetEqual(set(actual.keys()), set(query.keys()))
    self.assertEqual(cx.get_coordinate(actual['evap_rate']), self.lon_lat_grid)
    self.assertEqual(cx.get_coordinate(actual['turbulence_index']), full_coord)


class MultiObservationOperatorTest(parameterized.TestCase):
  """Tests MultiObservationOperator implementation."""

  def test_multiple_operators(self):

    coord_a = cx.LabeledAxis('a_ax', np.arange(3))
    coord_b = cx.LabeledAxis('b_ax', np.arange(4))
    coord_c = cx.LabeledAxis('c_ax', np.arange(5))

    field_a = cx.field(np.random.rand(3), coord_a)
    field_b = cx.field(np.random.rand(4), coord_b)
    field_c = cx.field(np.random.rand(5), coord_c)
    op1 = observation_operators.DataObservationOperator({'a': field_a})
    op2 = observation_operators.DataObservationOperator(
        {'b': field_b, 'c': field_c}
    )
    inputs = {}

    with self.subTest('all_keys_handled'):
      keys_to_operator = {
          ('a',): op1,
          ('b', 'c'): op2,
      }
      multi_op = observation_operators.MultiObservationOperator(
          keys_to_operator
      )
      query = {
          'a': coord_a,
          'b': coord_b,
          'c': coord_c,
      }
      expected_obs = {
          'a': field_a,
          'b': field_b,
          'c': field_c,
      }
      actual_obs = multi_op.observe(inputs, query)
      chex.assert_trees_all_equal(actual_obs, expected_obs)

    with self.subTest('query_key_not_handled_by_any_operator'):
      keys_to_operator = {
          ('a',): op1,
          ('b',): op2,
      }
      multi_op = observation_operators.MultiObservationOperator(
          keys_to_operator
      )
      query = {
          'a': coord_a,
          'b': coord_b,
          'c': coord_c,
      }
      supported_keys = set(sum(keys_to_operator.keys(), start=()))
      query_keys = set(query.keys())
      expected_message = (
          f'query keys {query_keys} are not a subset of supported keys'
          f' {supported_keys}'
      )
      with self.assertRaisesWithLiteralMatch(ValueError, expected_message):
        multi_op.observe(inputs, query)


if __name__ == '__main__':
  config.update('jax_traceback_filtering', 'off')
  absltest.main()
