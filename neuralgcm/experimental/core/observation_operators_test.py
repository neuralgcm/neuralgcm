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
from flax import nnx
from jax import config  # pylint: disable=g-importing-member
from neuralgcm.experimental import coordax as cx
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import feature_transforms
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
        'a': cx.wrap(np.ones(7), cx.LabeledAxis('x', np.arange(7))),
        'b': cx.wrap(np.arange(11), 'z'),
    }
    operator = observation_operators.DataObservationOperator(fields)
    query = {'a': cx.LabeledAxis('x', np.arange(7))}
    actual = operator.observe(inputs={}, query=query)
    expected = {
        'a': cx.wrap(np.ones(7), cx.LabeledAxis('x', np.arange(7))),
    }
    chex.assert_trees_all_equal(actual, expected)

  def test_raises_on_missing_field(self):
    fields = {'a': cx.wrap(np.ones(7), cx.LabeledAxis('x', np.arange(7)))}
    operator = observation_operators.DataObservationOperator(fields)
    query = {'d': cx.LabeledAxis('x', np.arange(7))}
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "Query contains k='d' not in ['a']",
    ):
      operator.observe(inputs={}, query=query)

  def test_raises_on_non_matching_coordinate(self):
    coord = cx.LabeledAxis('x', np.arange(7))
    fields = {'a': cx.wrap(np.ones(7), coord)}
    operator = observation_operators.DataObservationOperator(fields)
    q_coord = cx.LabeledAxis('x', np.linspace(0, 1, 7))
    query = {'a': q_coord}
    with self.assertRaisesWithLiteralMatch(
        ValueError, f'Query (a, {q_coord}) does not match field.{coord=}'
    ):
      _ = operator.observe(inputs={}, query=query)

  def test_raises_on_field_in_query(self):
    coord = cx.LabeledAxis('rel_x', np.arange(7))
    fields = {
        'a': cx.wrap(np.ones(7), coord),
        'x': cx.wrap(np.linspace(0, np.pi, 7), coord),
    }
    operator = observation_operators.DataObservationOperator(fields)
    query = {'a': coord, 'x': fields['x'] + 10.0}
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'DataObservationOperator only supports coordinate queries, got'
        f' {query["x"]}',
    ):
      _ = operator.observe(inputs={}, query=query)


class FixedLearnedObservationOperatorTest(parameterized.TestCase):
  """Tests FixedLearnedObservationOperator implementation."""

  def setUp(self):
    super().setUp()
    lon_lat_grid = coordinates.LonLatGrid.T21()
    sigma_levels = coordinates.SigmaLevels.equidistant(4)
    self.lon_lat_grid = lon_lat_grid
    self.sigma_levels = sigma_levels

    input_names = ('u', 'v', 't')
    full_coord = cx.compose_coordinates(sigma_levels, lon_lat_grid)
    self.inputs = {
        k: cx.wrap(np.ones(full_coord.shape), full_coord) for k in input_names
    }
    feature_module = transforms.Select('|'.join(input_names))
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
            targets={
                'turbulence_index': full_coord,
                'evap_rate': lon_lat_grid,
            },
            tower_factory=tower_factory,
            dims_to_align=(lon_lat_grid,),
            in_transform=feature_module,
            mesh=parallelism.Mesh(None),
            rngs=nnx.Rngs(0),
        )
    )

  def test_predictions_have_correct_coordinates(self):
    operator = observation_operators.FixedLearnedObservationOperator(
        self.observation_transform
    )
    full_coord = cx.compose_coordinates(self.sigma_levels, self.lon_lat_grid)
    query = {'turbulence_index': full_coord, 'evap_rate': self.lon_lat_grid}
    actual = operator.observe(inputs=self.inputs, query=query)
    self.assertSetEqual(set(actual.keys()), set(query.keys()))
    self.assertEqual(cx.get_coordinate(actual['evap_rate']), self.lon_lat_grid)
    self.assertEqual(cx.get_coordinate(actual['turbulence_index']), full_coord)


class LearnedSparseScalarObservationFromNeighborsTest(parameterized.TestCase):
  """Tests FixedLearnedObservationOperator implementation."""

  def setUp(self):
    super().setUp()
    self.grid = coordinates.LonLatGrid.T21()
    feature_module = feature_transforms.LatitudeFeatures(self.grid)
    layer_factory = functools.partial(
        standard_layers.MlpUniform, hidden_size=6, n_hidden_layers=2
    )
    tower_factory = functools.partial(
        towers.ForwardTower.build_using_factories,
        inputs_in_dims=('d',),
        out_dims=('d',),
        neural_net_factory=layer_factory,
    )
    prediction_targets = {'temperature': cx.Scalar(), 'wind_speed': cx.Scalar()}
    self.operator = (
        observation_operators.LearnedSparseScalarObservationFromNeighbors.build_using_factories(
            target_predictions=prediction_targets,
            grid=self.grid,
            grid_features=feature_module,
            tower_factory=tower_factory,
            input_shapes={},
            mesh=parallelism.Mesh(None),
            rngs=nnx.Rngs(0),
        )
    )

  def test_output_structure(self):
    sparse_coord = cx.LabeledAxis('id', np.arange(7))
    np.random.seed(0)
    longitudes = cx.wrap(np.random.uniform(0, 360, 7), sparse_coord)
    latitudes = cx.wrap(np.random.uniform(-90, 90, 7), sparse_coord)
    full_query = {
        'longitude': longitudes,
        'latitude': latitudes,
        'temperature': sparse_coord,
        'wind_speed': sparse_coord,
    }
    actual = self.operator.observe({}, full_query)
    self.assertSetEqual(
        set(actual.keys()),
        {'longitude', 'latitude', 'temperature', 'wind_speed'},
    )
    np.testing.assert_array_equal(actual['longitude'].data, longitudes.data)
    np.testing.assert_array_equal(actual['latitude'].data, latitudes.data)
    self.assertEqual(cx.get_coordinate(actual['temperature']), sparse_coord)
    self.assertEqual(cx.get_coordinate(actual['wind_speed']), sparse_coord)


class MultiObservationOperatorTest(parameterized.TestCase):
  """Tests MultiObservationOperator implementation."""

  def test_multiple_operators(self):

    coord_a = cx.LabeledAxis('a_ax', np.arange(3))
    coord_b = cx.LabeledAxis('b_ax', np.arange(4))
    coord_c = cx.LabeledAxis('c_ax', np.arange(5))

    field_a = cx.wrap(np.random.rand(3), coord_a)
    field_b = cx.wrap(np.random.rand(4), coord_b)
    field_c = cx.wrap(np.random.rand(5), coord_c)
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
