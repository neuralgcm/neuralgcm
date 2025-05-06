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
from neuralgcm.experimental import pytree_mappings
from neuralgcm.experimental import pytree_transforms
from neuralgcm.experimental import towers
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import observation_operators
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import pytree_utils
from neuralgcm.experimental.core import standard_layers
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
    self.coords = coordinates.DinosaurCoordinates(
        horizontal=coordinates.LonLatGrid.T21(),
        vertical=coordinates.SigmaLevels.equidistant(4),
    )
    input_names = ('u', 'v', 't')
    self.inputs = {
        k: cx.wrap(np.ones(self.coords.shape), self.coords) for k in input_names
    }
    feature_module = pytree_transforms.PrognosticFeatures(input_names)
    tower_factory = functools.partial(
        towers.ColumnTower,
        column_net_factory=functools.partial(
            standard_layers.MlpUniform, hidden_size=6, n_hidden_layers=2
        ),
    )
    mapping_factory = functools.partial(
        pytree_mappings.ChannelMapping,
        tower_factory=tower_factory,
    )
    embedding_factory = functools.partial(
        pytree_mappings.Embedding,
        feature_module=feature_module,
        mapping_factory=mapping_factory,
        input_state_shapes=pytree_utils.shape_structure(self.inputs),
        mesh=parallelism.Mesh(None),
    )
    volume_field_names = ('turbulence_index',)
    surface_field_names = ('evaporation_rate',)
    self.observation_mapping = pytree_mappings.CoordsStateMapping(
        coords=self.coords,
        surface_field_names=surface_field_names,
        volume_field_names=volume_field_names,
        embedding_factory=embedding_factory,
        rngs=nnx.Rngs(0),
        mesh=parallelism.Mesh(None),
    )

  def test_returns_only_queried_fields(self):
    operator = observation_operators.FixedLearnedObservationOperator(
        self.observation_mapping
    )
    query_a = {'evaporation_rate': self.coords.horizontal}
    actual = operator.observe(inputs=self.inputs, query=query_a)
    self.assertSetEqual(set(actual.keys()), {'evaporation_rate'})
    self.assertEqual(
        cx.get_coordinate(actual['evaporation_rate']), self.coords.horizontal
    )

    query_b = {
        'turbulence_index': self.coords,
        'evaporation_rate': self.coords.horizontal,
    }
    actual = operator.observe(inputs=self.inputs, query=query_b)
    self.assertSetEqual(
        set(actual.keys()), {'turbulence_index', 'evaporation_rate'}
    )
    self.assertEqual(
        cx.get_coordinate(actual['evaporation_rate']), self.coords.horizontal
    )
    self.assertEqual(cx.get_coordinate(actual['turbulence_index']), self.coords)

  def test_raises_on_missing_field(self):
    operator = observation_operators.FixedLearnedObservationOperator(
        self.observation_mapping
    )
    query = {'X': self.coords.horizontal}
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "Query contains k='X' not in ['evaporation_rate', 'turbulence_index']",
    ):
      operator.observe(inputs=self.inputs, query=query)

  def test_raises_on_non_matching_coordinate(self):
    operator = observation_operators.FixedLearnedObservationOperator(
        self.observation_mapping
    )
    query_coord = coordinates.LonLatGrid.TL31()
    query = {'evaporation_rate': query_coord}
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        f'Query (evaporation_rate, {query_coord}) is not compatible with'
        f' coord={self.coords}',
    ):
      _ = operator.observe(inputs=self.inputs, query=query)


class LearnedSparseScalarObservationFromNeighborsTest(parameterized.TestCase):
  """Tests FixedLearnedObservationOperator implementation."""

  def setUp(self):
    super().setUp()
    self.grid = coordinates.LonLatGrid.T21()
    feature_module = pytree_transforms.LatitudeFeatures(self.grid)
    layer_factory = functools.partial(
        standard_layers.MlpUniform, hidden_size=6, n_hidden_layers=2
    )
    scalar_names = ('temperature', 'wind_speed')
    self.operator = (
        observation_operators.LearnedSparseScalarObservationFromNeighbors(
            scalar_names=scalar_names,
            features_module=feature_module,
            grid=self.grid,
            input_state_shapes={},  # not used by specific feature_module.
            layer_factory=layer_factory,
            rngs=nnx.Rngs(0),
        )
    )

  def test_output_structure(self):
    sparse_coord = cx.LabeledAxis('id', np.arange(7))
    np.random.seed(0)
    longitudes = cx.wrap(np.random.uniform(0, 360, 7), sparse_coord)
    latitudes = cx.wrap(np.random.uniform(-90, 90, 7), sparse_coord)
    with self.subTest('full_query'):
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

    with self.subTest('temperature_only_query'):
      temperature_query = {
          'longitude': longitudes,
          'latitude': latitudes,
          'temperature': sparse_coord,
      }
      actual = self.operator.observe({}, temperature_query)
      self.assertSetEqual(
          set(actual.keys()),
          {'longitude', 'latitude', 'temperature'},
      )
      np.testing.assert_array_equal(actual['longitude'].data, longitudes.data)
      np.testing.assert_array_equal(actual['latitude'].data, latitudes.data)
      self.assertEqual(cx.get_coordinate(actual['temperature']), sparse_coord)


if __name__ == '__main__':
  config.update('jax_traceback_filtering', 'off')
  absltest.main()
