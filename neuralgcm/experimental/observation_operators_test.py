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

from absl.testing import absltest
from absl.testing import parameterized
import chex
from jax import config  # pylint: disable=g-importing-member
from neuralgcm.experimental import coordax as cx
from neuralgcm.experimental import observation_operators
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
    expected = {k: fields[k] for k in query.keys()}
    chex.assert_trees_all_equal(actual, expected)

  def test_raises_on_missing_field(self):
    fields = {'a': cx.wrap(np.ones(7), cx.LabeledAxis('x', np.arange(7)))}
    operator = observation_operators.DataObservationOperator(fields)
    query = {'d': cx.LabeledAxis('x', np.arange(7))}
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "Query contains k='d' not in ['a']",
    ):
      _ = operator.observe(inputs={}, query=query)

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


if __name__ == '__main__':
  config.update('jax_traceback_filtering', 'off')
  absltest.main()
