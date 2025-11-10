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
from neuralgcm.experimental.metrics import aggregation
from neuralgcm.experimental.metrics import base
from neuralgcm.experimental.metrics import deterministic_losses
from neuralgcm.experimental.metrics import deterministic_metrics
from neuralgcm.experimental.metrics import evaluators
import numpy as np


class AggregatorTest(parameterized.TestCase):

  def test_zeros_aggregation_state_constructor(self):
    dim = cx.SizedAxis('spatial', 2)
    predictions = {'x': cx.wrap(np.array([2.0, 3.0]), dim)}
    targets = {'x': cx.wrap(np.array([1.0, 1.0]), dim)}
    mse = deterministic_metrics.MSE()
    rmse = deterministic_metrics.RMSE()
    aggregator = aggregation.Aggregator(
        dims_to_reduce=['spatial'], weight_by=[]
    )
    evaluator = evaluators.Evaluator(
        metrics={'mse': mse, 'rmse': rmse},
        aggregators=aggregator,
    )
    agg_states = evaluator.evaluate(predictions, targets)
    zero_state_mse = aggregator.zeros_aggregation_state(
        mse, predictions, targets
    )
    zero_state_rmse = aggregator.zeros_aggregation_state(
        rmse, predictions, targets
    )

    chex.assert_trees_all_equal_structs(zero_state_mse, agg_states['mse'])
    chex.assert_trees_all_equal_structs(zero_state_rmse, agg_states['rmse'])


class AggregationUtilsTest(parameterized.TestCase):
  """Tests for aggregation helper functions."""

  def test_split_aggregation_state_for_metrics(self):
    # Generate AggregationState with unique statistics for metrics below.
    stat_a = deterministic_metrics.SquaredError()
    stat_b = deterministic_metrics.AbsoluteError()
    dummy_field = cx.wrap(np.array([1.0]))
    agg_state = aggregation.AggregationState(
        sum_weighted_statistics={
            stat_a.unique_name: {'var_name': dummy_field},
            stat_b.unique_name: {'var_name': dummy_field},
        },
        sum_weights={
            stat_a.unique_name: {'var_name': dummy_field},
            stat_b.unique_name: {'var_name': dummy_field},
        },
    )
    metric1 = deterministic_losses.MSE()
    metric2 = deterministic_losses.MAE()
    metric3 = base.SumLoss(terms={'mse': metric1, 'mae': metric2})
    metrics = {'m1': metric1, 'm2': metric2, 'm3': metric3}

    states = aggregation.split_aggregation_state_for_metrics(metrics, agg_state)

    # Assertions
    m1_struct = {stat_a.unique_name: {'var_name': dummy_field}}
    m2_struct = {stat_b.unique_name: {'var_name': dummy_field}}
    m3_struct = {
        stat_a.unique_name: {'var_name': dummy_field},
        stat_b.unique_name: {'var_name': dummy_field},
    }
    chex.assert_trees_all_equal_structs(states['m1'].sum_weights, m1_struct)
    chex.assert_trees_all_equal_structs(states['m2'].sum_weights, m2_struct)
    chex.assert_trees_all_equal_structs(states['m3'].sum_weights, m3_struct)

  def test_split_aggregation_state_missing_stats_raises(self):
    stat_a = deterministic_metrics.SquaredError()
    dummy_field = cx.wrap(np.array([1.0]))
    agg_state = aggregation.AggregationState(
        sum_weighted_statistics={stat_a.unique_name: {'var_name': dummy_field}},
        sum_weights={stat_a.unique_name: {'var_name': dummy_field}},
    )
    metric = base.SumLoss(
        terms={
            'mse': deterministic_losses.MSE(),
            'mae': deterministic_losses.MAE(),
        }
    )
    with self.assertRaisesRegex(ValueError, 'requires statistics'):
      aggregation.split_aggregation_state_for_metrics({'m1': metric}, agg_state)


if __name__ == '__main__':
  jax.config.update('jax_traceback_filtering', 'off')
  absltest.main()
