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
import coordax as cx
import jax
from neuralgcm.experimental.metrics import base
from neuralgcm.experimental.metrics import deterministic_losses
from neuralgcm.experimental.metrics import deterministic_metrics
import numpy as np


class DeterministicMetricsTest(parameterized.TestCase):

  def test_mse(self):
    x = {'x': cx.field(2.0), 'y': cx.field(2.0)}
    y = {'x': cx.field(1.0), 'y': cx.field(4.0)}
    mse = deterministic_metrics.MSE()
    mse_statistics = {
        s.unique_name: s.compute(x, y) for s in mse.statistics.values()
    }
    mse_values = mse.values_from_mean_statistics(mse_statistics)
    np.testing.assert_almost_equal(mse_values['x'].data, (2.0 - 1.0) ** 2)
    np.testing.assert_almost_equal(mse_values['y'].data, (2.0 - 4.0) ** 2)

  def test_mse_loss_total_weighted(self):
    loss = deterministic_losses.MSE(variable_weights={'x': 2.0, 'y': 0.5})
    x = {'x': cx.field(2.0), 'y': cx.field(2.0)}
    y = {'x': cx.field(1.0), 'y': cx.field(4.0)}
    mse_statistics = {
        s.unique_name: s.compute(x, y) for s in loss.statistics.values()
    }
    mse_values = loss.values_from_mean_statistics(mse_statistics)
    with self.subTest('total_loss'):
      total_loss = loss.total(mse_values)
      np.testing.assert_almost_equal(total_loss.data, 1.0 * 2.0 + 4.0 * 0.5)

    with self.subTest('debug_terms'):
      debug_terms = loss.debug_terms(mse_statistics, mse_values)
      np.testing.assert_almost_equal(debug_terms['relative_x'].data, 0.5)
      np.testing.assert_almost_equal(debug_terms['relative_y'].data, 0.5)

  def test_mae(self):
    x = {'x': cx.field(2.0), 'y': cx.field(2.0)}
    y = {'x': cx.field(1.0), 'y': cx.field(4.0)}
    mae = deterministic_metrics.MAE()
    mae_statistics = {
        s.unique_name: s.compute(x, y) for s in mae.statistics.values()
    }
    mae_values = mae.values_from_mean_statistics(mae_statistics)
    np.testing.assert_almost_equal(mae_values['x'].data, np.abs(2.0 - 1.0))
    np.testing.assert_almost_equal(mae_values['y'].data, np.abs(2.0 - 4.0))

  def test_mae_loss_total_weighted(self):
    mae_loss = deterministic_losses.MAE(variable_weights={'x': 0.4, 'y': 1.1})
    x = {'x': cx.field(2.0), 'y': cx.field(2.0)}
    y = {'x': cx.field(1.0), 'y': cx.field(4.0)}
    mae_statistics = {
        s.unique_name: s.compute(x, y) for s in mae_loss.statistics.values()
    }
    mae_values = mae_loss.values_from_mean_statistics(mae_statistics)
    with self.subTest('total_loss'):
      total_loss = mae_loss.total(mae_values)
      np.testing.assert_almost_equal(total_loss.data, 1.0 * 0.4 + 2.0 * 1.1, 5)

    with self.subTest('debug_terms'):
      debug_terms = mae_loss.debug_terms(mae_statistics, mae_values)
      np.testing.assert_almost_equal(debug_terms['relative_x'].data, 0.4 / 2.6)
      np.testing.assert_almost_equal(debug_terms['relative_y'].data, 2.2 / 2.6)

  def test_wind_vector_rmse(self):
    u_name = 'u'
    v_name = 'v'
    vector_name = 'test_wind'
    x = {'u': cx.field(2.0), 'v': cx.field(1.0)}
    y = {'u': cx.field(1.0), 'v': cx.field(3.0)}
    metric = deterministic_metrics.WindVectorRMSE(
        u_name=u_name, v_name=v_name, vector_name=vector_name
    )
    statistics = {
        s.unique_name: s.compute(x, y) for s in metric.statistics.values()
    }
    metric_values = metric.values_from_mean_statistics(statistics)
    expected_se = (2.0 - 1.0) ** 2 + (1.0 - 3.0) ** 2
    np.testing.assert_almost_equal(
        metric_values['test_wind'].data, np.sqrt(expected_se)
    )

  def test_sum_loss(self):
    mae_var_weights = {'x': 0.4, 'y': 1.1}
    loss = base.SumLoss(
        terms={
            'mse': deterministic_losses.MSE(),
            'mae': deterministic_losses.MAE(variable_weights=mae_var_weights),
        },
        term_weights={'mse': 0.3, 'mae': 0.7},
    )
    x = {'x': cx.field(2.0), 'y': cx.field(2.0)}
    y = {'x': cx.field(1.0), 'y': cx.field(4.0)}
    statistics = {
        s.unique_name: s.compute(x, y) for s in loss.statistics.values()
    }
    metric_values = loss.values_from_mean_statistics(statistics)
    expected_mse = (1.0 - 2.0) ** 2 + (4.0 - 2.0) ** 2
    expected_mae = 0.4 * np.abs(1.0 - 2.0) + 1.1 * np.abs(4.0 - 2.0)
    expected_total = 0.3 * expected_mse + 0.7 * expected_mae
    with self.subTest('total_loss'):
      total_loss = loss.total(metric_values)
      np.testing.assert_almost_equal(total_loss.data, expected_total, 5)

    with self.subTest('debug_terms'):
      debug_terms = loss.debug_terms(statistics, metric_values)
      np.testing.assert_almost_equal(
          debug_terms['relative_mse_total'].data,
          0.3 * expected_mse / expected_total,
      )
      np.testing.assert_almost_equal(
          debug_terms['relative_mae_total'].data,
          0.7 * expected_mae / expected_total,
      )


if __name__ == '__main__':
  jax.config.update('jax_traceback_filtering', 'off')
  absltest.main()
