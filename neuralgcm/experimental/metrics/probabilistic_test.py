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
from neuralgcm.experimental.metrics import probabilistic_losses
from neuralgcm.experimental.metrics import probabilistic_metrics
import numpy as np


class ProbabilisticMetricsTest(parameterized.TestCase):

  def test_crps_metric(self):
    e, d = cx.SizedAxis('ensemble', 2), cx.SizedAxis('d', 4)
    rng = np.random.default_rng(seed=0)
    predictions = {'x': cx.field(rng.random(e.shape + d.shape), e, d)}
    targets = {'x': cx.field(rng.random(d.shape), d)}
    crps_metric = probabilistic_metrics.CRPS()
    crps_statistics = base.compute_unique_statistics_for_all_metrics(
        {'crps': crps_metric}, predictions, targets
    )
    crps_values = crps_metric.values_from_mean_statistics(crps_statistics)
    skill = crps_statistics['EnergySkill_ensemble_beta_1.0']['x']
    spread = crps_statistics['EnergySpread_ensemble_beta_1.0']['x']
    expected_crps = skill - 0.5 * spread
    cx.testing.assert_fields_allclose(crps_values['x'], expected_crps)
    loss = probabilistic_losses.CRPS()
    cx.testing.assert_fields_allclose(loss.total(crps_values), expected_crps)

    with self.subTest('debug_terms'):
      debug_terms = loss.debug_terms(crps_statistics, crps_values)
      cx.testing.assert_fields_allclose(
          debug_terms['spread_skill_ratio_x'],
          spread / skill,
      )
      cx.testing.assert_fields_allclose(
          debug_terms['relative_crps_x'],
          cx.field(np.full(d.shape, 1.0), d),
      )

  def test_energy_spread_broadcast_target_nans_behavior(self):
    e, d = cx.SizedAxis('ensemble', 2), cx.SizedAxis('d', 4)
    predictions = {'x': cx.field(np.ones(e.shape + d.shape), e, d)}
    targets = {'x': cx.field(np.array([np.nan, 1.0, 2.0, 3.0]), d)}

    with self.subTest('no_broadcast_target'):
      spread = probabilistic_metrics.EnergySpread(broadcast_target_nans=False)
      spread_results = spread.compute(predictions, targets)
      self.assertEqual(spread_results['x'].data[0], 0.0)  # no nans.

    with self.subTest('broadcast_target'):
      spread = probabilistic_metrics.EnergySpread(broadcast_target_nans=True)
      spread_results = spread.compute(predictions, targets)
      self.assertTrue(np.isnan(spread_results['x'].data[0]))  # broadcasted.
      self.assertEqual(spread_results['x'].data[1], 0.0)

  def test_crps_large_ensemble(self):
    e, d = cx.SizedAxis('ensemble', 4), cx.SizedAxis('d', 2)
    predictions = {
        'x': cx.field(np.arange(1, 9).reshape((4, 2)).astype(float), e, d)
    }
    targets = {'x': cx.field(np.array([4.0, 5.0]), d)}

    metric = probabilistic_metrics.CRPS()
    stats = base.compute_unique_statistics_for_all_metrics(
        {'crps': metric}, predictions, targets
    )
    values = metric.values_from_mean_statistics(stats)

    expected_spread = cx.field(np.full(d.shape, 20.0 / 6.0), d)
    cx.testing.assert_fields_allclose(
        stats['EnergySpread_ensemble_beta_1.0']['x'], expected_spread
    )

    expected_skill = cx.field(np.full(d.shape, 2.0), d)
    cx.testing.assert_fields_allclose(
        stats['EnergySkill_ensemble_beta_1.0']['x'], expected_skill
    )

    expected_crps = expected_skill - 0.5 * expected_spread
    cx.testing.assert_fields_allclose(values['x'], expected_crps, atol=1e-6)

  def test_energy_spread_algorithms(self):
    stat_flip = probabilistic_metrics.EnergySpread(algorithm='flip')
    stat_sort = probabilistic_metrics.EnergySpread(algorithm='sort')
    stat_all_pairs = probabilistic_metrics.EnergySpread(algorithm='all_pairs')
    stat_ring_chain = probabilistic_metrics.EnergySpread(algorithm='ring_chain')
    stat_auto = probabilistic_metrics.EnergySpread(algorithm='auto')

    rng = np.random.default_rng(seed=0)
    d = cx.SizedAxis('d', 4)
    targets = {'x': cx.field(rng.random(d.shape), d)}

    with self.subTest('n=2'):
      e_size_2 = cx.SizedAxis('ensemble', 2)
      ens_2_predictions = {
          'x': cx.field(rng.random(e_size_2.shape + d.shape), e_size_2, d)
      }

      spread_flip = stat_flip.compute(ens_2_predictions, targets)['x']
      spread_sort = stat_sort.compute(ens_2_predictions, targets)['x']
      spread_all_pairs = stat_all_pairs.compute(ens_2_predictions, targets)['x']
      spread_ring_chain = stat_ring_chain.compute(ens_2_predictions, targets)[
          'x'
      ]
      spread_auto = stat_auto.compute(ens_2_predictions, targets)['x']

      cx.testing.assert_fields_allclose(spread_flip, spread_sort, atol=1e-6)
      cx.testing.assert_fields_allclose(
          spread_flip, spread_all_pairs, atol=1e-6
      )
      cx.testing.assert_fields_allclose(
          spread_flip, spread_ring_chain, atol=1e-6
      )
      cx.testing.assert_fields_allclose(spread_flip, spread_auto, atol=1e-6)

    with self.subTest('n=4'):
      e_size_4 = cx.SizedAxis('ensemble', 4)
      ens_4_predictions = {
          'x': cx.field(rng.random(e_size_4.shape + d.shape), e_size_4, d)
      }

      spread_sort4 = stat_sort.compute(ens_4_predictions, targets)['x']
      spread_pairs4 = stat_all_pairs.compute(ens_4_predictions, targets)['x']
      spread_chain4 = stat_ring_chain.compute(ens_4_predictions, targets)['x']
      spread_auto4 = stat_auto.compute(ens_4_predictions, targets)['x']

      cx.testing.assert_fields_allclose(spread_sort4, spread_pairs4, atol=1e-6)
      cx.testing.assert_fields_allclose(spread_sort4, spread_chain4, atol=1e-6)
      cx.testing.assert_fields_allclose(spread_sort4, spread_auto4, atol=1e-6)

  @parameterized.parameters(
      dict(n=2),
      dict(n=4),
      dict(n=10),
  )
  def test_unbiased_mse_and_ssr(self, n):
    e, d = cx.SizedAxis('ensemble', n), cx.SizedAxis('d', 1)
    rng = np.random.default_rng(seed=0)
    # Sample predictions and targets.
    pred_data = rng.standard_normal((n, 1))
    target_data = rng.standard_normal((1,))
    predictions = {'x': cx.field(pred_data, e, d)}
    targets = {'x': cx.field(target_data, d)}

    # Explicit calculation.
    pred_mean = np.mean(pred_data)
    pred_var = np.var(pred_data, ddof=1)
    biased_mse = (pred_mean - target_data) ** 2
    expected_unbiased_mse = cx.field(
        np.full(d.shape, biased_mse - pred_var / n), d
    )
    expected_ssr = cx.field(
        np.full(d.shape, np.sqrt(pred_var / expected_unbiased_mse.data)), d
    )

    mse_stat = probabilistic_metrics.UnbiasedEnsembleMeanSquaredError()
    mse_val = mse_stat.compute(predictions, targets)['x']
    cx.testing.assert_fields_allclose(mse_val, expected_unbiased_mse, atol=1e-6)

    ssr_metric = probabilistic_metrics.UnbiasedSpreadSkillRatio()
    ssr_stats = base.compute_unique_statistics_for_all_metrics(
        {'ssr': ssr_metric}, predictions, targets
    )
    ssr_val = ssr_metric.values_from_mean_statistics(ssr_stats)['x']
    cx.testing.assert_fields_allclose(ssr_val, expected_ssr, atol=1e-6)


if __name__ == '__main__':
  jax.config.update('jax_traceback_filtering', 'off')
  absltest.main()
