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
from neuralgcm.experimental.metrics import probabilistic_losses
from neuralgcm.experimental.metrics import probabilistic_metrics
import numpy as np


class ProbabilisticMetricsTest(parameterized.TestCase):

  def test_crps_metric(self):
    e, d = cx.SizedAxis('ensemble', 2), cx.SizedAxis('d', 4)
    rng = np.random.default_rng(seed=0)
    predictions = {'x': cx.field(rng.random(e.shape + d.shape), e, d)}
    targets = {'x': cx.field(rng.random(d.shape), d)}
    crps_metric = probabilistic_metrics.CRPS(ensemble_dim='ensemble')
    crps_statistics = {
        s.unique_name: s.compute(predictions, targets)
        for s in crps_metric.statistics.values()
    }
    crps_values = crps_metric.values_from_mean_statistics(crps_statistics)
    skill = crps_statistics['EnergySkill_ensemble_beta_1.0']['x'].data
    spread = crps_statistics['EnergySpread_ensemble_beta_1.0']['x'].data
    expected_crps = skill - 0.5 * spread
    np.testing.assert_allclose(crps_values['x'].data, expected_crps)
    loss = probabilistic_losses.CRPS(ensemble_dim='ensemble')
    np.testing.assert_allclose(loss.total(crps_values).data, expected_crps)

    with self.subTest('debug_terms'):
      debug_terms = loss.debug_terms(crps_statistics, crps_values)
      np.testing.assert_allclose(
          debug_terms['relative_skill_to_total'].data,
          skill / expected_crps,
      )
      np.testing.assert_allclose(
          debug_terms['relative_spread_to_total'].data,
          0.5 * spread / expected_crps,
      )
      np.testing.assert_allclose(
          debug_terms['relative_crps_x'].data,
          1.0,
      )


if __name__ == '__main__':
  jax.config.update('jax_traceback_filtering', 'off')
  absltest.main()
