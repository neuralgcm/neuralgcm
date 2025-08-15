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
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import spherical_transforms
from neuralgcm.experimental.core import transforms
from neuralgcm.experimental.metrics import aggregation
from neuralgcm.experimental.metrics import base
from neuralgcm.experimental.metrics import deterministic_losses
from neuralgcm.experimental.metrics import deterministic_metrics
from neuralgcm.experimental.metrics import evaluators
from neuralgcm.experimental.metrics import probabilistic_losses
from neuralgcm.experimental.metrics import weighting
import numpy as np


class EvaluatorsTest(parameterized.TestCase):

  def test_evaluator_mse(self):
    dim = cx.SizedAxis('spatial', 2)
    predictions = {
        'x': cx.wrap(np.array([2.0, 3.0]), dim),
        'y': cx.wrap(np.array([2.0, 2.0]), dim),
    }
    targets = {
        'x': cx.wrap(np.array([1.0, 1.0]), dim),
        'y': cx.wrap(np.array([4.0, 4.0]), dim),
    }
    mse = deterministic_metrics.MSE()
    evaluator = evaluators.Evaluator(
        metrics={'eval_metric': mse},
        getters=transforms.Identity(),
        aggregators=aggregation.Aggregator(
            dims_to_reduce=['spatial'], weight_by=[]
        ),
    )
    self.assertFalse(evaluator.is_loss_evaluator)  # Pass metrics MSE, not loss.
    agg_states = evaluator.evaluate(predictions, targets)
    self.assertEqual(list(agg_states.keys()), ['eval_metric'])
    aggregation_state = agg_states['eval_metric']
    expected_mse_stats = ['SquaredError']
    self.assertEqual(
        list(aggregation_state.sum_weighted_statistics.keys()),
        expected_mse_stats,
    )
    expected_mse_stats_components = ['x', 'y']
    self.assertEqual(
        list(aggregation_state.sum_weighted_statistics['SquaredError'].keys()),
        expected_mse_stats_components,
    )

  def test_evaluator_crps(self):
    ens_dim = cx.SizedAxis('ensemble', 2)
    spatial_dim = cx.SizedAxis('spatial', 2)
    predictions = {
        'z': cx.wrap(np.array([[1.0, 3.0], [2.0, 4.0]]), ens_dim, spatial_dim)
    }
    targets = {'z': cx.wrap(np.array([0.0, 5.0]), spatial_dim)}

    crps = probabilistic_losses.CRPS(ensemble_dim='ensemble')
    evaluator = evaluators.Evaluator(
        metrics={'loss_metric': crps},
        getters={'loss_metric': transforms.Identity()},
        aggregators=aggregation.Aggregator(
            dims_to_reduce=['spatial'], weight_by=[]
        ),
    )
    agg_states = evaluator.evaluate(predictions, targets)
    self.assertEqual(list(agg_states.keys()), ['loss_metric'])
    aggregation_state = agg_states['loss_metric']
    expected_crps_stats = [
        'EnergySkill_ensemble_beta_1.0',
        'EnergySpread_ensemble_beta_1.0',
    ]
    self.assertEqual(
        list(aggregation_state.sum_weighted_statistics.keys()),
        expected_crps_stats,
    )
    self.assertTrue(evaluator.is_loss_evaluator)
    with self.subTest('total_loss'):
      total_loss = evaluator.evaluate_total(predictions, targets)
      # skill per location = [1.5, 1.5]
      # spread per location = [1.0, 1.0]
      # crps per location = skill - 0.5 * spread = [1.0, 1.0]
      # aggregated crps = mean([1.0, 1.0]) = 1.0
      # total loss = 1.0
      np.testing.assert_almost_equal(total_loss.data, 1.0)

  def test_evaluator_sum_loss(self):
    dim = cx.SizedAxis('spatial', 2)
    predictions = {'x': cx.wrap(np.array([2.0, 3.0]), dim)}
    targets = {'x': cx.wrap(np.array([1.0, 1.0]), dim)}

    loss = base.SumLoss(
        terms={
            'mse': deterministic_losses.MSE(),
            'mae': deterministic_losses.MAE(),
        },
        term_weights={'mse': 0.3, 'mae': 0.7},
    )
    evaluator = evaluators.Evaluator(
        metrics={'mse_plus_mae': loss},
        getters=transforms.Identity(),
        aggregators=aggregation.Aggregator(
            dims_to_reduce=['spatial'], weight_by=[]
        ),
    )
    self.assertTrue(evaluator.is_loss_evaluator)
    total_loss = evaluator.evaluate_total(predictions, targets)
    # mse = ((2-1)**2 + (3-1)**2) / 2 = 2.5
    # mae = (abs(2-1) + abs(3-1)) / 2 = 1.5
    # total = 0.3 * 2.5 + 0.7 * 1.5 = 0.75 + 1.05 = 1.8
    np.testing.assert_almost_equal(total_loss.data, 1.8)

  def test_evaluator_with_skipna(self):
    dim = cx.SizedAxis('spatial', 3)
    predictions = {'x': cx.wrap(np.array([1.0, 2.0, 3.0]), dim)}
    targets = {'x': cx.wrap(np.array([1.0, 5.0, np.nan]), dim)}

    loss = deterministic_losses.MAE()
    with self.subTest('keep_weights_false'):
      evaluator = evaluators.Evaluator(
          metrics={'mae': loss},
          getters=transforms.Identity(),
          aggregators=aggregation.Aggregator(
              dims_to_reduce=['spatial'],
              weight_by=[],
              skipna=True,
              keep_weights_for_nans=False,
          ),
      )
      total_loss = evaluator.evaluate_total(predictions, targets)
      # MAE should only be computed for the first two entries, ignoring the NaN.
      # mae = (abs(1 - 1) + abs(2 - 5)) / 2 = 1.5
      np.testing.assert_almost_equal(total_loss.data, 1.5)

    with self.subTest('keep_weights_true'):
      evaluator = evaluators.Evaluator(
          metrics={'mae': loss},
          getters=transforms.Identity(),
          aggregators=aggregation.Aggregator(
              dims_to_reduce=['spatial'],
              weight_by=[],
              skipna=True,
              keep_weights_for_nans=True,
          ),
      )
      total_loss = evaluator.evaluate_total(predictions, targets)
      # When keep_weights_for_nans=True, weights are not adjusted, so
      # mae = (abs(1 - 1) + abs(2 - 5)) / 3 = 1.0
      np.testing.assert_almost_equal(total_loss.data, 1.0)

  def test_evaluator_multiple_terms_with_weighting(self):
    ens = cx.SizedAxis('ensemble', 2)
    grid = coordinates.LonLatGrid.T21()
    pressure = coordinates.PressureLevels.with_13_era5_levels()
    ones_like = lambda c: cx.wrap(np.ones(c.shape), c)
    zeros_like = lambda c: cx.wrap(np.zeros(c.shape), c)
    predictions = {
        'x': ones_like(cx.compose_coordinates(ens, pressure, grid)),
        'y': ones_like(cx.compose_coordinates(ens, grid)),
    }
    targets = {
        'x': zeros_like(cx.compose_coordinates(pressure, grid)),
        'y': zeros_like(cx.compose_coordinates(grid)),
    }

    #
    ylm_grid = coordinates.SphericalHarmonicGrid.T21()
    ylm_transform = spherical_transforms.FixedYlmMapping(
        lon_lat_grid=grid,
        ylm_grid=ylm_grid,
        mesh=parallelism.Mesh(),
        partition_schema_key=None,
    )
    nodal_getter = transforms.Identity()
    modal_getter = transforms.Sequential([
        transforms.ToModal(ylm_transform),
        transforms.ClipWavenumbers(grid=ylm_grid, wavenumbers_to_clip=2),
    ])
    area_weighting = weighting.GridAreaWeighting()
    variable_weighting = weighting.PerVariableWeighting(
        variable_weights={'x': 1.0, 'y': 1.0}
    )
    nodal_aggregator = aggregation.Aggregator(
        dims_to_reduce=('pressure', 'longitude', 'latitude'),
        weight_by=[variable_weighting, area_weighting],
    )
    modal_aggregator = aggregation.Aggregator(
        dims_to_reduce=('pressure', 'longitude_wavenumber', 'total_wavenumber'),
        weight_by=[variable_weighting, area_weighting],
    )
    nodal_crps = probabilistic_losses.CRPS()
    modal_crps = probabilistic_losses.CRPS()
    evaluator = evaluators.Evaluator(
        metrics={'nodal': nodal_crps, 'modal': modal_crps},
        getters={'nodal': nodal_getter, 'modal': modal_getter},
        aggregators={'nodal': nodal_aggregator, 'modal': modal_aggregator},
    )
    self.assertTrue(evaluator.is_loss_evaluator)
    total_loss = evaluator.evaluate_total(predictions, targets)
    self.assertEqual(total_loss.ndim, 0)


if __name__ == '__main__':
  jax.config.update('jax_traceback_filtering', 'off')
  absltest.main()
