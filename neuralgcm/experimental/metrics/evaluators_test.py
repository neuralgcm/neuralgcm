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

import operator

from absl.testing import absltest
from absl.testing import parameterized
import chex
import coordax as cx
from flax import nnx
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
from neuralgcm.experimental.metrics import scaling
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
    self.assertSameElements(
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

  def test_evaluator_with_scaling(self):
    dim = cx.SizedAxis('spatial', 2)
    predictions = {'x': cx.wrap(np.array([2.0, 3.0]), dim)}
    targets = {'x': cx.wrap(np.array([1.0, 1.0]), dim)}

    loss = deterministic_losses.MAE()
    scaler = scaling.ConstantScaler(cx.wrap(np.array([2.0, 1.0]), dim))
    evaluator = evaluators.Evaluator(
        metrics={'loss': loss},
        aggregators=aggregation.Aggregator(
            dims_to_reduce=['spatial'], weight_by=[], scale_by=[scaler]
        ),
    )
    self.assertTrue(evaluator.is_loss_evaluator)
    total_loss = evaluator.evaluate_total(predictions, targets)
    # mae = (2 * abs(2-1) + 1 * abs(3-1)) / 2 = 2.0
    np.testing.assert_almost_equal(total_loss.data, 2.0)

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
        transforms.ClipWavenumbers({ylm_grid: 2}),
    ])
    area_weighting = weighting.GridAreaWeighting()
    variable_weighting = weighting.PerVariableWeighting.from_constants(
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

  def test_evaluator_shared_statistics(self):
    dim = cx.SizedAxis('spatial', 2)
    predictions = {'x': cx.wrap(np.array([2.0, 3.0]), dim)}
    targets = {'x': cx.wrap(np.array([1.0, 1.0]), dim)}
    mse = deterministic_metrics.MSE()
    rmse = deterministic_metrics.RMSE()
    evaluator = evaluators.Evaluator(
        metrics={'mse': mse, 'rmse': rmse},
        aggregators=aggregation.Aggregator(
            dims_to_reduce=['spatial'], weight_by=[]
        ),
    )
    agg_states = evaluator.evaluate(predictions, targets)
    self.assertCountEqual(agg_states.keys(), ['mse', 'rmse'])
    # check values
    mse_values = agg_states['mse'].metric_values(mse)
    rmse_values = agg_states['rmse'].metric_values(rmse)
    # mse = ((2-1)**2 + (3-1)**2) / 2 = (1 + 4) / 2 = 2.5
    np.testing.assert_almost_equal(mse_values['x'].data, 2.5)
    np.testing.assert_almost_equal(rmse_values['x'].data, np.sqrt(2.5))

  def test_evaluation_through_scan_gives_same_results_as_default(self):
    length, n_spatial = 6, 10
    key_p, key_t = jax.random.split(jax.random.key(42))
    dt = coordinates.TimeDelta(np.arange(length) * np.timedelta64(1, 'h'))
    x = cx.SizedAxis('x', n_spatial)
    coord = cx.compose_coordinates(dt, x)
    predictions = {'u': cx.wrap(jax.random.uniform(key_p, coord.shape), coord)}
    targets = {'u': cx.wrap(jax.random.uniform(key_t, coord.shape), coord)}

    rmse = deterministic_metrics.RMSE()
    three_hour_mask_coord = coordinates.TimeDelta(np.timedelta64(3, 'h')[None])

    agg_total = aggregation.Aggregator(  # full RMSE.
        dims_to_reduce=('timedelta', 'x'), weight_by=[]
    )
    agg_3hr = aggregation.Aggregator(  # RMSE at dt == 3hr.
        dims_to_reduce=('timedelta', 'x'),
        weight_by=[weighting.CoordinateMaskWeighting(three_hour_mask_coord)],
    )
    evaluator = evaluators.Evaluator(
        metrics={'rmse_total': rmse, 'rmse_3hr': rmse},
        aggregators={'rmse_total': agg_total, 'rmse_3hr': agg_3hr},
    )
    # Note: 'timedelta' is included in `dims_to_reduce` to check that weighting
    # from context is applied correctly.

    # Single pass evaluation.
    agg_states = evaluator.evaluate(predictions, targets)
    metric_values = {
        k: agg_states[k].metric_values(rmse)['u'].data
        for k in ['rmse_total', 'rmse_3hr']
    }

    # Through scan from context evaluation.
    data_struct = {'u': cx.shape_struct_field(x)}  # same targets/predictions.
    init_agg_states = {
        'rmse_total': aggregation.AggregationState.zeros_for_metric(
            rmse, agg_total, data_struct, data_struct
        ),
        'rmse_3hr': aggregation.AggregationState.zeros_for_metric(
            rmse, agg_3hr, data_struct, data_struct
        ),
    }

    def scan_body(aggregation_carry, prediction_i, target_i, evaluator_slice):
      agg_state = evaluator_slice.evaluate(prediction_i, target_i)
      new_carry = jax.tree.map(
          operator.add,
          aggregation_carry,
          agg_state,
          is_leaf=lambda x: isinstance(x, aggregation.AggregationState),
      )
      return new_carry

    evaluate_in_scan_fn = nnx.scan(
        scan_body,
        length=length,
        in_axes=(nnx.Carry, 0, 0, 0),
        out_axes=nnx.Carry,
    )
    evaluator_with_dt_context = evaluator.with_context(
        {'timedelta': dt.fields['timedelta'].untag(dt)}
    )
    scanned_agg_states = evaluate_in_scan_fn(
        init_agg_states,
        cx.untag(predictions, dt),  # need to untag dt to be able to scan.
        cx.untag(targets, dt),
        evaluator_with_dt_context,
    )
    scanned_metric_values = {
        k: scanned_agg_states[k].metric_values(rmse)['u'].data
        for k in ['rmse_total', 'rmse_3hr']
    }
    chex.assert_trees_all_close(metric_values, scanned_metric_values)


class NestedAndFlattenedEvaluatorsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    dim = cx.SizedAxis('spatial', 2)
    self.predictions = {
        'atmosphere': {'t': cx.wrap(np.array([2.0, 3.0]), dim)},
        'ocean': {'sst': cx.wrap(np.array([10.0, 14.0]), dim)},
    }
    self.targets = {
        'atmosphere': {'t': cx.wrap(np.array([1.0, 1.0]), dim)},
        'ocean': {'sst': cx.wrap(np.array([12.0, 12.0]), dim)},
    }
    self.aggregator = aggregation.Aggregator(
        dims_to_reduce=['spatial'], weight_by=[]
    )

  def test_nested_evaluators_loss(self):
    eval_atmo = evaluators.Evaluator(
        metrics={'mse_t': deterministic_losses.MSE()},
        aggregators=self.aggregator,
    )
    eval_ocean = evaluators.Evaluator(
        metrics={'mae_sst': deterministic_losses.MAE()},
        aggregators=self.aggregator,
    )
    nested = evaluators.NestedEvaluators(
        evaluators={'atmosphere': eval_atmo, 'ocean': eval_ocean},
        evaluator_weights={'atmosphere': 0.4, 'ocean': 0.6},
    )
    self.assertTrue(nested.is_loss_evaluator)
    agg_states = nested.evaluate(self.predictions, self.targets)
    # check structure
    self.assertCountEqual(agg_states.keys(), ['atmosphere', 'ocean'])
    self.assertCountEqual(agg_states['atmosphere'].keys(), ['mse_t'])
    self.assertCountEqual(agg_states['ocean'].keys(), ['mae_sst'])
    # check total loss value
    total_loss = nested.evaluate_total(self.predictions, self.targets)
    # atmosphere mse_t = ((2-1)**2 + (3-1)**2) / 2 = 2.5
    # ocean mae_sst = (abs(10-12) + abs(14-12)) / 2 = 2.0
    # total = 0.4 * 2.5 + 0.6 * 2.0 = 1.0 + 1.2 = 2.2
    np.testing.assert_almost_equal(total_loss.data, 2.2)

  def test_flattened_evaluator_loss(self):
    loss = deterministic_losses.MSE(
        variable_weights={'atmosphere.t': 0.4, 'ocean.sst': 0.6}
    )
    evaluator = evaluators.Evaluator(
        metrics={'flat_mse': loss}, aggregators=self.aggregator
    )
    flat_eval = evaluators.FlattenedEvaluator(evaluator)
    self.assertTrue(flat_eval.is_loss_evaluator)
    agg_states = flat_eval.evaluate(self.predictions, self.targets)
    # check structure
    self.assertCountEqual(agg_states.keys(), ['flat_mse'])
    self.assertIn(
        'SquaredError', agg_states['flat_mse'].sum_weighted_statistics
    )
    # check total loss value
    total_loss = flat_eval.evaluate_total(self.predictions, self.targets)
    # atmosphere.t mse = 2.5
    # ocean.sst mse = ( (10-12)^2 + (14-12)^2 ) / 2 = 4.0
    # total = 0.4 * 2.5 + 0.6 * 4.0 = 1.0 + 2.4 = 3.4
    np.testing.assert_almost_equal(total_loss.data, 3.4)

  def test_nested_evaluators_with_default(self):
    eval_ocean = evaluators.Evaluator(
        metrics={'mae_sst': deterministic_losses.MAE()},
        aggregators=self.aggregator,
    )
    default_eval = evaluators.Evaluator(
        metrics={'mse': deterministic_losses.MSE()},
        aggregators=self.aggregator,
    )
    nested = evaluators.NestedEvaluators(
        evaluators={'ocean': eval_ocean, ...: default_eval},
        evaluator_weights={'ocean': 0.6},
    )
    self.assertTrue(nested.is_loss_evaluator)
    # 'atmosphere' uses default MSE, 'ocean' uses MAE
    total_loss = nested.evaluate_total(self.predictions, self.targets)
    # atmosphere mse_t = 2.5
    # ocean mae_sst = 2.0
    # total = 1.0 * 2.5 + 0.6 * 2.0 = 2.5 + 1.2 = 3.7
    np.testing.assert_almost_equal(total_loss.data, 3.7)

  def test_nested_evaluators_raises_on_missing_key(self):
    eval_ocean = evaluators.Evaluator(
        metrics={'mae_sst': deterministic_losses.MAE()},
        aggregators=self.aggregator,
    )
    nested = evaluators.NestedEvaluators(
        evaluators={'ocean': eval_ocean},
    )
    with self.assertRaisesRegex(
        ValueError, "No evaluator found for key 'atmosphere'"
    ):
      nested.evaluate(self.predictions, self.targets)


if __name__ == '__main__':
  jax.config.update('jax_traceback_filtering', 'off')
  absltest.main()
