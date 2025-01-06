# Copyright 2024 Google LLC
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
"""Tests that random processes generate values with expected stats."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
from flax import nnx
import jax
from neuralgcm.experimental import coordinates
from neuralgcm.experimental import random_processes
from neuralgcm.experimental import typing
from neuralgcm.experimental import units
import numpy as np
import tensorflow_probability.substrates.jax as tfp


@absltest.skipThisClass('Base class')
class BaseSphericalHarmonicRandomProcessTest(parameterized.TestCase):
  """Base class for testing variants of random fields on spherical harmonics."""

  def setUp(self):
    super().setUp()
    self.sim_units = units.DEFAULT_UNITS
    self.dt = self.sim_units.nondimensionalize(typing.Quantity('1 hour'))

  def check_correlation_length(
      self,
      samples,
      expected_correlation_length,
      grid,
  ):
    """Checks the correlation length of random field."""
    unused_n_samples, n_lngs, n_lats = samples.shape
    expected_corr_frac = expected_correlation_length / (2 * np.pi * grid.radius)
    acorr_lat = tfp.stats.auto_correlation(
        # Mean autocorrelation in the lat direction at the longitude=0 line.
        samples[:, 0, :],
        axis=-1,
    ).mean(axis=0)
    # There are 2 * n_lats points in the circumference.
    fractional_corr_len_lat = np.argmax(acorr_lat < 0) / (2 * n_lats)
    self.assertBetween(
        fractional_corr_len_lat,
        expected_corr_frac * 0.5,
        expected_corr_frac * 3,
    )
    acorr_lng = tfp.stats.auto_correlation(
        # Mean autocorrelation in the lng direction at the latitude=0 line.
        samples[:, :, n_lats // 2],
        axis=-1,
    ).mean(axis=0)
    fractional_corr_len_lng = np.argmax(acorr_lng < 0) / n_lngs
    self.assertBetween(
        fractional_corr_len_lng,
        expected_corr_frac * 0.2,
        expected_corr_frac * 3,
    )

  def check_mean(
      self,
      samples,
      grid,
      expected_mean,
      variance,
      correlation_length,
      mean_tol_in_standard_errs,
  ):
    """Checks the mean (at every point & average) of samples."""
    n_samples, unused_n_lngs, unused_n_lats = samples.shape

    # Pointwise mean should be with tol with high probability everywhere.
    # Since we're testing many (lat/lon) points, we allow for some deviations.
    standard_error = np.sqrt(variance) / np.sqrt(n_samples) if variance else 0.0
    np.testing.assert_allclose(
        # 95% of points are within specified tol. There may be outliers.
        np.percentile(np.mean(samples, axis=0), 95),
        expected_mean,
        atol=mean_tol_in_standard_errs * standard_error,
    )
    np.testing.assert_allclose(
        # 100% of points are within a looser tol.
        np.mean(samples, axis=0),
        expected_mean,
        atol=2 * mean_tol_in_standard_errs * standard_error,
    )

    # Check average mean over whole earth (standard_error will be lower so this
    # is a good second check).
    expected_corr_frac = correlation_length / grid.radius
    n_equivalent_integrated_samples = n_samples / expected_corr_frac**2
    if variance:
      standard_error = np.sqrt(variance) / np.sqrt(
          n_equivalent_integrated_samples
      )
    else:
      standard_error = 0.0
    np.testing.assert_allclose(
        np.mean(samples),
        expected_mean,
        atol=mean_tol_in_standard_errs * standard_error,
    )

  def check_variance(
      self,
      samples,
      grid,
      correlation_length,
      expected_variance,
      var_tol_in_standard_errs,
  ):
    expected_variance = expected_variance or 0.0
    expected_integrated_variance = (
        expected_variance * 4 * np.pi * grid.radius**2
    )

    n_samples, unused_n_lngs, unused_n_lats = samples.shape
    # Integrating over the sphere we get additional statistical power since
    # points decorrelate.
    expected_corr_frac = correlation_length / grid.radius
    n_equivalent_integrated_samples = n_samples / expected_corr_frac**2
    standard_error = np.sqrt(
        # The variance of a (normal) variance estimate is 2 σ⁴ / (n - 1).
        2
        * expected_integrated_variance**2
        / n_equivalent_integrated_samples
    )

    np.testing.assert_allclose(
        grid.ylm_grid.integrate(np.var(samples, axis=0)),
        expected_integrated_variance,
        atol=var_tol_in_standard_errs * standard_error,
        rtol=0.0,
    )

  def check_unconditional_and_trajectory_stats(
      self,
      random_field,
      mean,
      variance,
      grid,
      correlation_length,
      correlation_time,
      run_mean_check=True,
      run_variance_check=True,
      run_correlation_length_check=True,
      run_correlation_time_check=True,
      mean_tol_in_standard_errs=4,
      var_tol_in_standard_errs=4,
  ):
    dt = self.dt

    # generating multiple trajectories of random fields.
    n_samples = 500
    unroll_length = 40
    init_rngs = jax.random.split(jax.random.key(5), n_samples)
    graph, params = nnx.split(random_field)
    sample_fn = lambda x: graph.apply(params).unconditional_sample(x)[0]
    evaluate_fn = lambda x: graph.apply(params).state_values(grid, x)[0]
    advance_fn = lambda x: graph.apply(params).advance(x)[0]
    batch_sample_fn = jax.vmap(sample_fn)
    batch_evaluate_fn = jax.vmap(evaluate_fn)
    batch_advance_fn = jax.vmap(advance_fn)
    initial_states = batch_sample_fn(init_rngs)
    initial_values = batch_evaluate_fn(initial_states).data

    with self.subTest('unconditional_sample_shape'):
      self.assertEqual(initial_values.shape, (n_samples,) + grid.shape)

    # TODO(dkochkov): Consider adding sizes/size properties to coordinate/field.
    n_lats = grid.fields['latitude'].shape[0]
    n_lngs = grid.fields['longitude'].shape[0]
    self.assertTupleEqual((n_samples, n_lngs, n_lats), initial_values.shape)

    if run_correlation_length_check and variance is not None:
      with self.subTest('unconditional_sample_correlation_len'):
        self.check_correlation_length(initial_values, correlation_length, grid)

    if run_mean_check:
      with self.subTest('unconditional_sample_pointwise_mean'):
        self.check_mean(
            initial_values,
            grid,
            expected_mean=mean,
            variance=variance,
            correlation_length=correlation_length,
            mean_tol_in_standard_errs=mean_tol_in_standard_errs,
        )

    if run_variance_check:
      with self.subTest('unconditional_sample_integrated_var'):
        self.check_variance(
            initial_values,
            grid,
            correlation_length=correlation_length,
            expected_variance=variance,
            var_tol_in_standard_errs=var_tol_in_standard_errs,
        )

    def step_fn(c, _):
      next_c = batch_advance_fn(c)
      next_output = batch_evaluate_fn(next_c).data
      return (next_c, next_output)

    _, field_trajectory = jax.lax.scan(
        step_fn, initial_states, xs=None, length=unroll_length
    )
    field_trajectory = jax.device_get(field_trajectory)

    if run_correlation_time_check and variance is not None:
      with self.subTest('trajectory_correlation_time'):
        # Mean autocorrelation at the lat=lng=0 point.
        acorr = tfp.stats.auto_correlation(
            field_trajectory[:, :, n_lngs // 2, 0], axis=0
        ).mean(axis=1)
        sample_decorr_time = dt * np.argmax(acorr < 0)
        self.assertBetween(
            sample_decorr_time, correlation_time / 2, correlation_time * 2
        )

    final_sample = field_trajectory[-1]

    if run_correlation_length_check and variance is not None:
      with self.subTest('final_sample_correlation_len'):
        self.check_correlation_length(final_sample, correlation_length, grid)

    if run_mean_check:
      with self.subTest('final_sample_pointwise_mean'):
        self.check_mean(
            final_sample,
            grid,
            expected_mean=mean,
            variance=variance,
            correlation_length=correlation_length,
            mean_tol_in_standard_errs=mean_tol_in_standard_errs,
        )

    if run_variance_check:
      with self.subTest('final_sample_integrated_var'):
        self.check_variance(
            final_sample,
            grid,
            correlation_length=correlation_length,
            expected_variance=variance,
            var_tol_in_standard_errs=var_tol_in_standard_errs,
        )

  def check_independent(self, x, y):
    """Checks random field values x and y are independent."""
    self.assertEqual(x.ndim, 2)
    self.assertEqual(y.ndim, 2)
    # corr[i, j] = Correlation(x[:, i], y[:, j])
    corr = tfp.stats.correlation(x, y, sample_axis=0, event_axis=1)
    standard_error = 2 / np.sqrt(x.shape[0])  # product of two iid χ²
    np.testing.assert_array_less(corr, 4 * standard_error)

  def check_nnx_state_structure_is_invariant(self, grf, grid):
    """Checks that random process does not mutate nnx.state(grf) structure."""
    init_nnx_state = nnx.state(grf, nnx.Param)
    random_state = grf.unconditional_sample(jax.random.key(0))
    random_state = grf.advance(random_state)
    _ = grf.state_values(grid, random_state)
    nnx_state = nnx.state(grf, nnx.Param)
    chex.assert_trees_all_equal_shapes_and_dtypes(init_nnx_state, nnx_state)


class GaussianRandomFieldTest(BaseSphericalHarmonicRandomProcessTest):
  """Tests GaussianRandomField random process."""

  @parameterized.named_parameters(
      dict(
          testcase_name='T42_reasonable_corrs',
          variance=0.7,
          grid=coordinates.LonLatGrid.T42(),
          correlation_length=0.15,
          correlation_time=3,
      ),
      dict(
          testcase_name='T21_reasonable_corrs',
          variance=1.5,
          grid=coordinates.LonLatGrid.T21(),
          correlation_length=0.15,
          correlation_time=3,
      ),
      dict(
          testcase_name='T42_large_radius',
          variance=1.2,
          grid=coordinates.LonLatGrid.T42(radius=4),
          correlation_length=1.15,
          correlation_time=3,
      ),
      dict(
          testcase_name='T85_long_corrs',
          variance=2.7,
          grid=coordinates.LonLatGrid.T85(),
          correlation_length=0.5,
          correlation_time=3,
      ),
  )
  def test_unconditional_and_trajectory_stats(
      self,
      variance,
      grid,
      correlation_length,
      correlation_time,
  ):
    grf = random_processes.GaussianRandomField(
        grid=grid,
        dt=self.dt,
        sim_units=self.sim_units,
        correlation_time=correlation_time * typing.Quantity('1 hour'),
        correlation_length=correlation_length,
        variance=variance,
        rngs=nnx.Rngs(0),
    )
    self.check_unconditional_and_trajectory_stats(
        grf,
        0.0,  # mean = 0
        variance,
        grid,
        correlation_length,
        correlation_time,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='with_all_nnx_params',
          correlation_time_type=nnx.Param,
          correlation_length_type=nnx.Param,
          variance_type=nnx.Param,
      ),
      dict(
          testcase_name='with_fixed_params',
          correlation_time_type=random_processes.RandomnessParam,
          correlation_length_type=random_processes.RandomnessParam,
          variance_type=random_processes.RandomnessParam,
      ),
      dict(
          testcase_name='with_nnx_and_fixed_params',
          correlation_time_type=nnx.Param,
          correlation_length_type=nnx.Param,
          variance_type=random_processes.RandomnessParam,
      ),
  )
  def test_nnx_state_structure(
      self, correlation_time_type, correlation_length_type, variance_type
  ):
    """Tests that random process does not mutate structure of nnx.state."""
    grid = coordinates.LonLatGrid.T42()
    grf = random_processes.GaussianRandomField(
        grid=grid,
        dt=self.dt,
        sim_units=self.sim_units,
        correlation_time=3 * typing.Quantity('1 hour'),
        correlation_length=0.15,
        variance=1.5,
        correlation_time_type=correlation_time_type,
        correlation_length_type=correlation_length_type,
        variance_type=variance_type,
        rngs=nnx.Rngs(0),
    )
    with self.subTest('nnx_state_structure_invariance'):
      self.check_nnx_state_structure_is_invariant(grf, grid)

    with self.subTest('nnx_param_count'):
      params = nnx.state(grf, nnx.Param)
      actual_count = sum([np.size(x) for x in jax.tree.leaves(params)])
      expected_count = sum(
          x == nnx.Param
          for x in [
              correlation_time_type,
              correlation_length_type,
              variance_type,
          ]
      )
      self.assertEqual(actual_count, expected_count)


class BatchGaussianRandomFieldTest(BaseSphericalHarmonicRandomProcessTest):

  def setUp(self):
    super().setUp()
    self.grid = coordinates.LonLatGrid.T85()

  def _make_grf(
      self,
      variances,
      correlation_lengths,
      correlation_times,
  ):
    return random_processes.BatchGaussianRandomField(
        grid=self.grid,
        dt=self.dt,
        sim_units=self.sim_units,
        correlation_times=correlation_times,
        correlation_lengths=correlation_lengths,
        variances=variances,
        rngs=nnx.Rngs(0),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='reasonable_corrs',
          variances=(1.0, 2.7),
          correlation_lengths=(0.15, 0.2),
          correlation_times=(1, 2.1),
      ),
  )
  def test_stats(
      self,
      variances,
      correlation_lengths,
      correlation_times,
  ):
    random_field = self._make_grf(
        variances, correlation_lengths, correlation_times
    )
    n_fields = len(variances)
    unroll_length = 10
    n_samples = 1000
    rngs = jax.random.split(jax.random.key(802701), n_samples)

    ###
    graph, params = nnx.split(random_field)
    sample_fn = lambda x: graph.apply(params).unconditional_sample(x)[0]
    evaluate_fn = lambda x: graph.apply(params).state_values(self.grid, x)[0]
    advance_fn = lambda x: graph.apply(params).advance(x)[0]
    batch_sample_fn = jax.vmap(sample_fn)
    batch_evaluate_fn = jax.vmap(evaluate_fn)
    batch_advance_fn = jax.vmap(advance_fn)
    initial_states = batch_sample_fn(rngs)
    initial_values = batch_evaluate_fn(initial_states).data

    def step_fn(c, _):
      next_c = batch_advance_fn(c)
      next_output = batch_evaluate_fn(next_c).data
      return (next_c, next_output)

    _, field_trajectory = jax.lax.scan(
        step_fn, initial_states, xs=None, length=unroll_length
    )
    field_trajectory = jax.device_get(field_trajectory)
    ###
    self.assertEqual(
        (unroll_length, n_samples, n_fields) + self.grid.shape,
        field_trajectory.shape,
    )
    final_nodal_value = field_trajectory[-1, ...]

    self.assertEqual(
        (n_samples, n_fields) + self.grid.ylm_grid.modal_shape,
        initial_states.core.shape,
    )

    # Nodal values should have the right statistics.
    for i, (variance, correlation_length) in enumerate(
        zip(variances, correlation_lengths, strict=True)
    ):
      for x in [initial_values, final_nodal_value]:
        self.check_mean(
            x[:, i],
            self.grid,
            expected_mean=0.0,
            variance=variance,
            correlation_length=correlation_length,
            mean_tol_in_standard_errs=5,
        )
        self.check_variance(
            x[:, i],
            self.grid,
            correlation_length=correlation_length,
            expected_variance=variance,
            var_tol_in_standard_errs=5,
        )
        self.check_correlation_length(
            x[:, i],
            expected_correlation_length=correlation_length,
            grid=self.grid,
        )

      # Fields 0 and 1 should be independent.
      # Check a handful of groups of nodal values.
      self.check_independent(
          x[:, 0, 100:105, 60],
          x[:, 1, 100:105, 60],
      )
      self.check_independent(
          x[:, 0, 100:105, 0],
          x[:, 1, 100:105, 0],
      )
      self.check_independent(x[:, 0, 0:5, 0], x[:, 1, 0:5, 0])

    # Initial and final sample should be independent as well, since we unroll
    # for much longer than the correlation time.
    self.check_independent(
        initial_values[:, 0, 50:55, 60],
        final_nodal_value[:, 0, 50:55, 60],
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='with_all_nnx_params',
          correlation_time_type=nnx.Param,
          correlation_length_type=nnx.Param,
          variance_type=nnx.Param,
      ),
      dict(
          testcase_name='with_fixed_params',
          correlation_time_type=random_processes.RandomnessParam,
          correlation_length_type=random_processes.RandomnessParam,
          variance_type=random_processes.RandomnessParam,
      ),
      dict(
          testcase_name='with_nnx_and_fixed_params',
          correlation_time_type=nnx.Param,
          correlation_length_type=nnx.Param,
          variance_type=random_processes.RandomnessParam,
      ),
  )
  def test_nnx_state_structure(
      self, correlation_time_type, correlation_length_type, variance_type
  ):
    """Tests that random process does not mutate structure of nnx.state."""
    grid = coordinates.LonLatGrid.T42()
    grf = random_processes.BatchGaussianRandomField(
        grid=grid,
        dt=self.dt,
        sim_units=self.sim_units,
        correlation_times=(1.0, 2.7),
        correlation_lengths=(0.15, 0.2),
        variances=(1, 2.1),
        correlation_time_type=correlation_time_type,
        correlation_length_type=correlation_length_type,
        variance_type=variance_type,
        rngs=nnx.Rngs(0),
    )
    with self.subTest('nnx_state_structure_invariance'):
      self.check_nnx_state_structure_is_invariant(grf, grid)

    with self.subTest('nnx_param_count'):
      params = nnx.state(grf, nnx.Param)
      actual_count = sum([np.size(x) for x in jax.tree.leaves(params)])
      expected_count = sum(
          grf.n_fields
          for x in [
              correlation_time_type,
              correlation_length_type,
              variance_type,
          ]
          if x == nnx.Param
      )
      self.assertEqual(actual_count, expected_count)


if __name__ == '__main__':
  absltest.main()
