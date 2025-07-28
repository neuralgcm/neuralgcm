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
"""Tests for stochastic."""
from absl.testing import absltest
from absl.testing import parameterized
from dinosaur import coordinate_systems
from dinosaur import primitive_equations
from dinosaur import scales
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
import haiku as hk
import jax
import jax.numpy as jnp
from neuralgcm.legacy import stochastic
import numpy as np
import scipy.stats


tree_map = jax.tree_util.tree_map

# Effectively constant correlation time/length.
CONSTANT_CORRELATION_TIME_HRS = 24 * 365 * 1000  # 1000 years in hours
CONSTANT_CORRELATION_LENGTH_KM = 40_075 * 10  # 10x circumference of earth in km


def _auto_correlation_1d(series):
  centered = series - jnp.mean(series)
  autocov = jax.scipy.signal.correlate(centered, centered, precision='float32')
  autocov = autocov[series.shape[0] - 1 :]
  autocorr = autocov / autocov[0]
  return autocorr


def auto_correlation(series: jax.typing.ArrayLike, axis: int) -> jax.Array:
  """JAX based implementation of tfp.stats.auto_correlation."""
  if series.ndim != 2:
    raise ValueError('series must be 2D')
  i = (axis + 1) % 2  # vectorize along the other axis
  return jax.vmap(_auto_correlation_1d, in_axes=i, out_axes=i)(series)


@absltest.skipThisClass('Base class')
class BaseRandomFieldTest(parameterized.TestCase):
  """Base class for random field tests."""

  def setUp(self):
    super().setUp()
    self.physics_specs = primitive_equations.PrimitiveEquationsSpecs.from_si()
    self.dt = self.physics_specs.nondimensionalize(1 * scales.units.hour)

  def check_correlation_length(
      self,
      nodal_samples,
      expected_correlation_length,
      coords,
  ):
    """Checks the correlation length of random field."""
    unused_n_samples, n_lngs, n_lats = nodal_samples.shape
    expected_corr_frac = expected_correlation_length / (
        2 * np.pi * coords.horizontal.radius
    )
    # Mean autocorrelation in the lat direction at the longitude=0 line.
    acorr_lat = auto_correlation(nodal_samples[:, 0, :], axis=-1).mean(axis=0)
    # There are 2 * n_lats points in the circumference.
    fractional_corr_len_lat = np.argmax(acorr_lat < 0) / (2 * n_lats)
    self.assertBetween(
        fractional_corr_len_lat,
        expected_corr_frac * 0.5,
        expected_corr_frac * 3,
    )
    # Mean autocorrelation in the lng direction at the latitude=0 line.
    acorr_lng = auto_correlation(
        nodal_samples[:, :, n_lats // 2],
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
      nodal_samples,
      coords,
      expected_mean,
      variance,
      correlation_length,
      mean_tol_in_standard_errs,
  ):
    """Checks the mean (at every point & average) of nodal_samples."""
    n_samples, unused_n_lngs, unused_n_lats = nodal_samples.shape

    # Pointwise mean should be with tol with high probability everywhere.
    # Since we're testing many (lat/lon) points, we allow for some deviations.
    standard_error = np.sqrt(variance) / np.sqrt(n_samples) if variance else 0.0
    np.testing.assert_allclose(
        # 95% of points are within specified tol. There may be outliers.
        np.percentile(np.mean(nodal_samples, axis=0), 95),
        expected_mean,
        atol=mean_tol_in_standard_errs * standard_error,
    )
    np.testing.assert_allclose(
        # 100% of points are within a looser tol.
        np.mean(nodal_samples, axis=0),
        expected_mean,
        atol=2 * mean_tol_in_standard_errs * standard_error,
    )

    # Check average mean over whole earth (standard_error will be lower so this
    # is a good second check).
    expected_corr_frac = correlation_length / coords.horizontal.radius
    n_equivalent_integrated_samples = n_samples / expected_corr_frac**2
    if variance:
      standard_error = np.sqrt(variance) / np.sqrt(
          n_equivalent_integrated_samples
      )
    else:
      standard_error = 0.0
    np.testing.assert_allclose(
        np.mean(nodal_samples),
        expected_mean,
        atol=mean_tol_in_standard_errs * standard_error,
    )

  def check_variance(
      self,
      nodal_samples,
      coords,
      correlation_length,
      expected_variance,
      var_tol_in_standard_errs,
  ):
    expected_variance = expected_variance or 0.0
    expected_integrated_variance = (
        expected_variance * 4 * np.pi * coords.horizontal.radius**2
    )

    n_samples, unused_n_lngs, unused_n_lats = nodal_samples.shape
    # Integrating over the sphere we get additional statistical power since
    # points decorrelate.
    expected_corr_frac = correlation_length / coords.horizontal.radius
    n_equivalent_integrated_samples = n_samples / expected_corr_frac**2
    standard_error = np.sqrt(
        # The variance of a (normal) variance estimate is 2 σ⁴ / (n - 1).
        2
        * expected_integrated_variance**2
        / n_equivalent_integrated_samples
    )

    np.testing.assert_allclose(
        coords.horizontal.integrate(np.var(nodal_samples, axis=0)),
        expected_integrated_variance,
        atol=var_tol_in_standard_errs * standard_error,
        rtol=0.0,
    )

  def check_unconditional_and_trajectory_stats(
      self,
      coords,
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
    del grid  # unused.
    physics_specs = primitive_equations.PrimitiveEquationsSpecs.from_si()
    dt = physics_specs.nondimensionalize(1 * scales.units.hour)

    # generating multiple trajectories of random fields.
    n_samples = 500
    unroll_length = 40
    init_rngs = jax.random.split(jax.random.PRNGKey(5), n_samples)
    initial_states = jax.vmap(random_field.unconditional_sample)(init_rngs)
    initial_values = jax.vmap(random_field.to_nodal_values)(initial_states.core)

    with self.subTest('to_modal'):
      modal_initial_values = jax.vmap(random_field.to_modal_values)(
          initial_states.core
      )
      to_nodal = random_field.coords.horizontal.to_nodal
      to_modal = random_field.coords.horizontal.to_modal
      np.testing.assert_allclose(
          to_nodal(to_modal(initial_values)),
          to_nodal(modal_initial_values),
          rtol=1e-3,
          atol=1e-3,
      )

    with self.subTest('unconditional_sample_shape'):
      self.assertEqual(
          initial_values.shape, (n_samples,) + coords.horizontal.nodal_shape
      )

    n_lats = len(coords.horizontal.latitudes)
    n_lngs = len(coords.horizontal.longitudes)
    self.assertTupleEqual((n_samples, n_lngs, n_lats), initial_values.shape)

    if run_correlation_length_check and variance is not None:
      with self.subTest('unconditional_sample_correlation_len'):
        self.check_correlation_length(
            initial_values, correlation_length, coords
        )

    if run_mean_check:
      with self.subTest('unconditional_sample_pointwise_mean'):
        self.check_mean(
            initial_values,
            coords,
            expected_mean=mean,
            variance=variance,
            correlation_length=correlation_length,
            mean_tol_in_standard_errs=mean_tol_in_standard_errs,
        )

    if run_variance_check:
      with self.subTest('unconditional_sample_integrated_var'):
        self.check_variance(
            initial_values,
            coords,
            correlation_length=correlation_length,
            expected_variance=variance,
            var_tol_in_standard_errs=var_tol_in_standard_errs,
        )

    def step_fn(c, _):
      next_c = jax.vmap(random_field.advance)(c)
      next_output = jax.vmap(random_field.to_nodal_values)(next_c.core)
      return (next_c, next_output)

    _, field_trajectory = jax.lax.scan(
        step_fn, initial_states, xs=None, length=unroll_length
    )
    field_trajectory = jax.device_get(field_trajectory)

    if run_correlation_time_check and variance is not None:
      with self.subTest('trajectory_correlation_time'):
        # Mean autocorrelation at the lat=lng=0 point.
        acorr = auto_correlation(
            field_trajectory[:, :, n_lngs // 2, 0], axis=0
        ).mean(axis=1)
        sample_decorr_time = dt * np.argmax(acorr < 0)
        self.assertBetween(
            sample_decorr_time, correlation_time / 2, correlation_time * 2
        )

    final_sample = field_trajectory[-1]

    if run_correlation_length_check and variance is not None:
      with self.subTest('final_sample_correlation_len'):
        self.check_correlation_length(final_sample, correlation_length, coords)

    if run_mean_check:
      with self.subTest('final_sample_pointwise_mean'):
        self.check_mean(
            final_sample,
            coords,
            expected_mean=mean,
            variance=variance,
            correlation_length=correlation_length,
            mean_tol_in_standard_errs=mean_tol_in_standard_errs,
        )

    if run_variance_check:
      with self.subTest('final_sample_integrated_var'):
        self.check_variance(
            final_sample,
            coords,
            correlation_length=correlation_length,
            expected_variance=variance,
            var_tol_in_standard_errs=var_tol_in_standard_errs,
        )

  def check_independent(self, x, y):
    """Checks random field values x and y are independent."""
    self.assertEqual(x.ndim, 2)
    self.assertEqual(y.ndim, 2)
    corr = scipy.stats.pearsonr(x, y, axis=0).statistic
    standard_error = 2 / np.sqrt(x.shape[0])  # product of two iid χ²
    np.testing.assert_array_less(corr, 4 * standard_error)


class GaussianRandomFieldTest(BaseRandomFieldTest):
  """Tests GaussianRandomField."""

  @parameterized.named_parameters(
      dict(
          testcase_name='T42_reasonable_corrs',
          variance=0.7,
          grid=spherical_harmonic.Grid.T42(),
          correlation_length=0.15,
          correlation_time=3,
      ),
      dict(
          testcase_name='T21_reasonable_corrs',
          variance=1.5,
          grid=spherical_harmonic.Grid.T21(),
          correlation_length=0.15,
          correlation_time=3,
      ),
      dict(
          testcase_name='T42_large_radius',
          variance=1.2,
          grid=spherical_harmonic.Grid.T42(radius=4),
          correlation_length=1.15,
          correlation_time=3,
      ),
      dict(
          testcase_name='T85_long_corrs',
          variance=2.7,
          grid=spherical_harmonic.Grid.T85(),
          correlation_length=0.5,
          correlation_time=3,
      ),
      dict(
          testcase_name='variance_None',
          variance=None,
          grid=spherical_harmonic.Grid.T21(),
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
    coords = coordinate_systems.CoordinateSystem(
        grid, sigma_coordinates.SigmaCoordinates.equidistant(12)
    )
    aux_features = {}
    if variance is None:
      variance_arg = None
    else:
      variance_arg = variance * scales.units.dimensionless
    grf = stochastic.GaussianRandomField(
        coords,
        self.dt,
        self.physics_specs,
        aux_features,
        correlation_time=correlation_time * scales.units.hours,
        correlation_length=correlation_length * scales.units.dimensionless,
        variance=variance_arg,
    )
    self.assertEqual(
        grf.preferred_representation, stochastic.PreferredRepresentation.MODAL
    )
    self.check_unconditional_and_trajectory_stats(
        coords,
        grf,
        0.0,  # mean = 0
        variance,
        grid,
        correlation_length,
        correlation_time,
    )


class CenteredLognormalRandomFieldTest(BaseRandomFieldTest):
  """Tests CenteredLognormalRandomField."""

  @parameterized.named_parameters(
      dict(
          testcase_name='T42',
          grid=spherical_harmonic.Grid.T42(),
      ),
      dict(
          testcase_name='T21',
          grid=spherical_harmonic.Grid.T21(),
      ),
      dict(
          testcase_name='T42_large_radius',
          grid=spherical_harmonic.Grid.T42(radius=40),
      ),
      dict(
          testcase_name='T42_small_radius',
          grid=spherical_harmonic.Grid.T42(radius=0.5),
      ),
      dict(
          testcase_name='T85',
          grid=spherical_harmonic.Grid.T85(),
      ),
      dict(
          testcase_name='T85_small_radius',
          grid=spherical_harmonic.Grid.T85(radius=0.5),
      ),
  )
  def test_stats(self, grid):
    coords = coordinate_systems.CoordinateSystem(
        grid, sigma_coordinates.SigmaCoordinates.equidistant(12)
    )
    aux_features = {}

    variance = 3.0  # In the nonlinear regime.
    correlation_time = 1.1
    correlation_length = 0.15

    rf = stochastic.CenteredLognormalRandomField(
        coords,
        self.dt,
        self.physics_specs,
        aux_features,
        correlation_time=correlation_time,
        correlation_length=correlation_length,
        variance=variance,
    )
    self.assertEqual(
        rf.preferred_representation, stochastic.PreferredRepresentation.NODAL
    )
    self.check_unconditional_and_trajectory_stats(
        coords,
        rf,
        0.0,  # mean = 0 since our lognormals are centered.
        variance,
        grid,
        correlation_length,
        correlation_time,
        run_mean_check=True,
        run_variance_check=True,
        run_correlation_length_check=False,
        run_correlation_time_check=False,
        mean_tol_in_standard_errs=(
            # On coarser grids we will fail the pointwise tests due to the field
            # not being that homogeneous.
            # In all cases, the lognormal is heavy tailed, so we require looser
            # tolerance than the GaussianRandomField.
            10
            if grid.total_wavenumbers < 85
            else 5
        ),
        var_tol_in_standard_errs=(
            # The 4th moment of lognormal is quite large.
            # Nonetheless, due to the large number of effective samples
            # (considering 500 samples and averaging over the globe), the test
            # does indeed check something useful.
            80
            if grid.total_wavenumbers < 85
            else 9
        ),
    )


class BatchGaussianRandomFieldModuleTest(BaseRandomFieldTest):

  def setUp(self):
    super().setUp()
    grid = spherical_harmonic.Grid.T85()
    self.coords = coordinate_systems.CoordinateSystem(
        grid, sigma_coordinates.SigmaCoordinates.equidistant(12)
    )

  def _make_grf(
      self,
      variances,
      initial_correlation_lengths,
      initial_correlation_times,
      field_subset=None,
      n_fixed_fields=None,
  ):
    self.physics_specs = primitive_equations.PrimitiveEquationsSpecs.from_si()
    self.dt = self.physics_specs.nondimensionalize(1 * scales.units.hour)

    return stochastic.BatchGaussianRandomFieldModule(
        self.coords,
        self.dt,
        self.physics_specs,
        aux_features={},
        initial_correlation_times=initial_correlation_times,
        initial_correlation_lengths=initial_correlation_lengths,
        variances=variances,
        field_subset=field_subset,
        n_fixed_fields=n_fixed_fields,
    )

  def nondimensionalize(self, x):
    return stochastic.nondimensionalize(x, self.physics_specs)

  @parameterized.named_parameters(
      dict(
          testcase_name='reasonable_corrs',
          variances=(1.0, 2.7),
          initial_correlation_lengths=(0.15, 0.2),
          initial_correlation_times=(1, 2.1),
      ),
      dict(
          testcase_name='one_fixed_field',
          variances=(1.0, 2.7),
          initial_correlation_lengths=(0.15, 0.2),
          initial_correlation_times=(1, 2.1),
          n_fixed_fields=1,
      ),
      dict(
          testcase_name='reasonable_corrs_skip_middle',
          # Using NaN for the one that should be skipped as a extra means to
          # test it wasn't used.
          variances=(1.0, np.nan, 2.7),
          initial_correlation_lengths=(0.5, np.nan, 0.2),
          initial_correlation_times=(1, np.nan, 2.1),
          field_subset=(0, 2),
      ),
  )
  def test_stats(
      self,
      variances,
      initial_correlation_lengths,
      initial_correlation_times,
      field_subset=None,
      n_fixed_fields=None,
  ):
    unroll_length = 10

    @hk.transform
    def make_field_trajectory(key):
      grf = self._make_grf(
          variances=variances,
          initial_correlation_lengths=initial_correlation_lengths,
          initial_correlation_times=initial_correlation_times,
          # Do not specify the field names... Let the default naming happen.
          field_subset=field_subset,
          n_fixed_fields=n_fixed_fields,
      )
      initial_value = grf.unconditional_sample(key)

      def step_fn(c, _):
        next_c = grf.advance(c)
        next_output = next_c.nodal_value
        return (next_c, next_output)

      _, trajectory = jax.lax.scan(
          step_fn, initial_value, xs=None, length=unroll_length
      )
      return initial_value, jax.device_get(trajectory)

    n_fixed_fields = n_fixed_fields or 0
    n_samples = 1000
    rngs = jax.random.split(jax.random.PRNGKey(802701), n_samples)
    params = make_field_trajectory.init(rng=rngs[0], key=rngs[0])
    initial_value, trajectory = jax.vmap(
        lambda rng: make_field_trajectory.apply(params, rng, rng)
    )(rngs)

    if field_subset:
      subset = lambda seq: [seq[i] for i in field_subset]
      variances = subset(variances)
      initial_correlation_times = subset(initial_correlation_times)
      initial_correlation_lengths = subset(initial_correlation_lengths)
    n_fields = len(variances)

    self.assertEqual(
        (n_samples, unroll_length, n_fields)
        + self.coords.horizontal.nodal_shape,
        trajectory.shape,
    )
    final_nodal_value = trajectory[:, -1]

    self.assertEqual(
        (n_samples, n_fields) + self.coords.horizontal.modal_shape,
        initial_value.core.shape,
    )
    self.assertEqual(
        (n_samples, n_fields) + self.coords.horizontal.modal_shape,
        initial_value.modal_value.shape,
    )
    self.assertEqual(
        (n_samples, n_fields) + self.coords.horizontal.nodal_shape,
        initial_value.nodal_value.shape,
    )

    # Check stochastic params were initialized with hk parameters.
    self.assertCountEqual(
        ['batch_gaussian_random_field_module'],
        params.keys(),
    )
    self.assertCountEqual(
        ['correlation_times_raw', 'correlation_lengths_raw'],
        params['batch_gaussian_random_field_module'].keys(),
    )
    for name in ['correlation_times_raw', 'correlation_lengths_raw']:
      self.assertEqual(
          (n_fields - n_fixed_fields,),
          params['batch_gaussian_random_field_module'][name].shape,
      )

    # Core should be modal
    tree_map(
        np.testing.assert_array_equal,
        initial_value.core,
        initial_value.modal_value,
    )

    # Nodal values should have the right statistics.
    for i, (variance, correlation_length) in enumerate(
        zip(
            variances,
            initial_correlation_lengths,
            strict=True,
        )
    ):
      for x in [initial_value.nodal_value, final_nodal_value]:
        self.check_mean(
            x[:, i],
            self.coords,
            expected_mean=0.0,
            variance=variance,
            correlation_length=correlation_length,
            mean_tol_in_standard_errs=5,
        )
        self.check_variance(
            x[:, i],
            self.coords,
            correlation_length=correlation_length,
            expected_variance=variance,
            var_tol_in_standard_errs=5,
        )
        self.check_correlation_length(
            x[:, i],
            expected_correlation_length=correlation_length,
            coords=self.coords,
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
        initial_value.nodal_value[:, 0, 50:55, 60],
        final_nodal_value[:, 0, 50:55, 60],
    )

  def test_giant_correlations_give_constant_fields(self):
    unroll_length = 10

    initial_correlation_lengths = [
        # Include a moderate correlation batch member just to check that extreme
        # correlations in other batch members don't mess this up.
        0.2,
        # Include the default "CONSTANT" correlations
        f'{CONSTANT_CORRELATION_LENGTH_KM} km',
        # Include a much larger correlation, to check for numerical stability.
        f'{1000 * CONSTANT_CORRELATION_LENGTH_KM} km',
    ]
    initial_correlation_times = [
        1,
        f'{CONSTANT_CORRELATION_TIME_HRS} hours',
        f'{1000 * CONSTANT_CORRELATION_TIME_HRS} hours',
    ]
    variances = [1.0, 1.0, 1.0]

    @hk.transform
    def make_field_trajectory(key):
      grf = self._make_grf(
          variances=variances,
          initial_correlation_lengths=initial_correlation_lengths,
          initial_correlation_times=initial_correlation_times,
          # Do not specify the field names... Let the default naming happen.
      )
      initial_value = grf.unconditional_sample(key)

      def step_fn(c, _):
        next_c = grf.advance(c)
        next_output = next_c.nodal_value
        return (next_c, next_output)

      _, trajectory = jax.lax.scan(
          step_fn, initial_value, xs=None, length=unroll_length
      )
      return initial_value, jax.device_get(trajectory)

    n_samples = 100
    rngs = jax.random.split(jax.random.PRNGKey(802701), n_samples)
    params = make_field_trajectory.init(rng=rngs[0], key=rngs[0])
    initial_value, trajectory = jax.vmap(
        lambda rng: make_field_trajectory.apply(
            params, rng, jax.random.fold_in(rng, 1)
        )
    )(rngs)
    final_nodal_value = trajectory[:, -1]
    initial_nodal_value = initial_value.nodal_value

    self.assertTrue(np.all(np.isfinite(initial_nodal_value)))
    self.assertTrue(np.all(np.isfinite(final_nodal_value)))

    # All fields have the correct mean and variance.
    for i in range(3):
      for x in [initial_value.nodal_value, final_nodal_value]:
        self.check_mean(
            x[:, i],
            self.coords,
            expected_mean=0.0,
            variance=variances[i],
            correlation_length=self.nondimensionalize(
                initial_correlation_lengths[i]
            ),
            mean_tol_in_standard_errs=5,
        )
        self.check_variance(
            x[:, i],
            self.coords,
            correlation_length=self.nondimensionalize(
                initial_correlation_lengths[i]
            ),
            expected_variance=variances[i],
            var_tol_in_standard_errs=5,
        )

    # Field 0 (moderate correlation) has correct correlation length.
    for x in [initial_value.nodal_value, final_nodal_value]:
      self.check_correlation_length(
          x[:, 0],
          expected_correlation_length=initial_correlation_lengths[0],
          coords=self.coords,
      )

    # The variation in index 0 (moderate correlation) is much larger than that
    # in index 1 (CONSTANT_CORRELATION_*) or 2.
    # This checks the correlation length/time of fields 1, 2 is huge.
    diff = final_nodal_value - initial_nodal_value
    for i in [1, 2]:
      self.assertGreater(  # Variation in time
          np.max(np.abs(diff[:, 0])), 100 * np.max(np.abs(diff[:, i]))
      )
      self.assertGreater(  # Variation in lat/lon
          np.std(initial_nodal_value[:, 0], axis=(-1, -2)).max(),
          10000 * np.std(initial_nodal_value[:, i], axis=(-1, -2)).max(),
      )


class DictOfGaussianRandomFieldModulesTest(BaseRandomFieldTest):

  def setUp(self):
    super().setUp()
    grid = spherical_harmonic.Grid.T85()
    self.coords = coordinate_systems.CoordinateSystem(
        grid, sigma_coordinates.SigmaCoordinates.equidistant(12)
    )

  def _make_grf(
      self,
      variances,
      initial_correlation_lengths,
      initial_correlation_times,
      field_names=None,
      field_subset=None,
  ):
    physics_specs = primitive_equations.PrimitiveEquationsSpecs.from_si()
    self.dt = physics_specs.nondimensionalize(1 * scales.units.hour)

    return stochastic.DictOfGaussianRandomFieldModules(
        self.coords,
        self.dt,
        physics_specs,
        aux_features={},
        initial_correlation_times=initial_correlation_times,
        initial_correlation_lengths=initial_correlation_lengths,
        variances=variances,
        field_names=field_names,
        field_subset=field_subset,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='reasonable_corrs',
          variances=(1.0, 2.7),
          initial_correlation_lengths=(0.15, 0.25),
          initial_correlation_times=(2, 3),
      ),
      dict(
          testcase_name='reasonable_corrs_skip_middle',
          # Using NaN for the one that should be skipped as a extra means to
          # test it wasn't used.
          variances=(1.0, np.nan, 2.7),
          initial_correlation_lengths=(0.15, np.nan, 0.25),
          initial_correlation_times=(2, np.nan, 3),
          field_subset=(0, 2),
      ),
  )
  def test_initialization_and_stats(
      self,
      variances,
      initial_correlation_lengths,
      initial_correlation_times,
      field_subset=None,
  ):
    @hk.transform
    def unconditional_sample(key):
      grf = self._make_grf(
          variances=variances,
          initial_correlation_lengths=initial_correlation_lengths,
          initial_correlation_times=initial_correlation_times,
          # Do not specify the field names... Let the default naming happen.
          field_subset=field_subset,
      )
      return grf.unconditional_sample(key)

    n_samples = 500
    rngs = jax.random.split(jax.random.PRNGKey(802701), n_samples)
    params = unconditional_sample.init(rng=rngs[0], key=rngs[0])
    sample = jax.vmap(lambda rng: unconditional_sample.apply(params, rng, rng))(
        rngs
    )

    # Now that we are done using the GRF, trim the args to the length of args
    # that *should* have been used (in light of subsetting).
    field_names = [f'GRF{i}' for i in range(len(variances))]
    if field_subset:
      subset = lambda seq: [seq[i] for i in field_subset]
      field_names = subset(field_names)
      variances = subset(variances)
      initial_correlation_times = subset(initial_correlation_times)
      initial_correlation_lengths = subset(initial_correlation_lengths)

    # Check stochastic params were initialized with hk parameters.
    self.assertCountEqual(
        params.keys(),
        [
            f'dict_of_gaussian_random_field_modules/~/{name}'
            for name in field_names
        ],
    )
    for one_rf_name, one_rf_params in params.items():
      self.assertCountEqual(
          # Correlations are tuned, variance is fixed and so doesn't show up
          # here.
          ['correlation_time_raw', 'correlation_length_raw'],
          one_rf_params.keys(),
          msg=f'Failed at {one_rf_name=}',
      )

    # Core should be modal
    tree_map(np.testing.assert_array_equal, sample.core, sample.modal_value)

    # Nodal values should have the right statistics.
    for name, variance, correlation_length in zip(
        field_names,
        variances,
        initial_correlation_lengths,
        strict=True,
    ):
      self.check_mean(
          sample.nodal_value[name],
          self.coords,
          expected_mean=0.0,
          variance=variance,
          correlation_length=correlation_length,
          mean_tol_in_standard_errs=5,
      )
      self.check_variance(
          sample.nodal_value[name],
          self.coords,
          correlation_length=correlation_length,
          expected_variance=variance,
          var_tol_in_standard_errs=5,
      )
      self.check_correlation_length(
          sample.nodal_value[name],
          expected_correlation_length=correlation_length,
          coords=self.coords,
      )

    # Different fields should be independent.
    self.check_independent(
        sample.nodal_value[field_names[0]][:, 100:105, 64],
        sample.nodal_value[field_names[1]][:, 100:105, 64],
    )


class SumOfGaussianRandomFieldsTest(BaseRandomFieldTest):

  def test_sum_of_two_gaussians_that_only_differ_in_variance(self):
    # The sum of two independent RFs with the same correlation lengths results
    # in a RF with the sum of variance, but the same correlation lengths.
    grid = spherical_harmonic.Grid.T21()
    coords = coordinate_systems.CoordinateSystem(
        grid, sigma_coordinates.SigmaCoordinates.equidistant(12)
    )
    aux_features = {}

    correlation_time = 2.1
    correlation_length = 0.15
    variances = [1.0, 2.0]

    sum_rf = stochastic.SumOfGaussianRandomFields(
        coords,
        self.dt,
        self.physics_specs,
        aux_features,
        correlation_times=[correlation_time] * 2,
        correlation_lengths=[correlation_length] * 2,
        variances=variances,
    )

    self.assertEqual(
        sum_rf.preferred_representation,
        stochastic.PreferredRepresentation.MODAL,
    )

    self.check_unconditional_and_trajectory_stats(
        coords,
        sum_rf,
        0.0,  # mean = 0
        sum(variances),
        grid,
        correlation_length,
        correlation_time,
    )


class GaussianRandomFieldModuleTest(parameterized.TestCase):

  def _make_module_and_params_and_grf(
      self,
      initial_variance='0.5 dimensionless',
      variance_bound=None,
      tune_variance=True,
  ):
    """Makes hk.Module, and params / Loss."""
    coords = coordinate_systems.CoordinateSystem(
        spherical_harmonic.Grid.T21(),
        sigma_coordinates.SigmaCoordinates.equidistant(12),
    )
    physics_specs = primitive_equations.PrimitiveEquationsSpecs.from_si()
    self.dt = physics_specs.nondimensionalize(1 * scales.units.hour)

    @hk.transform
    def f():
      return stochastic.GaussianRandomFieldModule(
          coords,
          self.dt,
          physics_specs,
          aux_features={},
          initial_correlation_time='3 dimensionless',
          initial_correlation_length='50 km',
          initial_variance=initial_variance,
          variance_bound=variance_bound,
          tune_variance=tune_variance,
      )

    params = f.init(rng=None)
    grf = f.apply(params, rng=None)
    return f, params, grf

  @parameterized.named_parameters(
      dict(
          testcase_name='Variance0p5',
          initial_variance='0.5 dimensionless',
          expected_variance=0.5,
      ),
      dict(
          testcase_name='VarianceNone',
          initial_variance=None,
          expected_variance=None,
      ),
      dict(
          testcase_name='Variance0p5Bounded',
          initial_variance='0.5 dimensionless',
          expected_variance=0.5,
          variance_bound='1 dimensionless',
      ),
      dict(
          testcase_name='Variance0p5DoNotTune',
          initial_variance=0.5,
          expected_variance=0.5,
          tune_variance=False,
      ),
  )
  def test_parameter_initialization(
      self,
      initial_variance,
      expected_variance,
      variance_bound=None,
      tune_variance=True,
  ):
    _, params, grf = self._make_module_and_params_and_grf(
        initial_variance=initial_variance,
        variance_bound=variance_bound,
        tune_variance=tune_variance,
    )

    expected_raw_params = ['correlation_time_raw', 'correlation_length_raw']
    if tune_variance:
      expected_raw_params += ['variance_raw']

    self.assertCountEqual(['gaussian_random_field_module'], params)
    self.assertCountEqual(
        expected_raw_params,
        params['gaussian_random_field_module'],
    )
    self.assertEqual(
        grf.preferred_representation, stochastic.PreferredRepresentation.MODAL
    )

    np.testing.assert_array_equal(expected_variance, grf.variance)

    # C.f. correlation_time = '3 dimensionless'
    np.testing.assert_allclose(np.exp(-self.dt / 3), grf.phi)

    n_samples = 500
    init_rngs = jax.random.split(jax.random.PRNGKey(5), n_samples)
    initial_states = jax.vmap(grf.unconditional_sample)(init_rngs)
    initial_values = jax.vmap(grf.to_nodal_values)(initial_states.core)

    # Test stats...these are tested more carefully by
    # CenteredLognormalRandomFieldTest, but let's double check here.
    np.testing.assert_allclose(0.0, initial_values.mean(), atol=1e-2)
    np.testing.assert_allclose(
        expected_variance or 0.0,
        initial_values.var(axis=0).mean(),
        rtol=1e-2,
        atol=1e-5,
    )


class CenteredLognormalRandomFieldModuleTest(parameterized.TestCase):

  def _make_module_and_params_and_rf(
      self,
      initial_variance=0.5,
      variance_bound=None,
  ):
    """Makes hk.Module, and params / Loss."""
    coords = coordinate_systems.CoordinateSystem(
        spherical_harmonic.Grid.T21(),
        sigma_coordinates.SigmaCoordinates.equidistant(12),
    )
    physics_specs = primitive_equations.PrimitiveEquationsSpecs.from_si()
    self.dt = physics_specs.nondimensionalize(1 * scales.units.hour)

    @hk.transform
    def f():
      return stochastic.CenteredLognormalRandomFieldModule(
          coords,
          self.dt,
          physics_specs,
          aux_features={},
          initial_correlation_time='1 dimensionless',
          initial_correlation_length='5km',
          initial_variance=initial_variance,
          variance_bound=variance_bound,
      )

    params = f.init(rng=None)
    rf = f.apply(params, rng=None)
    return f, params, rf

  @parameterized.named_parameters(
      dict(
          testcase_name='Variance0p5',
          initial_variance=0.5,
          expected_variance=0.5,
      ),
      dict(
          testcase_name='VarianceNone',
          initial_variance=None,
          expected_variance=None,
      ),
      dict(
          testcase_name='Variance0p5Bounded',
          initial_variance=0.5,
          expected_variance=0.5,
          variance_bound=1,
      ),
  )
  def test_parameter_initialization(
      self,
      initial_variance,
      expected_variance,
      variance_bound=None,
  ):
    _, params, rf = self._make_module_and_params_and_rf(
        initial_variance=initial_variance,
        variance_bound=variance_bound,
    )

    self.assertCountEqual(['centered_lognormal_random_field_module'], params)
    self.assertCountEqual(
        ['correlation_time_raw', 'correlation_length_raw', 'variance_raw'],
        params['centered_lognormal_random_field_module'],
    )
    self.assertEqual(
        rf.preferred_representation, stochastic.PreferredRepresentation.NODAL
    )

    np.testing.assert_array_equal(expected_variance, rf.variance)

    # C.f. correlation_time = '1 dimensionless'
    np.testing.assert_allclose(np.exp(-self.dt / 1), rf.phi)

    n_samples = 500
    init_rngs = jax.random.split(jax.random.PRNGKey(5), n_samples)
    initial_states = jax.vmap(rf.unconditional_sample)(init_rngs)
    initial_values = jax.vmap(rf.to_nodal_values)(initial_states.core)

    # Recall this is Exp(ξ) - 1
    self.assertGreater(initial_values.min(), -1)

    # Test stats...these are tested more carefully by
    # CenteredLognormalRandomFieldTest, but let's double check here.
    np.testing.assert_allclose(0.0, initial_values.mean(), atol=2e-3)
    np.testing.assert_allclose(
        expected_variance or 0.0,
        initial_values.var(axis=0).mean(),
        rtol=1e-2,
        atol=1e-5,
    )


class SumOfGaussianRandomFieldsModuleTest(parameterized.TestCase):

  def _make_module_and_params_and_grf(
      self,
      initial_correlation_times,
      initial_correlation_lengths,
      initial_variances,
      variance_bounds,
  ):
    """Makes hk.Module, and params / Loss."""
    self.coords = coordinate_systems.CoordinateSystem(
        spherical_harmonic.Grid.T21(),
        sigma_coordinates.SigmaCoordinates.equidistant(12),
    )
    physics_specs = primitive_equations.PrimitiveEquationsSpecs.from_si()
    self.dt = physics_specs.nondimensionalize(1 * scales.units.hour)

    @hk.transform
    def f():
      return stochastic.SumOfGaussianRandomFieldsModule(
          self.coords,
          self.dt,
          physics_specs,
          aux_features={},
          initial_correlation_times=initial_correlation_times,
          initial_correlation_lengths=initial_correlation_lengths,
          initial_variances=initial_variances,
          variance_bounds=variance_bounds,
      )

    params = f.init(rng=None)
    grf = f.apply(params, rng=None)
    return f, params, grf

  @parameterized.named_parameters(
      dict(testcase_name='NoBound', variance_bound=None),
      dict(testcase_name='YesBound', variance_bound=5),
  )
  def test_parameter_initialization(self, variance_bound):
    initial_correlation_times = [1.0, 3.0]
    initial_correlation_lengths = [2.0, 4]
    initial_variances = [0.5, 1.5]
    _, params, grf = self._make_module_and_params_and_grf(
        initial_correlation_times,
        initial_correlation_lengths,
        initial_variances,
        variance_bounds=[variance_bound] * 2,
    )

    self.assertCountEqual(
        [
            'sum_of_gaussian_random_fields_module/~/gaussian_random_field_module',
            'sum_of_gaussian_random_fields_module/~/gaussian_random_field_module_1',
        ],
        params,
    )
    for top_key in params:
      self.assertCountEqual(
          ['correlation_time_raw', 'correlation_length_raw', 'variance_raw'],
          params[top_key],
      )
    self.assertEqual(
        grf.preferred_representation, stochastic.PreferredRepresentation.MODAL
    )

    for i, grf_i in enumerate(grf._random_fields):
      # Variance may not match exactly, if the bound is used.
      np.testing.assert_allclose(initial_variances[i], grf_i.variance)
      self.assertEqual(
          np.exp(-self.dt / initial_correlation_times[i]), grf_i.phi
      )
      self.assertEqual(
          (initial_correlation_lengths[i] / self.coords.horizontal.radius) ** 2
          / 2,
          grf_i.kt,
      )


class ParameterConversionTest(parameterized.TestCase):

  def test_convert_hk_param_to_positive_scalar(self):
    initial_value = 0.9

    def convert(param):
      return stochastic.convert_hk_param_to_positive_scalar(
          param, initial_value=initial_value
      )

    np.testing.assert_allclose(convert(0.0), initial_value)
    np.testing.assert_allclose(convert(-20.0), 0.0, atol=1e-7)
    for x in np.linspace(-2.0, 2.0, 10):
      self.assertGreater(convert(x), 0.0)

  def test_convert_hk_param_to_bounded_scalar(self):
    low = 0.1
    high = 2.2
    initial_value = 0.9

    def convert(param):
      return stochastic.convert_hk_param_to_bounded_scalar(
          param, initial_value=initial_value, low=low, high=high
      )

    np.testing.assert_allclose(convert(0.0), initial_value)
    np.testing.assert_allclose(convert(-20.0), low)
    np.testing.assert_allclose(convert(20.0), high)
    for x in np.linspace(-2.0, 2.0, 10):
      self.assertBetween(convert(x), low, high)


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
