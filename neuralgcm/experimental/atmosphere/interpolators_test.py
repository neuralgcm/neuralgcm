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

"""Tests for atmosphere-specific interpolation routines."""

from absl.testing import absltest
from absl.testing import parameterized
import coordax as cx
from coordax import testing as cx_testing
from dinosaur import vertical_interpolation
import jax
from neuralgcm.experimental.atmosphere import interpolators
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import typing
from neuralgcm.experimental.core import units
import numpy as np


class LinearOnPressureTest(parameterized.TestCase):
  """Tests LinearOnPressure interpolator module."""

  def test_to_sigma_matches_dinosaur_vertical_interpolation(self):
    """Tests LinearOnPressure against vertical_interpolation sigma out."""
    source_levels = coordinates.PressureLevels.with_13_era5_levels()
    target_levels = coordinates.SigmaLevels.equidistant(20)
    grid = coordinates.LonLatGrid.T21()
    rng = np.random.RandomState(42)
    coords = cx.compose_coordinates(source_levels, grid)
    field = cx.wrap(rng.randn(*coords.shape), coords)
    surface_pressure = cx.wrap(100000 + 10000 * rng.randn(*grid.shape), grid)
    inputs = {'field': field, 'surface_pressure': surface_pressure}

    regridder = interpolators.LinearOnPressure(target_levels, 'linear')
    actual = regridder(inputs)
    dino_interp_fn = vertical_interpolation.vectorize_vertical_interpolation(
        vertical_interpolation.linear_interp_with_linear_extrap
    )
    expected_data = vertical_interpolation.interp_pressure_to_sigma(
        {'field': field.data},
        pressure_coords=source_levels.pressure_levels,
        sigma_coords=target_levels.sigma_levels,
        surface_pressure=surface_pressure.data / 100,  # dinosaur uses hPa.
        interpolate_fn=dino_interp_fn,
    )
    expected_coords = cx.compose_coordinates(target_levels, grid)
    expected_field = cx.wrap(expected_data['field'], expected_coords)
    cx_testing.assert_fields_allclose(
        actual['field'], expected_field, atol=1e-5
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='linear',
          extrapolation='linear',
          interp_fn=vertical_interpolation.linear_interp_with_linear_extrap,
      ),
      dict(
          testcase_name='constant',
          extrapolation='constant',
          interp_fn=vertical_interpolation.interp,
      ),
  )
  def test_to_pressure_matches_dinosaur_vertical_interpolation(
      self, extrapolation, interp_fn
  ):
    """Tests LinearOnPressure against vertical_interpolation pressure out."""
    source_levels = coordinates.SigmaLevels.equidistant(20)
    target_levels = coordinates.PressureLevels.with_13_era5_levels()
    grid = coordinates.LonLatGrid.T21()
    rng = np.random.RandomState(42)
    coords = cx.compose_coordinates(source_levels, grid)
    field = cx.wrap(rng.randn(*coords.shape), coords)
    surface_pressure = cx.wrap(100000 + 10000 * rng.randn(*grid.shape), grid)
    inputs = {'field': field, 'surface_pressure': surface_pressure}

    regridder = interpolators.LinearOnPressure(target_levels, extrapolation)
    actual = regridder(inputs)
    dino_interp_fn = vertical_interpolation.vectorize_vertical_interpolation(
        interp_fn
    )
    expected_data = vertical_interpolation.interp_sigma_to_pressure(
        {'field': field.data},
        pressure_coords=target_levels.pressure_levels,
        sigma_coords=source_levels.sigma_levels,
        surface_pressure=surface_pressure.data / 100,  # dinosaur uses hPa.
        interpolate_fn=dino_interp_fn,
    )
    expected_coords = cx.compose_coordinates(target_levels, grid)
    expected_field = cx.wrap(expected_data['field'], expected_coords)
    cx_testing.assert_fields_allclose(
        actual['field'], expected_field, atol=1e-5
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='linear',
          extrapolation='linear',
          expected_data=np.array([10.0, 25.0, 35.0, 45.0, 55.0]),
      ),
      dict(
          testcase_name='constant',
          extrapolation='constant',
          expected_data=np.array([10.0, 25.0, 30.0, 30.0, 30.0]),
      ),
  )
  def test_to_sigma_from_pressure_1d(self, extrapolation, expected_data):
    """Tests interpolation to sigma from 1d pressure level inputs."""
    source_levels = coordinates.PressureLevels([100, 200, 400])
    target_levels = coordinates.SigmaLevels.equidistant(5)
    surface_pressure = 1000.0 * 100  # hPa to Pa.
    field = cx.wrap(np.array([10.0, 20.0, 30.0]), source_levels)
    inputs = {'field': field, 'surface_pressure': cx.wrap(surface_pressure)}

    regridder = interpolators.LinearOnPressure(target_levels, extrapolation)
    actual = regridder(inputs)['field']

    expected = cx.wrap(expected_data, target_levels)
    cx_testing.assert_fields_allclose(actual, expected, atol=1e-5)

  def test_to_sigma_from_pressure_1d_with_sim_units(self):
    """Tests interpolation to sigma with sim_units and nondim pressure."""
    source_levels = coordinates.PressureLevels([100, 200, 400])
    target_levels = coordinates.SigmaLevels.equidistant(5)
    surface_pressure_pa = 1000.0 * 100  # hPa to Pa.
    field = cx.wrap(np.array([10.0, 20.0, 30.0]), source_levels)
    inputs_hpa = {
        'field': field,
        'surface_pressure': cx.wrap(surface_pressure_pa),
    }

    regridder_hpa = interpolators.LinearOnPressure(target_levels, 'linear')
    result_hpa = regridder_hpa(inputs_hpa)['field']

    sim_units = units.DEFAULT_UNITS
    surface_pressure_nondim = sim_units.nondimensionalize(
        surface_pressure_pa * typing.units.Pa
    )
    inputs_nondim = {
        'field': field,
        'surface_pressure': cx.wrap(surface_pressure_nondim),
    }
    regridder_nondim = interpolators.LinearOnPressure(
        target_levels, 'linear', sim_units=sim_units
    )
    result_nondim = regridder_nondim(inputs_nondim)['field']

    cx_testing.assert_fields_allclose(result_nondim, result_hpa, atol=1e-5)

  def test_to_sigma_from_sigma_1d(self):
    """Tests interpolation to sigma from 1d sigma level inputs."""
    source_levels = coordinates.SigmaLevels.from_centers(
        np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    )
    target_levels = coordinates.SigmaLevels.from_centers(
        np.array([0.1, 0.25, 0.4, 0.6, 0.85])
    )
    surface_pressure = 1000.0 * 100  # hPa to Pa.
    field_data = np.array([1.0, 2.0, 4.0, 6.0, 8.0])
    field = cx.wrap(field_data, source_levels)
    inputs = {'field': field, 'surface_pressure': cx.wrap(surface_pressure)}

    regridder = interpolators.LinearOnPressure(target_levels=target_levels)
    actual = regridder(inputs)['field']

    expected_data = np.array([1.0, 1.75, 3.0, 5.0, 7.5])
    expected = cx.wrap(expected_data, target_levels)
    cx_testing.assert_fields_allclose(actual, expected, atol=1e-5)

  def test_to_pressure_from_pressure_1d(self):
    """Tests interpolation to pressure from pressure levels."""
    source_levels = coordinates.PressureLevels([100, 200, 400])
    target_levels = coordinates.PressureLevels([100, 150, 300])
    field = cx.wrap(np.array([10.0, 20.0, 30.0]), source_levels)
    inputs = {'field': field}

    regridder = interpolators.LinearOnPressure(target_levels=target_levels)
    actual = regridder(inputs)['field']
    expected_data = np.array([10.0, 15.0, 25.0])
    expected = cx.wrap(expected_data, target_levels)
    cx_testing.assert_fields_allclose(actual, expected, atol=1e-5)

  def test_to_pressure_from_sigma_1d(self):
    """Tests interpolation to pressure from sigma levels."""
    source_levels = coordinates.SigmaLevels.from_centers(
        np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    )
    target_levels = coordinates.PressureLevels([100, 250, 400, 600, 850])
    surface_pressure = 1000.0 * 100  # hPa to Pa.
    field_data = np.array([1.0, 2.0, 4.0, 6.0, 8.0])
    field = cx.wrap(field_data, source_levels)
    inputs = {'field': field, 'surface_pressure': cx.wrap(surface_pressure)}

    regridder = interpolators.LinearOnPressure(target_levels=target_levels)
    actual = regridder(inputs)['field']
    expected_data = np.array([1.0, 1.75, 3.0, 5.0, 7.5])
    expected = cx.wrap(expected_data, target_levels)
    cx_testing.assert_fields_allclose(actual, expected, atol=1e-5)

  @parameterized.named_parameters(
      dict(
          testcase_name='linear',
          extrapolation='linear',
          # P_target = [100, 300, 500, 700, 900]
          # P_source = [200, 500, 800]
          # Values   = [ 20,  50,  80]
          # Slope = 0.1
          # 100: 20 + (100-200)*0.1 = 10 (extrap)
          # 300: 20 + (300-200)*0.1 = 30 (interp)
          # 500: 50                 = 50 (exact)
          # 700: 50 + (700-500)*0.1 = 70 (interp)
          # 900: 80 + (900-800)*0.1 = 90 (extrap)
          expected_data=np.array([10.0, 30.0, 50.0, 70.0, 90.0]),
      ),
      dict(
          testcase_name='constant',
          extrapolation='constant',
          # 100: clamped to 200 -> 20 (extrap)
          # 300: 30 (interp)
          # 500: 50 (exact)
          # 700: 70 (interp)
          # 900: clamped to 800 -> 80 (extrap)
          expected_data=np.array([20.0, 30.0, 50.0, 70.0, 80.0]),
      ),
  )
  def test_to_sigma_from_hybrid_1d(self, extrapolation, expected_data):
    """Tests interpolation to sigma from 1d hybrid level inputs."""
    # Define Hybrid Levels (Source)
    # P = A + B * SurfacePressure
    # Boundaries:
    # 0: A=0,   B=0   -> P=0
    # 1: A=200, B=0.2 -> P=200 + 0.2*1000 = 400
    # 2: A=100, B=0.5 -> P=100 + 0.5*1000 = 600
    # 3: A=0,   B=1.0 -> P=0   + 1.0*1000 = 1000
    # Centers: 200, 500, 800
    a_boundaries = np.array([0.0, 200.0, 100.0, 0.0])
    b_boundaries = np.array([0.0, 0.2, 0.5, 1.0])
    source_levels = coordinates.HybridLevels(a_boundaries, b_boundaries)

    # Define Sigma Levels (Target)
    # 5 equidistant levels yield centers at 0.1, 0.3, 0.5, 0.7, 0.9
    target_levels = coordinates.SigmaLevels.equidistant(5)

    surface_pressure = 1000.0 * 100  # hPa to Pa.

    # Source Pressures: 200, 500, 800
    # Target Pressures: 100, 300, 500, 700, 900

    # Define field values linearly proportional to pressure (F = P/10)
    # Source Field: 20, 50, 80
    field = cx.wrap(np.array([20.0, 50.0, 80.0]), source_levels)
    inputs = {'field': field, 'surface_pressure': cx.wrap(surface_pressure)}

    regridder = interpolators.LinearOnPressure(target_levels, extrapolation)
    actual = regridder(inputs)['field']

    expected = cx.wrap(expected_data, target_levels)
    cx_testing.assert_fields_allclose(actual, expected, atol=1e-5)

  @parameterized.named_parameters(
      dict(
          testcase_name='linear',
          extrapolation='linear',
          expected_data=np.array([8.0, 20.0, 40.75]),
      ),
      dict(
          testcase_name='constant',
          extrapolation='constant',
          expected_data=np.array([10.0, 20.0, 40.75]),
      ),
  )
  def test_to_hybrid_from_sigma_1d(self, extrapolation, expected_data):
    """Tests interpolation to hybrid from 1d sigma level inputs."""
    source_levels = coordinates.SigmaLevels.equidistant(5)
    a_boundaries = np.array([0, 20, 80, 150])
    b_boundaries = np.array([0, 0.1, 0.4, 0.8])
    target_levels = coordinates.HybridLevels(a_boundaries, b_boundaries)
    surface_pressure = 1000.0 * 100  # hPa to Pa.
    field = cx.wrap(np.array([10.0, 20.0, 30.0, 40.0, 50.0]), source_levels)
    inputs = {'field': field, 'surface_pressure': cx.wrap(surface_pressure)}

    regridder = interpolators.LinearOnPressure(target_levels, extrapolation)
    actual = regridder(inputs)['field']

    expected = cx.wrap(expected_data, target_levels)
    cx_testing.assert_fields_allclose(actual, expected, atol=1e-5)


class ConservativeOnPressureTest(parameterized.TestCase):
  """Tests ConservativeOnPressure module."""

  def test_to_sigma_from_pressure_levels_1d(self):
    """Tests regridding from pressure with a simple case."""
    source_levels = coordinates.PressureLevels([100, 300, 500])
    # boundaries are: (0, 200), (200, 400), (400, 600).
    surface_pressure = 600.0 * 100  # hPa to Pa.
    target_levels = coordinates.SigmaLevels.equidistant(2)
    # sigma levels correspond to pressure boundaries: (0, 300), (300, 600).
    field_data = np.array([10.0, 20.0, 30.0])
    field = cx.wrap(field_data, source_levels)
    inputs = {'field': field, 'surface_pressure': cx.wrap(surface_pressure)}

    regridder = interpolators.ConservativeOnPressure(target_levels)
    actual = regridder(inputs)['field']
    # (10 * 200 + 20 * 100) / 300 = 40 / 3
    # (20 * 100 + 30 * 200) / 300 = 80 / 3
    expected_data = np.array([40.0 / 3.0, 80.0 / 3.0])
    expected = cx.wrap(expected_data, target_levels)
    cx_testing.assert_fields_allclose(actual, expected, atol=1e-6)

  def test_to_sigma_from_sigma_levels_1d(self):
    """Tests regridding from sigma levels with a simple case."""
    source_levels = coordinates.SigmaLevels.from_centers(
        np.array([0.1, 0.3, 0.7])
    )  # boundaries at [0.0, 0.2, 0.4, 1.0]
    target_levels = coordinates.SigmaLevels.from_centers(
        np.array([0.25, 0.75])
    )  # boundaries at [0.0, 0.5, 1.0]
    surface_pressure = 1000.0 * 100  # hPa to Pa.
    field_data = np.array([10.0, 20.0, 30.0])
    field = cx.wrap(field_data, source_levels)
    inputs = {'field': field, 'surface_pressure': cx.wrap(surface_pressure)}

    regridder = interpolators.ConservativeOnPressure(target_levels)
    actual = regridder(inputs)['field']

    # Source cells (sigma): (0, 0.2), (0.2, 0.4), (0.4, 1.0), values 10, 20, 30.
    # Target cell 1 (sigma): (0, 0.5).
    # Value = (10 * 0.2 + 20 * 0.2 + 30 * 0.1) / 0.5 = 18.0
    # Target cell 2 (sigma): (0.5, 1.0).
    # Value = (30 * 0.5) / 0.5 = 30.0
    expected_data = np.array([18.0, 30.0])
    expected = cx.wrap(expected_data, target_levels)
    cx_testing.assert_fields_allclose(actual, expected, atol=1e-6)


class GetSurfacePressureTest(parameterized.TestCase):
  """Tests get_surface_pressure function."""

  def test_get_surface_pressure(self):
    """Tests get_surface_pressure against dinosaur implementation."""
    levels = np.array([100, 200, 300, 400, 500])
    orography = np.array([[0, 5, 10, 15]])
    geopotential_data = np.moveaxis(
        [[
            [400, 250, 150, 50, -50],
            [1000, 900, 140, 40, 20],
            [500, 400, 300, 200, 100],
            [600, 500, 400, 300, 200],
        ]],
        -1,
        0,
    )
    pressure_levels = coordinates.PressureLevels(levels)
    grid = coordinates.LonLatGrid(longitude_nodes=1, latitude_nodes=4)
    coords = cx.compose_coordinates(pressure_levels, grid)
    geopotential = cx.wrap(geopotential_data, coords)
    gravity_acceleration = 10
    geopotential_at_surface = cx.wrap(orography * gravity_acceleration, grid)
    expected_data = np.array([[[45000, 39000, 50000, 55000]]])
    expected = cx.wrap(np.squeeze(expected_data).reshape(1, 4), grid)
    actual = interpolators.get_surface_pressure(
        geopotential, geopotential_at_surface
    )
    cx_testing.assert_fields_allclose(actual, expected, atol=1e-6)


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
