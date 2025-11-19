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

"""Tests for atmosphere-specific equations and helpers."""


from absl.testing import absltest
from absl.testing import parameterized
import chex
import coordax as cx
from coordax import testing as cx_testing
import jax
import jax.numpy as jnp
import jax_datetime as jdt
from neuralgcm.experimental.atmosphere import equations
from neuralgcm.experimental.core import coordinates
import numpy as np


class AtmosphereEquationsAndHelpersTests(parameterized.TestCase):

  def test_temperature_linearization_roundtrip_with_grid(self):
    """Tests linearization and delinearization with LonLatGrid."""
    levels = coordinates.SigmaLevels.equidistant(5)
    ref_temps = 280.0 + np.arange(levels.shape[0])
    grid = coordinates.LonLatGrid.T21()
    ylm_grid = coordinates.SphericalHarmonicGrid.T21()
    rng = np.random.RandomState(42)

    temperature = cx.wrap(rng.randn(*levels.shape, *grid.shape), levels, grid)
    inputs = {
        'temperature': temperature,
        'time': cx.wrap(jdt.to_datetime('2025-01-01')),
        'sh': cx.wrap(rng.randn(*ylm_grid.shape), ylm_grid),
    }

    linearize = equations.get_temperature_linearization_transform(
        ref_temperatures=ref_temps, levels=levels
    )
    actual_linearized = linearize(inputs)

    with self.subTest('direct'):
      self.assertNotIn('temperature', actual_linearized)
      self.assertIn('temperature_variation', actual_linearized)
      # check that other fields are untouched.
      expected_others = {k: v for k, v in inputs.items() if k != 'temperature'}
      actual_others = {
          k: v
          for k, v in actual_linearized.items()
          if k != 'temperature_variation'
      }
      chex.assert_trees_all_equal(actual_others, expected_others)
      expected_variation = temperature - cx.wrap(ref_temps, levels)
      cx_testing.assert_fields_allclose(
          actual_linearized['temperature_variation'], expected_variation
      )

    with self.subTest('roundtrip'):
      delinearize = equations.get_temperature_delinearization_transform(
          ref_temperatures=ref_temps, levels=levels
      )
      actual_roundtrip = delinearize(actual_linearized)
      expected = jax.tree.map(jnp.asarray, inputs)
      chex.assert_trees_all_close(actual_roundtrip, expected, atol=5e-5)

  def test_temperature_linearization_roundtrip_with_ylm_grid(self):
    """Tests linearization and delinearization with SphericalHarmonicGrid."""
    levels = coordinates.SigmaLevels.equidistant(5)
    ref_temps = 280.0 + np.arange(levels.shape[0])
    ylm_grid = coordinates.SphericalHarmonicGrid.T21()
    grid = coordinates.LonLatGrid.T21()
    rng = np.random.RandomState(42)

    temperature = cx.wrap(
        rng.randn(*levels.shape, *ylm_grid.shape), levels, ylm_grid
    )
    inputs = {
        'temperature': temperature,
        'time': cx.wrap(jdt.to_datetime('2025-01-01')),
        'lonlat': cx.wrap(rng.randn(*grid.shape), grid),
    }

    linearize = equations.get_temperature_linearization_transform(
        ref_temperatures=ref_temps, levels=levels
    )
    actual_linearized = linearize(inputs)

    with self.subTest('direct'):
      self.assertNotIn('temperature', actual_linearized)
      self.assertIn('temperature_variation', actual_linearized)
      # check that other fields are untouched.
      expected_others = {k: v for k, v in inputs.items() if k != 'temperature'}
      actual_others = {
          k: v
          for k, v in actual_linearized.items()
          if k != 'temperature_variation'
      }
      chex.assert_trees_all_equal(actual_others, expected_others)

    with self.subTest('roundtrip'):
      delinearize = equations.get_temperature_delinearization_transform(
          ref_temperatures=ref_temps, levels=levels
      )
      actual_roundtrip = delinearize(actual_linearized)
      expected = jax.tree.map(jnp.asarray, inputs)
      chex.assert_trees_all_close(actual_roundtrip, expected, atol=5e-5)


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
