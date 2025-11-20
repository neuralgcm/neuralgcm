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

"""Tests for atmosphere-specific diagnostics modules and utilities."""


from absl.testing import absltest
from absl.testing import parameterized
import chex
import coordax as cx
import jax
import jax.numpy as jnp
from neuralgcm.experimental.atmosphere import fixers
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import spherical_harmonics
from neuralgcm.experimental.core import units


class EnergyFixersTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.sim_units = units.DEFAULT_UNITS
    self.ylm_grid = coordinates.SphericalHarmonicGrid.T21()
    self.lon_lat_grid = coordinates.LonLatGrid.T21()
    self.sigma_levels = coordinates.SigmaLevels.equidistant(layers=8)
    self.mesh = parallelism.Mesh()
    self.ylm_map = spherical_harmonics.FixedYlmMapping(
        lon_lat_grid=self.lon_lat_grid,
        ylm_grid=self.ylm_grid,
        partition_schema_key=None,
        mesh=self.mesh,
    )
    full_modal = cx.compose_coordinates(self.sigma_levels, self.ylm_grid)
    ones_like = lambda c: cx.wrap(jnp.ones(c.shape), c)
    self.prognostics = {
        'divergence': ones_like(full_modal),
        'vorticity': ones_like(full_modal),
        'temperature': ones_like(full_modal),
        'specific_humidity': ones_like(full_modal),
        'specific_cloud_ice_water_content': ones_like(full_modal),
        'specific_cloud_liquid_water_content': ones_like(full_modal),
        'log_surface_pressure': ones_like(self.ylm_grid),
    }
    self.tendencies = {k: 0.1 * v for k, v in self.prognostics.items()}

  def test_temperature_energy_adjustment_shape_and_dtype(self):
    temp_adjustment = fixers.TemperatureAdjustmentForEnergyBalance(
        ylm_map=self.ylm_map,
        levels=self.sigma_levels,
        sim_units=self.sim_units,
    )
    imbalance = {
        'imbalance': cx.wrap(
            jnp.ones(self.lon_lat_grid.shape), self.lon_lat_grid
        )
    }
    tendencies = jax.tree.map(lambda x: x, self.tendencies)
    adjusted_tendencies = temp_adjustment(
        imbalance, tendencies, prognostics=self.prognostics
    )
    chex.assert_trees_all_equal_shapes_and_dtypes(
        adjusted_tendencies, self.tendencies
    )


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
