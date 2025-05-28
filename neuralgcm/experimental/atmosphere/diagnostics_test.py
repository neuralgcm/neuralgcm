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

import functools

from absl.testing import absltest
from absl.testing import parameterized
import chex
from flax import nnx
import jax
import jax.numpy as jnp
from neuralgcm.experimental import coordax as cx
from neuralgcm.experimental.atmosphere import diagnostics as atmos_diagnostics
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import learned_transforms
from neuralgcm.experimental.core import observation_operators
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import pytree_utils
from neuralgcm.experimental.core import spherical_transforms
from neuralgcm.experimental.core import towers
from neuralgcm.experimental.core import transforms
from neuralgcm.experimental.core import units


class MockMethod(nnx.Module):
  """Mock method to which diagnostics are attached for testing."""

  def custom_add_half_to_y(self, inputs):
    inputs['y'] += 0.5
    return inputs

  def __call__(self, inputs):
    result = {k: v for k, v in inputs.items()}
    result = self.custom_add_half_to_y(result)
    result = self.custom_add_half_to_y(result)
    return result


class PrecipitationPlusEvaporationTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.sim_units = units.DEFAULT_UNITS
    ylm_grid = coordinates.SphericalHarmonicGrid.T21()
    sigma_levels = coordinates.SigmaLevels.equidistant(layers=8)
    full_modal = cx.compose_coordinates(sigma_levels, ylm_grid)
    ones_like = lambda c: cx.wrap(jnp.ones(c.shape), c)
    self.prognostics = {
        'divergence': ones_like(full_modal),
        'vorticity': ones_like(full_modal),
        'temperature_variation': ones_like(full_modal),
        'specific_humidity': ones_like(full_modal),
        'specific_cloud_ice_water_content': ones_like(full_modal),
        'specific_cloud_liquid_water_content': ones_like(full_modal),
        'log_surface_pressure': ones_like(ylm_grid),
    }
    self.tendencies = {k: 0.1 * v for k, v in self.prognostics.items()}

  def test_extract_precipitation_plus_evaporation(self):
    ylm_transform = spherical_transforms.SphericalHarmonicsTransform(
        lon_lat_grid=coordinates.LonLatGrid.T21(),
        ylm_grid=coordinates.SphericalHarmonicGrid.T21(),
        partition_schema_key=None,
        mesh=parallelism.Mesh(),
    )
    grid = coordinates.LonLatGrid.T21()
    sigma = coordinates.SigmaLevels.equidistant(layers=8)
    precip_plus_evap = atmos_diagnostics.ExtractPrecipitationPlusEvaporation(
        ylm_transform=ylm_transform,
        levels=sigma,
        sim_units=self.sim_units,
    )
    ones_like = lambda c: cx.wrap(jnp.ones(c.shape), c)
    actual = precip_plus_evap(self.tendencies, prognostics=self.prognostics)
    expected_struct = {'precipitation_plus_evaporation_rate': ones_like(grid)}
    chex.assert_trees_all_equal_shapes_and_dtypes(actual, expected_struct)

  def test_extract_precipitation_and_evaporation(self):
    mesh = parallelism.Mesh()
    ylm_transform = spherical_transforms.SphericalHarmonicsTransform(
        lon_lat_grid=coordinates.LonLatGrid.T21(),
        ylm_grid=coordinates.SphericalHarmonicGrid.T21(),
        partition_schema_key=None,
        mesh=mesh,
    )
    grid = coordinates.LonLatGrid.T21()
    sigma = coordinates.SigmaLevels.equidistant(layers=8)
    # Setting up basic observation operator for evaporation.
    state_shapes = pytree_utils.shape_structure(self.prognostics)
    tower_factory = functools.partial(
        towers.UnaryFieldTower.build_using_factories,
        net_in_dims=('d',),
        net_out_dims=('d',),
        neural_net_factory=nnx.Linear,
    )
    surface_observation_operator_transform = (
        learned_transforms.UnaryFieldTowerTransform.build_using_factories(
            input_shapes=state_shapes,
            targets={'evaporation': grid},
            tower_factory=tower_factory,
            dims_to_align=(grid,),
            in_transform=transforms.ToNodal(ylm_transform),
            feature_sharding_schema=None,
            result_sharding_schema=None,
            mesh=mesh,
            rngs=nnx.Rngs(0),
        )
    )
    operator = observation_operators.FixedLearnedObservationOperator(
        surface_observation_operator_transform
    )
    precip_plus_evap = atmos_diagnostics.ExtractPrecipitationPlusEvaporation(
        ylm_transform=ylm_transform,
        levels=sigma,
        sim_units=self.sim_units,
    )
    precip_and_evap = atmos_diagnostics.ExtractPrecipitationAndEvaporation(
        observation_operator=operator,
        operator_query={'evaporation': ylm_transform.lon_lat_grid},
        extract_p_plus_e=precip_plus_evap,
        prognostics_arg_key='prognostics',
        sim_units=self.sim_units,
    )
    ones_like = lambda c: cx.wrap(jnp.ones(c.shape), c)
    actual = precip_and_evap(self.tendencies, prognostics=self.prognostics)
    expected_struct = {
        'precipitation': ones_like(grid),
        'evaporation': ones_like(grid),
    }
    chex.assert_trees_all_equal_shapes_and_dtypes(actual, expected_struct)


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
