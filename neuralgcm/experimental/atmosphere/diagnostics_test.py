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
from neuralgcm.experimental import pytree_mappings
from neuralgcm.experimental import pytree_transforms
from neuralgcm.experimental import towers
from neuralgcm.experimental.atmosphere import diagnostics as atmos_diagnostics
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import observation_operators
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import pytree_utils
from neuralgcm.experimental.core import spherical_transforms
from neuralgcm.experimental.core import units
import numpy as np


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
    modal_coords = coordinates.DinosaurCoordinates(
        horizontal=coordinates.SphericalHarmonicGrid.T21(),
        vertical=coordinates.SigmaLevels.equidistant(layers=8),
    )
    self.prognostics = {
        'divergence': np.ones(modal_coords.shape),
        'vorticity': np.ones(modal_coords.shape),
        'log_surface_pressure': np.zeros(modal_coords.horizontal.shape),
        'temperature_variation': np.ones(modal_coords.shape),
        'specific_humidity': np.ones(modal_coords.shape),
        'specific_cloud_ice_water_content': np.ones(modal_coords.shape),
        'specific_cloud_liquid_water_content': np.ones(modal_coords.shape),
    }
    self.tendencies = jax.tree.map(jnp.ones_like, self.prognostics)

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
    feature_module = pytree_transforms.ToNodalWithVelocity(ylm_transform)
    tower_factory = functools.partial(
        towers.ColumnTower,
        column_net_factory=nnx.Linear,
    )
    mapping_factory = functools.partial(
        pytree_mappings.ChannelMapping,
        tower_factory=tower_factory,
    )
    embedding_factory = functools.partial(
        pytree_mappings.Embedding,
        feature_module=feature_module,
        input_state_shapes=pytree_utils.shape_structure(self.prognostics),
        mapping_factory=mapping_factory,
        mesh=mesh,
    )
    observation_mapping = pytree_mappings.CoordsStateMapping(
        coords=coordinates.DinosaurCoordinates(
            horizontal=ylm_transform.lon_lat_grid,
            vertical=sigma,
        ),
        surface_field_names=tuple(['evaporation']),
        volume_field_names=tuple(),
        embedding_factory=embedding_factory,
        rngs=nnx.Rngs(0),
        mesh=mesh,
    )
    operator = observation_operators.FixedLearnedObservationOperator(
        observation_mapping
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
