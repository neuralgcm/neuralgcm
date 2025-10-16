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

"""Tests for idealized_states."""

from absl.testing import absltest
from absl.testing import parameterized
import coordax as cx
import jax
from neuralgcm.experimental.atmosphere import idealized_states
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import spherical_transforms
from neuralgcm.experimental.core import units


class TestStatesTest(parameterized.TestCase):
  """Tests idealized_states."""

  def setUp(self):
    super().setUp()
    self.sigma = coordinates.SigmaLevels.equidistant(8)
    self.grid = coordinates.LonLatGrid.T42()
    self.ylm_grid = coordinates.SphericalHarmonicGrid.T42()
    self.ylm_map = spherical_transforms.FixedYlmMapping(
        self.grid,
        self.ylm_grid,
        partition_schema_key=None,
        mesh=parallelism.Mesh(),
    )
    self.sim_units = units.get_si_units()

  @parameterized.product(
      as_nodal=[True, False],
      temperature_format=['absolute', 'variation'],
  )
  def test_isothermal_rest_atmosphere(
      self, as_nodal, temperature_format
  ):
    rng = jax.random.PRNGKey(42)
    state = idealized_states.isothermal_rest_atmosphere(
        self.ylm_map,
        self.sigma,
        rng,
        self.sim_units,
        as_nodal=as_nodal,
        temperature_format=temperature_format,
    )
    data_coord = cx.compose_coordinates(
        self.sigma, self.grid if as_nodal else self.ylm_grid
    )
    surface_coord = self.grid if as_nodal else self.ylm_grid
    expected_coords = {
        'vorticity': data_coord,
        'divergence': data_coord,
        'log_surface_pressure': surface_coord,
    }
    if temperature_format == 'absolute':
      expected_coords['temperature'] = data_coord
    else:
      expected_coords['temperature_variation'] = data_coord
    actual_coords = {k: v.coordinate for k, v in state.items()}
    self.assertEqual(actual_coords, expected_coords)

  @parameterized.product(
      as_nodal=[True, False],
      temperature_format=['absolute', 'variation'],
  )
  def test_steady_state_jw(
      self, as_nodal, temperature_format
  ):
    rng = jax.random.PRNGKey(42)
    state = idealized_states.steady_state_jw(
        self.ylm_map,
        self.sigma,
        rng,
        self.sim_units,
        as_nodal=as_nodal,
        temperature_format=temperature_format,
    )
    data_coord = cx.compose_coordinates(
        self.sigma, self.grid if as_nodal else self.ylm_grid
    )
    surface_coord = self.grid if as_nodal else self.ylm_grid
    expected_coords = {
        'vorticity': data_coord,
        'divergence': data_coord,
        'log_surface_pressure': surface_coord,
    }
    if temperature_format == 'absolute':
      expected_coords['temperature'] = data_coord
    else:
      expected_coords['temperature_variation'] = data_coord
    actual_coords = {k: v.coordinate for k, v in state.items()}
    self.assertEqual(actual_coords, expected_coords)

  @parameterized.product(
      as_nodal=[True, False],
      temperature_format=['absolute', 'variation'],
  )
  def test_perturbed_jw(
      self, as_nodal, temperature_format
  ):
    state = idealized_states.perturbed_jw(
        self.ylm_map,
        self.sigma,
        jax.random.key(42),
        self.sim_units,
        as_nodal=as_nodal,
        temperature_format=temperature_format,
    )
    data_coord = cx.compose_coordinates(
        self.sigma, self.grid if as_nodal else self.ylm_grid
    )
    surface_coord = self.grid if as_nodal else self.ylm_grid
    expected_coords = {
        'vorticity': data_coord,
        'divergence': data_coord,
        'log_surface_pressure': surface_coord,
    }
    if temperature_format == 'absolute':
      expected_coords['temperature'] = data_coord
    else:
      expected_coords['temperature_variation'] = data_coord
    actual_coords = {k: v.coordinate for k, v in state.items()}
    self.assertEqual(actual_coords, expected_coords)


if __name__ == '__main__':
  jax.config.update('jax_traceback_filtering', 'off')
  absltest.main()
