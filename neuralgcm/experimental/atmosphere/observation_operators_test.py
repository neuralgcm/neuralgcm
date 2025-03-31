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

"""Tests for atmosphere-specific observation operators."""


from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
from neuralgcm.experimental import coordax as cx
from neuralgcm.experimental.atmosphere import equations
from neuralgcm.experimental.atmosphere import observation_operators
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import orographies
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import units
import numpy as np


class ObservationOperatorsTest(parameterized.TestCase):

  def test_returns_pressure_level_outputs(self):
    sh_grid = coordinates.SphericalHarmonicGrid.T21()
    grid = coordinates.LonLatGrid.T21()
    sigma_levels = coordinates.SigmaLevels.equidistant(12)
    pressure_levels = coordinates.PressureLevels.with_13_era5_levels()
    source_coords = coordinates.DinosaurCoordinates(
        horizontal=sh_grid, vertical=sigma_levels
    )
    target_coords = coordinates.DinosaurCoordinates(
        horizontal=grid, vertical=pressure_levels
    )
    sim_units = units.DEFAULT_UNITS
    mesh = parallelism.Mesh(None)
    ref_temperatures = np.linspace(220, 250, num=sigma_levels.shape[0])
    orography_module = orographies.ModalOrography(
        grid=grid,
        mesh=mesh,
        rngs=nnx.Rngs(0),
    )
    primitive_equations = equations.PrimitiveEquations(
        coords=source_coords,
        sim_units=sim_units,
        reference_temperatures=ref_temperatures,
        orography_module=orography_module,
        mesh=mesh,
    )
    operator = observation_operators.PressureLevelObservationOperator(
        primitive_equations,
        orography_module,
        pressure_levels=pressure_levels,
        sim_units=sim_units,
        observation_correction=None,
        tracer_names=['specific_humidity'],
        mesh=mesh,
    )
    prognostic_fields = {
        'divergence': cx.wrap(np.zeros(source_coords.shape), source_coords),
        'vorticity': cx.wrap(np.zeros(source_coords.shape), source_coords),
        'specific_humidity': cx.wrap(
            np.zeros(source_coords.shape), source_coords
        ),
        'temperature_variation': cx.wrap(
            np.zeros(source_coords.shape), source_coords
        ),
        'log_surface_pressure': cx.wrap(np.zeros(sh_grid.shape), sh_grid),
    }
    query = {
        'temperature': target_coords,
        'u_component_of_wind': target_coords,
        'specific_humidity': target_coords,
    }
    actual = operator.observe(inputs=prognostic_fields, query=query)
    for key in query:
      self.assertEqual(cx.get_coordinate(actual[key]), query[key])


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
