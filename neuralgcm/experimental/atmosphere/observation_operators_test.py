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
from neuralgcm.experimental import jax_datetime as jdt
from neuralgcm.experimental.atmosphere import equations
from neuralgcm.experimental.atmosphere import observation_operators
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import orographies
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import units
import numpy as np


class ObservationOperatorsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    n_sigma = 12
    self.sh_grid = coordinates.SphericalHarmonicGrid.T21()
    self.grid = coordinates.LonLatGrid.T21()
    self.input_sigma_levels = coordinates.SigmaLevels.equidistant(n_sigma)
    self.source_coords = coordinates.DinosaurCoordinates(
        horizontal=self.sh_grid, vertical=self.input_sigma_levels
    )
    self.sim_units = units.DEFAULT_UNITS
    self.mesh = parallelism.Mesh(None)
    self.orography_module = orographies.ModalOrography(
        grid=self.grid,
        mesh=self.mesh,
        rngs=nnx.Rngs(0),
    )
    self.ref_temperatures = np.linspace(220, 250, num=n_sigma)
    self.primitive_equations = equations.PrimitiveEquations(
        coords=self.source_coords,
        sim_units=self.sim_units,
        reference_temperatures=self.ref_temperatures,
        orography_module=self.orography_module,
        mesh=self.mesh,
    )
    zero_like = lambda c: cx.wrap(np.zeros(c.shape), c)
    self.prognostic_fields = {
        'divergence': zero_like(self.source_coords),
        'vorticity': zero_like(self.source_coords),
        'specific_humidity': zero_like(self.source_coords),
        'temperature_variation': zero_like(self.source_coords),
        'log_surface_pressure': zero_like(self.sh_grid),
        'time': cx.wrap(jdt.to_datetime('2001-01-01')),
    }

  def test_returns_pressure_level_outputs(self):
    pressure_levels = coordinates.PressureLevels.with_13_era5_levels()
    target_coords = coordinates.DinosaurCoordinates(
        horizontal=self.grid, vertical=pressure_levels
    )
    operator = observation_operators.PressureLevelObservationOperator(
        self.primitive_equations,
        self.orography_module,
        pressure_levels=pressure_levels,
        sim_units=self.sim_units,
        observation_correction=None,
        tracer_names=['specific_humidity'],
        mesh=self.mesh,
    )
    query = {
        'temperature': target_coords,
        'u_component_of_wind': target_coords,
        'specific_humidity': target_coords,
    }
    actual = operator.observe(inputs=self.prognostic_fields, query=query)
    for key in query:
      self.assertEqual(cx.get_coordinate(actual[key]), query[key])

  def test_returns_sigma_level_outputs(self):
    target_sigma_levels = coordinates.SigmaLevels.equidistant(10)
    target_coords = coordinates.DinosaurCoordinates(
        horizontal=self.grid, vertical=target_sigma_levels
    )
    operator = observation_operators.SigmaLevelObservationOperator(
        self.primitive_equations,
        self.orography_module,
        sigma_levels=target_sigma_levels,
        sim_units=self.sim_units,
        observation_correction=None,
        tracer_names=['specific_humidity'],
        mesh=self.mesh,
    )
    query = {
        'temperature': target_coords,
        'u_component_of_wind': target_coords,
        'specific_humidity': target_coords,
    }
    actual = operator.observe(inputs=self.prognostic_fields, query=query)
    for key in query:
      self.assertEqual(cx.get_coordinate(actual[key]), query[key])


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
