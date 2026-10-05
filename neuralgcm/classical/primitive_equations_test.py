# Copyright 2026 Google LLC
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

"""Tests for primitive_equations model instantiation."""

import functools
from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import numpy as np
from neuralgcm.classical import primitive_equations
from terrax.atmosphere import equations as atmos_equations
from terrax.core import coordinates
from terrax.core import orographies
from terrax.core import spatial_filters
from terrax.core import spherical_harmonics
from terrax.core import time_integrators
from terrax.core import units


def _make_modal_filter(
    ylm_map: spherical_harmonics.FixedYlmMapping,
    dt: float,
    sim_units: units.SimUnits,
    smooth_timescale: str = '120 minutes',
    sharp_timescale: str = '6 minutes',
) -> spatial_filters.SequentialModalFilter:
  """Constructs a two-scale exponential modal filter for substep `dt`."""
  smooth = spatial_filters.ExponentialModalFilter.from_timescale(
      ylm_map=ylm_map,
      dt=dt,
      timescale=smooth_timescale,
      order=3,
      cutoff=0.0,
      sim_units=sim_units,
  )
  sharp = spatial_filters.ExponentialModalFilter.from_timescale(
      ylm_map=ylm_map,
      dt=dt,
      timescale=sharp_timescale,
      order=10,
      cutoff=0.4,
      sim_units=sim_units,
  )
  return spatial_filters.SequentialModalFilter([sharp, smooth], ylm_map=ylm_map)


class PrimitiveEquationsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.sim_units = units.SI_UNITS
    self.grid = coordinates.LonLatGrid.T21()
    self.ylm_grid = coordinates.SphericalHarmonicGrid.T21()
    self.ylm_map = spherical_harmonics.FixedYlmMapping(
        self.grid,
        self.ylm_grid,
        radius=self.sim_units.radius,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='semi_lagrangian_hybrid',
          levels=coordinates.HybridLevels.analytic_ecmwf_like(8),
          dycore_equation_cls=atmos_equations.SemiLagrangianPrimitiveEquations,
          time_integrator_cls=functools.partial(
              time_integrators.SemiLagrangianCrankNicolsonRK2,
              off_centering=0.1,
          ),
          model_timestep=np.timedelta64(30, 'm'),
          num_substeps=1,
          with_held_suarez=False,
      ),
      dict(
          testcase_name='eulerian_sigma',
          levels=coordinates.SigmaLevels.equidistant(8),
          dycore_equation_cls=atmos_equations.PrimitiveEquations,
          time_integrator_cls=time_integrators.ImexRk3Sil,
          model_timestep=np.timedelta64(10, 'm'),
          num_substeps=2,
          with_held_suarez=False,
      ),
      dict(
          testcase_name='semi_lagrangian_with_held_suarez',
          levels=coordinates.SigmaLevels.equidistant(8),
          dycore_equation_cls=atmos_equations.SemiLagrangianPrimitiveEquations,
          time_integrator_cls=functools.partial(
              time_integrators.SemiLagrangianCrankNicolsonRK2,
              off_centering=0.1,
          ),
          model_timestep=np.timedelta64(30, 'm'),
          num_substeps=1,
          with_held_suarez=True,
      ),
  )
  def test_model_instantiation(
      self,
      levels,
      dycore_equation_cls,
      time_integrator_cls,
      model_timestep,
      num_substeps,
      with_held_suarez,
  ):
    orography = orographies.ModalOrography(
        ylm_map=self.ylm_map, rngs=nnx.Rngs(0)
    )
    dt = self.sim_units.nondimensionalize_timedelta64(
        model_timestep / num_substeps
    )
    modal_filter = _make_modal_filter(self.ylm_map, dt, self.sim_units)

    if with_held_suarez:
      explicit_equations = (
          atmos_equations.HeldSuarezForcing(
              self.ylm_map,
              levels=levels,
              sim_units=self.sim_units,
              reference_temperatures=atmos_equations.get_reference_temperature(
                  levels
              ),
          ),
      )
    else:
      explicit_equations = ()

    model = primitive_equations.SpectralPrimitiveEquationsModel(
        ylm_map=self.ylm_map,
        levels=levels,
        model_timestep=model_timestep,
        orography=orography,
        time_integrator_cls=time_integrator_cls,
        modal_filter=modal_filter,
        sim_units=self.sim_units,
        num_substeps=num_substeps,
        dycore_equation_cls=dycore_equation_cls,
        explicit_equations=explicit_equations,
    )

    self.assertEqual(model.timestep, model_timestep)
    self.assertIn('prognostics', model.inputs_spec)
    self.assertIn('nondim_prognostics', model.inputs_spec)


if __name__ == '__main__':
  absltest.main()
