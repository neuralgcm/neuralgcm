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

"""Observation operators for NeuralGCM models."""

import coordax as cx
import jax.numpy as jnp
from terrax.core import observation_operators
from terrax.core import spherical_harmonics
from terrax.core import transforms


def primitive_equations_nodal_operator(
    ylm_map: spherical_harmonics.FixedYlmMapping,
) -> observation_operators.TransformObservationOperator:
  """Returns an operator observing nodal fields from primitive equation prognostics.

  Converts modal prognostic fields on `ylm_map.ylm_grid` to nodal fields on
  `ylm_map.lon_lat_grid` (passing through any already-nodal fields), computes
  `u_component_of_wind` and `v_component_of_wind` from `divergence` and
  `vorticity`, and computes `surface_pressure` by exponentiating
  `log_surface_pressure`.

  Args:
    ylm_map: Spherical harmonic mapping between the model's modal and nodal
      grids.
  """
  to_nodal = transforms.ModalToNodal(ylm_map, include_remaining=True)
  surface_pressure = transforms.Sequential([
      transforms.ApplyFnToKeys(cx.cpmap(jnp.exp), ['log_surface_pressure']),
      transforms.RenameKeys({'log_surface_pressure': 'surface_pressure'}),
  ])
  transform = transforms.Sequential([  # pyrefly: ignore[bad-argument-type]
      transforms.SelectKeys('time', invert=True),
      transforms.Merge({
          'uv': transforms.VelocityFromDivCurl(ylm_map),
          'nodal': to_nodal,
      }),
      transforms.Merge({
          'nodal': transforms.Identity(),  # pyrefly: ignore[bad-assignment]
          'sp': surface_pressure,
      }),
  ])
  return observation_operators.TransformObservationOperator(transform)
