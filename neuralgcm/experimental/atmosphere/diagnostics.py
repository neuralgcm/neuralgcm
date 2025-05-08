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

"""Module-based API for calculating diagnostics of NeuralGCM models."""

import dataclasses

from typing import Literal

from dinosaur import sigma_coordinates
from flax import nnx
import jax.numpy as jnp
from neuralgcm.experimental import coordax as cx
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import nnx_compat
from neuralgcm.experimental.core import observation_operators
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import spherical_transforms
from neuralgcm.experimental.core import typing
from neuralgcm.experimental.core import units


@nnx_compat.dataclass
class ExtractPrecipitationPlusEvaporation(nnx.Module):
  """Diagnoses precipitation plus evaporation rate from physics tendencies.

  The computation of P + E is based on the integration of non-dynamical moisture
  tendency over the vertical column. We define precipitation and evaporation
  rates as the rate of change of non-atmospheric moisture, i.e. resulting in
  positive values for precipitation and negative values for evaporation. This
  is in line with how these quantities are often defined in datasets like ERA5
  or IMERG. This is also in line with the convention of having "downward"
  fluxes as positive and "upward" fluxes as negative.
  """

  ylm_transform: spherical_transforms.SphericalHarmonicsTransform
  levels: coordinates.SigmaLevels
  sim_units: units.SimUnits
  moisture_species: tuple[str, ...] = (
      'specific_humidity',
      'specific_cloud_ice_water_content',
      'specific_cloud_liquid_water_content',
  )
  prognostics_arg_key: str | int = 'prognostics'

  def _compute_p_plus_e_rate(self, tendencies, prognostics) -> typing.Array:
    to_nodal = self.ylm_transform.to_nodal_array
    p_surface = jnp.exp(to_nodal(prognostics['log_surface_pressure']))
    scale = p_surface / self.sim_units.gravity_acceleration
    moisture_tendencies_nodal = [
        to_nodal(v) for k, v in tendencies.items() if k in self.moisture_species
    ]
    moisture_tendencies_sum = sum(moisture_tendencies_nodal)
    # TODO(dkochkov): add sigma integral method to SigmaLevels.
    p_plus_e = -scale * sigma_coordinates.sigma_integral(
        moisture_tendencies_sum,
        self.levels.sigma_levels,
        keepdims=False,
    )
    return p_plus_e

  def __call__(self, inputs, *args, **kwargs) -> dict[str, cx.Field]:
    tendencies = inputs
    if isinstance(self.prognostics_arg_key, int):
      prognostics = args[self.prognostics_arg_key]
    else:
      prognostics = kwargs.get(self.prognostics_arg_key)
    p_plus_e_rate = self._compute_p_plus_e_rate(tendencies, prognostics)
    p_plus_e_rate = cx.wrap(p_plus_e_rate, self.ylm_transform.nodal_grid)
    return {'precipitation_plus_evaporation_rate': p_plus_e_rate}


PrecipitationScales = Literal['rate', 'cumulative', 'mass_rate']


@nnx_compat.dataclass
class ExtractPrecipitationAndEvaporation(nnx.Module):
  """Extracts balanced precipitation and evaporation values.

  This module can be attached in diagnostics that have access to both
  parameterization tendencies and model state to infer balanced precipitation
  and evaporation. We use `observation_operator` to predict on of the
  two (either `precipitation` or `evaporation`) and infer the other from the
  precipitation_plus_evaporation calculation. The mode is defined by the
  provided operator, query and inference variable indicating which variable
  will be computed from the balance equations.

  Attributes:
    observation_operator: Observation operator used to predict one of the two
      variables from the balance equations.
    operator_query: Query used for the observation operator.
    extract_p_plus_e: Module that extracts precipitation plus evaporation from
      tendencies and prognostics.
    prognostics_arg_key: Key or index of the prognostics argument in the call
      signature.
    precipitation_scaling: Scaling strategy for the precipitation field. Must be
      one of `rate`, `mass_rate` or `cumulative`. If using `cumulative` scaling,
      `dt` must be set.
    evaporation_scaling: Scaling strategy for the evaporation field. Must be
      one of `rate`, `mass_rate` or `cumulative`. If using `cumulative` scaling,
      `dt` must be set.
    dt: Timestep by which the precipitation is scaled (only used when
      `precipitation_scaling` is set to `cumulative`).
    sim_units: Object defining nondimensionalization and physical constants.
    precipitation_key: Key under which the precipitation field is stored in the
      output.
    evaporation_key: Key under which the evaporation field is stored in the
      output.
  """

  observation_operator: observation_operators.ObservationOperator
  operator_query: dict[str, cx.Coordinate]
  extract_p_plus_e: ExtractPrecipitationPlusEvaporation
  prognostics_arg_key: str | int = 'prognostics'
  precipitation_scaling: PrecipitationScales = 'rate'
  evaporation_scaling: PrecipitationScales = 'rate'
  dt: float | None = None
  precipitation_key: str = 'precipitation'
  evaporation_key: str = 'evaporation'
  sim_units: units.SimUnits = dataclasses.field(kw_only=True)

  def __post_init__(self):
    valid_keys = set([self.precipitation_key, self.evaporation_key])
    query_keys = set(self.operator_query.keys())
    if len(query_keys.intersection(valid_keys)) != 1:
      raise ValueError(
          f'{self.operator_query=} should contain exactly on of {valid_keys=}.'
      )
    [self.observe_key] = valid_keys.intersection(query_keys)
    [self.diagnosed_key] = valid_keys.difference(query_keys)

  def _extract_prognostics(self, *args, **kwargs):
    if isinstance(self.prognostics_arg_key, int):
      prognostics = args[self.prognostics_arg_key]
    else:
      prognostics = kwargs.get(self.prognostics_arg_key)
    if not isinstance(prognostics, dict):
      raise ValueError(
          f'Prognostics must be a dictionary, got {type(prognostics)=} instead.'
      )
    # here we do a dummy wrap to interface with observation operator
    # interface. Once we start using Fields for intermediate representations
    # this won't be needed as prognostics will already be fields.
    return {k: cx.wrap(v) for k, v in prognostics.items()}

  def __call__(self, result, *args, **kwargs):
    tendencies = result
    [p_plus_e] = self.extract_p_plus_e(tendencies, *args, **kwargs).values()
    prognostics = self._extract_prognostics(*args, **kwargs)
    observation = self.observation_operator.observe(
        prognostics, query=self.operator_query
    )
    observation = observation[self.observe_key]
    precipitation_and_evaporation = {
        self.diagnosed_key: p_plus_e - observation,
        self.observe_key: observation,
    }
    water_density = self.sim_units.water_density
    for key, scaling in zip(
        [self.precipitation_key, self.evaporation_key],
        [self.precipitation_scaling, self.evaporation_scaling],
    ):
      if scaling == 'cumulative':
        if self.dt is None:
          raise ValueError(
              'dt must be provided when using cumulative precipitation scaling.'
          )
        precipitation_and_evaporation[key] *= self.dt / water_density
      elif scaling == 'rate':
        precipitation_and_evaporation[key] *= 1 / water_density
      elif scaling == 'mass_rate':
        continue
      else:
        raise ValueError(
            f'{scaling=} should be one of rate, mass_rate or cumulative.'
        )
    return precipitation_and_evaporation
