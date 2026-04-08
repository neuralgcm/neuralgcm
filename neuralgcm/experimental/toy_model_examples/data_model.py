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

"""Implementation of data model that wires pre-loaded data to model API."""

import coordax as cx
from flax import nnx
import jax.numpy as jnp
import jax_datetime as jdt
from neuralgcm.experimental.core import api
from neuralgcm.experimental.core import data_specs
from neuralgcm.experimental.core import diagnostics
from neuralgcm.experimental.core import dynamic_io
from neuralgcm.experimental.core import module_utils
from neuralgcm.experimental.core import observation_operators
from neuralgcm.experimental.core import transforms
from neuralgcm.experimental.core import typing
from neuralgcm.experimental.core import units
import numpy as np


@nnx.dataclass
class DataModel(api.Model):
  """A model that reads values from data via DynamicInputSlice."""

  keys_to_coords: dict[str, cx.Coordinate] = nnx.static()
  observation_key: str = nnx.static()
  model_timestep: np.timedelta64 = nnx.static(
      default=np.timedelta64(1, 'h'), kw_only=True
  )
  operators: dict[str, typing.ObservationOperator] = nnx.data(
      default_factory=dict, kw_only=True
  )
  state_diagnostics: dict[str, diagnostics.DiagnosticModule] = nnx.data(
      default_factory=dict, kw_only=True
  )
  tendency_dependent_diagnostics: dict[str, diagnostics.DiagnosticModule] = (
      nnx.data(default_factory=dict, kw_only=True)
  )
  sim_units: units.SimUnits | None = nnx.static(default=None, kw_only=True)
  nondim_transform: transforms.TransformABC = nnx.data(
      default_factory=transforms.Identity, kw_only=True
  )
  redim_transform: transforms.TransformABC = nnx.data(
      default_factory=transforms.Identity, kw_only=True
  )
  _dynamic_inputs_slice: dynamic_io.DynamicInputSlice = nnx.data(init=False)
  _prognostic_vars: typing.Prognostic = nnx.data(init=False)

  def __post_init__(self):
    self._dynamic_inputs_slice = dynamic_io.DynamicInputSlice(
        keys_to_coords=self.keys_to_coords,
        observation_key=self.observation_key,
    )
    prognostics = {}
    for k, c in self.keys_to_coords.items():
      value = jnp.nan * jnp.zeros(c.shape)
      prognostics[k] = cx.field(value, c)
    prognostics['time'] = cx.field(jdt.Datetime(jdt.Timedelta()))
    self._prognostic_vars = typing.Prognostic(prognostics)
    if self.observation_key not in self.operators:
      self.operators[self.observation_key] = (
          observation_operators.TransformObservationOperator(
              transforms.Identity()
          )
      )

  @property
  def prognostics(self) -> dict[str, cx.Field]:
    return self._prognostic_vars.get_value()

  @property
  def timestep(self) -> np.timedelta64:
    return self.model_timestep

  @module_utils.ensure_unchanged_state_structure
  def assimilate(self, inputs: dict[str, dict[str, cx.Field]]) -> None:
    if self.observation_key not in inputs:
      raise ValueError(f'Observation key {self.observation_key} not in inputs')
    fields_with_timedelta = inputs[self.observation_key]
    new_prognostics = {}
    for k in self.keys_to_coords:
      new_prognostics[k] = fields_with_timedelta[k].isel(timedelta=-1)
    new_prognostics = self.nondim_transform(new_prognostics)
    new_prognostics['time'] = fields_with_timedelta['time'].isel(timedelta=-1)
    self._prognostic_vars.set_value(new_prognostics)

  @module_utils.ensure_unchanged_state_structure
  def advance(self) -> None:
    current_prognostics = self._prognostic_vars.get_value()
    next_time = current_prognostics['time'] + self.timestep
    new_values = self.nondim_transform(self._dynamic_inputs_slice(next_time))
    new_prognostics = {k: new_values[k] for k in self.keys_to_coords}
    new_prognostics['time'] = next_time

    if self.sim_units is not None:
      dt_nondim = self.sim_units.nondimensionalize_timedelta64(self.timestep)
    else:
      dt_nondim = self.timestep / np.timedelta64(1, 's')
    tendencies = {
        k: (new_prognostics[k] - current_prognostics[k]) / dt_nondim
        for k in self.keys_to_coords
    }
    dt_timedelta = jdt.to_timedelta(self.timestep)
    for diagnostic in self.tendency_dependent_diagnostics.values():
      diagnostic(tendencies, prognostics=current_prognostics)
    for diagnostic in self.state_diagnostics.values():
      diagnostic({}, prognostics=current_prognostics)

    for diagnostic in self.state_diagnostics.values():
      if isinstance(diagnostic, diagnostics.TemporalDiagnosticModule):
        diagnostic.advance_diagnostic_clock({'timedelta': dt_timedelta})
    for diagnostic in self.tendency_dependent_diagnostics.values():
      if isinstance(diagnostic, diagnostics.TemporalDiagnosticModule):
        diagnostic.advance_diagnostic_clock({'timedelta': dt_timedelta})

    self._prognostic_vars.set_value(new_prognostics)

  @module_utils.ensure_unchanged_state_structure
  def observe(self, queries: typing.Query) -> typing.Observation:
    result = {}
    diagnostic_values = self.diagnostic_values()
    diagnostics_op = observation_operators.DataObservationOperator(
        diagnostic_values
    )
    obs_operators = {'diagnostics': diagnostics_op} | self.operators
    for k, q in queries.items():
      if k in obs_operators:
        obs = obs_operators[k].observe(self.prognostics, q)
        result[k] = self.redim_transform(obs)
      else:
        raise ValueError(f'No observation operator for key "{k}"')
    return result

  @property
  def inputs_spec(
      self,
  ) -> dict[str, dict[str, cx.Coordinate | data_specs.CoordSpec]]:
    specs = {}
    cs = {k: c for k, c in self.keys_to_coords.items()}
    cs['time'] = cx.Scalar()
    specs[self.observation_key] = {
        k: data_specs.CoordSpec.with_any_timedelta(v) for k, v in cs.items()
    }
    return specs
