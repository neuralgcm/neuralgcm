# Copyright 2024 Google LLC

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
"""ForecastSystem API."""

from __future__ import annotations

import abc
import dataclasses
from typing import Callable

import fiddle as fdl
from flax import nnx
import jax
import jax_datetime as jdt
from neuralgcm.experimental import coordax as cx
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import diagnostics
from neuralgcm.experimental.core import dynamic_io
from neuralgcm.experimental.core import fiddle_tags  # pylint: disable=unused-import
from neuralgcm.experimental.core import module_utils
from neuralgcm.experimental.core import nnx_compat
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import random_processes
from neuralgcm.experimental.core import time_integrators
from neuralgcm.experimental.core import typing
import numpy as np
import pandas as pd
import xarray


def calculate_sub_steps(
    timestep: np.timedelta64, duration: typing.TimedeltaLike
) -> int:
  """Calculate the number of time-steps required to simulate a time interval."""
  duration = pd.Timedelta(duration)
  time_step_ratio = duration / timestep
  if abs(time_step_ratio - round(time_step_ratio)) > 1e-6:
    raise ValueError(
        f'non-integral time-step ratio: {duration=} is not a multiple of '
        f'the internal model timestep {timestep}'
    )
  return round(time_step_ratio)


@nnx_compat.dataclass
class ForecastSystem(nnx.Module, abc.ABC):
  """Base class for forecast systems."""

  _metadata: dict = dataclasses.field(default_factory=dict, kw_only=True)  # pylint: disable=g-bare-generic
  mesh: parallelism.Mesh = dataclasses.field(
      default_factory=parallelism.Mesh, kw_only=True
  )

  @property
  def metadata(self):
    """Returns optional metadata associated with the model."""
    return self._metadata

  @property
  def timestep(self):
    """Returns the timestep of the model."""
    raise NotImplementedError()

  def update_metadata(self, key, value):
    """Adds metadata to the model."""
    self._metadata[key] = value

  @abc.abstractmethod
  def assimilate_prognostics(
      self,
      observations: typing.Pytree,
      dynamic_inputs: typing.Pytree | None = None,
      rng: typing.PRNGKeyArray | None = None,
      initial_state: typing.ModelState | None = None,
  ) -> typing.Pytree:
    raise NotImplementedError()

  @abc.abstractmethod
  def advance_prognostics(
      self, prognostics: typing.PyTreeState
  ) -> typing.PyTreeState:
    raise NotImplementedError()

  @abc.abstractmethod
  def observe_from_prognostics(
      self,
      prognostics: typing.Pytree,
      query: typing.Pytree,
      dynamic_inputs: typing.Pytree | None = None,
  ) -> typing.Pytree:
    raise NotImplementedError()

  def assimilate(
      self,
      observations: typing.Pytree,
      dynamic_inputs: typing.Pytree | None = None,
      rng: typing.PRNGKeyArray | None = None,
      initial_state: typing.ModelState | None = None,
  ) -> typing.ModelState:
    self.update_dynamic_inputs(dynamic_inputs)
    self.initialize_random_processes(rng)
    prognostics = self.assimilate_prognostics(observations, initial_state)
    diagnostic = nnx.state(self, diagnostics.DiagnosticValue)
    randomness = nnx.state(self, random_processes.RandomnessValue)
    return typing.ModelState(prognostics, diagnostic, randomness)

  def advance(
      self,
      state: typing.ModelState,
      dynamic_inputs: typing.Pytree | None = None,
  ) -> typing.ModelState:
    self.update_dynamic_inputs(dynamic_inputs)
    nnx.update(self, state.diagnostics)
    nnx.update(self, state.randomness)
    prognostics = self.advance_prognostics(state.prognostics)
    diagnostic = nnx.state(self, diagnostics.DiagnosticValue)
    randomness = nnx.state(self, random_processes.RandomnessValue)
    return typing.ModelState(prognostics, diagnostic, randomness)

  def observe(
      self,
      state: typing.ModelState,
      query: typing.Pytree,
      dynamic_inputs: typing.Pytree | None = None,
  ) -> typing.Pytree:
    self.update_dynamic_inputs(dynamic_inputs)
    nnx.update(self, state.diagnostics)
    nnx.update(self, state.randomness)
    return self.observe_from_prognostics(state.prognostics, query)

  def update_dynamic_inputs(self, dynamic_input: typing.Pytree | None = None):
    if dynamic_input is not None:
      for covariate_module in module_utils.retrieve_subclass_modules(
          self, dynamic_io.DynamicInputModule
      ):
        covariate_module.update_dynamic_inputs(dynamic_input)

  def initialize_random_processes(self, rng: typing.PRNGKeyArray) -> None:
    modules = module_utils.retrieve_subclass_modules(
        self, random_processes.RandomProcessModule
    )
    rngs = jax.random.split(rng, len(modules))
    for random_process, key in zip(modules, rngs):
      random_process.unconditional_sample(key)

  def format_diagnostics(
      self,
      state: typing.ModelState | None = None,
      time: jdt.Datetime | None = None,
  ) -> typing.Pytree:
    if state is not None:
      nnx.update(self, state.diagnostics)
    outputs = {}
    for diagnostic_module in module_utils.retrieve_subclass_modules(
        self, diagnostics.DiagnosticModule
    ):
      outputs |= diagnostic_module.format_diagnostics(time)
    return outputs

  def unroll(
      self,
      state: typing.ModelState,
      outer_steps: int,
      timedelta: typing.TimedeltaLike | None = None,
      start_with_input: bool = True,
      post_process_fn: Callable[..., typing.Pytree] = lambda x, **kwargs: x,
      dynamic_inputs: typing.Pytree | None = None,
  ) -> tuple[typing.ModelState, typing.Pytree]:
    self.update_dynamic_inputs(dynamic_inputs)
    nnx.update(self, state.diagnostics)
    nnx.update(self, state.randomness)

    if timedelta is None:
      timedelta = self.timestep
    inner_steps = calculate_sub_steps(self.timestep, timedelta)

    def _inner_step(model_and_prognostics):
      model, prognostics = model_and_prognostics
      next_prognostics = model.advance_prognostics(prognostics)
      return (model, next_prognostics)

    inner_step = time_integrators.repeated(_inner_step, inner_steps)

    def _step(model_and_state):
      model, model_state = model_and_state
      model, next_prognostics = inner_step((model, model_state.prognostics))
      diagnostic = nnx.state(model, diagnostics.DiagnosticValue)
      randomness = nnx.state(model, random_processes.RandomnessValue)
      next_model_state = typing.ModelState(
          next_prognostics, diagnostic, randomness
      )
      frame = model_state if start_with_input else next_model_state
      return (model, next_model_state), post_process_fn(frame, model=model)

    unroll_fn = nnx.scan(
        _step,
        length=outer_steps,
        in_axes=nnx.Carry,
        out_axes=(nnx.Carry, 0),
    )
    (_, final_state), intermediates = unroll_fn((self, state))
    steps = int(not start_with_input) + np.arange(outer_steps)
    time = coordinates.TimeDelta(steps * timedelta)
    intermediates = cx.tag(intermediates, time)
    return final_state, intermediates

  def inputs_from_xarray(
      self, nested_data: dict[str, xarray.Dataset]
  ) -> typing.Pytree:
    """Converts xarray dataset to inputs for the model."""
    raise NotImplementedError(
        f'Class {self.__class__.__name__} does not implement'
        ' inputs_from_xarray.'
    )

  def dynamic_inputs_from_xarray(
      self, nested_data: dict[str, xarray.Dataset]
  ) -> typing.Pytree:
    """Converts xarray dataset to dynamic covariates for the model."""
    raise NotImplementedError(
        f'Class {self.__class__.__name__} does not implement'
        ' dynamic_inputs_from_xarray.'
    )

  def data_from_xarray(
      self, nested_data: dict[str, xarray.Dataset]
  ) -> tuple[typing.Pytree, typing.Pytree]:
    """Converts xarray dataset to data for the model."""
    inputs = self.inputs_from_xarray(nested_data)
    dynamic_inputs = self.dynamic_inputs_from_xarray(nested_data)
    return inputs, dynamic_inputs

  @classmethod
  def from_fiddle_config(
      cls,
      config: fdl.Config[ForecastSystem],
      spmd_mesh_updates: (
          dict[parallelism.TagOrMeshType, jax.sharding.Mesh | None] | None
      ) = None,
      array_partitions_updates: (
          dict[parallelism.TagOrMeshType, parallelism.ArrayPartitions] | None
      ) = None,
      field_partitions_updates: (
          dict[parallelism.TagOrMeshType, parallelism.FieldPartitions] | None
      ) = None,
  ):
    """Builds a model from a fiddle config with updated mesh properties."""
    if not issubclass(config.__fn_or_cls__, ForecastSystem):
      raise ValueError(
          f'Fiddle config defines {config.__fn_or_cls__} '
          'which does not inherit from the ForecastSystem class'
      )
    model_config = parallelism.update_mesh_properties(
        config,
        spmd_mesh_updates=spmd_mesh_updates,
        array_partitions_updates=array_partitions_updates,
        field_partitions_updates=field_partitions_updates,
    )
    return fdl.build(model_config)
