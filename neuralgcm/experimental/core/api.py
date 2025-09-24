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
import functools
from typing import Any, Callable

import coordax as cx
import fiddle as fdl
from flax import nnx
import jax
import jax_datetime as jdt
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import data_specs
from neuralgcm.experimental.core import diagnostics
from neuralgcm.experimental.core import dynamic_io
from neuralgcm.experimental.core import fiddle_tags  # pylint: disable=unused-import
from neuralgcm.experimental.core import module_utils
from neuralgcm.experimental.core import nnx_compat
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import pytree_utils
from neuralgcm.experimental.core import random_processes
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


# Default model state axes specifies which model variables are carried through
# scan iterations and which are closed over.
DEFAULT_MODEL_STATE_AXES = nnx.StateAxes({
    typing.DynamicInput: None,  # close over the dynamic inputs.
    nnx.Param: None,
    ...: nnx.Carry,
})


@nnx_compat.dataclass
class ForecastSystem(nnx.Module, abc.ABC):
  """Base class for forecast systems."""

  _metadata: dict[str, Any] = dataclasses.field(
      default_factory=dict, init=False
  )
  mesh: parallelism.Mesh = dataclasses.field(
      default_factory=parallelism.Mesh, kw_only=True
  )
  fiddle_config: fdl.Config[ForecastSystem] | None = dataclasses.field(
      default=None, init=False
  )

  @property
  def metadata(self) -> dict[str, Any]:
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
      observations: typing.Observation,
      dynamic_inputs: typing.Observation | None = None,
      rng: typing.PRNGKeyArray | None = None,
      initial_state: typing.ModelState | None = None,
  ) -> typing.PrognosticsDict:
    raise NotImplementedError()

  @abc.abstractmethod
  def advance_prognostics(
      self, prognostics: typing.PrognosticsDict
  ) -> typing.PrognosticsDict:
    raise NotImplementedError()

  @abc.abstractmethod
  def observe_from_prognostics(
      self,
      prognostics: typing.PrognosticsDict,
      query: typing.Query,
      dynamic_inputs: typing.Observation | None = None,
  ) -> typing.Observation:
    raise NotImplementedError()

  def assimilate(
      self,
      observations: typing.Observation,
      dynamic_inputs: typing.Observation | None = None,
      rng: typing.PRNGKeyArray | None = None,
      initial_state: typing.ModelState | None = None,
  ) -> typing.ModelState:
    self.update_dynamic_inputs(dynamic_inputs)
    self.initialize_random_processes(rng)
    self.reset_diagnostic_state()
    prognostics = self.assimilate_prognostics(observations, initial_state)
    diagnostic = nnx.clone(nnx.state(self, typing.Diagnostic))
    randomness = nnx.clone(nnx.state(self, typing.Randomness))
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
    diagnostic = nnx.clone(nnx.state(self, typing.Diagnostic))
    randomness = nnx.clone(nnx.state(self, typing.Randomness))
    return typing.ModelState(prognostics, diagnostic, randomness)

  def observe(
      self,
      state: typing.ModelState,
      query: typing.Query,
      dynamic_inputs: typing.Observation | None = None,
  ) -> typing.Observation:
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
    if not modules:
      return
    rngs = jax.random.split(rng, len(modules))
    for random_process, key in zip(modules, rngs):
      random_process.unconditional_sample(key)

  def reset_diagnostic_state(self):
    for diagnostic_module in module_utils.retrieve_subclass_modules(
        self, diagnostics.DiagnosticModule
    ):
      diagnostic_module.reset_diagnostic_state()

  def format_diagnostics(
      self,
      state: typing.ModelState | None = None,
  ) -> typing.Pytree:
    if state is not None:
      nnx.update(self, state.diagnostics)
    outputs = {}
    for diagnostic_module in module_utils.retrieve_subclass_modules(
        self, diagnostics.DiagnosticModule
    ):
      outputs |= diagnostic_module.diagnostic_values()
    return outputs

  def unroll(
      self,
      state: typing.ModelState,
      outer_steps: int,
      timedelta: typing.TimedeltaLike | None = None,
      start_with_input: bool = True,
      post_process_fn: Callable[..., typing.Pytree] = lambda x, **kwargs: x,
      dynamic_inputs: typing.Pytree | None = None,
      model_state_axes_spec: nnx.StateAxes | None = None,
  ) -> tuple[typing.ModelState, typing.Pytree]:
    self.update_dynamic_inputs(dynamic_inputs)
    nnx.update(self, state.diagnostics)
    nnx.update(self, state.randomness)
    # model_state_axes_spec specifies which model variables are carried through
    # the time integration and which are closed over.
    if model_state_axes_spec is None:
      # By default we carry diagnostics and randomness, and close over the rest.
      model_state_axes_spec = DEFAULT_MODEL_STATE_AXES

    if timedelta is None:
      timedelta = self.timestep
    inner_steps = calculate_sub_steps(self.timestep, timedelta)

    def _inner_step(model, prognostics):
      return model.advance_prognostics(prognostics)

    if inner_steps > 1:
      inner_step_fn = nnx.scan(
          _inner_step,
          length=inner_steps,
          in_axes=(model_state_axes_spec, nnx.Carry),
          out_axes=nnx.Carry,
      )
    else:
      inner_step_fn = _inner_step

    def _to_model_state(prognostics, model):
      return typing.ModelState(
          prognostics,
          nnx.clone(nnx.state(model, typing.Diagnostic)),
          nnx.clone(nnx.state(model, typing.Randomness)),
      )

    def _step(model, prognostics):
      if start_with_input:
        model_state = _to_model_state(prognostics, model)
        intermediate = post_process_fn(model_state, model=model)
        next_prognostics = inner_step_fn(model, prognostics)
      else:
        next_prognostics = inner_step_fn(model, prognostics)
        model_state = _to_model_state(next_prognostics, model)
        intermediate = post_process_fn(model_state, model=model)
      # TODO(dkochkov): Consider calling post_process on prognostics directly.
      return next_prognostics, intermediate

    unroll_fn = nnx.scan(
        _step,
        length=outer_steps,
        in_axes=(model_state_axes_spec, nnx.Carry),
        out_axes=(nnx.Carry, 0),
    )
    final_prognostics, intermediates = unroll_fn(self, state.prognostics)
    final_state = _to_model_state(final_prognostics, self)
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
    # TODO(shoyer): Consider moving this method into a dedicated config module
    # (maybe also with some utilities from checkpointing).
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
    model = fdl.build(model_config)
    model.fiddle_config = config
    return model

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(pytree=False, **kwargs)


@nnx_compat.dataclass
class Model(nnx.Module, abc.ABC):
  """Base class for stateful, modular forecast systems."""

  mesh: parallelism.Mesh = dataclasses.field(
      default_factory=parallelism.Mesh, kw_only=True
  )
  fiddle_config: fdl.Config[Model] | None = dataclasses.field(
      default=None, init=False
  )

  @abc.abstractmethod
  def assimilate(self, inputs: typing.Observation) -> None:
    """Assimilates inputs into the model state."""
    raise NotImplementedError()

  @abc.abstractmethod
  def advance(self) -> None:
    """Advances the model state by one timestep."""
    raise NotImplementedError()

  @abc.abstractmethod
  def observe(self, query: typing.Query) -> typing.Observation:
    """Computes observations specified in `query` from the model state."""
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def timestep(self):
    """Returns the timestep of the model."""
    raise NotImplementedError()

  # TODO(dkochkov): Consider if creating VectorizedModel explicitly is better
  # than having a method that returns a VectorizedModel instance.
  def to_vectorized(
      self,
      vectorization_specs: dict[nnx.filterlib.Filter, cx.Coordinate],
  ) -> VectorizedModel:
    return VectorizedModel.from_model_with_vectorization(
        self, vectorization_specs
    )

  @property
  def simulation_state_filters(
      self,
  ) -> typing.SimulationState[nnx.filterlib.Filter]:
    """Returns SimulationState specs specifying filters for state components."""
    return typing.SimulationState(
        prognostics=typing.Prognostic,
        diagnostics=typing.Diagnostic,
        randomness=typing.Randomness,
        extras={'coupling': typing.Coupling},
    )

  @property
  def simulation_state(self) -> typing.SimulationState:
    """Returns the simulation state of the model."""
    extract_state = lambda k: nnx.clone(nnx.state(self, k))
    return typing.SimulationState(
        prognostics=extract_state(self.simulation_state_filters.prognostics),
        diagnostics=extract_state(self.simulation_state_filters.diagnostics),
        randomness=extract_state(self.simulation_state_filters.randomness),
        extras={
            k: extract_state(v)
            for k, v in self.simulation_state_filters.extras.items()
        },
    )

  def set_simulation_state(self, state: typing.SimulationState) -> None:
    """Sets the simulation state of the model."""
    nnx.update(self, state.prognostics)
    nnx.update(self, state.diagnostics)
    nnx.update(self, state.randomness)
    for v in state.extras.values():
      nnx.update(self, v)

  def update_dynamic_inputs(
      self,
      dynamic_inputs: dict[str, dict[str, cx.Field]],
  ) -> None:
    """Calls update_dynamic_inputs on all DynamicInputModule submodules."""
    for covariate_module in module_utils.retrieve_subclass_modules(
        self, dynamic_io.DynamicInputModule
    ):
      covariate_module.update_dynamic_inputs(dynamic_inputs)

  @module_utils.ensure_unchanged_state_structure
  def initialize_random_processes(self, rng: cx.Field) -> None:
    """Generates new unconditional samples for all random process submodules."""
    modules = module_utils.retrieve_subclass_modules(
        self, random_processes.RandomProcessModule
    )
    split_fn = lambda k: tuple(jax.random.split(k, len(modules)))
    rngs = cx.cmap(split_fn)(rng)
    for random_process, key in zip(modules, rngs):
      random_process.unconditional_sample(key)

  @module_utils.ensure_unchanged_state_structure
  def reset_diagnostic_state(self):
    for diagnostic_module in module_utils.retrieve_subclass_modules(
        self, diagnostics.DiagnosticModule
    ):
      diagnostic_module.reset_diagnostic_state()

  def diagnostic_values(self) -> typing.Pytree:
    outputs = {}
    for diagnostic_module in module_utils.retrieve_subclass_modules(
        self, diagnostics.DiagnosticModule
    ):
      outputs |= diagnostic_module.diagnostic_values()
    return outputs

  @property
  def required_dynamic_input_specs(
      self,
  ) -> dict[str, dict[str, data_specs.DataSpec]]:
    """Returns the required dynamic inputs for the given query."""
    raise NotImplementedError()

  @property
  def required_input_specs(self) -> dict[str, dict[str, data_specs.DataSpec]]:
    """Returns the required inputs for the given query."""
    raise NotImplementedError()

  @classmethod
  def from_fiddle_config(
      cls,
      config: fdl.Config[Model],
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
    # TODO(shoyer): Consider moving this method into a dedicated config module
    # (maybe also with some utilities from checkpointing).
    if not issubclass(config.__fn_or_cls__, Model):
      raise ValueError(
          f'Fiddle config defines {config.__fn_or_cls__} '
          'which does not inherit from the Model class'
      )
    updated_config = parallelism.update_mesh_properties(
        config,
        spmd_mesh_updates=spmd_mesh_updates,
        array_partitions_updates=array_partitions_updates,
        field_partitions_updates=field_partitions_updates,
    )
    model = fdl.build(updated_config)
    model.fiddle_config = config
    return model

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(pytree=False, **kwargs)


@nnx_compat.dataclass
class VectorizedModel(Model):
  """A wrapper for a vectorized model."""

  vectorized_model: Model
  _vector_axes: list[tuple[nnx.filterlib.Filter, cx.Coordinate]] = (
      dataclasses.field(default_factory=lambda: [], init=False)
  )

  @classmethod
  def from_model_with_vectorization(
      cls,
      model: Model | VectorizedModel,
      vectorization_specs: dict[nnx.filterlib.Filter, cx.Coordinate],
  ) -> VectorizedModel:
    """Returns a vectorized model with the given vectorization specs."""
    if isinstance(model, VectorizedModel):
      v_axes = model.vector_axes
      v_axes = module_utils.merge_vectorized_axes(vectorization_specs, v_axes)
    elif isinstance(model, Model):
      v_axes = vectorization_specs
    else:
      raise ValueError(
          f'Model type must be Model or VectorizedModel, got: {type(model)}'
      )

    vectorized_filters = tuple(k for k, v in v_axes.items() if v.ndim)
    graph_def, state_to_clone, state_to_merge = nnx.split(
        model, vectorized_filters, ...
    )
    state_to_clone = nnx.clone(state_to_clone)
    vectorized_model = nnx.merge(graph_def, state_to_clone, state_to_merge)
    module_utils.vectorize_module(vectorized_model, vectorization_specs)
    vectorized = VectorizedModel(vectorized_model)
    object.__setattr__(vectorized, '_vector_axes', list(v_axes.items()))
    return vectorized

  def _run_vectorized(self, fn, axes_to_vectorize, *args):
    vectorized_fn = module_utils.vectorize_module_fn(
        fn, self.vector_axes, axes_to_vectorize
    )
    return vectorized_fn(self.vectorized_model, *args)

  @property
  def vector_axes(self) -> dict[nnx.filterlib.Filter, cx.Coordinate]:
    """Dictionary mapping state filters to their vectorization coordinates."""
    axes = dict(self._vector_axes)
    axes[...] = axes.get(..., cx.Scalar())
    return axes

  def to_vectorized(
      self,
      vectorization_specs: dict[nnx.filterlib.Filter, cx.Coordinate],
  ) -> VectorizedModel:
    """Returns a vectorized model with additional vectorization specs."""
    return VectorizedModel.from_model_with_vectorization(
        self, vectorization_specs
    )

  def _vec_coord_for_filter(
      self, state_filter: nnx.filterlib.Filter
  ) -> cx.Coordinate:
    candidates = []
    for k, v in self.vector_axes.items():
      if k != ... and module_utils.is_filter_subset(state_filter, k):
        candidates.append(v)
    if not candidates and ... in self.vector_axes:
      # Handle reminder vectorization separately since ... matches all filters.
      candidates.append(self.vector_axes[...])
    if len(candidates) != 1:
      raise ValueError(
          f'cannot identify vectorization for {state_filter} from'
          f' {self.vector_axes=}'
      )
    return candidates[0]

  def assimilate(self, inputs: typing.Observation) -> None:
    """Assimilates observations into the model state."""

    def run_assimilate(model, inputs):
      return model.assimilate(inputs)

    v_axis = self._vec_coord_for_filter(typing.Prognostic)
    self._run_vectorized(run_assimilate, v_axis, inputs)

  def advance(self) -> None:
    """Advances the model state by one timestep."""

    def run_advance(model):
      return model.advance()

    v_axis = self._vec_coord_for_filter(typing.Prognostic)
    self._run_vectorized(run_advance, v_axis)

  def observe(self, query: typing.Query) -> typing.Observation:
    """Computes observations specified in `query` from the model state."""

    def run_observe(model, q):
      return model.observe(q)

    v_axis = self._vec_coord_for_filter(typing.Prognostic)
    return self._run_vectorized(run_observe, v_axis, query)

  @property
  def timestep(self):
    """Returns the timestep of the model."""
    return self.vectorized_model.timestep

  @property
  def simulation_state(self) -> typing.SimulationState:
    """Returns the simulation state of the model."""
    return self.vectorized_model.simulation_state

  def set_simulation_state(self, state: typing.SimulationState) -> None:
    """Sets the simulation state of the model."""
    # This method supports auto-vectorization, so no mapping is necessary.
    # Alternatively, we could inspect `state`` and vmap set_simulation_state.
    self.vectorized_model.set_simulation_state(state)

  def update_dynamic_inputs(
      self,
      dynamic_inputs: dict[str, dict[str, cx.Field]],
  ) -> None:
    """Calls update_dynamic_inputs on all DynamicInputModule submodules."""

    def update_dynamic(model, inputs):
      model.update_dynamic_inputs(inputs)

    v_axis = self._vec_coord_for_filter(typing.DynamicInput)
    self._run_vectorized(update_dynamic, v_axis, dynamic_inputs)

  @module_utils.ensure_unchanged_state_structure
  def initialize_random_processes(self, rng: cx.Field) -> None:
    """Generates new unconditional samples for all random process submodules."""

    def init_rng(model, inputs):
      model.initialize_random_processes(inputs)

    v_axis = self._vec_coord_for_filter(typing.Randomness)
    self._run_vectorized(init_rng, v_axis, rng)

  @module_utils.ensure_unchanged_state_structure
  def reset_diagnostic_state(self):
    """Resets diagnostic state of the model."""

    def run_reset_diagnostic(model):
      model.reset_diagnostic_state()

    v_axis = self._vec_coord_for_filter(typing.Diagnostic)
    self._run_vectorized(run_reset_diagnostic, v_axis)

  def diagnostic_values(self) -> typing.Pytree:
    """Returns diagnostic values of the model."""

    def run_get_diagnostic_values(model):
      return model.diagnostic_values()

    v_axis = self._vec_coord_for_filter(typing.Diagnostic)
    self._run_vectorized(run_get_diagnostic_values, v_axis)

  @property
  def required_dynamic_input_specs(
      self,
  ) -> dict[str, dict[str, data_specs.DataSpec]]:
    """Returns the required dynamic inputs for the given query."""
    # TODO(dkochkov): Consider returning vectorized specs.
    return self.vectorized_model.required_dynamic_input_specs

  @property
  def required_input_specs(self) -> dict[str, dict[str, data_specs.DataSpec]]:
    """Returns the required inputs for the given query."""
    # TODO(dkochkov): Consider returning vectorized specs.
    return self.vectorized_model.required_input_specs


# TODO(dkochkov): Consider renaming InferenceModel to ForecastSystem.
@dataclasses.dataclass(frozen=True)
class InferenceModel:
  """Inference wrapper for Model that uses fixed parameters."""

  model_graph_def: nnx.GraphDef
  model_state: typing.Pytree  # this is model parameters which is a pytree;
  dummy_simulation_state: typing.Pytree
  fiddle_config: fdl.Config

  def _dummy_model(self) -> Model:
    return nnx.merge(
        self.model_graph_def,
        self.model_state,
        self.dummy_simulation_state,
        copy=True,
    )

  @functools.cached_property
  def timestep(self) -> typing.Timestep:
    """Time interval of a single `self.advance` step."""
    return self._dummy_model().timestep

  def _prepare_model(
      self,
      simulation_state: typing.SimulationState | None = None,
      dynamic_inputs: dict[str, dict[str, cx.Field]] | None = None,
      rng: cx.Field | None = None,
  ):
    """Initializes a temporary model with given simulation parameters."""
    model = self._dummy_model()
    if dynamic_inputs is not None:
      model.update_dynamic_inputs(dynamic_inputs)
    if simulation_state is not None:
      model.set_simulation_state(simulation_state)
    if rng is not None:
      model.initialize_random_processes(rng)
    return model

  @jax.jit
  def assimilate(
      self,
      inputs: dict[str, dict[str, cx.Field]],
      dynamic_inputs: typing.Pytree | None = None,
      rng: cx.Field | typing.PRNGKeyArray | None = None,
      previous_estimate: typing.SimulationState | None = None,
  ) -> typing.SimulationState:
    """Returns simulation state after assimilating inputs."""
    if isinstance(rng, typing.PRNGKeyArray):
      rng = cx.wrap(rng, cx.Scalar())
    model = self._prepare_model(previous_estimate, dynamic_inputs, rng)
    model.assimilate(inputs)
    return model.simulation_state

  @jax.jit
  def advance(
      self,
      simulation_state: typing.SimulationState,
      dynamic_inputs: typing.Pytree | None = None,
  ) -> typing.SimulationState:
    """Returns simulation state after advancing by one timestep."""
    model = self._prepare_model(simulation_state, dynamic_inputs)
    model.advance()
    return model.simulation_state

  @jax.jit
  def observe(
      self,
      simulation_state: typing.SimulationState,
      query: typing.Query,
      dynamic_inputs: typing.Pytree | None = None,
  ) -> dict[str, dict[str, cx.Field]]:
    """Returns model observations given the simulation state and a query."""
    model = self._prepare_model(simulation_state, dynamic_inputs)
    return model.observe(query)

  @classmethod
  def from_model_api(
      cls, model: Model, fiddle_config: fdl.Config[Model] | None = None
  ) -> InferenceModel:
    """Constructs InferenceModel from Model instance."""
    if model.fiddle_config is not None and fiddle_config is not None:
      if model.fiddle_config != fiddle_config:
        raise ValueError(
            'Both model or fiddle_config are provided and are different.'
        )
    if fiddle_config is None:
      fiddle_config = model.fiddle_config
    graph_def, model_sim_state, model_state = nnx.split(
        model, typing.SimulationVariable, ...
    )
    dummy_model_sim_state = pytree_utils.shape_structure(model_sim_state)
    return cls(graph_def, model_state, dummy_model_sim_state, fiddle_config)

  @functools.cached_property
  def required_dynamic_input_specs(
      self,
  ) -> dict[str, dict[str, data_specs.DataSpec]]:
    """Returns the required dynamic inputs for the given query."""
    return self._dummy_model().required_dynamic_input_specs

  @functools.cached_property
  def required_input_specs(self) -> dict[str, dict[str, data_specs.DataSpec]]:
    """Returns the required inputs for the given query."""
    return self._dummy_model().required_input_specs


jax.tree_util.register_dataclass(
    InferenceModel,
    data_fields=['model_state', 'dummy_simulation_state'],
    meta_fields=['model_graph_def', 'fiddle_config'],
)


@functools.partial(
    jax.jit,
    static_argnames=[
        'timedelta',
        'steps',
        'process_observations_fn',
        'start_with_input',
    ],
)
def unroll_from_advance(
    forecast_system: InferenceModel,
    initial_state: typing.SimulationState,
    timedelta: np.timedelta64,
    steps: int,
    query: typing.Query,
    process_observations_fn: Callable[[typing.Observation], Any] = lambda x: x,
    dynamic_inputs: typing.Pytree | None = None,
    start_with_input: bool = True,
) -> tuple[typing.SimulationState, typing.Pytree]:
  """Unrolls a forecast using functional advance and observe calls."""
  inner_steps = calculate_sub_steps(forecast_system.timestep, timedelta)

  def _advance_n_steps(
      simulation_state: typing.SimulationState,
  ) -> typing.SimulationState:
    def body_fn(state, _):
      next_state = forecast_system.advance(state, dynamic_inputs)
      return next_state, None

    if inner_steps > 1:
      final_state, _ = jax.lax.scan(
          body_fn, simulation_state, xs=None, length=inner_steps
      )
      return final_state
    else:
      return forecast_system.advance(simulation_state, dynamic_inputs)

  def _scan_step(state, _):
    next_state = _advance_n_steps(state)
    if start_with_input:
      raw_obs = forecast_system.observe(state, query, dynamic_inputs)
    else:
      raw_obs = forecast_system.observe(next_state, query, dynamic_inputs)
    observation = process_observations_fn(raw_obs)
    return next_state, observation

  final_state, observations = jax.lax.scan(
      _scan_step, initial_state, xs=None, length=steps
  )
  time_steps = int(not start_with_input) + np.arange(steps)
  time_coord = coordinates.TimeDelta(time_steps * timedelta)
  observations = cx.tag(observations, time_coord)
  return final_state, observations


# TODO(dkochkov): Figure out why having dummy state conflicts with the jit.
@functools.partial(
    jax.jit,
    static_argnames=[
        'timedelta',
        'steps',
        'process_observations_fn',
        'start_with_input',
    ],
)
def forecast_steps(
    forecast_system: InferenceModel,
    inputs: dict[str, dict[str, cx.Field]],
    timedelta: np.timedelta64,
    steps: int,
    query: typing.Query,
    dynamic_inputs: typing.Pytree | None = None,
    rng: typing.PRNGKeyArray | None = None,
    start_with_input: bool = True,
    process_observations_fn: Callable[[typing.Observation], Any] = lambda x: x,
) -> tuple[typing.ModelState, typing.Pytree]:
  """Runs a forecast from an inputs for a specified number of steps."""
  if rng is not None:
    rng = cx.wrap(rng, cx.Scalar())
  initial_state = forecast_system.assimilate(inputs, dynamic_inputs, rng)
  return unroll_from_advance(
      forecast_system=forecast_system,
      initial_state=initial_state,
      query=query,
      steps=steps,
      timedelta=timedelta,
      dynamic_inputs=dynamic_inputs,
      start_with_input=start_with_input,
      process_observations_fn=process_observations_fn,
  )
