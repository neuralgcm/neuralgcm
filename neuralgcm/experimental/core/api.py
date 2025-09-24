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
from typing import Any, Callable, Sequence

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
      time: jdt.Datetime | None = None,
  ) -> typing.Pytree:
    del time  # unused.
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


CoordAndAxes = tuple[cx.Coordinate, tuple[int, ...]]


@nnx_compat.dataclass
class Model(nnx.Module, abc.ABC):
  """Base class for stateful, modular forecast systems."""

  mesh: parallelism.Mesh = dataclasses.field(
      default_factory=parallelism.Mesh, kw_only=True
  )
  fiddle_config: fdl.Config[Model] | None = dataclasses.field(
      default=None, init=False
  )
  _vector_axes: list[tuple[nnx.filterlib.Filter, cx.Coordinate]] = (
      dataclasses.field(default_factory=lambda: [], init=False)
  )

  @abc.abstractmethod
  def assimilate(self, observations: typing.Observation) -> None:
    """Assimilates observations into the model state."""
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

  @property
  def vector_axes(self) -> dict[nnx.filterlib.Filter, cx.Coordinate]:
    """Dictionary mapping state filters to their vectorization coordinates."""
    return dict(self._vector_axes)

  def vectorize(
      self,
      filter_axes: dict[nnx.filterlib.Filter, cx.Coordinate | CoordAndAxes],
  ) -> None:
    vector_axes = self.vector_axes
    for k, v in filter_axes.items():
      if isinstance(v, cx.Coordinate):
        ax = v
        indices = tuple(range(v.ndim))
      else:
        ax, indices = v
        if ax.ndim != len(indices):
          raise ValueError(f'ndim mismatch between {ax} with {indices=}')

      def broadcast(x: cx.Field) -> cx.Field:
        to_insert = dict(zip(indices, ax.axes))  # pylint: disable=cell-var-from-loop
        new_x_coord = cx.insert_axes_to_coordinate(x.coordinate, to_insert)
        return x.broadcast_like(new_x_coord)

      k_state = jax.tree.map(broadcast, nnx.state(self, k), is_leaf=cx.is_field)
      nnx.update(self, k_state)
      c = vector_axes.get(k, cx.Scalar())
      vector_axes[k] = cx.compose_coordinates(v, c)
    # TODO(dkochkov): Avoid this hack by modifying `with_callback` class.
    # Currently this is a workaround for updating metadata on instances that are
    # generated by `with_callback` method.
    obj = self
    while hasattr(obj, 'wrapped_instance'):
      obj = self.wrapped_instance
    object.__setattr__(obj, '_vector_axes', list(vector_axes.items()))

  def _index_for_axis_in_state(
      self,
      axis: cx.Coordinate,
      k: nnx.filterlib.Filter,
  ) -> int | None:
    if axis.ndim != 1:
      raise ValueError(f'idx can be computed only for 1d axis, got {axis}')
    dim = axis.dims[0]
    k_leaves = jax.tree.leaves(nnx.state(self, k), is_leaf=cx.is_field)
    indices = [leaf.named_axes.get(dim, None) for leaf in k_leaves] or [None]
    if len(set(indices)) > 1:
      raise ValueError(f'{axis=} is present at different indices in {k_leaves}')
    return indices[0]  # returns unique index or None if not present.

  def model_state_axes(
      self, axes: cx.Coordinate | Sequence[cx.Coordinate]
  ) -> nnx.StateAxes | list[nnx.StateAxes]:
    """Computes the `in_axes` for `vmap` transformations over the model state.

    This method is designed to handle both a single `vmap` and a sequence of
    nested `vmap` calls. When a sequence of axes is provided, it calculates
    the correct `in_axes` for each level of nesting, accounting for the fact
    that outer `vmap` operations effectively remove a dimension for the inner
    operations.

    Args:
      axes: A single `cx.Coordinate` or a sequence of `cx.Coordinate` objects.
        Each coordinate must be 1-dimensional and represents an axis to be
        mapped over. The sequence order determines the nesting order of `vmap`
        calls (e.g., `axes=[c1, c2]` corresponds to `vmap(vmap(f, c2), c1)`).

    Returns:
      If a single axis is provided, returns a single `nnx.StateAxes` object.
      If a sequence of axes is provided, returns a list of `nnx.StateAxes`
      objects, one for each corresponding axis in the input sequence.
    """
    is_single_axis = isinstance(axes, cx.Coordinate)
    axes = [axes] if is_single_axis else axes
    if not all([x.ndim == 1 for x in axes]):
      raise ValueError(f'Requested state_axes for non-1d coordinate {axes}')

    all_state_axes = []
    consumed_indices = {k: [] for k in self.vector_axes}

    for axis in axes:
      current_state_axes = {}
      for k in self.vector_axes:
        idx = self._index_for_axis_in_state(axis, k)
        if idx is None:
          current_state_axes[k] = None
        else:
          shift = sum(1 for i in consumed_indices[k] if i < idx)  # vmap shifts.
          adjusted_idx = idx - shift
          current_state_axes[k] = adjusted_idx
          consumed_indices[k].append(idx)
      current_state_axes[...] = None  # Default for non-vectorized state.
      all_state_axes.append(nnx.StateAxes(current_state_axes))

    return all_state_axes[0] if is_single_axis else all_state_axes

  def unvectorize(
      self,
      indices: dict[nnx.filterlib.Filter, tuple[int, ...]] | None = None,
  ) -> None:
    """Unvectorizes the model state slicing vector dimensions at `indices`."""
    # TODO(dkochkov): consider parameterizing this differently where we specify
    # correspondance between coordinates and indices.
    if indices is None:
      indices = {k: tuple([0] * v.ndim) for k, v in self.vector_axes.items()}
    for k, v in self.vector_axes.items():
      slice_index = cx.cmap(lambda x: x[*indices[k]])  # pylint: disable=cell-var-from-loop
      take = lambda f: slice_index(f.untag(v))  # pylint: disable=cell-var-from-loop
      k_state = jax.tree.map(take, nnx.state(self, k), is_leaf=cx.is_field)
      nnx.update(self, k_state)

  def untag_state(self, coordinate: cx.Coordinate) -> None:
    """Untags `coordinate` from model state where it is expected by metadata."""
    for k, v in self.vector_axes.items():
      untag_components = [ax for ax in coordinate.axes if ax in v.axes]
      if untag_components:
        untag_axis = cx.compose_coordinates(*untag_components)
        state_to_untag = nnx.state(self, k)
        nnx.update(self, cx.untag(state_to_untag, untag_axis))

  def tag_state(self, coordinate: cx.Coordinate) -> None:
    """Retags coordinate to model state where it is expected by metadata."""
    for k, v in self.vector_axes.items():
      tag_components = [ax for ax in coordinate.axes if ax in v.axes]
      if tag_components:
        retag_axis = cx.compose_coordinates(*tag_components)
        state_to_retag = nnx.state(self, k)
        nnx.update(self, cx.tag(state_to_retag, retag_axis))

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
    if model.fiddle_config is None and fiddle_config is None:
      raise ValueError('Either model or fiddle_config must be provided.')
    if model.fiddle_config is not None and fiddle_config is not None:
      if model.fiddle_config != fiddle_config:
        raise ValueError(
            'Both model or fiddle_config are provided and are different.'
        )
    if fiddle_config is None:
      fiddle_config = model.fiddle_config
    if model.vector_axes.values():
      raise ValueError('Converting vectorized Model is not supported.')
    graph_def, model_sim_state, model_state = nnx.split(
        model, typing.SimulationVariable, ...
    )
    dummy_model_sim_state = pytree_utils.shape_structure(model_sim_state)
    # TODO(dkochkov): Add verification that fiddle_config is in sync with the
    # model instance.
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
