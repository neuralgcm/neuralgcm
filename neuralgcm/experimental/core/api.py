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
"""Modeling API."""

from __future__ import annotations

import abc
import dataclasses
import functools
from typing import Any, Callable

import coordax as cx
import fiddle as fdl
from flax import nnx
import jax
import jax.numpy as jnp
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import data_specs
from neuralgcm.experimental.core import diagnostics
from neuralgcm.experimental.core import dynamic_io
from neuralgcm.experimental.core import fiddle_tags  # pylint: disable=unused-import
from neuralgcm.experimental.core import module_utils
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import pytree_utils
from neuralgcm.experimental.core import random_processes
from neuralgcm.experimental.core import scan_utils
from neuralgcm.experimental.core import typing
import numpy as np
import pandas as pd


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


@nnx.dataclass
class Model(nnx.Module, abc.ABC):
  """Base class for stateful, modular forecast systems."""

  mesh: parallelism.Mesh = nnx.static(
      default_factory=parallelism.default_mesh, kw_only=True
  )
  fiddle_config: fdl.Config[Model] | None = nnx.static(default=None, init=False)

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
  def dynamic_inputs_spec(
      self,
  ) -> dict[str, dict[str, cx.Coordinate | data_specs.CoordLikeSpec]]:
    """Returns inputs spec for all DynamicInputModule components."""
    dynamic_input_modules = module_utils.retrieve_subclass_modules(
        self, dynamic_io.DynamicInputModule
    )
    all_inputs_spec = {}
    for module in dynamic_input_modules:
      for k, v in module.inputs_spec.items():
        if k not in all_inputs_spec:  # new data key, add all specs.
          all_inputs_spec[k] = v
        else:
          # existing data key, check that all overlapping specs are consistent.
          current = all_inputs_spec[k]
          for var_name, var_spec in v.items():
            if var_name in current and var_spec != current[var_name]:
              raise ValueError(
                  f'Inconsistent specs for {k}/{var_name}: {var_spec} vs'
                  f' {current[var_name]}'
              )
            else:
              all_inputs_spec[k][var_name] = var_spec
    return all_inputs_spec

  @property
  def inputs_spec(
      self,
  ) -> dict[str, dict[str, cx.Coordinate | data_specs.CoordLikeSpec]]:
    """Returns inputs spec supported by this model."""
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


@nnx.dataclass
class VectorizedModel(Model):
  """A wrapper for a vectorized model."""

  vectorized_model: Model
  _vector_axes: list[tuple[nnx.filterlib.Filter, cx.Coordinate]] = nnx.static(
      default_factory=lambda: [], init=False
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
      model_to_vectorize = model.vectorized_model
    elif isinstance(model, Model):
      v_axes = vectorization_specs
      model_to_vectorize = model
    else:
      raise ValueError(
          f'Model type must be Model or VectorizedModel, got: {type(model)}'
      )

    vectorized_filters = tuple(k for k, v in v_axes.items() if v.ndim)
    graph_def, state_to_clone, state_to_merge = nnx.split(
        model_to_vectorize, vectorized_filters, ...
    )
    state_to_clone = nnx.clone(state_to_clone)
    vectorized_model = nnx.merge(graph_def, state_to_clone, state_to_merge)
    module_utils.vectorize_module(vectorized_model, vectorization_specs)
    vectorized = VectorizedModel(vectorized_model)
    object.__setattr__(vectorized, '_vector_axes', list(v_axes.items()))
    return vectorized

  def run_vectorized(
      self,
      fn,
      axes_to_vectorize,
      *args,
      custom_spmd_axis_names: None | str | tuple[str, ...] = None,
      custom_axis_names: None | str | tuple[str, ...] = None,
  ):
    """Runs a `fn(model, *args)` vectorized over `axes_to_vectorize`."""
    custom_names = (
        custom_spmd_axis_names is not None or custom_axis_names is not None
    )
    vectorized_fn = module_utils.vectorize_module_fn(
        fn,
        self.vector_axes,
        axes_to_vectorize,
        custom_spmd_axis_names,
        custom_axis_names,
        None if custom_names else getattr(self.vectorized_model, 'mesh', None),
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
    self.run_vectorized(run_assimilate, v_axis, inputs)

  def advance(self) -> None:
    """Advances the model state by one timestep."""

    def run_advance(model):
      return model.advance()

    v_axis = self._vec_coord_for_filter(typing.Prognostic)
    self.run_vectorized(run_advance, v_axis)

  def observe(self, query: typing.Query) -> typing.Observation:
    """Computes observations specified in `query` from the model state."""

    def run_observe(model, q):
      return model.observe(q)

    v_axis = self._vec_coord_for_filter(typing.Prognostic)
    return self.run_vectorized(run_observe, v_axis, query)

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
    self.run_vectorized(update_dynamic, v_axis, dynamic_inputs)

  @module_utils.ensure_unchanged_state_structure
  def initialize_random_processes(self, rng: cx.Field) -> None:
    """Generates new unconditional samples for all random process submodules."""

    def init_rng(model, inputs):
      model.initialize_random_processes(inputs)

    v_axis = self._vec_coord_for_filter(typing.Randomness)
    self.run_vectorized(init_rng, v_axis, rng)

  @module_utils.ensure_unchanged_state_structure
  def reset_diagnostic_state(self):
    """Resets diagnostic state of the model."""

    def run_reset_diagnostic(model):
      model.reset_diagnostic_state()

    v_axis = self._vec_coord_for_filter(typing.Diagnostic)
    self.run_vectorized(run_reset_diagnostic, v_axis)

  def diagnostic_values(self) -> typing.Pytree:
    """Returns diagnostic values of the model."""

    def run_get_diagnostic_values(model):
      return model.diagnostic_values()

    v_axis = self._vec_coord_for_filter(typing.Diagnostic)
    self.run_vectorized(run_get_diagnostic_values, v_axis)

  @property
  def dynamic_inputs_spec(
      self,
  ) -> dict[str, dict[str, cx.Coordinate | data_specs.CoordLikeSpec]]:
    """Returns the required dynamic inputs for the given query."""
    # TODO(dkochkov): Consider returning vectorized specs.
    return self.vectorized_model.dynamic_inputs_spec

  @property
  def inputs_spec(
      self,
  ) -> dict[str, dict[str, cx.Coordinate | data_specs.CoordLikeSpec]]:
    """Returns the required inputs for the given query."""
    # TODO(dkochkov): Consider returning vectorized specs.
    return self.vectorized_model.inputs_spec


def _checked_jit(func=None, *, static_argnums=(), static_argnames=()):
  """Wrapper around jax.jit that adds checks for ShapeDtypeStruct errors."""

  def _find_sds_paths(tree, prefix=''):
    paths = []
    try:
      leaves = jax.tree_util.tree_leaves_with_path(tree)
    except Exception:  # pylint: disable=broad-except
      leaves = []  # tree_leaves_with_path can fail on weird inputs.
    for path, leaf in leaves:
      if isinstance(leaf, jax.ShapeDtypeStruct):
        paths.append(f'{prefix}{jax.tree_util.keystr(path)}')
    return paths

  def decorator(fn):
    jitted_fn = jax.jit(
        fn, static_argnums=static_argnums, static_argnames=static_argnames
    )

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
      try:
        return jitted_fn(*args, **kwargs)
      except TypeError as e:
        if 'ShapeDtypeStruct' in str(e) or 'abstract value' in str(e):
          sds_paths = []
          sds_paths.extend(_find_sds_paths(args, prefix='args'))
          sds_paths.extend(_find_sds_paths(kwargs, prefix='kwargs'))
          try:
            result = fn(*args, **kwargs)
            sds_paths.extend(_find_sds_paths(result, prefix='outputs'))
          except Exception as e_unjitted:  # pylint: disable=broad-except
            sds_paths.append(
                f'Call to unjitted function failed with: {e_unjitted}'
            )

          error_message = (
              f"JAX raised TypeError: '{e}'. This often indicates "
              'uninitialized state (e.g., missing `rng` in assimilate, '
              'or incomplete `simulation_state` in advance/observe).'
          )
          if sds_paths:
            error_message += (
                '\nFurther inspection found ShapeDtypeStruct in:\n  '
                + '\n  '.join(sds_paths)
            )
          raise TypeError(error_message) from e
        raise

    return wrapper

  if func is not None:
    return decorator(func)
  return decorator


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
    else:
      model.reset_diagnostic_state()
    if rng is not None:
      model.initialize_random_processes(rng)
    return model

  @_checked_jit
  def assimilate(
      self,
      inputs: dict[str, dict[str, cx.Field]],
      dynamic_inputs: typing.Pytree | None = None,
      rng: cx.Field | typing.PRNGKeyArray | None = None,
      previous_estimate: typing.SimulationState | None = None,
  ) -> typing.SimulationState:
    """Returns simulation state after assimilating inputs."""
    if isinstance(rng, typing.PRNGKeyArray):
      rng = cx.field(rng, cx.Scalar())
    model = self._prepare_model(previous_estimate, dynamic_inputs, rng)
    model.assimilate(inputs)
    sim_state = model.simulation_state
    return sim_state

  @_checked_jit
  def advance(
      self,
      simulation_state: typing.SimulationState,
      dynamic_inputs: typing.Pytree | None = None,
  ) -> typing.SimulationState:
    """Returns simulation state after advancing by one timestep."""
    model = self._prepare_model(simulation_state, dynamic_inputs)
    model.advance()
    return model.simulation_state

  @_checked_jit
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
  def dynamic_inputs_spec(
      self,
  ) -> dict[str, dict[str, cx.Coordinate | data_specs.CoordLikeSpec]]:
    """Returns the required dynamic inputs for the given query."""
    return self._dummy_model().dynamic_inputs_spec

  @functools.cached_property
  def inputs_spec(
      self,
  ) -> dict[str, dict[str, cx.Coordinate | data_specs.CoordLikeSpec]]:
    """Returns the required inputs for the given query."""
    return self._dummy_model().inputs_spec


def _inference_model_flatten(model: InferenceModel):
  """Flattens InferenceModel."""
  children = (model.model_state,)
  dummy_simulation_state_coords = jax.tree.map(
      lambda x: x.coordinate,
      model.dummy_simulation_state,
      is_leaf=cx.is_field,
  )
  aux_data = (
      model.model_graph_def,
      dummy_simulation_state_coords,
      model.fiddle_config,
  )
  return children, aux_data


def _inference_model_unflatten(
    aux_data: tuple[Any, ...], children: tuple[Any, ...]
) -> InferenceModel:
  """Unflattens InferenceModel."""
  model_graph_def, dummy_simulation_state_coords, fiddle_config = aux_data
  (model_state,) = children
  dummy_simulation_state = jax.tree.map(
      cx.shape_struct_field,
      dummy_simulation_state_coords,
      is_leaf=cx.is_coord,
  )
  return InferenceModel(
      model_graph_def=model_graph_def,
      model_state=model_state,
      dummy_simulation_state=dummy_simulation_state,
      fiddle_config=fiddle_config,
  )


jax.tree_util.register_pytree_node(
    InferenceModel,
    _inference_model_flatten,
    _inference_model_unflatten,
)


def _unroll_for_queries(
    model: InferenceModel,
    initial_state: typing.SimulationState,
    queries: tuple[typing.Query, ...],
    timedelta: tuple[np.timedelta64, ...],
    final_leadtime: np.timedelta64,
    process_observations_fn: Callable[[typing.Observation], Any] = lambda x: x,
    dynamic_inputs: typing.Pytree | None = None,
    prepend_init: bool = False,
    trim_last: bool = False,
) -> tuple[typing.SimulationState, typing.Pytree]:
  """Unrolls a forecast for given queries supporting nested scans.

  This is a helper function that implements the core nested scan logic.

  Assumptions about `queries` and `timedelta`:
    - `queries`: Must be a tuple of queries, one per nesting level.
    - `timedelta`: Must be a tuple of timedeltas, one per nesting level,
      ordered from finest to coarsest frequency.
    - Leaves in `queries` that are Coordinates must already have a full
      `TimeDelta` coordinate attached, defining the exact lead times at which
      they should be observed.
    - Leaves in `queries` that are Fields must either have a full `TimeDelta`
      coordinate (indicating they are time-varying queries to be scanned over)
      or NO `TimeDelta` coordinate (indicating they apply to that specific
      nesting level and should be closed over in that level' computation).
    - All time-varying leaves must share a unique final lead time.

  Args:
    model: The inference model to unroll.
    initial_state: The starting state for the simulation.
    queries: A tuple of queries, one per nesting level.
    timedelta: A tuple of timedeltas, one per nesting level.
    final_leadtime: The shared final lead time for all queries.
    process_observations_fn: A function to post-process raw observations.
    dynamic_inputs: Optional time-varying inputs for the model.
    prepend_init: If True, includes observations at t=0 in the output.
    trim_last: If True, excludes the last observation from the output.

  Returns:
    A tuple containing:
      - final_state: The state of the simulation after the last step.
      - trajectory: A pytree of collected observations.
  """
  dt = model.timestep
  if dt != timedelta[0]:
    timedelta = (dt,) + timedelta
    queries = ({},) + queries

  if np.any(np.diff(timedelta) <= np.timedelta64(0, 's')):
    raise ValueError(
        'Timedeltas must be strictly increasing (finest to coarsest).'
    )
  coord_or_field = lambda x: cx.is_field(x) or cx.is_coord(x)  # tree helper.
  timedelta_coords = [  # coordinates for each nesting level.
      coordinates.TimeDelta((1 + np.arange(final_leadtime // td)) * td)
      for td in timedelta
  ]

  def _out_spec(x, td):
    coord = x.coordinate if cx.is_field(x) else x
    if 'timedelta' not in coord.dims:
      coord = cx.coords.compose(td, coord)
    return coord

  outputs_spec = {}  # contains flat coordinates spec for the final outputs.
  # pylint: disable=cell-var-from-loop
  for q, td in zip(queries, timedelta_coords, strict=True):
    spec = jax.tree.map(lambda x: _out_spec(x, td), q, is_leaf=coord_or_field)
    outputs_spec = pytree_utils.merge_nested_dicts(outputs_spec, spec)

  all_steps = scan_utils.nested_scan_steps(outputs_spec, dt=dt)

  # Timedeltas in queries have been saved in outputs_spec. We can trim them away
  # from coordinates to make them directly compatible with `observe` call.
  def _remove_td(x, td):
    if cx.is_field(x):
      return x
    return cx.coords.replace_axes(x, td, cx.Scalar()) if td in x.axes else x

  queries = tuple(
      jax.tree.map(lambda x: _remove_td(x, td), q, is_leaf=coord_or_field)
      for q, td in zip(queries, timedelta_coords, strict=True)
  )
  # pylint: enable=cell-var-from-loop

  is_scannable_field = lambda x: cx.is_field(x) and 'timedelta' in x.dims
  is_close_over_query = lambda x: not is_scannable_field(x)
  scannable_field_queries_specs = tuple(
      pytree_utils.filter_nested_dict(is_scannable_field, q) for q in queries
  )
  queries_to_close_over = tuple(
      pytree_utils.filter_nested_dict(is_close_over_query, q) for q in queries
  )
  # To nest data for scans we combine all queries, extract scannable fields and
  # nest them according to the full scan structure.
  merged_query = functools.reduce(pytree_utils.merge_nested_dicts, queries, {})
  fields_in_queries_to_scan_over = scan_utils.nest_data_for_scans(
      pytree_utils.filter_nested_dict(is_scannable_field, merged_query),
      scan_steps=all_steps,
      scan_specs=scannable_field_queries_specs,
  )

  # Build and run the nested scan function.
  def build_scan_fn(nest_level: int):
    if nest_level == -1:  # Base case body_fn simply calls advance.

      def _innermost(state, scannable_inputs):
        del scannable_inputs  # unused.
        next_state = model.advance(state, dynamic_inputs)
        return next_state, {}  # no nested sub-timestep observations.

      return _innermost

    length = all_steps[nest_level]
    static_q = queries_to_close_over[nest_level]
    inner_fn = build_scan_fn(nest_level - 1)

    def _with_obs(state, fields_in_q_to_scan):
      # scannable_inputs is a tuple of pytrees. At the current nest_level,
      # the leaves have a leading dimension of length `length`.
      scanned_q = fields_in_q_to_scan[-1]
      inner_fields_in_q_to_scan = fields_in_q_to_scan[:-1]
      next_state, inner_obs = inner_fn(state, inner_fields_in_q_to_scan)
      in_queries = pytree_utils.merge_nested_dicts(static_q, scanned_q)
      raw_obs = model.observe(next_state, in_queries, dynamic_inputs)
      obs = process_observations_fn(raw_obs)
      merged_obs = pytree_utils.merge_nested_dicts(inner_obs, obs)
      return next_state, merged_obs

    def _scan_level(state, fields_in_q_to_scan):
      return jax.lax.scan(
          _with_obs, state, fields_in_q_to_scan[: nest_level + 1], length=length
      )

    return _scan_level

  scan_fn = build_scan_fn(len(all_steps) - 1)
  final, observations = scan_fn(initial_state, fields_in_queries_to_scan_over)
  unroll = scan_utils.ravel_data_from_nested_scans(observations, outputs_spec)

  if prepend_init:  # prepends t0 observations to raveled result.
    dt0 = np.array([np.timedelta64(0, 's')])
    t0_query = jax.tree.map(
        lambda x: x.isel(timedelta=0) if 'timdelta' in x.dims else x,
        merged_query,
        is_leaf=coord_or_field,
    )
    t0_raw_obs = model.observe(initial_state, t0_query, dynamic_inputs)
    t0_obs = process_observations_fn(t0_raw_obs)
    # expand leading dimension. Coordax auto-adds unlabeled dimension.
    t0_obs_expanded = jax.tree.map(lambda x: x[None, ...], t0_obs)
    concat_array_fn = lambda x, y: jnp.concatenate([x, y], axis=0)
    concat_trees_fn = lambda tx, ty: jax.tree.map(concat_array_fn, tx, ty)
    combined = jax.tree.map(
        cx.cpmap(concat_trees_fn),
        t0_obs_expanded,
        cx.untag(unroll, 'timedelta'),
        is_leaf=cx.is_field,
    )

    def _tag_td(f: cx.Field, coord: cx.Coordinate) -> cx.Field:
      orig_td = cx.coords.extract(coord, coordinates.TimeDelta)
      new_td = coordinates.TimeDelta(np.concatenate([dt0, orig_td.deltas]))
      return f.tag(new_td)

    unroll = jax.tree.map(_tag_td, combined, outputs_spec, is_leaf=cx.is_field)

  if trim_last:
    unroll = jax.tree.map(
        lambda x: x.isel(timedelta=slice(None, -1)), unroll, is_leaf=cx.is_field
    )

  return final, unroll


@_checked_jit(
    static_argnames=[
        'timedelta',
        'steps',
        'process_observations_fn',
        'prepend_init',
        'trim_last',
    ],
)
def unroll_from_advance(
    model: InferenceModel,
    initial_state: typing.SimulationState,
    queries: tuple[typing.Query, ...] | typing.Query,
    timedelta: tuple[np.timedelta64, ...] | np.timedelta64,
    steps: int,
    *,
    process_observations_fn: Callable[[typing.Observation], Any] = lambda x: x,
    dynamic_inputs: typing.Pytree | None = None,
    prepend_init: bool = False,
    trim_last: bool = False,
) -> tuple[typing.SimulationState, typing.Pytree]:
  """Unrolls a forecast using functional advance and observe calls.

  This function performs a simulation rollout by repeatedly calling `advance`
  on the `model` and collecting observations at specified intervals
  defined by `timedelta` and `query`. It supports nested scan loops for
  collecting observations at different frequencies. When multiple timedeltas are
  requested, they must be ordered from finest to coarsest frequency and must be
  congruent.

  Args:
    model: The inference model to unroll.
    initial_state: The starting state for the simulation.
    queries: Single or nested queries specifying what to observe at each level
      of nesting. If a tuple, it must have the same length as `timedelta`.
    timedelta: A single timedelta or a tuple of timedeltas specifying the
      observation frequency. If a tuple, it must be ordered from finest to
      coarsest frequency. Timedeltas must be congruent.
    steps: The number of steps to take at the outermost timedelta level.
    process_observations_fn: A function to post-process raw observations.
    dynamic_inputs: Optional time-varying inputs for the model.
    prepend_init: If True, includes observations at t=0 in the output.
    trim_last: If True, excludes the last observation from the output.

  Returns:
    A tuple containing:
      - final_state: The state of the simulation at final lead time.
      - unroll: A trajectory of observations matching the requested queries and
        timedeltas.
  """
  is_nested = list({isinstance(timedelta, tuple), isinstance(queries, tuple)})
  if len(is_nested) != 1:
    raise ValueError(
        'timedelta and query must be both tuples or both not tuples'
    )
  [is_nested] = is_nested  # pylint: disable=unbalanced-tuple-unpacking

  if not is_nested:
    timedelta = (timedelta,)
    queries = (queries,)

  final_timedelta = timedelta[-1] * steps
  return _unroll_for_queries(
      model,
      initial_state,
      queries,
      timedelta,
      final_timedelta,
      process_observations_fn,
      dynamic_inputs,
      prepend_init,
      trim_last,
  )


@_checked_jit(
    static_argnames=[
        'timedelta',
        'steps',
        'process_observations_fn',
        'prepend_init',
        'trim_last',
    ],
)
def forecast_steps(
    model: InferenceModel,
    inputs: dict[str, dict[str, cx.Field]],
    queries: typing.Query | tuple[typing.Query, ...],
    timedelta: np.timedelta64 | tuple[np.timedelta64, ...],
    steps: int,
    *,
    dynamic_inputs: typing.Pytree | None = None,
    rng: cx.Field | typing.PRNGKeyArray | None = None,
    prepend_init: bool = False,
    trim_last: bool = False,
    process_observations_fn: Callable[[typing.Observation], Any] = lambda x: x,
) -> tuple[typing.SimulationState, typing.Pytree]:
  """Runs a forecast from inputs for a specified number of steps.

  This function first assimilates the provided `inputs` to create an initial
  simulation state, and then calls `unroll_from_advance` to perform the rollout.

  Args:
    model: The inference model to unroll.
    inputs: Input fields used for assimilation.
    queries: Single or nested queries specifying what to observe at each level
      of nesting. If a tuple, it must have the same length as `timedelta`.
    timedelta: A single timedelta or a tuple of timedeltas specifying the
      observation frequency. If a tuple, it must be ordered from finest to
      coarsest frequency. Timedeltas must be congruent.
    steps: The number of steps at the outermost level.
    dynamic_inputs: Optional time-varying inputs for the model.
    rng: Optional random number generator state.
    prepend_init: If True, includes observations at t=0 in the output.
    trim_last: If True, excludes the last observation from the output.
    process_observations_fn: A function to post-process raw observations.

  Returns:
    A tuple containing:
      - final_state: The state of the simulation at final lead time.
      - unroll: A trajectory of observations matching the requested queries and
        timedeltas.
  """
  if rng is not None and not isinstance(rng, cx.Field):
    rng = cx.field(rng, cx.Scalar())
  initial_state = model.assimilate(inputs, dynamic_inputs, rng)
  return unroll_from_advance(
      model=model,
      initial_state=initial_state,
      queries=queries,
      timedelta=timedelta,
      steps=steps,
      process_observations_fn=process_observations_fn,
      dynamic_inputs=dynamic_inputs,
      prepend_init=prepend_init,
      trim_last=trim_last,
  )


def unroll_for_template(
    model: InferenceModel,
    initial_state: typing.SimulationState,
    template: dict[str, dict[str, cx.Field | cx.Coordinate]],
    dynamic_inputs: typing.Pytree | None = None,
    process_observations_fn: Callable[[typing.Observation], Any] = lambda x: x,
) -> tuple[typing.SimulationState, typing.Pytree]:
  """Unrolls a forecast based on a template specifying desired outputs.

  This function infers the required observation frequencies and queries from the
  `template` argument. Template is specified as a nested dictionary of
  Fields or Coordinates. Coordinate leaves must have Timedelta axes with
  congruent timedeltas and share a unique final lead time. Field leaves are
  used to specify field in queries components and are also expected to have a
  TimeDelta axis.

  Args:
    model: The inference model to unroll.
    initial_state: The starting state for the simulation.
    template: A nested dictionary of Fields or Coordinates that defines the
      desired output structure and time coordinates. Coordinate leaves indicate
      the queried fields and Field leaves specify dynamic queries data.
    dynamic_inputs: Optional time-varying inputs for the model.
    process_observations_fn: A function to post-process raw observations.

  Returns:
    A tuple containing:
      - final_state: The state of the simulation at final lead time.
      - unroll: A trajectory of observations matching the template structure.
  """
  dt = model.timestep
  to_coord = lambda x: x.coordinate if cx.is_field(x) else x
  combined_coords = jax.tree.map(
      to_coord, template, is_leaf=lambda x: cx.is_field(x) or cx.is_coord(x)
  )
  final_leadtime = scan_utils.shared_final_leadtime(combined_coords)

  dt_and_specs = scan_utils.group_by_timedeltas(combined_coords, dt=dt)
  # Construct queries and timedeltas replacing coordinates with template values.
  timedeltas = []
  queries = []
  for td, spec in dt_and_specs:
    timedeltas.append(td)
    q = pytree_utils.replace_with_matching_or_default(
        spec, template, check_used_all_replace_keys=False
    )
    queries.append(q)

  return _unroll_for_queries(
      model=model,
      initial_state=initial_state,
      queries=tuple(queries),
      timedelta=tuple(timedeltas),
      final_leadtime=final_leadtime,
      process_observations_fn=process_observations_fn,
      dynamic_inputs=dynamic_inputs,
      prepend_init=False,
  )
