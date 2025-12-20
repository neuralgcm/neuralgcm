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
# pylint: disable=line-too-long
"""Trainer logic for rollout experiment."""

import abc
import collections
from collections.abc import Callable, Sequence
import dataclasses
import functools
import json
import logging
import math
import operator
import os.path
from typing import Any, NamedTuple, TypeAlias, TypeVar

import coordax as cx
from etils import epath
from flax import nnx
import jax
from jax import checkpoint_policies as cp  # pylint: disable=g-importing-member
from jax.ad_checkpoint import checkpoint_name  # pylint: disable=g-importing-member
from jax.experimental import compute_on
import jax.sharding
from neuralgcm.experimental.core import api
from neuralgcm.experimental.core import checkpointing as model_checkpointing
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import data_specs
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import typing
from neuralgcm.experimental.inference import streaming
from neuralgcm.experimental.inference import timing
from neuralgcm.experimental.metrics import aggregation
from neuralgcm.experimental.metrics import base as metrics_base
from neuralgcm.experimental.metrics import evaluators
from neuralgcm.experimental.metrics import weighting
from neuralgcm.experimental.training import checkpointing
from neuralgcm.experimental.training import data_loading
from neuralgcm.experimental.training import train_utils
import numpy as np
import optax
import orbax.checkpoint as ocp
from orbax.checkpoint import checkpoint_managers as ocp_managers


# pylint: disable=logging-fstring-interpolation

Params: TypeAlias = typing.Pytree
PyTree: TypeAlias = typing.Pytree


TrainEvalIteratorTuple = tuple[
    Any,
    train_utils.TrainStepFunction,
    Callable[..., Any],
]


def is_coordinator() -> bool:
  """Returns `True` on the coordinator host, otherwise `False`."""
  return jax.process_index() == 0


def create_spmd_mesh(
    ensemble_shards: int,
    z_shards: int,
    x_shards: int,
    y_shards: int,
) -> jax.sharding.Mesh:
  """Creates JAX sharding mesh used for training."""
  global_batch = jax.device_count() // (
      ensemble_shards * z_shards * x_shards * y_shards
  )
  if global_batch == 0:
    raise ValueError(
        f'{jax.device_count()=} is insufficient for '
        f'{ensemble_shards=} {z_shards=} {x_shards=} {y_shards=}'
    )
  return train_utils.create_spmd_mesh({
      'batch': global_batch,
      'ensemble': ensemble_shards,
      'z': z_shards,
      'x': x_shards,
      'y': y_shards,
  })


def create_training_mesh(
    spmd_mesh: jax.sharding.Mesh, base_field_partitions: dict[str, Any]
) -> parallelism.Mesh:
  """Creates parallel mesh with batch and ensemble sharding for training."""
  field_partitions = {
      k: v | {'ensemble': 'ensemble', 'batch': 'batch'}
      for k, v in base_field_partitions.items()
  }
  return parallelism.Mesh(
      spmd_mesh=spmd_mesh, field_partitions=field_partitions
  )


def create_batch_axis(
    spmd_mesh: jax.sharding.Mesh,
    batch_size_per_device: int,
) -> cx.SizedAxis:
  """Computes batch axis."""
  degree_of_model_parallelism = math.prod(
      v for k, v in spmd_mesh.shape.items() if k != 'batch'
  )
  global_batch_size = (
      jax.device_count() // degree_of_model_parallelism * batch_size_per_device
  )
  batch_axis = cx.SizedAxis('batch', global_batch_size)
  return batch_axis


K = TypeVar('K')
V = TypeVar('V')


def _flatten_dict(
    nested: dict[str, dict[str, cx.Field]],
    sep='/',
) -> dict[str, cx.Field]:
  """Flattens a nested dictionary of fields into a single level."""
  result = {}
  for k, v in nested.items():
    for inner_k, f in v.items():
      result[sep.join([k, inner_k])] = f
  return result


def _run_evaluator(evaluator, prediction, target):
  """Helper to run evaluation on the host."""
  return evaluator.evaluate(prediction, target)


def _combine_agg_states(agg_states, new_aggregations):
  """Helper to combine aggregation states."""
  agg_states = jax.tree.map(
      operator.add,
      agg_states,
      new_aggregations,
      is_leaf=lambda x: isinstance(x, aggregation.AggregationState),
  )
  return agg_states


def _on_host[T: Callable[..., Any]](fn: T) -> T:
  """Wraps `fn` to run on host."""

  @functools.wraps(fn)
  def _fn(*args, **kwargs):
    with compute_on.compute_on('device_host'):
      return fn(*args, **kwargs)

  return _fn


def _maybe_on_host[T: Callable[..., Any]](fn: T, compute_on_host: bool) -> T:
  if compute_on_host:
    return _on_host(fn)
  return fn


class EMAParamsTree(nnx.Module):
  """Captures the exponentially moving average values of a pytree."""

  def __init__(self, pytree: typing.Pytree, num_steps: int):
    # https://en.wikipedia.org/wiki/Moving_average#Relationship_between_SMA_and_EMA
    self.decay_rate = 1 - 2 / (num_steps + 1)
    self.pytree_ema = nnx.Param(pytree)

  def __call__(self, inputs):
    pytree_ema = jax.tree.map(
        lambda ema, x: self.decay_rate * ema + (1 - self.decay_rate) * x,
        self.pytree_ema.get_value(),
        inputs,
    )
    self.pytree_ema.set_value(pytree_ema)
    return pytree_ema


class ExperimentState(NamedTuple):
  """Pytree compatible experiment state for use with JIT-compiled functions.

  Attributes:
    opt_state: State of the Optax optimizer.
    params: Model parameters (nnx.Param variables) optimized by the optimizer.
    ema_params: Exponential moving average of model parameters.
    non_params: Model variables not optimized by the optimizer (nnx.Variable
      objects that are not params or temporary variables), typically set via
      pretraining. Examples include normalization statistics, and static
      orography and land/sea masks loaded from data.
  """

  opt_state: PyTree
  params: nnx.GraphState
  ema_params: nnx.GraphState
  non_params: nnx.GraphState


@dataclasses.dataclass(frozen=True)
class AutoRestart:
  """Status of the auto-restart process after encountering NaN loss."""

  iteration: int = 0
  began_at: int | None = None


@dataclasses.dataclass
class OnlineEvalMetrics:
  train: dict[str, float] = dataclasses.field(default_factory=dict)
  eval: dict[str, float] = dataclasses.field(default_factory=dict)
  eval_ema: dict[str, float] = dataclasses.field(default_factory=dict)
  seconds_per_evaluation: float | None = None
  seconds_per_train_step: float | None = None


class OnlineMetricsSaver(abc.ABC):
  """Abstract base class for saving online metrics."""

  @abc.abstractmethod
  def save(self, step: int, online_metrics: OnlineEvalMetrics) -> None:
    pass


@dataclasses.dataclass
class JsonMetricsSaver(OnlineMetricsSaver):
  """Saves online metrics to JSON files."""

  metrics_dir: str

  def save(self, step: int, online_metrics: OnlineEvalMetrics) -> None:
    online_metrics_dict = dataclasses.asdict(online_metrics)
    output_dir = epath.Path(self.metrics_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'metrics_{step:06d}.json'
    output_path.write_text(json.dumps(online_metrics_dict, indent=4))


PretrainOp = Callable[..., Any]


@dataclasses.dataclass
class PretrainingOps:
  pretrain_fns: collections.OrderedDict[str, PretrainOp]


@dataclasses.dataclass(frozen=True)
class TrainSchedule:
  """Schedule for training."""

  schedule_time_steps: Sequence[int]
  schedule_boundaries: Sequence[int]
  total_steps: int
  batch_size_per_device: int
  time_sample_offset: int
  time_slice: tuple[str, str] | list[tuple[str, str]] | None = None
  shuffle_buffer_size: int = 0
  dataset_rng_seed: int = 0
  random_process_rng_seed: int = 0


@dataclasses.dataclass(frozen=True)
class EvalSchedule:
  """Schedule for evaluation."""

  time_steps: Sequence[int]
  batch_size_per_device: int
  num_batches: int
  steps_between_evals: int
  time_slice: tuple[str, str] | list[tuple[str, str]] | None = None


@dataclasses.dataclass(frozen=True)
class OptimizationConfig:
  """Configuration for optimization."""

  optimizer: optax.GradientTransformation
  ema_num_steps: int


@dataclasses.dataclass(frozen=True)
class AutoRestartConfig:
  """Configuration for auto-restart on NaN loss."""

  max_nan_restarts: int = 0
  restart_lookback_steps: int = 0
  error_with_nan_loss: bool = True


@dataclasses.dataclass(frozen=True)
class CheckpointConfig:
  """Configuration for checkpointing."""

  save_interval_steps: int
  keep_every_n_steps: int
  model_config_str: str
  metadata: dict[str, Any] | None = None


@dataclasses.dataclass(frozen=True)
class InitialCheckpoint:
  """Options for initial checkpoint loading."""

  checkpoint_dir: str | None = None
  checkpoint_step: int | None = None
  reset_optimizer_state: bool = False


@dataclasses.dataclass(frozen=True)
class RematConfig:
  """Options for JAX checkpointing (re-materialization)."""

  step_policy: str | None = None
  observe_policy: str | None = None
  process_policy: str | None = None
  remat_collect_statistics: bool = False


@dataclasses.dataclass
class RolloutTrainer:
  """Trainer logic for rollout experiment."""

  experiment_dir: str
  model: api.Model
  data_loader: data_loading.DataLoader
  loss: evaluators.Evaluator[metrics_base.Loss]
  eval_metrics: evaluators.Evaluator[metrics_base.Metric]
  process_observations: nnx.Module
  pretraining: PretrainingOps | None
  train_schedule: TrainSchedule
  eval_schedule: EvalSchedule
  optimization_config: OptimizationConfig
  initial_checkpoint: InitialCheckpoint | None
  checkpoint_config: CheckpointConfig
  auto_restart_config: AutoRestartConfig
  remat_config: RematConfig
  ensemble_axis: cx.SizedAxis
  online_metrics_saver: OnlineMetricsSaver
  compute_loss_on_host: bool = False

  def __post_init__(self):
    self.run_evaluator = _maybe_on_host(
        _run_evaluator, self.compute_loss_on_host
    )
    self.combine_agg_states = _maybe_on_host(
        _combine_agg_states, self.compute_loss_on_host
    )

    # TODO(dkochkov): Remove this padding once we parameterize timedeltas in the
    # input data specs.
    def _add_dummy(c):
      return c.coord

    is_coord_spec = lambda x: isinstance(x, data_specs.CoordSpec)
    padded_input_specs = jax.tree.map(
        _add_dummy, self.data_loader.input_data_specs, is_leaf=is_coord_spec
    )
    try:
      data_specs.validate_inputs(padded_input_specs, self.model.inputs_spec)
    except ValueError as e:
      raise ValueError(
          f'{self.data_loader.input_data_specs=} is not compatible with the'
          f' model requirements {self.model.inputs_spec=}. Please check that'
          ' the `input_data_specs_config` in the config is consistent with the'
          ' model.'
      ) from e
    try:
      data_specs.validate_inputs(
          padded_input_specs, self.model.dynamic_inputs_spec
      )
    except ValueError as e:
      raise ValueError(
          f'{self.data_loader.input_data_specs=} is not compatible with the'
          f' model requirements {self.model.dynamic_inputs_spec=}. Please check'
          ' that the `input_data_specs_config` in the config is consistent'
          ' with the model.'
      ) from e

    self._checkpoint_dir = os.path.join(self.experiment_dir, 'checkpoints')
    epath.Path(self._checkpoint_dir).mkdir(
        parents=True, exist_ok=True, mode=0o775
    )

    self.train_inner_steps = self.data_loader.inner_steps
    self.eval_inner_steps = self.train_inner_steps  # simplified for now.

    self._eval_trajectory_length = self._outer_steps(
        max(self.eval_schedule.time_steps)
    )
    self._trajectory_lengths = [
        1 + n // self.train_inner_steps  # num_init_frames=1
        for n in self.train_schedule.schedule_time_steps
    ]
    self._max_trajectory_length = max(
        self._trajectory_lengths + [self._eval_trajectory_length]
    )

    self.init_params = nnx.state(self.model, nnx.Param)

    # exponentially moving average params tracking.
    ema_module = EMAParamsTree(
        self.init_params, self.optimization_config.ema_num_steps
    )
    ema_static = nnx.graphdef(ema_module)

    def update_ema(inputs, ema_state):
      ema_params, (_, new_ema_state) = ema_static.apply(ema_state)(inputs)
      return ema_params, new_ema_state

    def init_ema(rng, inputs):
      del rng  # unused.
      module = EMAParamsTree(inputs, 1)  # num_steps doesn't affect init.
      return inputs, nnx.state(module, nnx.Param)

    self._ema_init = jax.jit(init_ema)
    self._ema_update = jax.jit(update_ema)

  @property
  def queries_spec(self) -> dict[str, Any]:
    return self.data_loader.queries_spec

  @property
  def batch_axis(self) -> cx.SizedAxis:
    return self.data_loader.batch_axis

  @property
  def training_mesh(self) -> parallelism.Mesh:
    return self.data_loader.training_mesh

  @property
  def spmd_mesh(self) -> jax.sharding.Mesh:
    assert self.training_mesh.spmd_mesh is not None
    return self.training_mesh.spmd_mesh

  @functools.cached_property
  def degree_of_model_parallelism(self) -> int:
    """Product of mesh dimensions excluding 'batch'."""
    return math.prod(v for k, v in self.spmd_mesh.shape.items() if k != 'batch')

  @functools.cached_property
  def spatial_parallelism(self) -> int:
    """Product of spatial mesh dimensions ('z', 'x', 'y')."""
    return math.prod(
        self.spmd_mesh.shape[k] for k in 'zxy' if k in self.spmd_mesh.shape
    )

  @property
  def global_batch_size(self) -> int:
    return self.batch_axis.size

  @property
  def global_eval_batch_size(self) -> int:
    return (
        jax.device_count()
        // self.degree_of_model_parallelism
        * self.eval_schedule.batch_size_per_device
    )

  def _outer_steps(self, trajectory_length_in_steps: int) -> int:
    """Number of outer (saved) steps for this time step."""
    if trajectory_length_in_steps is None:
      return None
    if trajectory_length_in_steps % self.train_inner_steps:
      raise ValueError(
          f'{trajectory_length_in_steps=} was not a multiple of '
          f'{self.train_inner_steps=}'
      )
    return trajectory_length_in_steps // self.train_inner_steps + 1

  def build_train_and_eval_iterators(
      self, schedule_idx: int
  ) -> TrainEvalIteratorTuple:
    """Build new iterators for training at schedule_idx."""
    num_train_time_steps = self.train_schedule.schedule_time_steps[schedule_idx]
    trajectory_length = self._trajectory_lengths[schedule_idx]

    if self.eval_schedule.num_batches:
      eval_data = self.data_loader.build_eval_inputs(
          dataset_time_slices=self.eval_schedule.time_slice,
          train_trajectory_length=trajectory_length,
          num_init_frames=1,  # num_init_frames=1
          eval_trajectory_length=self._eval_trajectory_length,
          batch_size_per_device=self.eval_schedule.batch_size_per_device,
          global_batch_size=self.global_eval_batch_size,
          batch_count=self.eval_schedule.num_batches,
      )
      train_data = self.data_loader.build_eval_inputs(
          dataset_time_slices=self.train_schedule.time_slice,
          train_trajectory_length=trajectory_length,
          num_init_frames=1,  # num_init_frames=1
          eval_trajectory_length=self._eval_trajectory_length,
          batch_size_per_device=self.eval_schedule.batch_size_per_device,
          global_batch_size=self.global_eval_batch_size,
          batch_count=self.eval_schedule.num_batches,
      )

      evaluate_fn = functools.partial(
          self.evaluate,
          eval_batch_fn=self.get_eval_batch_fn(trajectory_length, seed=0),
          eval_data=eval_data,
          train_data=train_data,
      )
    else:
      evaluate_fn = self.dummy_evaluate

    training_data = self.data_loader.build_train_inputs(
        time_series_length=trajectory_length,
        batch_size_per_device=self.train_schedule.batch_size_per_device,
        global_batch_size=self.global_batch_size,
        shuffle_buffer_size_in_bytes=self.train_schedule.shuffle_buffer_size,
        dataset_rng_seed=self.train_schedule.dataset_rng_seed,
        time_sample_offset=self.train_schedule.time_sample_offset,
        dataset_time_slice=self.train_schedule.time_slice,
    )

    # TODO(shoyer): consider adding a hook for adding a profiler around
    # train_step_fn.
    train_step_fn = self.get_train_step_fn(num_train_time_steps)

    return training_data, train_step_fn, evaluate_fn

  def run_pretraining(self):
    """Runs pretraining."""
    if self.pretraining is None:
      logging.info('Skipping pretraining (no pretraining config provided)')
      return

    assert hasattr(self.pretraining, 'pretrain_fns')
    for k, pretraining_step in self.pretraining.pretrain_fns.items():
      logging.info('Running pretraining step: %s', k)
      self.model = pretraining_step(
          self.model, self.data_loader.all_data, self.queries_spec
      )

  def _make_initial_experiment_state(
      self, init_params, non_params
  ) -> ExperimentState:
    """Makes initial parameters and optimizer state for the experiment."""

    @jax.jit
    def init():
      params = init_params
      opt_state = self.optimization_config.optimizer.init(params)
      _, ema_state = self._ema_init(None, params)
      experiment_state = ExperimentState(
          opt_state, params, ema_state, non_params
      )
      experiment_state = train_utils.ensure_replicated(
          experiment_state, mesh=self.spmd_mesh
      )
      return experiment_state

    return init()

  def _predictions_and_targets_structs(
      self,
      eb_model: Any,
      process_obs: Any,
      data_slice_struct: PyTree,
  ) -> tuple[PyTree, PyTree]:
    """Returns shape structs for predictions and targets for aggregators."""
    target_slice_struct = data_loading.select_targets(
        data_slice_struct, self.queries_spec
    )
    process_fn = lambda proc_module, target_slice: proc_module(target_slice)
    target_struct = nnx.eval_shape(
        process_fn, process_obs, _flatten_dict(target_slice_struct)
    )
    query_struct = data_specs.construct_query(
        data_slice_struct, self.queries_spec
    )
    dummy_observe = lambda model, proc_module: proc_module(
        _flatten_dict(model.observe(query_struct))
    )
    prediction_struct = nnx.eval_shape(dummy_observe, eb_model, process_obs)
    return prediction_struct, target_struct

  def _initialize_vectorized_model(
      self,
      model: api.Model,
      rng: PyTree,
      init_slice: PyTree,
      dynamic_data: PyTree,
  ) -> Any:
    """Initializes a vectorized model for a rollout."""
    b_model = model.to_vectorized({
        typing.SimulationVariable: self.batch_axis,
        typing.DynamicInput: self.batch_axis,
    })
    b_model.update_dynamic_inputs(dynamic_data)
    b_model.assimilate(init_slice)
    eb_model = b_model.to_vectorized(
        {typing.SimulationVariable: self.ensemble_axis}
    )
    eb_model.initialize_random_processes(rng)
    return eb_model

  def _get_step_observe_process_fns(self):
    """Returns inner step and observe functions with checkpointing policies."""

    def step_fn(model: api.Model) -> None:
      model.advance()
      model_state = nnx.state(model, typing.SimulationVariable)
      model_state = self.training_mesh.with_sharding_constraint(
          model_state, 'physics'
      )
      add_name = lambda x: checkpoint_name(x, 'sim_state')
      model_state = jax.tree.map(add_name, model_state)
      nnx.update(model, model_state)

    def observe_fn(model: api.Model, query: typing.Queries):
      predictions = model.observe(query)
      predictions = self.training_mesh.with_sharding_constraint(
          predictions, 'physics'
      )
      add_name = lambda x: checkpoint_name(x, 'predictions')
      predictions = jax.tree.map(add_name, predictions)
      return predictions

    def process_fn(process_module: nnx.Module, inputs: dict[str, cx.Field]):
      processed = process_module(inputs)
      add_name = lambda x: checkpoint_name(x, 'processed')
      processed = jax.tree.map(add_name, processed)
      return processed

    if self.remat_config.step_policy == 'offload':
      step_policy = cp.save_and_offload_only_these_names(
          names_which_can_be_saved=[],  # No values stored on device.
          names_which_can_be_offloaded=['sim_state'],  # Simulation state.
          offload_src='device',  # Move from device memory.
          offload_dst='pinned_host',  # To pinned host memory.
      )
    else:
      step_policy = None

    if self.remat_config.observe_policy == 'offload':
      obs_policy = cp.save_and_offload_only_these_names(
          names_which_can_be_saved=[],  # No values stored on device.
          names_which_can_be_offloaded=['predictions'],  # Predictions.
          offload_src='device',  # Move from device memory.
          offload_dst='pinned_host',  # To pinned host memory.
      )
    else:
      obs_policy = None

    if self.remat_config.process_policy == 'offload':
      process_policy = cp.save_and_offload_only_these_names(
          names_which_can_be_saved=[],  # No values stored on device.
          names_which_can_be_offloaded=['processed'],  # Procsess outputs.
          offload_src='device',  # Move from device memory.
          offload_dst='pinned_host',  # To pinned host memory.
      )
    else:
      process_policy = None

    prevent_cse = False
    if self.remat_config.step_policy is not None:
      step_fn = nnx.remat(step_fn, prevent_cse=prevent_cse, policy=step_policy)

    if self.remat_config.observe_policy is not None:
      observe_fn = nnx.remat(
          observe_fn, prevent_cse=prevent_cse, policy=obs_policy
      )

    if self.remat_config.process_policy is not None:
      process_fn = nnx.remat(
          process_fn, prevent_cse=prevent_cse, policy=process_policy
      )
    return step_fn, observe_fn, process_fn

  def get_train_step_fn(
      self, unroll_length: int
  ) -> train_utils.TrainStepFunction:
    """Makes a function that performs a single training step.

    Args:
      unroll_length: Number of time slices in targets to compute loss against.

    Returns:
      train_step_fn: Function that performs a single training step given an
        (experiment state, step, inputs, dynamic_data) tuple and returning
        the updated experiment state and loss.
    """
    model_state_scan_axes = nnx.StateAxes(
        {typing.SimulationVariable: nnx.Carry, ...: None}
    )
    outer_steps = self._outer_steps(unroll_length)
    inner_steps = self.train_inner_steps
    timedelta = inner_steps * self.model.timestep

    # Setting up model components and helper functions.
    data_slice_struct = self.data_loader.data_slice_struct()

    process_def, process_params = nnx.split(self.process_observations)
    model_def, _, temporaries, _ = nnx.split(
        self.model,
        nnx.Param,  # params.
        (typing.SimulationVariable, typing.DynamicInput),  # temporaries.
        ...,  # non-params.
    )
    step_fn, observe_fn, process_fn = self._get_step_observe_process_fns()

    if self.data_loader.load_data_via_callback:
      targets_via_callback = self.data_loader.setup_targets_via_callback(
          data_slice_struct
      )
    else:
      targets_via_callback = None

    def batched_parameter_loss_fn(
        params, non_params, rng, inputs, dynamic_data
    ):
      """Computes evaluation metrics for a batch of targets."""
      time = coordinates.TimeDelta(np.arange(outer_steps) * timedelta)
      if self.data_loader.load_data_via_callback:
        init_slice = inputs
        loaded_targets = None
      else:
        init_slice = data_loading.slice_leading_timedelta(inputs, 1)
        loaded_targets = data_loading.select_targets(inputs, self.queries_spec)
        loaded_targets = cx.untag(loaded_targets, self.batch_axis, time)
      # Initializing the model state.
      process_obs = nnx.merge(process_def, process_params, copy=True)
      model = nnx.merge(model_def, params, non_params, temporaries, copy=True)
      eb_model = self._initialize_vectorized_model(
          model,
          rng,
          init_slice,
          dynamic_data,
      )
      eb_model_state = nnx.state(eb_model, typing.SimulationVariable)
      eb_model_state = self.training_mesh.with_sharding_constraint(
          eb_model_state, 'physics'
      )
      nnx.update(eb_model, eb_model_state)

      inner_step_fn = nnx.scan(
          step_fn,
          length=inner_steps,
          in_axes=model_state_scan_axes,
          out_axes=0,
      )

      def collect_statistics_step(
          model,
          carry,
          loss_evaluator_slice,
          loaded_targets_slice=None,
      ):
        """Collects loss and metrics aggregations for one step."""
        idx, loss_agg_state = carry
        if self.data_loader.load_data_via_callback:
          targets_slice = targets_via_callback(idx)
        else:
          targets_slice = cx.tag(loaded_targets_slice, self.batch_axis)
        query = data_specs.construct_query(targets_slice, self.queries_spec)
        prediction = process_fn(
            process_obs, _flatten_dict(observe_fn(model, query))
        )
        prediction = self.training_mesh.with_sharding_constraint(
            prediction, 'physics'
        )
        target = process_fn(process_obs, _flatten_dict(targets_slice))
        target = self.training_mesh.with_sharding_constraint(target, 'physics')
        loss_agg = self.run_evaluator(loss_evaluator_slice, prediction, target)
        loss_agg_state = self.combine_agg_states(loss_agg_state, loss_agg)
        inner_step_fn(model)  # integrates model to the next data slice.
        model_state = nnx.state(model, typing.SimulationVariable)
        model_state = self.training_mesh.with_sharding_constraint(
            model_state, 'physics'
        )
        nnx.update(model, model_state)
        return (idx + 1, loss_agg_state)

      predictions_struct, target_struct = self._predictions_and_targets_structs(
          eb_model, process_obs, data_slice_struct
      )
      init_loss_aggregations = self.loss.zeros_aggregation_states(
          predictions_struct, target_struct
      )
      # Setup context for loss evaluator.
      timedeltas = time.fields['timedelta']  # values of `time` at slices.
      dummy = cx.DummyAxis(cx.new_axis_name(timedeltas), outer_steps)
      replicate_t_coord = cx.coords.compose(dummy, time)
      replicated_deltas = timedeltas.broadcast_like(replicate_t_coord)
      loss_evaluator_with_context = self.loss.with_context({
          'timedelta': timedeltas.untag('timedelta'),
          'times': replicated_deltas.untag(dummy),
      })
      if self.remat_config.remat_collect_statistics:
        collect_statistics_step = nnx.remat(collect_statistics_step)
      scan_fn = nnx.scan(
          collect_statistics_step,
          length=outer_steps,
          in_axes=(model_state_scan_axes, nnx.Carry, 0, 1),
          out_axes=nnx.Carry,
      )
      _, loss_agg_state = scan_fn(
          eb_model,
          (0, init_loss_aggregations),
          loss_evaluator_with_context,
          loaded_targets,
      )
      loss_value = self.loss.evaluate_total({}, {}, loss_agg_state).data
      return loss_value

    # We would use donate_argnums here to update experiment_state in-place, but
    # that would mean we could not save the checkpoint in a separable thread.
    # Fortunately experiment_state is usually not too big (~100 MB).
    @train_utils.jit_once
    def train_step(experiment_state, step, inputs, dynamic_data):
      opt_state, params, ema_state, non_params = experiment_state
      rng = train_utils.batch_and_ensemble_parallel_rng_key(
          batch_size=self.global_batch_size,
          ensemble_size=self.ensemble_axis.size,
          seeds=(self.train_schedule.random_process_rng_seed, step),
          mesh=self.spmd_mesh,
      )
      rng = cx.field(rng, self.batch_axis, self.ensemble_axis)
      loss, grad = jax.value_and_grad(batched_parameter_loss_fn)(
          params, non_params, rng, inputs, dynamic_data
      )

      updates, opt_state = self.optimization_config.optimizer.update(
          grad, opt_state, params
      )
      params = optax.apply_updates(params, updates)
      _, ema_state = self._ema_update(params, ema_state)
      experiment_state = ExperimentState(
          opt_state, params, ema_state, non_params
      )
      experiment_state = train_utils.ensure_replicated(
          experiment_state, mesh=self.spmd_mesh
      )
      return experiment_state, loss

    return train_step

  def get_eval_batch_fn(
      self,
      train_trajectory_length: int,
      seed: int = 0,
  ) -> Callable[..., Any]:
    """Makes a function that performs a single evaluation pass."""
    outer_steps = self._eval_trajectory_length
    inner_steps = self.train_inner_steps  # equal to eval inner steps.
    model_state_scan_axes = nnx.StateAxes(
        {typing.SimulationVariable: nnx.Carry, ...: None}
    )
    timedelta = inner_steps * self.model.timestep  # time between data slices.

    # We adjust loss_evaluator to only include contributions from
    # train_trajectory_length time steps.
    loss_evaluator = self.loss
    train_timedelta = coordinates.TimeDelta(
        np.arange(train_trajectory_length) * timedelta
    )
    timedelta_masking = weighting.CoordinateMaskWeighting(
        train_timedelta, masked_value=1.0, unmasked_value=0.0
    )
    add_timedelta_mask = lambda aggr: aggregation.Aggregator(
        dims_to_reduce=aggr.dims_to_reduce,
        weight_by=aggr.weight_by + tuple([timedelta_masking]),
        scale_by=aggr.scale_by,
        bin_by=aggr.bin_by,
        skip_missing=aggr.skip_missing,
        skipna=aggr.skipna,
        keep_weights_for_nans=aggr.keep_weights_for_nans,
        context=aggr.context,
    )
    updated_aggregators = jax.tree.map(
        add_timedelta_mask,
        loss_evaluator.aggregators,
        is_leaf=lambda x: isinstance(x, aggregation.Aggregator),
    )
    loss_evaluator = dataclasses.replace(
        loss_evaluator, aggregators=updated_aggregators
    )

    # Setting up model components and helper functions.
    data_slice_struct = self.data_loader.data_slice_struct()

    process_def, process_params = nnx.split(self.process_observations)
    model_def, _, temporaries, _ = nnx.split(
        self.model,
        nnx.Param,  # params.
        (typing.SimulationVariable, typing.DynamicInput),  # temporaries.
        ...,  # non-params.
    )
    step_fn, observe_fn, process_fn = self._get_step_observe_process_fns()

    if self.data_loader.load_data_via_callback:
      targets_via_callback = self.data_loader.setup_targets_via_callback(
          data_slice_struct
      )
    else:
      targets_via_callback = None

    @train_utils.jit_once
    def batch_mean_eval_fn(params, non_params, eval_step, inputs, dynamic_data):
      """Computes evaluation metrics for a batch of targets."""
      time = coordinates.TimeDelta(np.arange(outer_steps) * timedelta)
      if self.data_loader.load_data_via_callback:
        init_slice = inputs
        loaded_targets = None
      else:
        init_slice = data_loading.slice_leading_timedelta(inputs, 1)
        loaded_targets = data_loading.select_targets(inputs, self.queries_spec)
        loaded_targets = cx.untag(loaded_targets, self.batch_axis, time)
      # Initializing the model state.
      [batch_size], [ensemble_size] = (
          self.batch_axis.shape,
          self.ensemble_axis.shape,
      )
      rng = train_utils.batch_and_ensemble_parallel_rng_key(
          batch_size=batch_size,
          ensemble_size=ensemble_size,
          seeds=(seed, eval_step),
          mesh=self.spmd_mesh,
      )
      rng = cx.field(rng, self.batch_axis, self.ensemble_axis)
      process_obs = nnx.merge(process_def, process_params, copy=True)
      model = nnx.merge(model_def, params, non_params, temporaries, copy=True)
      eb_model = self._initialize_vectorized_model(
          model,
          rng,
          init_slice,
          dynamic_data,
      )

      inner_step_fn = nnx.scan(
          step_fn,
          length=inner_steps,
          in_axes=model_state_scan_axes,
          out_axes=0,
      )

      def _collect_statistics_step(
          model,
          carry,
          metrics_evaluator_slice,
          loss_evaluator_slice,
          targets_slice_from_scan=None,
      ):
        """Collects loss and metrics aggregations for one step."""
        idx, metrics_agg_state, loss_agg_state = carry
        if targets_via_callback is not None:
          targets_slice = targets_via_callback(idx)
        else:
          targets_slice = cx.tag(targets_slice_from_scan, self.batch_axis)
        query = data_specs.construct_query(targets_slice, self.queries_spec)
        prediction = observe_fn(model, query)
        prediction = process_fn(process_obs, _flatten_dict(prediction))
        target = process_fn(process_obs, _flatten_dict(targets_slice))
        metrics_agg = self.run_evaluator(
            metrics_evaluator_slice, prediction, target
        )
        metrics_agg_state = self.combine_agg_states(
            metrics_agg_state, metrics_agg
        )
        loss_agg = self.run_evaluator(loss_evaluator_slice, prediction, target)
        loss_agg_state = self.combine_agg_states(loss_agg_state, loss_agg)
        inner_step_fn(model)  # integrates model to the next data slice.
        return (idx + 1, metrics_agg_state, loss_agg_state)

      prediction_struct, target_struct = self._predictions_and_targets_structs(
          eb_model, process_obs, data_slice_struct
      )
      init_metrics_aggregations = self.eval_metrics.zeros_aggregation_states(
          prediction_struct, target_struct
      )
      init_loss_aggregations = loss_evaluator.zeros_aggregation_states(
          prediction_struct, target_struct
      )
      timedeltas = time.fields['timedelta']  # values of `time` at slices.
      eval_metrics_with_context = self.eval_metrics.with_context(
          {'timedelta': timedeltas.untag('timedelta')}
      )
      # For loss context we provide timedelta and slices of full time axis that
      # corresponds only to the training slice to correctly account for any
      # normalization constants.
      train_timedeltas = train_timedelta.fields['timedelta']
      dummy = cx.DummyAxis(cx.new_axis_name(train_timedeltas), outer_steps)
      replicate_coord = cx.coords.compose(dummy, train_timedelta)
      replicated_deltas = train_timedeltas.broadcast_like(replicate_coord)
      loss_evaluator_with_context = loss_evaluator.with_context({
          'timedelta': timedeltas.untag('timedelta'),
          'times': replicated_deltas.untag(dummy),
      })
      # in targets time is axis 1, if loading data via callback, it is ignored.
      scan_fn = nnx.scan(
          _collect_statistics_step,
          length=outer_steps,
          in_axes=(model_state_scan_axes, nnx.Carry, 0, 0, 1),
          out_axes=nnx.Carry,
      )
      _, metrics_agg_state, loss_agg_state = scan_fn(
          eb_model,
          (0, init_metrics_aggregations, init_loss_aggregations),
          eval_metrics_with_context,
          loss_evaluator_with_context,
          loaded_targets,
      )
      # record metrics
      values_to_record = {}
      metric_values = self.eval_metrics.evaluate_metrics(
          {}, {}, metrics_agg_state
      )
      for metric_key, metric_dict in metric_values.items():
        for scalar_name, v in metric_dict.items():
          values_to_record['.'.join([metric_key, scalar_name])] = v.data
      # record loss
      loss_val = loss_evaluator.evaluate_total({}, {}, loss_agg_state).data
      values_to_record['loss'] = loss_val
      for term_key, term_metric in loss_evaluator.metrics.items():
        if loss_evaluator.term_weights is not None:
          w = loss_evaluator.term_weights.get(term_key, 1.0)
        else:
          w = 1.0
        mean_stats = loss_agg_state[term_key].mean_statistics()
        term_metric_values = loss_agg_state[term_key].metric_values(term_metric)
        relative_value_key = '.'.join(['relative', term_key])
        loss_term_value = w * term_metric.total(term_metric_values).data
        values_to_record[relative_value_key] = loss_term_value / loss_val

        debug_terms = term_metric.debug_terms(mean_stats, term_metric_values)
        for debug_term_name, v in debug_terms.items():
          key = '.'.join([term_key, debug_term_name])
          values_to_record[key] = v.data
      return values_to_record

    return batch_mean_eval_fn

  def save_online_metrics(self, step: int, online_metrics: OnlineEvalMetrics):
    # only save from the coordinator process to avoid redundant copies.
    if is_coordinator():
      self.online_metrics_saver.save(step, online_metrics)

  def run_training(self):
    """Runs the training experiment."""
    start_step, auto_restart, experiment_state = self.initialize_experiment()

    if start_step >= self.train_schedule.total_steps:
      logging.warning(
          f'Attempting to start training at {start_step=} >='
          f' {self.train_schedule.total_steps=}. Will simply return.'
      )
      return

    def logging_callback(step, schedule_idx, loss, auto_restart):
      loss = float(jax.device_get(loss))
      if step % max(self.eval_schedule.steps_between_evals // 100, 1) == 0:
        logging.info(f'{schedule_idx=}, {step=}, {loss=}')
      if (
          self.auto_restart_config.error_with_nan_loss
          and auto_restart.iteration > self.auto_restart_config.max_nan_restarts
      ):
        raise RuntimeError(
            f'NaN loss detected at {step=}, after too many restarts since'
            f' {auto_restart.iteration=} >'
            f' {self.auto_restart_config.max_nan_restarts=}. Aborting.'
        )

    # monitor loss using a separate thread, so it doesn't block execution
    logging_stream = streaming.SingleTaskExecutor(logging_callback)
    logging.info('starting training from step=%s', start_step)
    train_step_timer = timing.Timer()

    prev_loss = loss = 1.0  # initialize with non-NaN value
    schedule_idx = None

    train_step_fn, evaluate_fn, train_iter = None, None, None  # pytype happy.
    step = start_step
    while step < self.train_schedule.total_steps:
      old_schedule_idx = schedule_idx
      schedule_idx = np.sum(  # compute which leg of the schedule we are at.
          step > np.asarray(self.train_schedule.schedule_boundaries)
      )
      if schedule_idx != old_schedule_idx:
        (
            train_iter,
            train_step_fn,
            evaluate_fn,
        ) = self.build_train_and_eval_iterators(schedule_idx)

      # restart on NaN loss
      if (
          np.isnan(prev_loss)
          # TODO(langmore) This means if config.max_nan_restarts=0 we still have
          # one restart! Instead we should keep track of times_encountered_nan
          # or something like that.
          and auto_restart.iteration
          <= self.auto_restart_config.max_nan_restarts
      ):
        # See also logging_callback, which may raise RuntimeError for NaN loss.
        logging.warning(
            f'NaN loss encountered at {step=}. Re-initializing and incrementing'
            f' auto-restart iteration to {auto_restart.iteration + 1}'
        )
        auto_restart = AutoRestart(
            iteration=auto_restart.iteration + 1,
            began_at=auto_restart.began_at or step,
        )
        lookback = (
            auto_restart.iteration
            * self.auto_restart_config.restart_lookback_steps
        )
        target_step = step - 1 - lookback
        step, experiment_state = self.reinitialize_for_nan_restart(target_step)

      if (
          auto_restart.iteration
          and step - self.max_lookback_interval > auto_restart.began_at
      ):
        logging.info(
            f'Significant progress made since {auto_restart.began_at=}.'
            f' In particular, {step=}. Therefore reset auto_restart.'
        )
        auto_restart = AutoRestart()

      # save checkpoint
      if start_step == 0 or step > start_step:
        # don't override initial checkpoints
        self.save_checkpoint(step, auto_restart, experiment_state)

      # evaluate
      if (step + 1) % self.eval_schedule.steps_between_evals == 0:
        if step > start_step:
          with train_step_timer:
            # train_step is non-blocking, so we need to block on the output
            # of the previous training step to reliably time it.
            experiment_state = jax.block_until_ready(experiment_state)
          training_time = train_step_timer.total
          logging.info(
              f'training for {self.eval_schedule.steps_between_evals} steps'
              f' took {training_time:.1f} seconds'
          )
          train_step_timer = timing.Timer()  # reset
        else:
          training_time = None

        online_metrics = evaluate_fn(experiment_state)
        if step > start_step:
          assert training_time is not None
          online_metrics.seconds_per_train_step = (
              training_time / self.eval_schedule.steps_between_evals
          )
        self.save_online_metrics(step, online_metrics)
        logging.info(
            'evaluation pass took %.1f seconds',
            online_metrics.seconds_per_evaluation,
        )

      # train
      with train_step_timer:
        # go/xprof-instrument-jax
        with jax.profiler.StepTraceAnnotation('train', step_num=step):
          inputs, dynamic_data = next(train_iter)
          prev_loss = loss
          experiment_state, loss = train_step_fn(
              experiment_state, step, inputs, dynamic_data
          )
          logging_stream.wait()  # don't allow training to run ahead of logging.
          logging_stream.submit(step, schedule_idx, loss, auto_restart)
      step += 1
    # End of while step < self.config.num_training_steps:

    logging.info('finished training')
    eval_metrics = evaluate_fn(experiment_state)
    self.save_online_metrics(self.train_schedule.total_steps, eval_metrics)
    self.save_checkpoint(
        self.train_schedule.total_steps, auto_restart, experiment_state
    )
    self.checkpoint_manager.close()

  @functools.cached_property
  def max_lookback_interval(self) -> int:
    """Returns the maximum number of steps to look back for a restart."""
    return (
        self.auto_restart_config.max_nan_restarts
        * self.auto_restart_config.restart_lookback_steps
    )

  @functools.cached_property
  def checkpoint_manager(self) -> ocp.CheckpointManager:
    """Orbax CheckpointManager."""
    options = ocp.CheckpointManagerOptions(
        file_options=ocp.checkpoint_manager.FileOptions(
            path_permission_mode=0o775,  # world-readable
        ),
        save_interval_steps=self.checkpoint_config.save_interval_steps,
        preservation_policy=ocp_managers.AnyPreservationPolicy([
            checkpointing.LastNSteps(self.max_lookback_interval),
            ocp_managers.EveryNSteps(
                interval_steps=self.checkpoint_config.keep_every_n_steps
            ),
            ocp_managers.CustomSteps(
                steps=self.train_schedule.schedule_boundaries
            ),
        ]),
        save_decision_policy=ocp_managers.AnySavePolicy([
            ocp_managers.ContinuousCheckpointingPolicy(
                minimum_interval_secs=60
            ),
            ocp_managers.PreemptionCheckpointingPolicy(),
            ocp_managers.FixedIntervalPolicy(
                interval=self.eval_schedule.steps_between_evals
            ),
            ocp_managers.SpecificStepsPolicy(
                steps=self.train_schedule.schedule_boundaries
            ),
        ]),
    )
    return checkpointing.training_manager(
        self._checkpoint_dir,
        options=options,
        model_config_str=self.checkpoint_config.model_config_str,
        metadata=self.checkpoint_config.metadata,
    )

  def initialize_experiment(self) -> tuple[int, AutoRestart, ExperimentState]:
    """Initialize this experiment at the start of the training loop."""
    latest_step = self.checkpoint_manager.latest_step()
    if self.initial_checkpoint is not None:
      initial_dir = self.initial_checkpoint.checkpoint_dir
    else:
      initial_dir = None

    if latest_step is not None:
      logging.info('Resuming training from saved checkpoint')
      args = self.get_checkpoint_state(restore=True)
      ckpt = self.checkpoint_manager.restore(latest_step, args=args)
      return self._initialize_from_checkpoint(ckpt)

    elif initial_dir is not None:
      logging.info('Starting training with weights from another experiment')
      initial_checkpoint = self.initial_checkpoint
      assert initial_checkpoint is not None
      args = self.get_checkpoint_state(restore=True)
      with checkpointing.read_only_manager(initial_dir) as manager:
        ckpt = manager.restore(initial_checkpoint.checkpoint_step, args=args)

      if initial_checkpoint.reset_optimizer_state:
        logging.info('Resetting optimizer state, only using EMA params')
        return self._initialize_for_fresh_start(
            ckpt.ema_params, ckpt.non_params
        )
      else:
        return self._initialize_from_checkpoint(ckpt)

    else:
      logging.info('Starting training with new weights')
      self.run_pretraining()
      split_state = model_checkpointing.split_model_state_for_saving(self.model)
      return self._initialize_for_fresh_start(
          split_state.params, split_state.non_params
      )

  def _initialize_for_fresh_start(
      self, params: nnx.GraphState, non_params: nnx.GraphState
  ) -> tuple[int, AutoRestart, ExperimentState]:
    """Convert params and non_params into training experiment state."""
    start_step = 0
    auto_restart = AutoRestart()
    exp_state = self._make_initial_experiment_state(params, non_params)
    return (start_step, auto_restart, exp_state)

  def _initialize_from_checkpoint(
      self, ckpt: ocp.args.Composite
  ) -> tuple[int, AutoRestart, ExperimentState]:
    """Convert orbax composite args into training experiment state."""
    start_step = ckpt.metadata['step']
    logging.info(f'Restored experiment from checkpoint at step={start_step}')
    auto_restart = AutoRestart(**ckpt.metadata['auto_restart'])
    experiment_state = ExperimentState(
        ckpt.opt_state, ckpt.params, ckpt.ema_state, ckpt.non_params
    )
    experiment_state = train_utils.ensure_replicated(
        experiment_state, mesh=self.spmd_mesh
    )
    return (start_step, auto_restart, experiment_state)

  def reinitialize_for_nan_restart(
      self, target_step: int | None = None
  ) -> tuple[int, ExperimentState]:
    """Reinitialize this experiment for an auto-restart after NaN loss."""
    all_steps = np.array(self.checkpoint_manager.all_steps())
    logging.info(f'Found buffered checkpoints at steps {all_steps}')
    saved_step = int(all_steps[np.argmin(np.abs(all_steps - target_step))])
    logging.info(
        f'Using buffered checkpoint, which has step={saved_step}. Ideally '
        f'would have used {target_step=}'
    )
    args = self.get_checkpoint_state(restore=True)
    ckpt = self.checkpoint_manager.restore(saved_step, args=args)
    start_step, _, experiment_state = self._initialize_from_checkpoint(ckpt)
    return (start_step, experiment_state)

  def save_checkpoint(
      self,
      step: int,
      auto_restart: AutoRestart,
      experiment_state: ExperimentState,
  ) -> None:
    """Saves an experiment checkpoint using Orbax."""
    experiment_state = jax.block_until_ready(experiment_state)
    # Orbax requires saving from all Python processes, even with multiple hosts.
    args = self.get_checkpoint_state(
        step, experiment_state, auto_restart=auto_restart, save=True
    )
    self.checkpoint_manager.save(step, args=args)

  def get_checkpoint_state(
      self,
      step: int | None = None,
      experiment_state: ExperimentState | None = None,
      *,
      auto_restart: AutoRestart | None = None,
      save: bool = False,
      restore: bool = False,
  ) -> ocp.args.Composite:
    """Returns a checkpoint state for a given experiment_state."""
    if save and not restore:
      if step is None or experiment_state is None or auto_restart is None:
        raise ValueError(
            'step, experiment_state, and auto_restart must be '
            'provided when saving a checkpoint.'
        )
      wrap_pytree = ocp.args.StandardSave
      metadata = ocp.args.JsonSave({
          'step': step,
          'auto_restart': dataclasses.asdict(auto_restart),
      })
    elif restore and not save:
      wrap_pytree = ocp.args.StandardRestore
      metadata = ocp.args.JsonRestore()
    else:
      raise ValueError(f'must either save or restore, got {save=} {restore=}')

    if experiment_state is None:
      assert restore
      split_state = model_checkpointing.split_model_state_for_saving(self.model)
      experiment_state = self._make_initial_experiment_state(
          split_state.params, split_state.non_params
      )

    opt_state, params, ema_state, non_params = experiment_state
    ema_params, _ = self._ema_update(params, ema_state)  # pytype: disable=attribute-error  # jax-api-types

    ckpt_state = ocp.args.Composite(
        # state required for inference
        params=wrap_pytree(params),
        ema_params=wrap_pytree(ema_params),
        non_params=wrap_pytree(non_params),
        # state only used for training
        opt_state=wrap_pytree(opt_state),
        ema_state=wrap_pytree(ema_state),
        metadata=metadata,
    )
    return ckpt_state

  def dummy_evaluate(
      self, experiment_state: ExperimentState
  ) -> OnlineEvalMetrics:
    """Dummy evaluation step, for use with num_eval_batches=0 in unit tests."""
    del experiment_state  # unused
    assert self.eval_schedule.num_batches == 0
    logging.warning(f'skipping evaluation: {self.eval_schedule.num_batches=}')
    return OnlineEvalMetrics()

  def evaluate(
      self,
      experiment_state: ExperimentState,
      eval_batch_fn,
      eval_data,
      train_data,
  ) -> OnlineEvalMetrics:
    """Evaluates the model on train and eval data and writes summaries.

    Args:
      experiment_state: tuple of replicated step, optimizer state and EMA
        (exponentially moving average) state for model parameters.
      eval_batch_fn: function that, given parameters; rng; batch of data,
        computes evaluation metric of interest on the given samples.
      eval_data: iterable over evaluation data that is used for produce
        summaries on unseen evaluation data.
      train_data: iterable over training data that is used for produce summaries
        on training data.

    Returns:
      Online evaluation metrics.
    """
    if self._eval_trajectory_length == 0:  # pytype: disable=attribute-error  # jax-api-types
      return OnlineEvalMetrics()  # no evaluation to do.
    assert self.eval_schedule.num_batches > 0

    with timing.Timer() as eval_timer:
      _, params, ema_state, non_params = experiment_state
      ema_params, _ = self._ema_update(params, ema_state)  # pytype: disable=attribute-error  # jax-api-types

      # TODO(dkochkov): Rather than using train_utils.streaming_mean, use
      # aggregation state aware computation. This would increase the scope of
      # metrics that we can compute correctly.
      results = {}

      logging.info('evaluating on train dataset')
      metrics_ = train_utils.streaming_mean(
          train_data,
          functools.partial(eval_batch_fn, params, non_params),
      )
      results['train'] = {k: float(v) for k, v in metrics_.items()}

      logging.info('evaluating on test dataset')
      metrics_ = train_utils.streaming_mean(
          eval_data,
          functools.partial(eval_batch_fn, params, non_params),
      )
      results['eval'] = {k: float(v) for k, v in metrics_.items()}

      logging.info('evaluating EMA model on test dataset')
      metrics_ = train_utils.streaming_mean(
          eval_data,
          functools.partial(eval_batch_fn, ema_params, non_params),
      )
      results['eval_ema'] = {k: float(v) for k, v in metrics_.items()}

    return OnlineEvalMetrics(
        train=results['train'],
        eval=results['eval'],
        eval_ema=results['eval_ema'],
        seconds_per_evaluation=eval_timer.total,
    )
