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
from typing import Any, NamedTuple, TypeAlias

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
from neuralgcm.experimental.core import scan_utils
from neuralgcm.experimental.core import typing
from neuralgcm.experimental.inference import streaming
from neuralgcm.experimental.inference import timing
from neuralgcm.experimental.metrics import aggregation
from neuralgcm.experimental.metrics import base as metrics_base
from neuralgcm.experimental.metrics import evaluators
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

DataSpec: TypeAlias = dict[str, dict[str, cx.Coordinate]]
NestedAggStates: TypeAlias = tuple[aggregation.AggregationState | None, ...]

TrainEvalIteratorTuple = tuple[
    Any,
    train_utils.TrainStepFunction,
    list[tuple[int, Callable[..., Any]]],
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
  # TODO(dkochkov): Consider evaluator default behavior with no targets.
  if not prediction and not target:
    return {}
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


def _is_coord(x: Any) -> bool:
  """Returns True if x is a coordinate."""
  return isinstance(x, cx.Coordinate)


def _remove_timedelta(c: cx.Coordinate) -> cx.Coordinate:
  """Removes TimeDelta axes from a coordinate."""
  return cx.coords.compose(
      *[ax for ax in c.axes if not isinstance(ax, coordinates.TimeDelta)]
  )


def _untag_positional(inputs: PyTree, axis: cx.Coordinate) -> PyTree:
  """Untags `axis` from fields in `inputs` even if they have positional axes."""
  # TODO(dkochkov): Consider supporting untag on Fields with positional axes.

  def _untag(f: cx.Field) -> cx.Field:
    tmp_axes = []
    for _ in f.positional_shape:
      name = cx.new_axis_name(f, set(tmp_axes))
      tmp_axes.append(name)
    f = f.tag(*tmp_axes)
    f = f.untag(axis, *tmp_axes)
    return f

  return jax.tree.map(_untag, inputs, is_leaf=cx.is_field)


def _prepare_inputs_and_targets(
    inputs: PyTree,
    model_dt: np.timedelta64,
    retrieve_fns: tuple[Callable[[int], PyTree], ...] | None,
    queries_spec: data_specs.QueriesSpec,
    batch_axis: cx.SizedAxis,
) -> tuple[PyTree, PyTree | None]:
  """Prepares initial slice and targets for training/eval step."""
  if retrieve_fns is not None:
    init_slice = inputs
    loaded_targets = None
  else:
    init_slice = data_loading.slice_leading_timedelta(inputs, 1)
    # Select targets starting from t > 0
    targets_inputs = data_loading.sel_target_fields(inputs)
    loaded_targets = data_loading.select_targets(targets_inputs, queries_spec)
    # We use already computed scan_steps and specs here since loaded targets
    # might contain a single timedelta value which is not sufficient to
    # resolve the number of steps.
    loaded_targets = scan_utils.nest_data_for_scans(
        loaded_targets, model_dt, ref_t0=np.timedelta64(0, 's')
    )
    loaded_targets = _untag_positional(loaded_targets, batch_axis)
  return init_slice, loaded_targets


def _compute_aggregation(
    step_idx: int,
    model: api.Model,
    current_agg_states: tuple[aggregation.AggregationState | None, ...],
    evaluator_slices: tuple[evaluators.Evaluator | None, ...],
    loaded_targets_slice: PyTree | None,
    retrieve_fn: Callable[[int], PyTree] | None,
    query_spec: data_specs.QueriesSpec,
    observe_fn: Callable[..., PyTree],
    process_fn: Callable[..., PyTree],
    process_obs: Any,
    run_evaluator_fn: Callable[..., aggregation.AggregationState],
    combine_agg_states_fn: Callable[..., aggregation.AggregationState],
    training_mesh: parallelism.Mesh,
    batch_axis: cx.SizedAxis,
) -> tuple[aggregation.AggregationState | None, ...]:
  """Computes aggregation state update for one or more evaluators.

  This function computes the predictions from the model, retrieves the targets
  (either from pre-loaded slices or via a callback), computes the metrics using
  the provided evaluators, and updates the aggregation states.

  Args:
    step_idx: The step index representing number of model steps taken so far.
    model: The model to use for making predictions.
    current_agg_states: A tuple of aggregation states to update with
      contributions from predictions and targets. Aggregation states are
      expected to be matched with evaluators in `evaluator_slices`.
    evaluator_slices: A tuple of evaluators to compute aggregation state updates
      with. If entry is None, the corresponding update is skipped.
    loaded_targets_slice: A PyTree containing the targets for the current step.
    retrieve_fn: An optional callable that retrieves targets for a given step
      index. loaded_targets and retrieve_fn are mutually exclusive.
    query_spec: A specification for the model query for this evaluation.
    observe_fn: A function that generates observations from the model and query.
    process_fn: A callable that processes observations and targets.
    process_obs: A module or parameters used by `process_fn`.
    run_evaluator_fn: A callable that computes aggregations.
    combine_agg_states_fn: A callable that combines aggregation states.
    training_mesh: The parallelism mesh used for sharding constraints.
    batch_axis: The axis representing the batch dimension.

  Returns:
    A tuple of updated aggregation states, corresponding to the input
    `current_agg_states`.
  """
  if retrieve_fn is not None:
    targets_slice = retrieve_fn(step_idx)
  else:
    targets_slice = cx.tag(loaded_targets_slice, batch_axis)

  query = data_specs.construct_query(targets_slice, query_spec)
  prediction = process_fn(process_obs, _flatten_dict(observe_fn(model, query)))
  prediction = training_mesh.with_sharding_constraint(prediction, 'physics')
  target = process_fn(process_obs, _flatten_dict(targets_slice))
  target = training_mesh.with_sharding_constraint(target, 'physics')

  new_agg_states = []
  for current_agg_state, evaluator in zip(current_agg_states, evaluator_slices):
    if evaluator is None:
      new_agg_states.append(None)
      continue

    agg_update = run_evaluator_fn(evaluator, prediction, target)
    updated_agg_state = combine_agg_states_fn(current_agg_state, agg_update)
    new_agg_states.append(updated_agg_state)

  return tuple(new_agg_states)


def create_nested_evaluators(
    evaluator: evaluators.Evaluator,
    nested_targets_specs: tuple[DataSpec, ...],
    dt: np.timedelta64,
) -> tuple[evaluators.Evaluator, ...]:
  """Creates a tuple of evaluators with temporal context for nested scans.

  To support calculation of "timedelta"-dependent scaling or weighting,
  evaluators must be provided with context fields that have positional shapes
  compatible with the nested scans used to calculate statistics. This function
  creates "timedelta" and "times" context fields for each scan level defined via
  `nested_targets_specs` and return a evaluators with context fields for each.

  Args:
    evaluator: Evaluator for which to create nested evaluators with context.
    nested_targets_specs: Targets specs for each scan level from inner to outer.
    dt: Model timestep that will be used as most inner step.

  Returns:
    Tuple of nested evaluators with "timedelta" and "times" context fields.
  """
  extract_td = lambda c: cx.coords.extract(c, coordinates.TimeDelta)
  timedelta_replica_name = 'timedelta_replica'

  def _swap_replica(f: cx.Field) -> cx.Field:
    """Swaps dummy axis with TimeDelta."""
    if timedelta_replica_name in f.dims:
      dummy = f.axes[timedelta_replica_name]
      td = coordinates.TimeDelta(dummy.ticks)
      return cx.field(f.data, cx.coords.replace_axes(f.coordinate, dummy, td))
    return f

  nested_prediction_specs = data_loading.sel_target_timedeltas(
      nested_targets_specs
  )
  context_fields = {}
  for nest_level, prediction_spec in enumerate(nested_prediction_specs):
    timedeltas = jax.tree.map(extract_td, prediction_spec, is_leaf=_is_coord)
    timedeltas = jax.tree.leaves(timedeltas, is_leaf=_is_coord)
    timedeltas = set(timedeltas)
    if len(timedeltas) > 1:
      raise ValueError(
          f'Timedeltas must be unique for each scan level, got {timedeltas=}'
      )
    if timedeltas:
      timedelta_coord = timedeltas.pop()
      timedeltas = timedelta_coord.fields['timedelta']
      # add suffix to avoid name collisions, will be removed below.
      context_data = {f'timedelta_{nest_level}': timedeltas}
      replica = cx.LabeledAxis(timedelta_replica_name, timedelta_coord.deltas)
      replicated_coord = cx.coords.compose(timedelta_coord, replica)
      replicated_deltas = timedeltas.broadcast_like(replicated_coord)
      context_data[f'times_{nest_level}'] = replicated_deltas
      context_fields |= context_data

  nested_context_fields = scan_utils.nest_data_for_scans(
      context_fields, dt=dt, ref_t0=np.timedelta64(0, 's')
  )
  nested_context_fields = tuple(
      {k.removesuffix(f'_{i}'): _swap_replica(v) for k, v in d.items()}
      for i, d in enumerate(nested_context_fields)
  )
  return tuple(
      evaluator.with_context(fields) for fields in nested_context_fields
  )


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


def _merge_online_metrics(
    metrics_list: Sequence[OnlineEvalMetrics]
) -> OnlineEvalMetrics:
  """Merges a sequence of online metrics into a single OnlineEvalMetrics."""
  if not metrics_list:
    return OnlineEvalMetrics()

  merged_train = {}
  merged_eval = {}
  merged_eval_ema = {}
  total_eval_time = 0.0
  seconds_per_train_step = None

  for m in metrics_list:
    for k, v in m.train.items():
      if k in merged_train:
        raise ValueError(
            f'Metric key collision detected in train: {k}. '
            'Ensure EvalSchemas have unique names.'
        )
      merged_train[k] = v
    for k, v in m.eval.items():
      if k in merged_eval:
        raise ValueError(
            f'Metric key collision detected in eval: {k}. '
            'Ensure EvalSchemas have unique names.'
        )
      merged_eval[k] = v
    for k, v in m.eval_ema.items():
      if k in merged_eval_ema:
        raise ValueError(
            f'Metric key collision detected in eval_ema: {k}. '
            'Ensure EvalSchemas have unique names.'
        )
      merged_eval_ema[k] = v

    if m.seconds_per_evaluation is not None:
      total_eval_time += m.seconds_per_evaluation
    if m.seconds_per_train_step is not None:
      seconds_per_train_step = m.seconds_per_train_step

  return OnlineEvalMetrics(
      train=merged_train,
      eval=merged_eval,
      eval_ema=merged_eval_ema,
      seconds_per_evaluation=total_eval_time,
      seconds_per_train_step=seconds_per_train_step,
  )


PretrainOp = Callable[..., Any]


@dataclasses.dataclass
class PretrainingOps:
  pretrain_fns: collections.OrderedDict[str, PretrainOp]


@dataclasses.dataclass(frozen=True)
class TrainStage:
  """Specifies a single stage of training."""
  duration: int  # Number of training steps to run this stage for.
  inputs_spec: DataSpec
  dynamic_inputs_spec: DataSpec
  queries_spec: data_specs.QueriesSpec
  loss: evaluators.Evaluator[metrics_base.Loss]
  time_sample_offset: np.timedelta64
  batch_size_per_device: int = 1
  time_slice: tuple[str, str] | list[tuple[str, str]] | None = None
  shuffle_buffer_size: int = 0
  dataset_rng_seed: int = 0
  random_process_rng_seed: int = 0


@dataclasses.dataclass(frozen=True)
class TrainSchedule:
  """Specifies a sequence of training stages."""
  stages: Sequence[TrainStage]


@dataclasses.dataclass(frozen=True)
class EvalSchema:
  """Specifies a single evaluation run."""
  cadence: int  # how frequently to run this eval stage.
  inputs_spec: DataSpec
  dynamic_inputs_spec: DataSpec
  queries_spec: data_specs.QueriesSpec
  metrics_evaluator: evaluators.Evaluator[metrics_base.Metric] | None
  batch_size_per_device: int
  num_batches: int
  time_sample_offset: np.timedelta64
  loss_evaluator: evaluators.Evaluator[metrics_base.Loss] | None = None
  time_slice: tuple[str, str] | list[tuple[str, str]] | None = None
  name: str | None = None


@dataclasses.dataclass(frozen=True)
class EvalSchedule:
  """Specifies a sequence of evaluation runs."""
  stages: Sequence[EvalSchema]


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
  use_data_loading_callback: bool = False
  callback_pinned_host: bool = False
  callback_spatial_dims_layout: tuple[str, ...] = ()

  def __post_init__(self):
    if self.data_loader.parallelism_mesh is None:
      raise ValueError(
          'RolloutTrainer requires {data_loader.training_mesh=} to be'
          ' specified.'
      )
    self.run_evaluator = _maybe_on_host(
        _run_evaluator, self.compute_loss_on_host
    )
    self.combine_agg_states = _maybe_on_host(
        _combine_agg_states, self.compute_loss_on_host
    )

    # Validating compatibility of the data loader with the model.
    for stage in self.train_schedule.stages:
      try:
        data_specs.validate_inputs(stage.inputs_spec, self.model.inputs_spec)
      except ValueError as e:
        raise ValueError(
            f'{stage.inputs_spec=} is not compatible with the model'
            f' requirements {self.model.inputs_spec=}. Please check that the '
            ' `train_schedule_config` defines consistent inputs_spec.'
        ) from e
      try:
        data_specs.validate_inputs(
            stage.dynamic_inputs_spec, self.model.dynamic_inputs_spec
        )
      except ValueError as e:
        raise ValueError(
            f'{stage.dynamic_inputs_spec=} is not compatible with the model'
            f' requirements {self.model.dynamic_inputs_spec=}. Please check'
            ' that the `train_schedule_config` defines consistent'
            ' dynamic_inputs_spec.'
        ) from e

    self._checkpoint_dir = os.path.join(self.experiment_dir, 'checkpoints')
    epath.Path(self._checkpoint_dir).mkdir(
        parents=True, exist_ok=True, mode=0o775
    )

    self._train_schedule_boundaries = np.cumsum(
        [s.duration for s in self.train_schedule.stages]
    )
    self._total_train_steps = (
        self._train_schedule_boundaries[-1]
        if len(self._train_schedule_boundaries) > 0  # pylint: disable=g-explicit-length-test
        else 0
    )
    self.frequent_eval_stage = min(
        self.eval_schedule.stages, key=lambda s: s.cadence
    )

    if not self.eval_schedule.stages:
      raise ValueError('EvalSchedule must have at least one stage.')

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
  def training_mesh(self) -> parallelism.Mesh:
    assert isinstance(self.data_loader.parallelism_mesh, parallelism.Mesh)
    return self.data_loader.parallelism_mesh

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

  def _get_nested_steps_and_specs(
      self, flat_spec: DataSpec
  ) -> tuple[tuple[int, ...], tuple[DataSpec, ...]]:
    """Returns nested steps and specs from `flat_spec` with timedelta coords."""
    dt = self.model.timestep
    nested_specs = scan_utils.nested_scan_specs(flat_spec, dt=dt)
    steps = scan_utils.nested_scan_steps(flat_spec, dt=dt)
    return steps, nested_specs

  def build_train_and_eval_iterators(
      self, schedule_idx: int
  ) -> TrainEvalIteratorTuple:
    """Build new iterators for training at schedule_idx."""
    train_stage = self.train_schedule.stages[schedule_idx]

    def setup_callbacks(stage) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
      """Returns tuples of retrieve_fns and buffers for a given stage."""
      data_spec = data_loading.select_targets(
          stage.inputs_spec, stage.queries_spec
      )  # TODO(dkochkov) ensure this retains FieldInQuery entries.
      nested_steps, nested_data_specs = self._get_nested_steps_and_specs(
          data_spec
      )
      # idx_steps starts with 1 for most frequent and follows the product of
      # nested step frequencies for each subsequent level.
      idx_steps = [1] + list(map(int, np.cumprod(nested_steps)[:-1]))
      retrieve_fns, buffers = [], []
      for stride, data_spec in zip(idx_steps, nested_data_specs):
        data_slice_struct = self.data_loader.data_slice_struct(
            data_spec, batch_size_per_device=stage.batch_size_per_device
        )  # computes structure of a single time slice at this scan level.
        retrieve_fn, data_buffer = self.data_loader.setup_targets_via_callback(
            data_slice_struct=data_slice_struct,
            callback_pinned_host=self.callback_pinned_host,
            callback_spatial_dims_layout=self.callback_spatial_dims_layout,
            idx_step=stride,
        )
        retrieve_fns.append(retrieve_fn)
        buffers.append(data_buffer)
      return tuple(retrieve_fns), tuple(buffers)

    if self.use_data_loading_callback:
      train_retrieve_fns, train_buffers = setup_callbacks(train_stage)
    else:
      train_retrieve_fns, train_buffers = None, None

    evaluation_fns = []
    loss_eval_schema = EvalSchema(
        cadence=self.frequent_eval_stage.cadence,
        inputs_spec=train_stage.inputs_spec,
        dynamic_inputs_spec=train_stage.dynamic_inputs_spec,
        queries_spec=train_stage.queries_spec,
        metrics_evaluator=None,
        loss_evaluator=train_stage.loss,
        time_sample_offset=self.frequent_eval_stage.time_sample_offset,
        batch_size_per_device=self.frequent_eval_stage.batch_size_per_device,
        num_batches=self.frequent_eval_stage.num_batches,
        time_slice=self.frequent_eval_stage.time_slice,
        name='loss',
    )
    stages = [loss_eval_schema] + list(self.eval_schedule.stages)
    for eval_stage in stages:
      if self.use_data_loading_callback:
        eval_retrieve_fns, eval_buffers = setup_callbacks(eval_stage)
      else:
        eval_retrieve_fns, eval_buffers = None, None

      if eval_stage.num_batches:
        eval_data = self.data_loader.build_eval_inputs(
            input_data_specs=eval_stage.inputs_spec,
            dynamic_input_specs=eval_stage.dynamic_inputs_spec,
            dataset_time_slices=eval_stage.time_slice,
            batch_size_per_device=eval_stage.batch_size_per_device,
            time_sample_offset=eval_stage.time_sample_offset,
            batch_count=eval_stage.num_batches,
            data_buffer=eval_buffers,
        )
        train_data = self.data_loader.build_eval_inputs(
            input_data_specs=eval_stage.inputs_spec,
            dynamic_input_specs=eval_stage.dynamic_inputs_spec,
            dataset_time_slices=train_stage.time_slice,
            batch_size_per_device=eval_stage.batch_size_per_device,
            time_sample_offset=eval_stage.time_sample_offset,
            batch_count=eval_stage.num_batches,
            data_buffer=eval_buffers,
        )
        evaluate_fn = functools.partial(
            self.evaluate,
            eval_statistics_fn=self.get_eval_statistics_fn(
                eval_schema=eval_stage,
                seed=0,
                retrieve_fns=eval_retrieve_fns,
            ),
            eval_data=eval_data,
            train_data=train_data,
            eval_schema=eval_stage,
        )
      else:
        evaluate_fn = self.dummy_evaluate

      evaluation_fns.append((eval_stage.cadence, evaluate_fn))

    training_data = self.data_loader.build_train_inputs(
        input_data_specs=train_stage.inputs_spec,
        dynamic_input_specs=train_stage.dynamic_inputs_spec,
        batch_size_per_device=train_stage.batch_size_per_device,
        shuffle_buffer_size_in_bytes=train_stage.shuffle_buffer_size,
        dataset_rng_seed=train_stage.dataset_rng_seed,
        time_sample_offset=train_stage.time_sample_offset,
        dataset_time_slice=train_stage.time_slice,
        data_buffer=train_buffers,
    )

    # TODO(shoyer): consider adding a hook for adding a profiler around
    # train_step_fn.
    train_step_fn = self.get_train_step_fn(
        train_stage=train_stage, retrieve_fns=train_retrieve_fns
    )

    return training_data, train_step_fn, evaluation_fns

  def run_pretraining(self, train_stage: TrainStage):
    """Runs pretraining."""
    if self.pretraining is None:
      logging.info('Skipping pretraining (no pretraining config provided)')
      return

    assert hasattr(self.pretraining, 'pretrain_fns')
    for k, pretraining_step in self.pretraining.pretrain_fns.items():
      logging.info('Running pretraining step: %s', k)
      self.model = pretraining_step(
          self.model, self.data_loader.all_data, train_stage.queries_spec
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
      eb_model: api.Model,
      process_obs: nnx.Module,
      data_slice_struct: PyTree,
      queries_spec: data_specs.QueriesSpec,
  ) -> tuple[PyTree, PyTree]:
    """Returns shape structs for predictions and targets for aggregators."""
    target_slice_struct = data_loading.select_targets(
        data_slice_struct, queries_spec
    )
    process_fn = lambda proc_module, target_slice: proc_module(target_slice)
    target_struct = nnx.eval_shape(
        process_fn, process_obs, _flatten_dict(target_slice_struct)
    )
    query_struct = data_specs.construct_query(data_slice_struct, queries_spec)
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
      batch_axis: cx.SizedAxis,
      ensemble_axis: cx.SizedAxis,
  ) -> api.Model:
    """Initializes a vectorized model for a rollout."""
    b_model = model.to_vectorized({
        typing.SimulationVariable: batch_axis,
        typing.DynamicInput: batch_axis,
    })
    b_model.update_dynamic_inputs(dynamic_data)
    b_model.assimilate(init_slice)
    eb_model = b_model.to_vectorized({typing.SimulationVariable: ensemble_axis})
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

  def _create_initial_nested_agg_states(
      self,
      evaluator: evaluators.Evaluator,
      model: api.Model,
      process_obs: nnx.Module,
      nested_specs: tuple[DataSpec, ...],
      nested_queries_spec: tuple[data_specs.QueriesSpec, ...],
      batch_size_per_device: int | None,
  ) -> tuple[aggregation.AggregationState, ...]:
    """Creates a tuple of initial aggregation states for each nested level."""
    initial_nested_agg_states = []
    for input_spec, query_spec in zip(nested_specs, nested_queries_spec):
      time_slice_struct = self.data_loader.data_slice_struct(
          input_spec, batch_size_per_device
      )
      prediction_struct, target_struct = self._predictions_and_targets_structs(
          model, process_obs, time_slice_struct, query_spec
      )
      # TODO(dkochkov): Update evaluators to drop the if condition below.
      initial_nested_agg_states.append(
          evaluator.zeros_aggregation_states(prediction_struct, target_struct)
          if prediction_struct and target_struct else {}
      )
    return tuple(initial_nested_agg_states)

  def _recursive_scan(
      self,
      collect_inner: Callable[..., Any],
      in_axes: Any,
      out_axes: Any,
      nest_level: int,
      nested_steps: tuple[int, ...],
      nested_queries_spec: tuple[data_specs.QueriesSpec, ...],
      retrieve_fns: tuple[Callable[[int], PyTree], ...] | None,
      nested_evaluators_groups: tuple[tuple[evaluators.Evaluator, ...], ...],
      nested_targets: tuple[PyTree, ...] | None,
      observe_fn: Callable[..., PyTree],
      process_fn: Callable[..., PyTree],
      process_obs: nnx.Module,
      batch_axis: cx.Coordinate | None,
  ) -> Callable[..., Any]:
    """Recursively builds a nested scan function for collecting stats."""
    if nest_level == len(nested_steps):
      return collect_inner

    length = nested_steps[nest_level]
    query_spec = nested_queries_spec[nest_level]

    def _collect_stats(carry, model, loaded_targets_slice, evaluators_tuple):
      # Unpack carry: idx, then tuple of agg_state one per evaluator.
      step_idx, agg_states_tuple = carry

      # Extracting arguments for `fn` by separating current level values for
      # arguments that capture nesting: agg_states, targets, each evaluator tpl.
      inner_scan_aggs_tuple = tuple(states[:-1] for states in agg_states_tuple)
      inner_scan_targets = (
          loaded_targets_slice[:-1] if loaded_targets_slice else ()
      )
      inner_scan_evaluators = tuple(
          evaluators[:-1] for evaluators in evaluators_tuple
      )

      # Invariant signature: (carry, model, targets, evaluators)
      step_idx, inner_scan_agg_results = collect_inner(
          (step_idx, inner_scan_aggs_tuple),
          model,
          inner_scan_targets,
          inner_scan_evaluators,
      )  # step index is incremented at the lowest level.

      # Compute aggregation updates for the step at current level of nesting.
      current_level_agg_states = tuple(x[-1] for x in agg_states_tuple)
      current_level_evaluators = tuple(x[-1] for x in evaluators_tuple)
      current_level_targets = (
          loaded_targets_slice[-1] if loaded_targets_slice else None
      )
      retrieve_fn = retrieve_fns[nest_level] if retrieve_fns else None

      updated_current_level_aggs = _compute_aggregation(
          step_idx=step_idx,
          model=model,
          current_agg_states=current_level_agg_states,
          evaluator_slices=current_level_evaluators,
          loaded_targets_slice=current_level_targets,
          retrieve_fn=retrieve_fn,
          query_spec=query_spec,
          observe_fn=observe_fn,
          process_fn=process_fn,
          process_obs=process_obs,
          run_evaluator_fn=self.run_evaluator,
          combine_agg_states_fn=self.combine_agg_states,
          training_mesh=self.training_mesh,
          batch_axis=batch_axis,
      )

      # Combine inner and current agg states.
      combined_agg_states = []
      for inner_agg_states, agg_state in zip(
          inner_scan_agg_results, updated_current_level_aggs
      ):
        combined_agg_states.append(tuple([*inner_agg_states, agg_state]))

      return (step_idx, tuple(combined_agg_states))

    if self.remat_config.remat_collect_statistics:
      # TODO(dkochkov): Consider adding more granular control for this remat.
      _collect_stats = nnx.remat(_collect_stats)

    collect_stats_scan = nnx.scan(
        _collect_stats, length=length, in_axes=in_axes, out_axes=out_axes
    )
    return self._recursive_scan(
        collect_stats_scan,
        in_axes,
        out_axes,
        nest_level + 1,
        nested_steps,
        nested_queries_spec,
        retrieve_fns,
        nested_evaluators_groups,
        nested_targets,
        observe_fn,
        process_fn,
        process_obs,
        batch_axis,
    )

  def _split_model_and_process_obs(self):
    """Splits model and process_observations for checkpointing/sharding."""
    process_def, process_params = nnx.split(self.process_observations)
    model_def, _, temporaries, _ = nnx.split(
        self.model,
        nnx.Param,  # params.
        (typing.SimulationVariable, typing.DynamicInput),  # temporaries.
        ...,  # non-params.
    )
    return process_def, process_params, model_def, temporaries

  def _merge_model_and_process_obs(
      self,
      process_def,
      process_params,
      model_def,
      params,
      non_params,
      temporaries,
  ) -> tuple[nnx.Module, api.Model]:
    """Merges model and process_observations components."""
    process_obs = nnx.merge(process_def, process_params, copy=True)
    model = nnx.merge(model_def, params, non_params, temporaries, copy=True)
    return process_obs, model

  def get_train_step_fn(
      self,
      train_stage: TrainStage,
      retrieve_fns: tuple[Callable[[int], PyTree], ...] | None = None,
  ) -> train_utils.TrainStepFunction:
    """Makes a function that performs a single training step.

    Args:
      train_stage: Training stage for which to build a training step function.
      retrieve_fns: Functions to retrieve targets via callback.

    Returns:
      train_step_fn: Function that performs a single training step given an
        (experiment state, step, inputs, dynamic_data) tuple and returning
        the updated experiment state and loss.
    """
    inputs_spec = train_stage.inputs_spec
    loss_evaluator = train_stage.loss
    batch_axis = self.data_loader.make_batch_axis(
        train_stage.batch_size_per_device
    )
    ensemble_axis = self.ensemble_axis
    model_state_scan_axes = nnx.StateAxes(
        {typing.SimulationVariable: nnx.Carry, ...: None}
    )
    targets_spec = data_loading.select_targets(
        inputs_spec, train_stage.queries_spec
    )  # TODO(dkochkov): I think this should filter FieldInQuerySpec.
    nested_steps, nested_targets_specs = self._get_nested_steps_and_specs(
        targets_spec
    )
    nested_queries_spec = jax.tree.map(
        _remove_timedelta, nested_targets_specs, is_leaf=_is_coord
    )

    # Setting up model components and helper functions.
    (
        process_def,
        process_params,
        model_def,
        temporaries,
    ) = self._split_model_and_process_obs()
    step_fn, observe_fn, process_fn = self._get_step_observe_process_fns()

    def batched_parameter_loss_fn(
        params, non_params, rng, inputs, dynamic_data
    ):
      """Computes evaluation metrics for a batch of targets."""
      init_slice, loaded_targets = _prepare_inputs_and_targets(
          inputs, self.model.timestep, retrieve_fns, train_stage.queries_spec, batch_axis
      )

      # Initializing the model state.
      process_obs, model = self._merge_model_and_process_obs(
          process_def,
          process_params,
          model_def,
          params,
          non_params,
          temporaries,
      )
      eb_model = self._initialize_vectorized_model(
          model,
          rng,
          init_slice,
          dynamic_data,
          batch_axis,
          ensemble_axis,
      )
      eb_model_state = nnx.state(eb_model, typing.SimulationVariable)
      eb_model_state = self.training_mesh.with_sharding_constraint(
          eb_model_state, 'physics'
      )
      nnx.update(eb_model, eb_model_state)

      nested_evaluators = create_nested_evaluators(
          loss_evaluator, nested_targets_specs, self.model.timestep
      )
      init_agg_states = self._create_initial_nested_agg_states(
          loss_evaluator,
          eb_model,
          process_obs,
          nested_targets_specs,
          nested_queries_spec,
          train_stage.batch_size_per_device,
      )

      def collect_statistics_step(
          carry, model, loaded_targets_slice, evaluator_slice
      ) -> tuple[int, tuple[tuple[aggregation.AggregationState, ...]]]:  # pylint: disable=g-one-element-tuple
        del evaluator_slice, loaded_targets_slice
        idx, (agg,) = carry
        step_fn(model)  # advances model by 1 step.
        model_state = nnx.state(model, typing.SimulationVariable)
        model_state = self.training_mesh.with_sharding_constraint(
            model_state, 'physics'
        )
        nnx.update(model, model_state)
        return (idx + 1, (agg,))

      scan_fn = self._recursive_scan(
          collect_statistics_step,
          in_axes=(nnx.Carry, model_state_scan_axes, 1, 0),
          out_axes=nnx.Carry,
          nest_level=0,
          nested_steps=nested_steps,
          nested_queries_spec=nested_queries_spec,
          retrieve_fns=retrieve_fns,
          nested_evaluators_groups=(nested_evaluators,),
          nested_targets=loaded_targets,
          observe_fn=observe_fn,
          process_fn=process_fn,
          process_obs=process_obs,
          batch_axis=batch_axis,
      )

      _, (loss_agg_state_tuple,) = scan_fn(
          (0, (init_agg_states,)),
          eb_model,
          loaded_targets if loaded_targets else (None,) * len(nested_steps),
          (nested_evaluators,),
      )
      # Flatten aggregation states from all levels.
      full_agg_state = functools.reduce(
          lambda c, x: c | x, loss_agg_state_tuple, {}
      )
      loss_value = loss_evaluator.evaluate_total({}, {}, full_agg_state).data
      return loss_value

    # We would use donate_argnums here to update experiment_state in-place, but
    # that would mean we could not save the checkpoint in a separable thread.
    # Fortunately experiment_state is usually not too big (~100 MB).
    @train_utils.jit_once
    def train_step(experiment_state, step, inputs, dynamic_data):
      opt_state, params, ema_state, non_params = experiment_state
      rng = train_utils.batch_and_ensemble_parallel_rng_key(
          batch_size=batch_axis.size,
          ensemble_size=self.ensemble_axis.size,
          seeds=(train_stage.random_process_rng_seed, step),
          mesh=self.spmd_mesh,
      )
      rng = cx.field(rng, batch_axis, self.ensemble_axis)
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

  def get_eval_statistics_fn(
      self,
      eval_schema: EvalSchema,
      seed: int = 0,
      retrieve_fns: tuple[Callable[[int], PyTree], ...] | None = None,
  ) -> Callable[..., Any]:
    """Makes a function that performs a single evaluation pass."""
    batch_axis = self.data_loader.make_batch_axis(
        eval_schema.batch_size_per_device
    )
    ensemble_axis = self.ensemble_axis
    model_state_scan_axes = nnx.StateAxes(
        {typing.SimulationVariable: nnx.Carry, ...: None}
    )
    targets_spec = data_loading.select_targets(
        eval_schema.inputs_spec, eval_schema.queries_spec
    )  # TODO(dkochkov): I think this should filter out FieldInQuerySpec.
    nested_steps, nested_targets_specs = self._get_nested_steps_and_specs(
        targets_spec
    )
    nested_queries_spec = jax.tree.map(
        _remove_timedelta, nested_targets_specs, is_leaf=_is_coord
    )

    # Setting up model components and helper functions.
    (
        process_def,
        process_params,
        model_def,
        temporaries,
    ) = self._split_model_and_process_obs()
    step_fn, observe_fn, process_fn = self._get_step_observe_process_fns()

    @train_utils.jit_once
    def batch_eval_statistics_fn(
        params, non_params, eval_step, inputs, dynamic_data
    ):
      """Computes evaluation statistics for a batch of targets."""
      init_slice, loaded_targets = _prepare_inputs_and_targets(
          inputs,
          self.model.timestep,
          retrieve_fns,
          eval_schema.queries_spec,
          batch_axis,
      )

      # Initializing the model state.
      [batch_size], [ensemble_size] = (
          batch_axis.shape,
          self.ensemble_axis.shape,
      )
      rng = train_utils.batch_and_ensemble_parallel_rng_key(
          batch_size=batch_size,
          ensemble_size=ensemble_size,
          seeds=(seed, eval_step),
          mesh=self.spmd_mesh,
      )
      rng = cx.field(rng, batch_axis, self.ensemble_axis)
      process_obs, model = self._merge_model_and_process_obs(
          process_def,
          process_params,
          model_def,
          params,
          non_params,
          temporaries,
      )
      eb_model = self._initialize_vectorized_model(
          model,
          rng,
          init_slice,
          dynamic_data,
          batch_axis,
          ensemble_axis,
      )
      evaluators_seq, agg_states_seq = [], []
      if eval_schema.metrics_evaluator:
        nested_metrics = create_nested_evaluators(
            eval_schema.metrics_evaluator,
            nested_targets_specs,
            self.model.timestep,
        )
        init_metrics_agg_states = self._create_initial_nested_agg_states(
            eval_schema.metrics_evaluator,
            eb_model,
            process_obs,
            nested_targets_specs,
            nested_queries_spec,
            eval_schema.batch_size_per_device,
        )
        evaluators_seq.append(nested_metrics)
        agg_states_seq.append(init_metrics_agg_states)
      if eval_schema.loss_evaluator:
        nested_losses = create_nested_evaluators(
            eval_schema.loss_evaluator, nested_targets_specs, self.model.timestep
        )
        init_loss_agg_states = self._create_initial_nested_agg_states(
            eval_schema.loss_evaluator,
            eb_model,
            process_obs,
            nested_targets_specs,
            nested_queries_spec,
            eval_schema.batch_size_per_device,
        )
        evaluators_seq.append(nested_losses)
        agg_states_seq.append(init_loss_agg_states)

      evaluators_seq = tuple(evaluators_seq)
      agg_states_seq = tuple(agg_states_seq)

      def collect_statistics_step(
          carry,
          model,
          loaded_targets_slice,
          evaluators_tuple,
      ) -> tuple[int, tuple[NestedAggStates, ...]]:
        del evaluators_tuple, loaded_targets_slice
        idx, agg_states_tuples_seq = carry  # aggs here are empty tuples.
        step_fn(model)  # advances model by 1 step.
        return (idx + 1, agg_states_tuples_seq)

      scan_fn = self._recursive_scan(
          collect_statistics_step,
          in_axes=(nnx.Carry, model_state_scan_axes, 1, 0),
          out_axes=nnx.Carry,
          nest_level=0,
          nested_steps=nested_steps,
          nested_queries_spec=nested_queries_spec,
          retrieve_fns=retrieve_fns,
          nested_evaluators_groups=evaluators_seq,
          nested_targets=loaded_targets,
          observe_fn=observe_fn,
          process_fn=process_fn,
          process_obs=process_obs,
          batch_axis=batch_axis,
      )

      _, agg_states_tuples_seq = scan_fn(
          (0, agg_states_seq),
          eb_model,
          loaded_targets if loaded_targets else (None,) * len(nested_steps),
          evaluators_seq,
      )
      n_evaluators = len(evaluators_seq)
      # pylint: disable=unbalanced-tuple-unpacking
      if n_evaluators == 2:  # both metrics and loss evaluators.
        metrics_agg_tuple, loss_agg_tuple = agg_states_tuples_seq
      elif n_evaluators == 1 and eval_schema.metrics_evaluator:  # only metrics.
        metrics_agg_tuple, = agg_states_tuples_seq
        loss_agg_tuple = None
      elif n_evaluators == 1 and eval_schema.loss_evaluator:  # only loss.
        metrics_agg_tuple = None
        loss_agg_tuple, = agg_states_tuples_seq
      else:
        raise ValueError(
            'At least one of metrics_evaluator or loss_evaluator must be set.'
        )
      # pylint: enable=unbalanced-tuple-unpacking

      full_metrics_agg = functools.reduce(
          lambda c, x: c | x, metrics_agg_tuple, {}
      ) if metrics_agg_tuple else None
      full_loss_agg = functools.reduce(
          lambda c, x: c | x, loss_agg_tuple, {}
      ) if loss_agg_tuple else None
      return full_metrics_agg, full_loss_agg

    return batch_eval_statistics_fn

  def save_online_metrics(self, step: int, online_metrics: OnlineEvalMetrics):
    # only save from the coordinator process to avoid redundant copies.
    if is_coordinator():
      self.online_metrics_saver.save(step, online_metrics)

  def run_training(self):
    """Runs the training experiment."""
    start_step, auto_restart, experiment_state = self.initialize_experiment()

    if start_step >= self._total_train_steps:
      logging.warning(
          f'Attempting to start training at {start_step=} >='
          f' {self._total_train_steps=}. Will simply return.'
      )
      return

    def logging_callback(step, schedule_idx, loss, auto_restart):
      loss = float(jax.device_get(loss))
      if step % max(self.frequent_eval_stage.cadence // 100, 1) == 0:
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

    train_step_fn, evaluation_fns, train_iter = None, [], None  # pytype happy.
    step = start_step
    while step < self._total_train_steps:
      old_schedule_idx = schedule_idx
      schedule_idx = np.sum(  # compute which leg of the schedule we are at.
          step >= np.asarray(self._train_schedule_boundaries)
      )
      if schedule_idx != old_schedule_idx:
        (
            train_iter,
            train_step_fn,
            evaluation_fns,
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
      min_cadence = min(c for c, _ in evaluation_fns)
      training_time = None
      if (step + 1) % min_cadence == 0:
        # We default to timing the training step and min_cadence intervals.
        if step > start_step:
          with train_step_timer:
            # train_step is non-blocking, so we need to block on the output
            # of the previous training step to reliably time it.
            experiment_state = jax.block_until_ready(experiment_state)
          training_time = train_step_timer.total
          logging.info(
              f'training for {min_cadence} steps'
              f' took {training_time:.1f} seconds'
          )
          train_step_timer = timing.Timer()  # reset
        else:
          training_time = None

      current_step_metrics = []
      for cadence, evaluate_fn in evaluation_fns:
        if (step + 1) % cadence == 0:
          online_metrics = evaluate_fn(experiment_state)
          if cadence == min_cadence and step > start_step:
            assert training_time is not None
            online_metrics.seconds_per_train_step = (
                training_time / cadence
            )
          current_step_metrics.append(online_metrics)
          logging.info(
              'evaluation pass for cadence %d took %.1f seconds',
              cadence,
              online_metrics.seconds_per_evaluation,
          )

      if current_step_metrics:
        self.save_online_metrics(
            step, _merge_online_metrics(current_step_metrics)
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
    final_metrics = []
    for _, evaluate_fn in evaluation_fns:
      final_metrics.append(evaluate_fn(experiment_state))

    if final_metrics:
      self.save_online_metrics(
          self._total_train_steps, _merge_online_metrics(final_metrics)
      )
    self.save_checkpoint(
        self._total_train_steps, auto_restart, experiment_state
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
                steps=list(self._train_schedule_boundaries)
            ),
        ]),
        save_decision_policy=ocp_managers.AnySavePolicy([
            ocp_managers.ContinuousCheckpointingPolicy(
                minimum_interval_secs=60
            ),
            ocp_managers.PreemptionCheckpointingPolicy(),
            *[
                ocp_managers.FixedIntervalPolicy(interval=s.cadence)
                for s in self.eval_schedule.stages
            ],
            ocp_managers.SpecificStepsPolicy(
                steps=list(self._train_schedule_boundaries)
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
      # TODO(dkochkov): Consider making pretraining independent from training.
      self.run_pretraining(self.train_schedule.stages[0])
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
          'step': int(step),
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
    return OnlineEvalMetrics()

  def _compute_metrics_from_stats(
      self,
      eval_statistics_fn: Callable[..., Any],
      data_iterator: Any,
      eval_schema: EvalSchema,
  ) -> dict[str, float]:
    """Computes metrics from aggregated statistics."""
    total_metrics_agg = None
    total_loss_agg = None
    count = 0
    for step, (inputs, dynamic_data) in enumerate(data_iterator):
      metrics_agg, loss_agg = eval_statistics_fn(step, inputs, dynamic_data)
      if total_metrics_agg is None:
        total_metrics_agg = metrics_agg
      else:
        total_metrics_agg = self.combine_agg_states(
            total_metrics_agg, metrics_agg
        )
      if total_loss_agg is None:
        total_loss_agg = loss_agg
      else:
        total_loss_agg = self.combine_agg_states(total_loss_agg, loss_agg)
      count += 1

    if not count:
      raise RuntimeError('no batches to iterate over')

    prefix = [eval_schema.name] if eval_schema.name else []
    # Recording metrics.
    values_to_record = {}
    if eval_schema.metrics_evaluator:
      metric_values = eval_schema.metrics_evaluator.evaluate_metrics(
          {}, {}, total_metrics_agg
      )
      for metric_key, metric_dict in metric_values.items():
        for scalar_name, v in metric_dict.items():
          key = '.'.join(prefix + [metric_key, scalar_name])
          values_to_record[key] = float(v.data)

    # Recording loss.
    if eval_schema.loss_evaluator is not None:
      loss_val = float(
          eval_schema.loss_evaluator.evaluate_total({}, {}, total_loss_agg).data
      )
      key = '.'.join(prefix + ['total'])
      values_to_record[key] = loss_val
      for term_key, term_metric in eval_schema.loss_evaluator.metrics.items():
        # Recording relative value of loss terms.
        assert isinstance(term_metric, metrics_base.Loss)
        if eval_schema.loss_evaluator.term_weights is not None:
          w = eval_schema.loss_evaluator.term_weights.get(term_key, 1.0)
        else:
          w = 1.0
        term_metric_values = total_loss_agg[term_key].metric_values(term_metric)
        relative_value_key = '.'.join(prefix + ['relative', term_key])
        loss_term_value = w * float(term_metric.total(term_metric_values).data)
        if loss_val != 0:
          values_to_record[relative_value_key] = loss_term_value / loss_val
        else:
          values_to_record[relative_value_key] = 0.0

        # Recording debug terms for each loss term.
        mean_stats = total_loss_agg[term_key].mean_statistics()
        debug_terms = term_metric.debug_terms(mean_stats, term_metric_values)
        for debug_term_name, v in debug_terms.items():
          key = '.'.join(prefix + [term_key, debug_term_name])
          values_to_record[key] = float(v.data)

    return values_to_record

  def evaluate(
      self,
      experiment_state: ExperimentState,
      eval_statistics_fn,
      eval_data,
      train_data,
      eval_schema: EvalSchema,
  ) -> OnlineEvalMetrics:
    """Evaluates the model on train and eval data and writes summaries.

    Args:
      experiment_state: tuple of replicated step, optimizer state and EMA
        (exponentially moving average) state for model parameters.
      eval_statistics_fn: function for computing eval statistics per batch.
      eval_data: iterable that samples data from the evaluation dataset.
      train_data: iterable that samples data from the training dataset.
      eval_schema: the eval schema to use for evaluation.

    Returns:
      Online evaluation metrics.
    """
    with timing.Timer() as eval_timer:
      _, params, ema_state, non_params = experiment_state
      ema_params, _ = self._ema_update(params, ema_state)  # pytype: disable=attribute-error  # jax-api-types

      results = {}

      logging.info('evaluating on train dataset')
      results['train'] = self._compute_metrics_from_stats(
          functools.partial(eval_statistics_fn, params, non_params),
          train_data,
          eval_schema,
      )

      logging.info('evaluating on test dataset')
      results['eval'] = self._compute_metrics_from_stats(
          functools.partial(eval_statistics_fn, params, non_params),
          eval_data,
          eval_schema,
      )

      logging.info('evaluating EMA model on test dataset')
      results['eval_ema'] = self._compute_metrics_from_stats(
          functools.partial(eval_statistics_fn, ema_params, non_params),
          eval_data,
          eval_schema,
      )

    return OnlineEvalMetrics(
        train=results['train'],
        eval=results['eval'],
        eval_ema=results['eval_ema'],
        seconds_per_evaluation=eval_timer.total,
    )
