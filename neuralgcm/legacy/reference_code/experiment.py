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
# pylint: disable=line-too-long
r"""Pseudocode for training NeuralGCM models."""
from collections.abc import Iterable, Iterator, Mapping, Sequence
import dataclasses
import functools
import logging
import math
from typing import Any, Callable, NamedTuple, Optional, cast

from absl import app
import datasets
from dinosaur import typing
from dinosaur import xarray_utils
import gin
from google_proprietary_code import checkpoint
from google_proprietary_code import experiment
from google_proprietary_code import experiment_utils
# proprietary imports
from google_proprietary_code import profiling_util
from google_proprietary_code import streaming
from google_proprietary_code import timing_util
import haiku as hk
import jax
import jax.numpy as jnp
import jax.sharding
import metrics
import metrics_base
import metrics_util
from ml_collections import config_dict
from neuralgcm.legacy import model_builder
from neuralgcm.legacy import model_utils
from neuralgcm.legacy import optimization
from neuralgcm.legacy import physics_specifications
import numpy as np
import optax
import pandas as pd
import reader
import stochastic_losses
import tensorflow as tf
import train_utils
import xarray

Params = typing.Params
PyTree = Any
TrajectoryRepresentations = typing.TrajectoryRepresentations
TrajectoryFn = Callable[
    [Params, jax.Array, PyTree, PyTree],
    tuple[TrajectoryRepresentations, TrajectoryRepresentations],
]

tree_map = jax.tree_util.tree_map

# pylint: disable=logging-format-interpolation
# pylint: disable=logging-fstring-interpolation


@gin.configurable(allowlist=['constructor'])
def get_loss_obj(
    trajectory_spec: metrics_util.TrajectorySpec,
    constructor: Callable[..., metrics_base.Loss] = gin.REQUIRED,
) -> metrics_base.Loss:
  """Returns configured loss_fn on first `trajectory_length` time slices."""
  return constructor(trajectory_spec)


EvaluatorDict = dict[str, metrics_base.Evaluator]
TrainEvalIteratorTuple = tuple[
    Iterator[Any],
    train_utils.TrainStepFunction,
    Callable[..., Any],
    dict[str, Any],
]


@gin.configurable(allowlist=['constructor'])
def get_metrics_dict(
    trajectory_spec: metrics_util.TrajectorySpec,
    eval_time_steps: Sequence[int],
    loss: metrics_base.Loss,
    constructor: Callable[..., EvaluatorDict] = metrics.default_metrics,
    is_ensemble_data: bool = False,
) -> EvaluatorDict:
  """Returns configured loss_fn on first `trajectory_length` time slices."""
  return constructor(
      trajectory_spec, eval_time_steps, loss, is_ensemble_data=is_ensemble_data
  )


# start legacy configurables
#
# Keep these around for now (even though they are no-ops) so we can run
# inference on old models.


@gin.configurable
def get_loss_fn(loss_fn):
  raise NotImplementedError


@gin.register
def weighted_l2_cumulative_loss(weights, scale):
  raise NotImplementedError


# end legacy configurables


def ema_params_tree(num_steps):
  """Creates an EMAParamsTree object based on num_steps.

  Args:
    num_steps: average number of optimization steps to include in the
      exponential moving average of model weights.

  Returns:
    Haiku module.
  """
  # https://en.wikipedia.org/wiki/Moving_average#Relationship_between_SMA_and_EMA
  decay = 1 - 2 / (num_steps + 1)
  return hk.EMAParamsTree(decay)


def _model_inner_steps_for_data(
    ds: xarray.Dataset,
    model_specs: model_builder.ModelSpecs,
    rtol: float = 1e-6,
) -> int:
  """Calculates model inner steps based on data time step."""
  data_dt = xarray_utils.nondim_time_delta_from_time_axis(
      ds.time.data, model_specs.physics_specs
  )
  inner_steps = round(data_dt / model_specs.dt)
  if abs(inner_steps * model_specs.dt - data_dt) / data_dt > rtol:
    raise RuntimeError(  # pylint: disable=g-doc-exception
        f'{model_specs.dt=} does not divide evenly into {data_dt=}'
    )
  return inner_steps


def _get_datetime_forecast_starts(
    sample_count: int,
    first_start: pd.Timestamp,
    last_start: pd.Timestamp,
) -> pd.DatetimeIndex:
  """Get equispaced forecast start times for evaluating against ERA5."""
  if first_start.hour != 0:
    raise ValueError(f'dataset times must start at midnight: {first_start=}')
  # Round-up to midnight following the last forecast day (e.g., the start of
  # the next year).
  stop = last_start.ceil('1D')
  # Equally spaced from start (inclusive) to stop (exclusive).
  start_times = pd.date_range(first_start, stop, periods=sample_count + 1)[:-1]
  # To match ECMWF, all forecasts should be initialized at 0z or 12z. Here we
  # alternate start times.
  parity = np.arange(sample_count) % 2
  return start_times.round('1D') + parity * pd.Timedelta('12H')


P = jax.sharding.PartitionSpec


class ExperimentState(NamedTuple):
  opt_state: PyTree
  params: PyTree
  ema_params: PyTree


class Experiment(experiment.AbstractExperiment):
  """Training experiment based on trajectory loss minimization."""

  def __init__(
      self,
      experiment_dir: str,
      config: Optional[config_dict.ConfigDict] = None,
  ):
    """Creates an instance of a training scheme class.

    Args:
      experiment_dir: Path to experiment directory.
      config: config struct setting up the experiment.
    """
    if config is None:
      config = experiment_config.get_config()

    super().__init__(
        experiment_dir,
        config.distributed_training,
        writer_names=['train', 'eval', 'eval_ema'],
    )
    logging.info('Experiment config:\n%s', config)
    self.config = config

    self.train_ds = xarray_utils.open_dataset(config.train_dataset_path)
    self.eval_ds = xarray_utils.open_dataset(config.eval_dataset_path)

    if 'sample' in self.train_ds.dims:
      logging.warning('only using the first sample!')
      self.train_ds = self.train_ds.isel(sample=0, drop=True)
      self.eval_ds = self.eval_ds.isel(sample=0, drop=True)

    train_attrs = self.train_ds.attrs

    # Model instantiation and trajectory unroll functions
    # Note: we use interactive mode in experiments to split gin-configurations
    # into separate, distinct parts provided in the config_dict.
    gin.enter_interactive_mode()
    self.is_nodal = self.config.is_nodal

    # parse and override all the gin things
    gin.parse_config(config.model_gin_config)
    gin.parse_config(config.optimizer_gin_config)
    experiment_utils.parse_config_dict(config.gin_overrides)

    logging.info('Parsed gin config string:\n%s', gin.config_str())

    full_model_gin_config = gin.config_str()  # do not include physics config.
    logging.info('With overrides gin config string:\n%s', gin.config_str())

    self.data_coords = model_builder.coordinate_system_from_dataset(
        self.train_ds
    )
    logging.info(f'{self.model_parallel_training=}')

    if self.model_parallel_training:
      # It does not make sense to use spatial parallelism with batch size per
      # device larger than 1. Instead, you would get better performance from
      # using less model parallelism.
      if self.spatial_parallelism > 1 and self.config.batch_size_per_device > 1:
        raise NotImplementedError(
            f'{self.config.batch_size_per_device=} is not supported for model '
            'parallel training'
        )
      self.data_coords = dataclasses.replace(
          self.data_coords, spmd_mesh=self.spmd_mesh
      )

    # try getting aux_features from dataset, if not included we rely on
    # `model_builder.get_model_specs` to supply necessary values.
    try:
      data_aux_features = xarray_utils.aux_features_from_xarray(self.train_ds)
    except KeyError:
      data_aux_features = {}

    # when available, we parse physics_config_str from metadata in train_attrs.
    if 'physics_config_str' in train_attrs:
      physics_config_str = train_attrs['physics_config_str']
      experiment_utils.parse_gin_config_without_imports(physics_config_str)
    else:
      logging.info(
          'physics_config_str was not provided in the dataset, '
          'hence it is expected to be specified in model_gin_config.'
      )

    self.physics_specs = physics_specifications.get_physics_specs()
    self.model_specs = model_builder.get_model_specs(
        self.data_coords, self.physics_specs, data_aux_features
    )
    logging.info(f'{self.model_specs=}')

    self.train_inner_steps = _model_inner_steps_for_data(
        self.train_ds, self.model_specs
    )
    self.eval_inner_steps = _model_inner_steps_for_data(
        self.eval_ds, self.model_specs
    )
    if (
        len(self.config.train_schedule_time_steps)
        != len(self.config.train_schedule_boundaries) + 1
    ):
      raise ValueError(
          f'{self.config.train_schedule_time_steps} should be one longer '
          f'than {self.config.train_schedule_boundaries} but was not.'
      )
    if any(
        t % self.train_inner_steps
        for t in self.config.train_schedule_time_steps
    ):
      raise ValueError(
          f'{self.train_inner_steps=} does not divide '
          f'{self.config.train_schedule_time_steps=}'
      )
    if any(t % self.eval_inner_steps for t in self.config.eval_time_steps):
      raise ValueError(
          f'{self.eval_inner_steps=} does not divide '
          f'{self.config.eval_time_steps=}'
      )
    if max(self.config.eval_time_steps) < max(
        self.config.train_schedule_time_steps
    ):
      raise ValueError(
          f'Training will not work since {max(self.config.eval_time_steps)=} <'
          f' {max(self.config.train_schedule_time_steps)=}'
      )

    self._eval_trajectory_length = (
        max(self.config.eval_time_steps) // self.eval_inner_steps + 1
    )
    self._trajectory_lengths = [
        self.config.num_init_frames + n // self.train_inner_steps
        for n in self.config.train_schedule_time_steps
    ]
    self._max_trajectory_length = max(
        self._trajectory_lengths + [self._eval_trajectory_length]
    )

    self.reference_datetime = self.model_specs.aux_features[
        xarray_utils.REFERENCE_DATETIME_KEY
    ]

    self.whirl_model = model_builder.WhirlModel(
        **self.model_specs,
        input_coords=self.data_coords,
        output_coords=self.data_coords,
    )
    self.from_xarray_fn = self.whirl_model.from_xarray_fn

    def trajectory_fwd(x, forcing_data, model, outer_steps, inner_steps):
      trajectory_fn = model_utils.trajectory_with_inputs_and_forcing(
          model, config.num_init_frames, start_with_input=True
      )
      return trajectory_fn(x, forcing_data, outer_steps, inner_steps)

    self._trajectory_fwd = trajectory_fwd

    # Checkpoint items.
    self._model_dt = self.model_specs.dt
    self._model_gin_config = full_model_gin_config

    # optimizer configuration.
    self.optimizer = optimization.optimizer()

    # exponentially moving average params tracking.
    ema_fn = hk.without_apply_rng(
        hk.transform_with_state(
            lambda x: ema_params_tree(config.ema_num_steps)(x)  # pylint: disable=unnecessary-lambda
        )
    )
    self._ema_init = jax.jit(ema_fn.init)

    def ema_update(params, ema_state):
      return ema_fn.apply(None, ema_state, params)

    self._ema_update = jax.jit(ema_update)

    logging.info('Final active config string:\n%s', gin.config_str())

  #
  # Data inputs methods.
  #

  @functools.cached_property
  def spmd_mesh(self) -> jax.sharding.Mesh:
    n = self.config.model_parallelism.ensemble_shards
    z = self.config.model_parallelism.z_shards
    x = self.config.model_parallelism.x_shards
    y = self.config.model_parallelism.y_shards
    global_batch = jax.device_count() // (n * z * x * y)
    if global_batch == 0:
      raise ValueError(
          f'{jax.device_count()=} is insufficient for '
          f'{self.config.model_parallelism=}'
      )
    return train_utils.create_spmd_mesh(
        {'batch': global_batch, 'ensemble': n, 'z': z, 'x': x, 'y': y}
    )

  @functools.cached_property
  def degree_of_model_parallelism(self) -> int:
    return math.prod(v for k, v in self.spmd_mesh.shape.items() if k != 'batch')

  @functools.cached_property
  def model_parallel_training(self) -> bool:
    return self.degree_of_model_parallelism > 1

  @functools.cached_property
  def spatial_parallelism(self) -> int:
    return math.prod(self.spmd_mesh.shape[k] for k in 'zxy')

  def to_global_array(self, pytree: PyTree, global_batch_size: int) -> PyTree:
    """Create a pytree of global JAX arrays from a pytree of NumPy arrays."""
    # partition arrays along batch and spatial dimensions
    return train_utils.make_distributed_array_from_local_arrays(
        pytree,
        self.spmd_mesh,
        self.data_coords.physics_partition_spec,
        global_batch_size,
    )

  def num_eval_batches(self, large_eval: bool) -> int:
    if large_eval:
      return self.config.num_eval_batches[-1]
    else:
      return self.config.num_eval_batches[0]

  def steps_between_evals(self, large_eval: bool) -> int:
    if large_eval:
      return self.config.steps_between_evals[-1]
    else:
      return self.config.steps_between_evals[0]

  def eval_batch_size_per_device(self, large_eval: bool) -> int:
    if large_eval:
      return self.config.eval_batch_size_per_device[-1]
    else:
      return self.config.eval_batch_size_per_device[0]

  @functools.cached_property
  def global_batch_size(self) -> int:
    return (
        jax.device_count()
        // self.degree_of_model_parallelism
        * self.config.batch_size_per_device
    )

  def global_eval_batch_size(self, large_eval: bool) -> int:
    return (
        jax.device_count()
        // self.degree_of_model_parallelism
        * self.eval_batch_size_per_device(large_eval)
    )

  def local_eval_batch_size(self, large_eval: bool) -> int:
    return jax.local_device_count() * self.eval_batch_size_per_device(
        large_eval
    )

  def _to_dataset_iter(
      self, data: tf.data.Dataset, template: xarray.Dataset
  ) -> Callable[[], Iterable[Any]]:
    """Convert a tf.data.Dataset into a function that makes a data iterator."""
    leading_dims_set = {x.shape[:2] for x in data.element_spec.values()}
    assert len(leading_dims_set) == 1, leading_dims_set
    local_batch_size, time_series_length = leading_dims_set.pop()
    template = (
        template.drop_vars('time')
        .head(time=time_series_length)
        .pipe(xarray.zeros_like)  # replace data with zeros
        .pipe(datasets.drop_static_vars)
        .transpose('time', ...)
        .expand_dims(batch=local_batch_size)
    )

    def make_iterator():
      for example_dict in data.as_numpy_iterator():
        yield self.from_xarray_fn(template.copy(data=example_dict))

    return make_iterator

  def _read_shuffled_shard(
      self,
      dataset: xarray.Dataset,
      time_series_length: int,
      min_buffer_blocks: int,
      shard_index: int,
      shard_count: int,
  ) -> tf.data.Dataset:
    sampler = reader.Windower(
        window_size=time_series_length,
        stride_between_windows=self.config.train_time_sample_offset,
    )
    local_shard_count = max(
        self.degree_of_model_parallelism, jax.local_device_count()
    )
    seed = train_utils.combine_rng_seeds(
        self.config.dataset_rng_seed, shard_index, time_series_length
    )
    data = reader.read_shuffled_shard(
        dataset,
        sampler,
        block_size_in_bytes=self.config.block_size_in_bytes / local_shard_count,
        buffer_size_in_bytes=(
            self.config.shuffle_buffer_size_in_bytes / local_shard_count
        ),
        min_buffer_blocks=min_buffer_blocks,
        shard_index=shard_index,
        shard_count=shard_count,
        seed=seed,
    )
    return data

  def _get_train_dataset(self) -> xarray.Dataset:
    train_dataset = xarray_utils.ds_with_sim_time(
        self.train_ds, self.physics_specs, self.reference_datetime
    )

    if self.config.train_dataset_time_slice:
      time_slice = slice(*self.config.train_dataset_time_slice)
      train_dataset = train_dataset.sel(time=time_slice)

    if self.config.time_subsample_rate != 1:
      raise NotImplementedError('subsampling on the fly is not supported yet')
    if self.config.add_noise_to_input:
      raise NotImplementedError('add_noise_to_input not supported yet')

    return train_dataset

  def _spatial_addressable_indices_map(
      self, spatial_dim_sizes: tuple[int, int, int]
  ) -> Mapping[jax.Device, tuple[slice, slice, slice, slice]]:
    """Get slices for indexing global arrays to local devices."""
    spec = P('batch', *self.data_coords.physics_partition_spec)
    sharding = jax.sharding.NamedSharding(self.spmd_mesh, spec)
    global_shape = (self.global_batch_size,) + spatial_dim_sizes
    indices_map = sharding.addressable_devices_indices_map(global_shape)
    indices_map = cast(
        Mapping[jax.Device, tuple[slice, slice, slice, slice]], indices_map
    )
    return indices_map

  def _read_model_parallel_dataset(
      self,
      dataset: xarray.Dataset,
      read_shard: Callable[[xarray.Dataset, int], tf.data.Dataset],
  ) -> tuple[tf.data.Dataset, xarray.Dataset]:
    """Read a shard of a training dataset into tf.data.Dataset."""
    indices_map = self._spatial_addressable_indices_map(
        tuple(dataset.sizes[k] for k in ['level', 'longitude', 'latitude'])
    )

    shard_data: list[tf.data.Dataset] = []
    for device in jax.local_devices():
      indices = indices_map[device]
      batch_index = indices[0].start or 0
      selection = dict(zip(['level', 'longitude', 'latitude'], indices[1:]))
      shard_dataset = dataset.isel(selection)
      shard_data.append(read_shard(shard_dataset, batch_index))

    choices = tf.data.Dataset.range(jax.local_device_count()).repeat()
    data = tf.data.Dataset.choose_from_datasets(shard_data, choices)
    template = shard_dataset
    return data, template

  def _build_train_inputs(
      self, time_series_length: int
  ) -> tuple[Callable[[], Any], dict[str, Any]]:
    """Loads the training dataset and returns an iterator and train_attrs."""
    train_dataset = self._get_train_dataset()
    local_batch_size = (
        # Size of data needed to satisfy batch_size_per_device.
        self.config.batch_size_per_device
        * jax.local_device_count()
    )
    if self.model_parallel_training:

      def read_shard(shard_dataset, batch_index):
        return self._read_shuffled_shard(
            shard_dataset,
            time_series_length,
            min_buffer_blocks=1,
            shard_index=batch_index,
            shard_count=self.global_batch_size,
        )

      data, template = self._read_model_parallel_dataset(
          train_dataset, read_shard
      )

    else:
      data = self._read_shuffled_shard(
          train_dataset,
          time_series_length,
          shard_index=jax.process_index(),
          shard_count=jax.process_count(),
          min_buffer_blocks=local_batch_size,
      )
      template = train_dataset

    data = data.repeat()
    data = data.batch(local_batch_size, drop_remainder=True)
    data = data.prefetch(tf.data.AUTOTUNE)
    train_iter = self._to_dataset_iter(data, template)
    data_attrs = datasets.attrs_from_dataset(train_dataset, time_series_length)
    return train_iter, data_attrs

  def build_train_and_eval_iterators(
      self,
      schedule_idx: int,
      start_step: int,
      large_eval: bool,
  ) -> TrainEvalIteratorTuple:
    """Build new iterators for training at schedule_idx.

    Args:
      schedule_idx: Index into the rollout schedule.
      start_step: Step at which this training run started at. This does not
        change unless the Borg job dies and restarts.
      large_eval: Whether this evaluation should be done over a larger set of
        data.

    Returns:
      TrainEvalIteratorTuple: Tuple consisting of
        get_train_data. Iterator providing next set of training data.
        train_step_fn. train_utils.TrainStepFunction to update weights.
        evaluate_fn. Callable to evalate metrics and write results.
        ckpt_kwargs. dict[str, Any] of kwargs to add to the checkpoint.
    """
    num_train_time_steps = self.config.train_schedule_time_steps[schedule_idx]
    trajectory_length = self._trajectory_lengths[schedule_idx]

    train_traj_spec = metrics_util.TrajectorySpec(
        trajectory_length,
        self._max_trajectory_length,
        self.train_inner_steps,
        coords=self.model_specs.coords,
        data_coords=self.data_coords,
    )

    # These only change on the first and last rollout, but re-make them anyways.
    get_eval_data, eval_attrs = self.build_eval_inputs(
        self.config.eval_dataset_time_slice,
        large_eval,
    )
    get_eval_on_train, _ = self.build_eval_inputs(
        self.config.train_dataset_time_slice,
        large_eval,
    )

    evaluate_fn = functools.partial(
        self.evaluate,
        eval_batch_fn=self.get_eval_batch_fn(train_traj_spec),
        get_eval_data=get_eval_data,
        get_train_data=get_eval_on_train,
        large_eval=large_eval,
    )
    get_train_data, train_attrs = self._build_train_inputs(trajectory_length)
    ckpt_kwargs = {
        'train_attrs': train_attrs,
        'eval_attrs': eval_attrs,
    }

    train_step_fn = self.get_train_step_fn(
        num_train_time_steps, train_traj_spec
    )

    if (
        self.config.profile_with_xprof
        and schedule_idx == 0
        and experiment_utils.is_coordinator()
        and start_step == 0
    ):
      train_step_fn = profiling_util.Traced(
          train_step_fn,  # only the initial train_step is profiled.
          trace_name='train step',
          skip_steps=2,  # avoid JIT compilation
          num_trace_steps=3,
          enable_python_tracer=True,
          host_trace_level=3,
      )
    return get_train_data(), train_step_fn, evaluate_fn, ckpt_kwargs

  def _get_eval_dataset(self) -> xarray.Dataset:
    return xarray_utils.ds_with_sim_time(
        self.eval_ds, self.physics_specs, self.reference_datetime
    )

  def build_eval_inputs(
      self,
      dataset_time_slice: tuple[str, str] | None,
      large_eval: bool,
  ) -> tuple[Callable[[], Any], Any]:
    """Returns an iterable over the data and data attrs for evaluation."""
    eval_dataset = self._get_eval_dataset()

    num_eval_batches = self.num_eval_batches(large_eval)
    eval_batch_size_per_device = self.eval_batch_size_per_device(large_eval)
    time_series_length = (
        self._eval_trajectory_length + self.config.num_init_frames - 1
    )
    local_batch_size = eval_batch_size_per_device * jax.local_device_count()

    if isinstance(eval_dataset.indexes['time'], pd.DatetimeIndex):
      # For real world training data from ERA5, carefully sample starting
      # times to ensure they are equally spaced across the year.
      assert 'sample' not in eval_dataset.dims
      logging.info('Using eval data loader for build_eval_inputs')
      sample_count = self.global_eval_batch_size(large_eval) * num_eval_batches
      if dataset_time_slice:
        time_source = eval_dataset.time.loc[slice(*dataset_time_slice)]
      else:
        time_source = eval_dataset
      first_start = time_source.indexes['time'][0]
      last_start = time_source.indexes['time'][-1]
      starts = _get_datetime_forecast_starts(
          sample_count, first_start, last_start
      )
      logging.info(f'determined evaluation data for {sample_count=}: {starts=}')
      offsets = eval_dataset.indexes['time'].get_indexer(starts)
      sampler = reader.WindowerAtOffsets(
          window_size=time_series_length, window_offsets=offsets
      )
      if self.model_parallel_training:

        def read_shard(shard_dataset, batch_index):
          selector = reader.ShardSelector(batch_index, len(starts))
          return reader.read_timeseries(shard_dataset, sampler, selector)

        data, template = self._read_model_parallel_dataset(
            eval_dataset, read_shard
        )

      else:
        selector = reader.ShardSelector(
            jax.process_index(), jax.process_count()
        )
        data = reader.read_timeseries(eval_dataset, sampler, selector)
        template = eval_dataset

      data = data.batch(local_batch_size, drop_remainder=True)
      data = data.cache()
    else:
      # For synthetic datasets (e.g., from Held-Suarez), use the same shuffling
      # we use for reading training data.
      assert not self.model_parallel_training
      if dataset_time_slice:
        eval_dataset = eval_dataset.sel(time=slice(*dataset_time_slice))
      data = self._read_shuffled_shard(
          eval_dataset,
          time_series_length,
          shard_index=jax.process_index(),
          shard_count=jax.process_count(),
          min_buffer_blocks=local_batch_size * num_eval_batches,
      )
      data = data.batch(local_batch_size, drop_remainder=True)
      data = data.take(num_eval_batches)
      template = eval_dataset

    logging.info(f'created eval data: {data}')
    eval_iter = self._to_dataset_iter(data, template)
    data_attrs = datasets.attrs_from_dataset(eval_dataset, time_series_length)
    return eval_iter, data_attrs

  #
  # Training and evaluation methods.
  #

  def _make_initial_experiment_state(
      self,
      rng: typing.PRNGKeyArray,
      init_example,
      init_forcing_data: typing.ForcingData,
      init_params: Optional[typing.Params] = None,
  ) -> ExperimentState:
    """Makes initial parameters (via hk.Module.init)."""
    if self.eval_inner_steps != self.train_inner_steps:
      raise ValueError(
          'KroneckerCorrelatedL2LossModule stddev will be ill defined since '
          f'{self.eval_inner_steps=} != {self.train_inner_steps=}'
      )
    trajectory_length = list(init_example.values())[0].shape[0]

    @jax.jit
    def init(rng, init_example, init_forcing_data):
      outer_steps = (trajectory_length - self.config.num_init_frames) + 1
      # We need an "ensemble" dimension for stochastic losses, but parameters
      # are fully replicated across the ensemble.
      if init_params is None:
        init_fn = jax.vmap(
            self._make_unbatched_trajectory_fn(outer_steps).init,
            in_axes=None,
            out_axes=0,
            spmd_axis_name='ensemble',
            axis_size=1,
        )
        unsqueezd_params = init_fn(rng, init_example, init_forcing_data)
        params = tree_map(lambda x: jnp.squeeze(x, axis=0), unsqueezd_params)
      else:
        params = init_params
      opt_state = self.optimizer.init(params)
      _, ema_state = self._ema_init(None, params)
      experiment_state = ExperimentState(opt_state, params, ema_state)
      experiment_state = train_utils.ensure_replicated(
          experiment_state, mesh=self.spmd_mesh
      )
      return experiment_state

    return init(rng, init_example, init_forcing_data)

  def _make_unbatched_trajectory_fn(self, outer_steps: int):
    """Haiku transformation of func giving (prediction, target) trajectories.

    Args:
      outer_steps: Number of outer steps the trajectory should take.

    Returns:
      hk transformed object. The .apply member maps
        (params, rng, target, forcing_data) --> (prediction, target)
    """
    if self.train_inner_steps != self.eval_inner_steps:
      # We share a trajectory for train/eval...so the spacing better be equal.
      raise ValueError(f'{self.train_inner_steps=} != {self.eval_inner_steps=}')

    @hk.transform
    def unbatched_trajectory_fn(target, forcing_data):
      """Compute Fwd(target[0]) on one single batch/device."""
      # Shapes(target) ~ (n_t, n_z, n_m, n_l)
      model = self.whirl_model.model_cls()
      _, predicted_trajectory = self._trajectory_fwd(
          x=target,
          forcing_data=forcing_data,
          model=model,
          outer_steps=outer_steps,
          inner_steps=self.train_inner_steps,
      )
      prediction, target = (
          model_utils.compute_prediction_and_target_representations(
              predicted_trajectory, target, forcing_data, model
          )
      )
      return prediction, target

    return unbatched_trajectory_fn

  def _make_batch_trajectory_fn(
      self,
      outer_steps: int,
  ) -> TrajectoryFn:
    """Target, prediction representations with shape (batch, ensemble, ...)."""

    ensembled_fn = jax.vmap(
        # (params, rng, target, forcing_data) --> (prediction, target)
        self._make_unbatched_trajectory_fn(outer_steps).apply,
        in_axes=(None, 0, None, None),
        spmd_axis_name='ensemble',
    )

    batch_ensembled_fn = jax.vmap(
        # (params, rng, target, forcing_data) --> (prediction, target)
        # Input shapes are:
        #   params: (...)
        #   rng: (batch, ensemble, ...)
        #   target: (batch, time, ...)
        #   forcing_data: (batch, time, ...)
        ensembled_fn,
        in_axes=(None, 0, 0, 0),
        spmd_axis_name='batch',
    )
    return batch_ensembled_fn

  def get_train_step_fn(
      self,
      num_train_time_steps: int,
      traj_spec: metrics_util.TrajectorySpec,
  ) -> train_utils.TrainStepFunction:
    """Makes a function to update weights via gradient descent.

    This function makes use of on-device batching. Multiple devices are combined
    via an all reduce step whereby the average (across devices) gradient is
    applied to each (on device) params.

    Args:
      num_train_time_steps: Number of time steps for the trajectory in this
        training step.
      traj_spec: Specification of training trajectory.

    Returns:
      all_reduced_train_step: Function to exectute one training step.
        Params are injected into self._trajectory_fwd and Loss (if Loss requires
        Haiku params).
    """

    batch_trajectory_fn = self._make_batch_trajectory_fn(
        # (params, rng, target, forcing_data) --> [prediction, target]
        outer_steps=num_train_time_steps // self.train_inner_steps
        + 1,
    )

    loss_fn = get_loss_obj(traj_spec).evaluate
    ensembled_loss_fn = jax.vmap(
        loss_fn,
        axis_name='ensemble',
        spmd_axis_name='ensemble',
    )
    batch_ensembled_loss_fn = jax.vmap(
        ensembled_loss_fn,
        axis_name='batch',
        spmd_axis_name='batch',
    )

    def batched_parameter_loss_fn(params, rng, target, forcing_data):
      """Mean (over on-device batch members) of loss w.r.t parameters."""
      # Input shapes are:
      #   params: (...)
      #   rng: (batch, ensemble, ...)
      #   target: (batch, time, ...)
      #   forcing_data: (batch, time, ...)
      prediction, target = batch_trajectory_fn(
          # The `target` and prediction returned are TrajectoryRepresentations.
          # So don't just re-use the arg `target`.
          params,
          rng,
          target,
          forcing_data,
      )
      # dimensions (batch, ensemble)
      per_example_loss = batch_ensembled_loss_fn(prediction, target)
      # Average over ensemble and batch dimensions (technically, we don't have
      # to average over ensemble with our current stochastic losses, but these
      # values are already identical and this is cleaner than using array
      # indexing)
      overall_loss = jnp.mean(per_example_loss, axis=(0, 1))
      assert overall_loss.ndim == 0
      return overall_loss

    # We would use donate_argnums here to update experiment_state in-place, but
    # that would mean we could not save the checkpoint in a separable thread.
    # Fortunately experiment_state is usually not too big (~100 MB).
    @train_utils.jit_once
    def train_step(experiment_state, rng, target_trajectory, forcing_data):
      opt_state, params, ema_state = experiment_state
      rng = train_utils.ensure_sharded_rng_key(rng, mesh=self.spmd_mesh)
      loss, grad = jax.value_and_grad(batched_parameter_loss_fn)(
          params, rng, target_trajectory, forcing_data
      )
      updates, opt_state = self.optimizer.update(grad, opt_state, params)
      params = optax.apply_updates(params, updates)
      _, ema_state = self._ema_update(params, ema_state)
      experiment_state = ExperimentState(opt_state, params, ema_state)
      experiment_state = train_utils.ensure_replicated(
          experiment_state, mesh=self.spmd_mesh
      )
      return experiment_state, loss

    return train_step

  def get_eval_batch_fn(
      self,
      train_traj_spec: metrics_util.TrajectorySpec,
  ) -> train_utils.EvalStepFunction:
    """Makes a function that performs a single evaluation pass.

    Args:
      train_traj_spec: TrajectorySpec for training. Used to add the "loss"
        evaluation metrics.

    Returns:
      Function mapping (params, rng, target, forcing_data) to dictionary of
        scalar metric values. Parameters are injected into self._trajectory_fwd
        and Loss (if Loss requires Haiku params).
    """
    eval_traj_spec = metrics_util.TrajectorySpec(
        self._eval_trajectory_length,
        self._max_trajectory_length,
        steps_per_save=self.eval_inner_steps,
        coords=self.model_specs.coords,
        data_coords=self.data_coords,
    )

    eval_time_steps = [
        t // self.eval_inner_steps for t in self.config.eval_time_steps
    ]
    if any(t % self.eval_inner_steps for t in self.config.eval_time_steps):
      raise ValueError(
          f'cannot evaluate {self.config.eval_time_steps=} with '
          f'{self.eval_inner_steps=}'
      )

    batch_trajectory_fn = self._make_batch_trajectory_fn(
        # (params, rng, target, forcing_data) --> [prediction, target]
        outer_steps=max(
            self._eval_trajectory_length,
            train_traj_spec.trajectory_length,
        ),
    )

    def unbatched_eval_fn(
        prediction: TrajectoryRepresentations, target: TrajectoryRepresentations
    ):
      """Evaluate(target, Fwd(target[0])) on one single batch/device."""
      metrics_dict = get_metrics_dict(
          eval_traj_spec,
          eval_time_steps,
          get_loss_obj(train_traj_spec),
          is_ensemble_data=bool(self.config.ensemble_size),
      )
      return train_utils.flatten_dict({
          k: metric.evaluate(prediction, target)
          for k, metric in metrics_dict.items()
      })

    ensembled_fn = jax.vmap(
        unbatched_eval_fn, axis_name='ensemble', spmd_axis_name='ensemble'
    )

    batch_ensembled_fn = jax.vmap(
        ensembled_fn,
        axis_name='batch',
        spmd_axis_name='batch',
    )

    @train_utils.jit_once
    def batch_mean_eval_fn(params, rng, target, forcing_data):
      """Computes mean (over batch members) of evaluation."""
      # Input shapes for this function are:
      #   params: (...)
      #   rng: (batch, ensemble, ...)
      #   target: (batch, time, ...)
      #   forcing_data: (batch, time, ...)
      rng = train_utils.ensure_sharded_rng_key(rng, mesh=self.spmd_mesh)
      prediction, target = batch_trajectory_fn(
          # The `target` and prediction returned are TrajectoryRepresentations.
          # So don't just re-use the arg `target`.
          params,
          rng,
          target,
          forcing_data,
      )
      batch_eval_values = batch_ensembled_fn(prediction, target)
      return tree_map(jnp.mean, batch_eval_values)

    return batch_mean_eval_fn

  def run_training(self):
    """See base class."""
    (
        start_step,
        times_restarted_on_nan,
        step_auto_restart_began_at,
        experiment_state,
    ) = self.initialize_experiment(
        initial_checkpoint_path=self.config.initial_checkpoint_path
    )

    if start_step >= self.config.num_training_steps:
      logging.warning(
          f'Attempting to start training at {start_step=} >='
          f' {self.config.num_training_steps=}. Will simply return'
      )
      return

    def logging_callback(step, loss, times_restarted_on_nan):
      loss = float(jax.device_get(loss))
      if step % max(self.steps_between_evals(False) // 100, 1) == 0:
        logging.info(f'{step=}, {loss=}')
      if (
          self.config.error_with_nan_loss
          and times_restarted_on_nan > self.config.max_nan_restarts
      ):
        raise RuntimeError(
            f'NaN loss detected at {step=}, after too many restarts since'
            f' {times_restarted_on_nan=} >'
            f' {self.config.max_nan_restarts=}. Aborting.'
        )

    # monitor loss using a separate thread, so it doesn't block execution
    logging_stream = streaming.SingleThreadExecutor(logging_callback)
    logging.info('starting training from step=%s', start_step)
    train_step_timer = timing_util.Timer()

    rng_stream = train_utils.BatchedPRNGSequence(
        jax.random.PRNGKey(self.config.init_rng_seed),
        batch_shape=(self.global_batch_size, self.config.ensemble_size or 1),
    )

    loss = 1.0  # setting to a non-nan value when starting an experiment.
    schedule_idx = None
    ckpt_kwargs = {}

    step = start_step
    while step < self.config.num_training_steps:
      old_schedule_idx = schedule_idx
      schedule_idx = np.sum(  # compute which leg of the schedule we are at.
          step > np.asarray(self.config.train_schedule_boundaries)
      )
      large_eval = (
          schedule_idx == len(self.config.train_schedule_time_steps) - 1
      )
      if schedule_idx != old_schedule_idx:
        (
            train_iter,
            train_step_fn,
            evaluate_fn,
            ckpt_kwargs,
        ) = self.build_train_and_eval_iterators(
            schedule_idx=schedule_idx,
            start_step=start_step,
            large_eval=large_eval,
        )

      if (
          np.isnan(loss)
          # No sense re-initializing if we've restarted a bunch already. Also,
          # note that if error_with_nan_loss=True, we should raise and not have
          # worry about the times_restarted_on_nan < max_nan_restarts here.
          and times_restarted_on_nan <= self.config.max_nan_restarts
      ):
        # See also logging_callback, which may raise RuntimeError for NaN loss.
        logging.warning(
            f'NaN loss encountered at {step=}. Re-initializing and incrementing'
            f' times_restarted_on_nan to {times_restarted_on_nan + 1}'
        )
        times_restarted_on_nan += 1
        step, _, _, experiment_state = self.initialize_experiment(
            target_step=step
            - times_restarted_on_nan * self.config.restart_lookback_steps
        )
        step_auto_restart_began_at = step_auto_restart_began_at or step

      # checkpoint
      if step % self.steps_between_evals(large_eval) == 0:
        self.save_checkpoint(
            step,
            experiment_state,
            checkpoint_buffer_size=self.config.checkpoint_buffer_size,
            times_restarted_on_nan=times_restarted_on_nan,
            step_auto_restart_began_at=step_auto_restart_began_at,
            **ckpt_kwargs,
        )
      elif step % self.config.steps_between_checkpoints == 0:
        max_lookback_step = (
            step
            - self.config.max_nan_restarts * self.config.restart_lookback_steps
        )
        if (
            # If there is no chance of a restart sequence overlapping with
            # previously used checkpoints...
            times_restarted_on_nan
            and max_lookback_step > step_auto_restart_began_at
        ):
          logging.info(
              f'Significant progress made since {step_auto_restart_began_at=}.'
              f' In particular, {step=} Therefore set times_restarted_on_nan'
              ' to 0'
          )
          times_restarted_on_nan = 0
          step_auto_restart_began_at = None
        self.save_checkpoint(
            step,
            experiment_state,
            update_latest_only=True,
            checkpoint_buffer_size=self.config.checkpoint_buffer_size,
            times_restarted_on_nan=times_restarted_on_nan,
            step_auto_restart_began_at=step_auto_restart_began_at,
            **ckpt_kwargs,
        )

      # evaluate
      if (step + 1) % self.steps_between_evals(large_eval) == 0:
        if step > start_step:
          with train_step_timer:
            # train_step is non-blocking, so we need to block on the output
            # of the previous training step to reliably time it.
            experiment_state = jax.block_until_ready(experiment_state)
          eval_interval = self.steps_between_evals(large_eval)
          training_time = train_step_timer.total
          logging.info(
              f'training for {eval_interval} steps took '
              f'{training_time:.1f} seconds'
          )
          self.record_scalar(
              'train',
              tag='seconds_per_train_step',
              step=step,
              value=training_time / eval_interval,
          )
          train_step_timer = timing_util.Timer()  # reset

          if isinstance(train_step_fn, profiling_util.Traced):
            memory_usage = train_step_fn.tracer.memory_usage  # pytype: disable=attribute-error
            if memory_usage is not None:
              self.record_scalar(
                  'train',
                  tag='peak_memory_usage_mib',
                  step=step,
                  value=memory_usage,
              )

        with timing_util.Timer() as eval_timer:
          evaluate_fn(step, experiment_state, seed=step)
        self.record_scalar(
            'train',
            tag='seconds_per_evaluation',
            step=step,
            value=eval_timer.average,
        )
        logging.info('evaluation pass took %.1f seconds', eval_timer.average)
        self.flush_writers()  # flush all writers for this training step.

      # train
      with train_step_timer:
        # go/xprof-instrument-jax
        with jax.profiler.StepTraceAnnotation('train', step_num=step):
          batch, forcing_data = self.to_global_array(
              next(train_iter), self.global_batch_size
          )
          # This is necessary else ValueError.
          # See http://sponge2/960c64a7-703f-4a19-8572-2c97dd9c01f3
          with self.spmd_mesh:
            experiment_state, loss = train_step_fn(
                experiment_state, next(rng_stream), batch, forcing_data
            )
            # If we don't device_get (or similar), asynchronous execution
            # resultsin this timed block taking almost no time. device_get does
            # not result in longer runs, since each loop must eventually compute
            # the loss, one way or another.
            loss = jax.device_get(loss)
          logging_stream.wait()
          logging_stream.put(step, loss, times_restarted_on_nan)
      step += 1
    # End of while step < self.config.num_training_steps:

    evaluate_fn(self.config.num_training_steps, experiment_state)
    self.finalize_training(
        self.config.num_training_steps,
        experiment_state,
        times_restarted_on_nan=times_restarted_on_nan,
        step_auto_restart_began_at=step_auto_restart_began_at,
        **ckpt_kwargs,
    )

  def make_dummy_inputs(self) -> tuple[Any, Any]:
    train_dataset = xarray_utils.ds_with_sim_time(
        self.train_ds, self.physics_specs, self.reference_datetime
    )
    dummy_ds = (
        train_dataset.drop_vars('time')
        .head(time=self.config.num_init_frames)
        .pipe(xarray.zeros_like)  # replace data with zeros
        .pipe(datasets.drop_static_vars)
        .transpose('time', ...)
    )
    return self.from_xarray_fn(dummy_ds)

  def initialize_experiment(
      self,
      initial_checkpoint_path: Optional[str] = None,
      target_step: Optional[int] = None,
  ) -> tuple[int, int, int | None, ExperimentState]:
    """Returns training step and experiment state from checkpoint or init."""

    if target_step:
      ckpt = self.load_buffered_checkpoint(target_step=target_step)
      if ckpt is None:
        logging.info(
            'No acceptable buffered checkpoint found for {target_step=}'
        )
      else:
        logging.info(
            f'Using buffered checkpoint, which has step={ckpt.step}. Ideally '
            f'would have used {target_step=}'
        )
    else:
      ckpt = self.load_latest_checkpoint()
      if ckpt is None:
        logging.info('No latest checkpoint found')
      else:
        logging.info(f'Using latest checkpoint, which has step={ckpt.step}')

    init_params = None
    if ckpt is None and initial_checkpoint_path is not None:
      ckpt = checkpoint.load_checkpoint(initial_checkpoint_path)
      if self.config.reset_initial_optimizer_state:
        init_params = ckpt.eval_params
        ckpt = None  # if resetting optimizer, carry over only init_params.

    if ckpt is not None:
      start_step = ckpt.step
      logging.info(f'resuming from checkpoint at step={start_step}')
      times_restarted_on_nan = getattr(ckpt, 'times_restarted_on_nan', 0)
      step_auto_restart_began_at = getattr(
          ckpt, 'step_auto_restart_began_at', None
      )
      experiment_state = ExperimentState(
          ckpt.opt_state, ckpt.train_params, ckpt.ema_state
      )
    else:
      logging.info('starting training with new weights')
      start_step = 0
      times_restarted_on_nan = 0
      step_auto_restart_began_at = None
      rng = jax.random.PRNGKey(self.config.init_rng_seed)
      init_example, init_forcing_data = self.make_dummy_inputs()
      experiment_state = self._make_initial_experiment_state(
          rng, init_example, init_forcing_data, init_params=init_params
      )

    return (
        start_step,
        times_restarted_on_nan,
        step_auto_restart_began_at,
        experiment_state,
    )

  def _checkpoint_state(
      self,
      step: int,
      experiment_state: ExperimentState,
      times_restarted_on_nan: int,
      step_auto_restart_began_at: int,
      train_attrs: dict[str, Any],
      eval_attrs: dict[str, Any],
  ) -> checkpoint.CheckpointState:
    """Returns a checkpoint state for a given experiment_state."""
    opt_state, params, ema_state = experiment_state
    ema_params, _ = self._ema_update(params, ema_state)
    ckpt_state = checkpoint.CheckpointState(
        train_params=params,
        eval_params=ema_params,
        opt_state=opt_state,
        ema_state=ema_state,
        step=step,
        model_time_step=self._model_dt,
        model_config_str=self._model_gin_config,
        train_dataset_path=self.config.train_dataset_path,
        eval_dataset_path=self.config.eval_dataset_path,
        times_restarted_on_nan=times_restarted_on_nan,
        step_auto_restart_began_at=step_auto_restart_began_at,
        train_attrs=train_attrs,
        eval_attrs=eval_attrs,
    )
    return ckpt_state

  def evaluate(
      self,
      step,
      experiment_state,
      eval_batch_fn,
      get_eval_data,
      get_train_data,
      large_eval: bool,
      seed=0,
  ):
    """Evaluates the model on train and eval data and writes summaries.

    Args:
      step: global training step.
      experiment_state: tuple of replicated step, optimizer state and EMA
        (exponentially moving average) state for model parameters.
      eval_batch_fn: function that, given parameters; rng; batch of data,
        computes evaluation metric of interest on the given samples.
      get_eval_data: callable that returns an iterator over evaluation data that
        is used for produce summaries on unseen evaluation data.
      get_train_data: callable that returns an iterator over training data that
        is used for produce summaries on training data.
      large_eval: Whether this evaluation is on the larger size eval data.
      seed: seed for the random number generator to be used for evaluation.
    """
    num_eval_batches = self.num_eval_batches(large_eval)
    if num_eval_batches == 0:
      logging.warning(f'skipping evaluation: {num_eval_batches=}')
      return

    _, params, ema_state = experiment_state
    ema_params, _ = self._ema_update(params, ema_state)

    global_batch_size = self.global_eval_batch_size(large_eval)
    rng_stream = train_utils.BatchedPRNGSequence(
        jax.random.PRNGKey(seed),
        batch_shape=(global_batch_size, self.config.ensemble_size or 1),
    )
    to_global_array = functools.partial(
        self.to_global_array, global_batch_size=global_batch_size
    )

    # In theory, the mesh context manager should not be necessary because we use
    # jit with sharded arrays (rather than xmap or pjit), but it seems to be
    # required to avoid triggering bugs in JAX.
    with self.spmd_mesh:
      logging.info('evaluating on train dataset')
      metrics_ = train_utils.streaming_mean(
          rng_stream,
          map(to_global_array, get_train_data()),
          functools.partial(eval_batch_fn, params),
      )
      for tag, value in metrics_.items():
        self.record_scalar('train', tag=tag, value=value, step=step)

      logging.info('evaluating on test dataset')
      metrics_ = train_utils.streaming_mean(
          rng_stream,
          map(to_global_array, get_eval_data()),
          functools.partial(eval_batch_fn, params),
      )
      for tag, value in metrics_.items():
        self.record_scalar('eval', tag=tag, value=value, step=step)

      logging.info('evaluating EMA model on test dataset')
      metrics_ = train_utils.streaming_mean(
          rng_stream,
          map(to_global_array, get_eval_data()),
          functools.partial(eval_batch_fn, ema_params),
      )
      for tag, value in metrics_.items():
        self.record_scalar('eval_ema', tag=tag, value=value, step=step)


if __name__ == '__main__':
  app.run(functools.partial(run_training.main, Experiment))
