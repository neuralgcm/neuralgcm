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
"""Training utility functions for NeuralGCM."""

import collections
from collections import abc
import functools
import logging
import math
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
)
from dinosaur import pytree_utils
from dinosaur import typing
import einops
import gin
import haiku as hk
import jax
from jax.experimental import mesh_utils
import jax.numpy as jnp
from neuralgcm.legacy import optimization
import numpy as np
import optax


# pylint: disable=logging-fstring-interpolation


PRNGKeyArray = typing.PRNGKeyArray
Array = Union[np.ndarray, jnp.ndarray]
PyTree = Any
Forcing = typing.Forcing

IntOrArray = Union[int, Array]
OptState = optimization.OptState
ModelParams = Any
ModelGradients = ModelParams
EMAParams = ModelParams
StepAndOptState = Tuple[IntOrArray, OptState]
StepOptAndEMAState = Tuple[IntOrArray, OptState, ModelParams]
LossValue = Array
LossFunction = Callable[[PyTree, PyTree], LossValue]
LossAndGradFunction = Callable[
    [ModelParams, PRNGKeyArray, PyTree, Forcing],
    Tuple[LossValue, ModelGradients],
]
MetricFunction = Callable[[PyTree, PyTree], Union[Array, Mapping[str, Array]]]
TrainStepFunction = Callable[
    [PRNGKeyArray, StepAndOptState, PyTree, Forcing],
    Tuple[StepAndOptState, LossValue],
]
EvalStepFunction = Callable[
    [ModelParams, PRNGKeyArray, PyTree, Forcing], Mapping[str, Array]
]
TrajectoryFunction = Callable[
    [ModelParams, PRNGKeyArray, PyTree, Forcing], Tuple[PyTree, PyTree]
]


def flatten_dict(
    inputs: Mapping[str, Any],
    parent_key: str = '',
    sep: str = ' ',
) -> Mapping[str, Array]:
  """Returns a flattened version of `inputs` dictionary."""
  items = []
  for k, v in inputs.items():
    new_key = parent_key + sep + k if parent_key else k
    if isinstance(v, Mapping):
      items.extend(flatten_dict(v, new_key, sep=sep).items())
    else:
      items.append((new_key, v))
  keys, counts = np.unique(np.array([x[0] for x in items]), return_counts=True)
  if (counts > 1).any():
    raise ValueError(f'got duplicate keys {keys[counts > 1]}')
  return dict(items)


#
#  Note that all functions below deal with *batched* inputs.
#


def loss_and_gradient(
    trajectory_fn: TrajectoryFunction,
    loss_fn: LossFunction,
) -> LossAndGradFunction:
  """Returns a function that computes loss and the gradient of the loss.

  Args:
    trajectory_fn: a function that accepts `params` and `initial_velocity` and
      returns a trajectory of velocities.
    loss_fn: a function that accepts a predicted trajectory and a ground truth
      trajectory, returning a scalar loss value.

  Returns:
    A function that accepts `params, initial_velocity, target_trajectory` and
    returns the loss and the gradient of the loss.
  """

  def _loss(
      params: ModelParams,
      rng: PRNGKeyArray,
      target_trajectory: PyTree,
      forcing_data: typing.ForcingData,
  ) -> LossValue:
    """Returns loss value and gradient with respect to model parameters."""
    _, predicted_trajectory = trajectory_fn(
        params, rng, target_trajectory, forcing_data
    )
    loss = loss_fn(predicted_trajectory, target_trajectory)  # type: ignore
    return loss

  return jax.value_and_grad(_loss)


def train_step(
    loss_and_grad_fn: LossAndGradFunction,
    optimizer: optax.GradientTransformation,
) -> TrainStepFunction:
  """Returns a function that performs a single training step.

  Args:
    loss_and_grad_fn: a function that accepts `params, initial_velocity,
      target_trajectory` and returns the loss and the gradient of the loss.
    optimizer: Optax optimizer to update params and internal state.

  Returns:
    A function that performs a single training step.
  """

  def _train_step(
      rng: PRNGKeyArray,
      step_and_state: StepAndOptState,
      target_trajectory: PyTree,
      forcing_data: typing.ForcingData,
  ) -> Tuple[StepAndOptState, LossValue]:
    """A function that performs a single training step."""
    step, opt_state = step_and_state
    loss, grad = loss_and_grad_fn(
        opt_state.params, rng, target_trajectory, forcing_data
    )

    updates, new_state = optimizer.update(
        grad, opt_state.state, opt_state.params
    )
    new_params = optax.apply_updates(opt_state.params, updates)
    new_opt_state = OptState(state=new_state, params=new_params)

    return (step + 1, new_opt_state), loss

  return _train_step


def eval_batch(
    trajectory_fn: TrajectoryFunction,
    metric_funcs: Mapping[str, MetricFunction],
) -> EvalStepFunction:
  """Returns a function that performs a single evaluation step.

  Args:
    trajectory_fn: a function that accepts `params` and `initial_velocity` and
      returns a trajectory of velocities.
    metric_funcs: a dictionary mapping strings to metric funcutils, each
      returning either a metric scalar or a dictionary of such.

  Returns:
    A function that performs a single evaluation step.
  """

  def _eval_batch(
      params: ModelParams,
      rng: PRNGKeyArray,
      target_trajectory: PyTree,
      forcing_data: typing.ForcingData,
  ) -> Mapping[str, Array]:
    """A function that performs a single evaluation step."""
    _, predicted_trajectory = trajectory_fn(
        params, rng, target_trajectory, forcing_data
    )
    metric_values = {
        k: metric(predicted_trajectory, target_trajectory)
        for k, metric in metric_funcs.items()
    }
    results = flatten_dict(metric_values)
    return results

  return _eval_batch


def streaming_mean(
    rngs: Iterable[PRNGKeyArray],
    batch_and_forcing: Iterable[Tuple[PyTree, Forcing]],
    eval_fn: Callable[[PRNGKeyArray, PyTree, Forcing], Mapping[str, Array]],
    data_preprocess_fn: Callable[..., PyTree] = lambda x: x,
) -> Mapping[str, Array]:
  """Runs evaluation on `eval_data`.

  Args:
    rngs: an iterable of random number keys to be used for evaluation.
    batch_and_forcing: an iterable of batched velocity trajectories and forcing.
    eval_fn: a function that performs a single evaluation step.
    data_preprocess_fn: a preprocessing function be applied to each batch.

  Returns:
    A dict mapping strings to metric values.

  Raises:
    RuntimeError: if there are no batches to iterate over.
  """
  eval_metrics = collections.defaultdict(float)
  count = 0
  for rng, (batch, forcing) in zip(rngs, batch_and_forcing):
    batch = data_preprocess_fn(batch)
    batch_metrics = eval_fn(rng, batch, forcing)
    for k, v in batch_metrics.items():
      eval_metrics[k] += v
    count += 1
  if not count:
    raise RuntimeError('no batches to iterate over')
  return {k: v / count for k, v in eval_metrics.items()}


@gin.register
def identity(batch: Tuple[Array, ...], rng: Array = None) -> Tuple[Array, ...]:  # pytype: disable=annotation-type-mismatch  # jax-ndarray
  """Identity preprocessing function that does not modify the `batch`."""
  del rng  # unused.
  return batch


@gin.configurable
def add_noise_to_input_frame(
    batch: Tuple[Array, ...], rng: Array, scale: float = 1e-2, **kwargs
) -> Tuple[Array, ...]:
  """Adds noise to the 0th time frame in the `batch`.

  Args:
    batch: original batch to which the noise will be added.
    rng: random number key to be used to generate noise.
    scale: scale of the normal noise to be added.
    **kwargs: other keyword arguments. Not used.

  Returns:
    batch with noise added along the 0th time slice.
  """
  del kwargs  # unused.
  time_zero_slice = pytree_utils.slice_along_axis(batch, 1, 0)
  shapes = jax.tree.map(np.shape, time_zero_slice)
  rngs = jax.random.split(rng, len(jax.tree.leaves(time_zero_slice)))
  rngs = jax.tree.unflatten(jax.tree.structure(time_zero_slice), rngs)

  def noise_fn(key, s):
    return scale * jax.random.truncated_normal(key, -2.0, 2.0, s)

  noise = jax.tree.map(noise_fn, rngs, shapes)
  add_noise_fn = lambda x, n: x.at[:, 0, ...].add(n)
  return jax.tree.map(add_noise_fn, batch, noise)


def preprocess(
    data_iterator: Iterator[Tuple[Array, ...]],
    rng_stream: Iterator[Array],
    preprocess_fn: Callable[..., Tuple[Array, ...]],
):
  """Generator that applies `preprocess_fn` to entries of the `data_iterator`.

  Args:
    data_iterator: numpy iterator holding the data.
    rng_stream: stream of random numbers to be used by `preprocess_fn`.
    preprocess_fn: preprocessing function to be applied to each batch of data.

  Yields:
    Batch of data from `data_iterator` preprocessed with `preprocess_fn`.
  """
  preprocess_fn = jax.jit(preprocess_fn)
  while True:
    rng = next(rng_stream)
    yield preprocess_fn(next(data_iterator), rng)


def split_rngs(rngs: PRNGKeyArray, num: int) -> PRNGKeyArray:
  """Splits `rngs` into `num` along the last batch axis."""
  ndim = rngs.ndim
  split_fn = jax.random.split
  for _ in range(ndim - 1):
    split_fn = jax.vmap(split_fn, (0, None), 1)
  return split_fn(rngs, num)


@functools.partial(jax.jit, static_argnames=['batch_shape'])
def _split_rmgs_by_batch_shape(
    rngs: PRNGKeyArray,
    batch_shape: tuple[int, ...],
) -> PRNGKeyArray:
  for batch_size in batch_shape[::-1]:
    rngs = split_rngs(rngs, batch_size)
  return rngs


class BatchedPRNGSequence(Iterator):
  """Iterator of JAX different random keys split by `batch_shape`."""

  def __init__(
      self,
      key_or_seed: Union[int, PRNGKeyArray],
      batch_shape: Optional[Tuple[int, ...]] = None,
  ):
    """Creates an instance a class.

    Args:
      key_or_seed: Key or seed to initialize the random sequence.
      batch_shape: Batch shape of the sequence.
    """
    self._key = hk.PRNGSequence(key_or_seed)
    self.batch_shape = batch_shape

  def reserve(self, num: int):
    """Splits an additional ``num`` keys for later use."""
    self._key = self._key.reserve(num)

  def __next__(self):
    rngs = next(self._key)
    return _split_rmgs_by_batch_shape(rngs, self.batch_shape)


@jax.jit
def _combine_rng_seeds(seeds: jax.Array) -> jax.Array:
  key = jax.random.PRNGKey(seeds[0])
  for seed in seeds[1:]:
    key = jax.random.fold_in(key, seed)
  return jax.random.bits(key, shape=(), dtype=jnp.uint32)


def combine_rng_seeds(*seeds: int) -> int:
  """Combine uint32 seeds into a single Python integer RNG seed."""
  # Put the seeds on the first CPU device so that JAX runs the entire
  # computation on the CPU.
  seeds = jax.device_put(
      np.array(seeds), device=jax.local_devices(backend='cpu')[0]
  )
  return int(_combine_rng_seeds(seeds))


def ensure_sharded_rng_key(
    rng_key: jax.Array, *, mesh: jax.sharding.Mesh
) -> jax.Array:
  """Ensure that a batched PRNG key is sharded across all devices."""
  spec = P('batch', 'ensemble', None)
  sharding = jax.sharding.NamedSharding(mesh, spec)
  return jax.lax.with_sharding_constraint(rng_key, sharding)


def get_tpu_physical_mesh_shape() -> tuple[int, int, int] | None:
  """Get the shape of the TPU connectivity torus for v4 or v5 chips."""
  jax_devices = jax.devices()
  try:
    device_coords = [d.coords for d in jax_devices]
  except AttributeError:
    return None  # no "coords" attribute (e.g., using CPU devices)
  dims = tuple(d + 1 for d in max(device_coords))
  if len(dims) != 3 or math.prod(dims) != len(jax_devices):
    return None
  return dims


# dict of dicts of indicating how to rearrange from physical TPU mesh layouts
# (X, Y, Z) into logical mesh layouts (batch, ensemble, z, x, y) with
# einops.rearrange for model training.
# {tpu_topology: {(ensemble_shards, z_shard, x_shards, y_shards): ...}}
_TPU_LAYOUT_REARRANGEMENTS = {
    '2x2x2': {
        (1, 1, 1, 1): 'b0 b1 b2 -> (b0 b1 b2) () () () ()',
        (1, 2, 1, 1): 'z b0 b1 -> (b0 b1) () z () ()',
        (2, 1, 1, 1): 'e b0 b1 -> (b0 b1) e () () ()',
    },
    '2x2x4': {
        (1, 1, 1, 1): 'b0 b1 b2 -> (b0 b1 b2) () () () ()',
        (1, 2, 1, 1): 'z b0 b1 -> (b0 b1) () z () ()',
        (1, 4, 1, 1): 'b0 b1 z -> (b0 b1) () z () ()',
        (2, 1, 1, 1): 'e b0 b1 -> (b0 b1) e () () ()',
    },
    '2x4x4': {
        (1, 1, 1, 1): 'b0 b1 b2 -> (b0 b1 b2) () () () ()',
        (1, 2, 1, 1): 'z b0 b1 -> (b0 b1) () z () ()',
        (1, 4, 1, 1): 'b0 b1 z -> (b0 b1) () z () ()',
        (2, 1, 1, 1): 'e b0 b1 -> (b0 b1) e () () ()',
        (2, 2, 1, 1): 'z (b0 e) b1 -> (b0 b1) e z () ()',
    },
    '4x4x4': {
        (1, 1, 1, 1): 'b0 b1 b2 -> (b0 b1 b2) () () () ()',
        (1, 2, 1, 1): '(b0 z) b1 b2 -> (b0 b1 b2) () z () ()',
        (1, 4, 1, 1): 'b0 b1 z -> (b0 b1) () z () ()',
        (1, 2, 2, 1): '(b0 z) (b1 x) b2 -> (b0 b1 b2) () z x ()',
        (2, 1, 1, 1): '(b0 e) b1 b2 -> (b0 b1 b2) e () () ()',
        (2, 2, 1, 1): '(b0 e) (b1 z) b2 -> (b0 b1 b2) e z () ()',
        (2, 2, 2, 1): '(b0 e) (b1 z) (b2 x) -> (b0 b1 b2) e z x ()',
    },
    '4x4x8': {
        (1, 1, 1, 1): 'b0 b1 b2 -> (b0 b1 b2) () () () ()',
        (1, 4, 2, 1): 'z (b0 x) b1 -> (b0 b1) () z x ()',
        (1, 4, 2, 2): 'z (b0 x) (b1 y) -> (b0 b1) () z x y',
        (2, 4, 2, 1): 'z (b0 x) (b1 e) -> (b0 b1) e z x ()',
    },
    '4x8x8': {
        (1, 1, 1, 1): 'b0 b1 b2 -> (b0 b1 b2) () () () ()',
        (1, 4, 2, 1): 'z (b0 x) b1 -> (b0 b1) () z x ()',
        (1, 4, 2, 2): 'z (b0 x) (b1 y) -> (b0 b1) () z x y',
        (2, 4, 2, 1): 'z (b0 e) (b1 x) -> (b0 b1) e z x ()',
        (2, 4, 2, 2): 'z (b0 e x) (b1 y) -> (b0 b1) e z x y',
    },
    '2x2x1': {
        (1, 1, 1, 1): 'b0 b1 () -> (b0 b1) () () () ()',
        (1, 2, 1, 1): 'z b0 () -> b0 () z () ()',
        (2, 1, 1, 1): 'e b0 () -> b0 e () () ()',
        (2, 2, 1, 1): 'e z () -> () e z () ()',
    },
    '2x4x1': {
        (1, 1, 1, 1): 'b0 b1 () -> (b0 b1) () () () ()',
        (1, 2, 1, 1): 'z b0 () -> b0 () z () ()',
        (2, 1, 1, 1): 'e b0 () -> b0 e () () ()',
        (2, 2, 1, 1): 'z (b0 e) -> b0 e z () ()',
    },
    '4x4x1': {
        (1, 1, 1, 1): 'b0 b1 () -> (b0 b1) () () () ()',
        (1, 2, 1, 1): '(b0 z) b1 () -> (b0 b1) () z () ()',
        (2, 1, 1, 1): '(b0 e) b1 () -> (b0 b1) e () () ()',
        (2, 2, 1, 1): '(b0 e) (b1 z) () -> (b0 b1) e z () ()',
    },
    '4x8x1': {
        (1, 1, 1, 1): 'b0 b1 () -> (b0 b1) () () () ()',
        (1, 2, 1, 1): '(b0 z) b1 () -> (b0 b1) () z () ()',
        (1, 2, 2, 1): '(b0 z) (b1 x) () -> (b0 b1) () z x ()',
        (2, 1, 1, 1): '(b0 e) b1 () -> (b0 b1) e () () ()',
        (2, 2, 1, 1): '(b0 e) (b1 z) () -> (b0 b1) e z () ()',
    },
    '8x8x1': {
        (1, 1, 1, 1): 'b0 b1 () -> (b0 b1) () () () ()',
        (1, 2, 1, 1): '(b0 z) b1 () -> (b0 b1) () z () ()',
        (1, 2, 2, 1): '(b0 z) (b1 x) () -> (b0 b1) () z x ()',
        (2, 1, 1, 1): '(b0 e) b1 () -> (b0 b1) e () () ()',
        (2, 2, 1, 1): '(b0 e) (b1 z) () -> (b0 b1) e z () ()',
        (2, 2, 2, 1): '(b0 e z) (b1 x) () -> (b0 b1) e z x ()',
    },
    '8x16x1': {
        (1, 1, 1, 1): 'b0 b1 () -> (b0 b1) () () () ()',
        (1, 2, 1, 1): '(b0 z) b1 () -> (b0 b1) () z () ()',
        (1, 2, 2, 1): '(b0 z) (b1 x) () -> (b0 b1) () z x ()',
        (2, 1, 1, 1): '(b0 e) b1 () -> (b0 b1) e () () ()',
        (2, 2, 1, 1): '(b0 e) (b1 z) () -> (b0 b1) e z () ()',
        (2, 2, 2, 1): '(b0 e z) (b1 x) () -> (b0 b1) e z x ()',
    },
}


def create_spmd_mesh(sizes: dict[str, int]) -> jax.sharding.Mesh:
  """Create an SPMD mesh suitable for data & model parallelism.

  Args:
    sizes: dictionary mapping from dimension names (batch, z, x, and y) to the
      number of devices desired along that axis in the parallel mesh.

  Returns:
    Mesh with axis names ['batch', 'ensemble', 'x', 'y', 'z'] and the desired
    axis sizes.
  """
  axis_names = ['batch', 'ensemble', 'z', 'x', 'y']
  for name in sizes:
    if name not in axis_names:
      raise ValueError(f'unrecognized {name!r} not in {axis_names}')

  logical_mesh_shape = tuple(
      sizes.get(axis_name, 1) for axis_name in axis_names
  )
  if math.prod(logical_mesh_shape) != jax.device_count():
    raise ValueError(
        f'{logical_mesh_shape=} is incompatible with {jax.device_count()=}'
    )

  physical_mesh_shape = get_tpu_physical_mesh_shape()
  if physical_mesh_shape is None:
    try:
      # only succeeds if the logical mesh shape perfectly matches the physical
      # mesh, e.g., in the case of pure data parallelism
      mesh_devices = mesh_utils.create_device_mesh(logical_mesh_shape)
    except (AssertionError, NotImplementedError):
      mesh_devices = np.reshape(jax.devices(), logical_mesh_shape)
  else:
    devices = np.empty(physical_mesh_shape, dtype=object)
    for device in jax.devices():
      devices[tuple(device.coords)] = device

    topology = 'x'.join(map(str, physical_mesh_shape))
    logical_mesh_shape = tuple(
        sizes[dim] for dim in ['ensemble', 'z', 'x', 'y']
    )
    rearrangement = _TPU_LAYOUT_REARRANGEMENTS[topology][logical_mesh_shape]

    abbreviated_sizes = {
        'e': sizes['ensemble'],
        'z': sizes['z'],
        'x': sizes['x'],
        'y': sizes['y'],
    }
    abbreviated_sizes = {k: v for k, v in abbreviated_sizes.items() if v != 1}
    mesh_devices = einops.rearrange(devices, rearrangement, **abbreviated_sizes)

  return jax.sharding.Mesh(mesh_devices, axis_names)


P = jax.sharding.PartitionSpec


def make_distributed_array_from_local_arrays(
    pytree: PyTree,
    mesh: jax.sharding.Mesh,
    spatial_partitions: jax.sharding.PartitionSpec,
    global_batch_size: int,
) -> PyTree:
  """Creates a pytree of global jax arrays for data/model parallelsm.

  This function exists for loading spatially partitioned data, which is assumed
  to be replicated across the ensemble dimension.

  Args:
    pytree: PyTree of NumPy arrays to convert into distributed JAX arrays. The
      leading "batch" dimension is divided between different local devices.
    mesh: SPDM sharding mesh.
    spatial_partitions: JAX partition spec (of length 3) to use for partitioning
      spatial dimensions (z, x, y).
    global_batch_size: number distinct examples in a single batch across all
      devices. Does not include the ensemble.

  Returns:
    Pytree with the same structure as the inputs, but with arrays replaced by
    distributed JAX arrays.
  """
  if len(spatial_partitions) != 3:
    raise ValueError(f'invalid {spatial_partitions=}')

  def get_shard_count(spec_part: None | str | tuple[str, ...]) -> int:
    # calculate the number of shards corresponding to an element in a
    # PartitionSpec
    if spec_part is None:
      return 1
    elif isinstance(spec_part, str):
      return mesh.shape[spec_part]
    else:
      return math.prod(mesh.shape[x] for x in spec_part)

  def shard_array(x: np.ndarray) -> jax.Array:
    if x.ndim <= 3:
      # handle sim_time [batch]
      global_shape = (global_batch_size,) + x.shape[1:]
      partition_spec = P('batch', *([None] * (x.ndim - 1)))
    elif x.ndim == 4:
      # This is currently needed to handle surface data that has shape:
      # [batch, time, x, y].
      _, x_shards, y_shards = map(get_shard_count, spatial_partitions)
      global_shape = (
          global_batch_size,
          x.shape[1],
          x.shape[2] * x_shards,
          x.shape[3] * y_shards,
      )
      partition_spec = P('batch', None, *spatial_partitions[1:])
    else:
      # everything else has dimensions [batch, time, z, x, y]
      assert x.ndim == 5, x.shape
      z_shards, x_shards, y_shards = map(get_shard_count, spatial_partitions)
      if x.shape[2] == 1:
        z_shards = 1
      global_shape = (
          global_batch_size,
          x.shape[1],
          x.shape[2] * z_shards,
          x.shape[3] * x_shards,
          x.shape[4] * y_shards,
      )
      partition_spec = P('batch', None, *spatial_partitions)

    sharding = jax.sharding.NamedSharding(mesh, partition_spec)
    single_device_arrays = put_to_devices(x, jax.local_devices(), axis=0)
    return jax.make_array_from_single_device_arrays(
        global_shape, sharding, single_device_arrays
    )

  try:
    return jax.tree_util.tree_map(shard_array, pytree)
  except Exception as e:
    shape_tree = jax.tree_util.tree_map(jnp.shape, pytree)
    raise RuntimeError(
        f'failed to shard arrays with shapes {shape_tree!r}'
    ) from e


def put_to_devices(
    host_array: np.ndarray, local_devices: abc.Sequence[Any], axis: int
) -> list[Any]:
  """Transfers a host array to local devices, split on the first dimension."""
  local_device_count = len(local_devices)
  try:
    per_device_arrays = np.split(host_array, local_device_count, axis=axis)
  except ValueError as array_split_error:
    raise ValueError(
        f'Unable to put to devices shape {host_array.shape} with '
        f'local device count {local_device_count}'
    ) from array_split_error
  device_buffers = [
      jax.device_put(arr, d) for arr, d in zip(per_device_arrays, local_devices)
  ]
  return device_buffers


def ensure_replicated(pytree: PyTree, *, mesh: jax.sharding.Mesh) -> PyTree:
  """Ensure that a pytree is replicated across all devices."""

  def replicate(x):
    x = jnp.asarray(x)
    spec = jax.sharding.PartitionSpec(*([None] * x.ndim))
    sharding = jax.sharding.NamedSharding(mesh, spec)
    return jax.lax.with_sharding_constraint(x, sharding)

  return jax.tree_util.tree_map(replicate, pytree)


T = TypeVar('T')


def jit_once(f: T, **jit_kwargs) -> T:
  """Like jax.jit, but raises an error instead of compiling multiple times."""
  compiled = None

  def g(*args, **kwargs):
    nonlocal compiled
    if compiled is None:
      logging.info(f'lowering {f}')
      lowered = jax.jit(f, **jit_kwargs).lower(*args, **kwargs)
      logging.info(f'compiling {f}')
      compiled = lowered.compile()
      logging.info(f'finishing compiling {f}')
    return compiled(*args, **kwargs)

  return g
