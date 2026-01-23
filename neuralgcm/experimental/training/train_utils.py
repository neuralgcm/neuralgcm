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
"""Training utility functions for NeuralGCM."""

from collections.abc import Callable
import logging
import math
import time
from typing import Any, TypeVar, Union

from dinosaur import typing
import einops
import jax
from jax.experimental import mesh_utils
import jax.numpy as jnp
import numpy as np


# pylint: disable=logging-fstring-interpolation


Array = Union[np.ndarray, jnp.ndarray]
PyTree = Any
Forcing = typing.Forcing
ExperimentState = Any
LossValue = Array
TrainStepFunction = Callable[
    [ExperimentState, int, PyTree, Forcing],
    tuple[ExperimentState, LossValue],
]


def _to_cpu(array):
  cpu_device = jax.local_devices(backend='cpu')[0]
  return jax.device_put(array, device=cpu_device)


@jax.jit
def _combine_rng_seeds(seeds: jax.Array) -> jax.Array:
  key = jax.random.key(seeds[0])
  for seed in seeds[1:]:
    key = jax.random.fold_in(key, seed)
  return jax.random.bits(key, shape=(), dtype=jnp.uint32)


def combine_rng_seeds(*seeds: int) -> int:
  """Combine uint32 seeds into a single Python integer RNG seed."""
  # Put the seeds on the first CPU device so that JAX runs the entire
  # computation on the CPU.
  seeds = _to_cpu(np.array(seeds))
  return int(_combine_rng_seeds(seeds))


def ensure_sharded_rng_key(
    rng_key: jax.Array, *, mesh: jax.sharding.Mesh
) -> jax.Array:
  """Ensure that a batched PRNG key is sharded across all devices."""
  spec = jax.sharding.PartitionSpec('batch', 'ensemble')
  sharding = jax.sharding.NamedSharding(mesh, spec)
  return jax.lax.with_sharding_constraint(rng_key, sharding)


def batch_and_ensemble_parallel_rng_key(
    batch_size: int,
    ensemble_size: int,
    seeds: tuple[int, ...],
    mesh: jax.sharding.Mesh,
) -> jax.Array:
  """Create a PRNG key for each batch and ensemble member."""
  rng = jax.random.key(seeds[0])
  for seed in seeds[1:]:
    rng = jax.random.fold_in(rng, seed)
  rng = jax.random.split(rng, (batch_size, ensemble_size))
  return ensure_sharded_rng_key(rng, mesh=mesh)


def get_tpu_physical_mesh_shape() -> tuple[int, int, int] | None:
  """Get the shape of the TPU connectivity torus for v4 or v5 chips."""
  jax_devices = jax.devices()
  try:
    device_coords = [d.coords for d in jax_devices]
  except AttributeError:
    return None  # no "coords" attribute (e.g., using CPU devices)
  dims = tuple(d + 1 for d in max(device_coords))
  if len(dims) != 3 or math.prod(dims) != len(jax_devices):
    return None  # using a v3 or older TPUs
  return dims


# dict of dicts of indicating how to rearrange from physical TPU mesh layouts
# (X, Y, Z) into logical mesh layouts (batch, ensemble, z, x, y) with
# einops.rearrange for model training.
# {tpu_topology: {(ensemble_shards, z_shard, x_shards, y_shards): ...}}
_TPU_LAYOUT_REARRANGEMENTS = {
    # v4 & v5p
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
    # v5e
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
    '4x2x1': {
        (1, 1, 1, 1): 'b0 b1 () -> (b0 b1) () () () ()',
        (1, 2, 1, 1): 'b0 z () -> b0 () z () ()',
        (1, 1, 2, 1): 'b0 x () -> b0 () () x ()',
        (1, 1, 1, 2): 'b0 y () -> b0 () () () y',
        (2, 1, 1, 1): 'b0 e () -> b0 e () () ()',
        (2, 2, 1, 1): '(b0 e) z -> b0 e z () ()',
    },
    '4x4x1': {
        (1, 1, 1, 1): 'b0 b1 () -> (b0 b1) () () () ()',
        (1, 2, 1, 1): '(b0 z) b1 () -> (b0 b1) () z () ()',
        (2, 1, 1, 1): '(b0 e) b1 () -> (b0 b1) e () () ()',
        (2, 2, 1, 1): '(b0 e) (b1 z) () -> (b0 b1) e z () ()',
        (2, 2, 2, 1): '(e z) (b0 x) () -> b0 e z x ()',
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
    logging.warning(
        'Not running on a TPU v4 or v5 mesh. The mesh being '
        'used for model parallelism is likely highly suboptimal.'
    )
    try:
      # only succeeds if the logical mesh shape perfectly matches the physical
      # mesh, e.g., in the case of pure data parallelism
      mesh_devices = mesh_utils.create_device_mesh(logical_mesh_shape)
    except (AssertionError, NotImplementedError):
      # create_device_mesh() doesn't work for non-megacore Pufferfish, but
      # that's what we use for testing.
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


def ensure_replicated(pytree: PyTree, *, mesh: jax.sharding.Mesh) -> PyTree:
  """Ensure that a pytree is replicated across all devices."""

  def replicate(x):
    x = jnp.asarray(x)
    spec = jax.sharding.PartitionSpec(*([None] * x.ndim))
    sharding = jax.sharding.NamedSharding(mesh, spec)
    return jax.lax.with_sharding_constraint(x, sharding)

  return jax.tree.map(replicate, pytree)


T = TypeVar('T')


def jit_once(f: T, **jit_kwargs) -> T:
  """Like jax.jit, but raises an error instead of compiling multiple times."""
  compiled = None

  def g(*args, **kwargs):
    nonlocal compiled
    if compiled is None:
      start_time = time.perf_counter()
      logging.info(f'Lowering {f}')
      lowered = jax.jit(f, **jit_kwargs).lower(*args, **kwargs)
      lower_time = time.perf_counter()
      logging.info(f'Compiling {f}')
      compiled = lowered.compile()
      compile_time = time.perf_counter()
      logging.info(
          f'Lowering and compiling {f} took'
          f' {compile_time - start_time:.1f} seconds'
          f' ({lower_time - start_time:.1f} for lowering,'
          f' {compile_time - lower_time:.1f} for compiling)'
      )
    return compiled(*args, **kwargs)

  return g
