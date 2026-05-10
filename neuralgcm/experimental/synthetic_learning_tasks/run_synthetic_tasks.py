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
"""Runs synthetic tasks with a simple model."""

import functools
import os

from absl import app
from absl import flags
import coordax as cx
from flax import nnx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import learned_transforms
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import spherical_harmonics
from neuralgcm.experimental.core import standard_layers
from neuralgcm.experimental.core import towers
from neuralgcm.experimental.core import transforms
from neuralgcm.experimental.synthetic_learning_tasks import spherical_geometry_tasks
from neuralgcm.experimental.synthetic_learning_tasks import synthetic_tasks
import optax
import xarray as xr


FLAGS = flags.FLAGS
flags.DEFINE_integer('train_steps', 20, 'Number of training steps.')
flags.DEFINE_integer('eval_every', 10, 'Evaluate every N steps.')
flags.DEFINE_integer('batch_size', 8, 'Batch size. Use 0 for None.')
flags.DEFINE_string(
    'output_dir',
    '/tmp/synthetic_tasks_output',
    'Directory to save plots and dataset.',
)


def default_model_factory(input_shapes, target_split_axes):
  # Determine if we need to flatten sigma levels
  has_sigma = False
  sigma_coord = None
  for f in input_shapes.values():
    if isinstance(f, cx.Field):
      coords = (
          f.coordinate.coordinates
          if hasattr(f.coordinate, 'coordinates')
          else [f.coordinate]
      )
      for coord in coords:
        if isinstance(coord, coordinates.SigmaLevels):
          has_sigma = True
          sigma_coord = coord
          break
    if has_sigma:
      break

  transform_list = [transforms.InsertAxis(cx.SizedAxis('channel', 1), loc=-1)]
  concat_dim_name = 'channel'

  if has_sigma:
    # Ravel sigma and channel into flat_channel
    concat_dim_name = 'flat_channel'
    transform_list.append(
        transforms.RavelDims(
            dims_to_ravel=(sigma_coord, 'channel'), out_dim=concat_dim_name
        )
    )

  inputs_transform = transforms.Sequential(transform_list)
  concat_dims = (concat_dim_name,)

  nn_factory = functools.partial(
      standard_layers.MlpUniform, hidden_size=32, n_hidden_layers=2
  )

  tower_factory = functools.partial(
      towers.ForwardTower.build_using_factories,
      inputs_in_dims=(concat_dim_name,),
      out_dims=('out_channel',),
      neural_net_factory=nn_factory,
  )

  no_shard_mesh = parallelism.Mesh(spmd_mesh=None)

  return learned_transforms.ForwardTowerTransform.build_using_factories(
      input_shapes=input_shapes,
      target_split_axes=target_split_axes,
      tower_factory=tower_factory,
      concat_dims=concat_dims,
      inputs_transform=inputs_transform,
      mesh=no_shard_mesh,
      rngs=nnx.Rngs(42),
  )


def generate_report(ds: xr.Dataset, output_dir: str):
  os.makedirs(output_dir, exist_ok=True)
  print(f'Generating report in {output_dir}')

  # Group variables by task
  task_vars = {}
  for var_name in ds.data_vars:
    if '/' not in str(var_name):
      continue
    task_name, metric = str(var_name).split('/', 1)
    if task_name not in task_vars:
      task_vars[task_name] = []
    task_vars[task_name].append(str(var_name))

  for task_name, vars in task_vars.items():
    # Scalar plots
    scalar_vars = [v for v in vars if set(ds[v].dims) == {'train_step'}]
    if scalar_vars:
      plt.figure(figsize=(10, 6))
      for v in scalar_vars:
        ds[v].plot(label=v.split('/', 1)[1])
      plt.title(f'{task_name} Scalars')
      plt.legend()
      plt.savefig(os.path.join(output_dir, f'{task_name}_scalars.png'))
      plt.close()

    # Zonal mean plots at final step
    zonal_vars = [
        v for v in vars if set(ds[v].dims) == {'train_step', 'latitude'}
    ]
    if zonal_vars:
      plt.figure(figsize=(10, 6))
      for v in zonal_vars:
        final_step = ds[v].isel(train_step=-1)
        final_step.plot(label=v.split('/', 1)[1])
      plt.title(f'{task_name} Zonal Means (Final Step)')
      plt.legend()
      plt.savefig(os.path.join(output_dir, f'{task_name}_zonal_means.png'))
      plt.close()

    # Map plots at final step
    map_vars = [
        v
        for v in vars
        if set(ds[v].dims) == {'train_step', 'longitude', 'latitude'}
    ]
    if map_vars:
      n_maps = len(map_vars)
      _, axes = plt.subplots(1, n_maps, figsize=(5 * n_maps, 4))
      if n_maps == 1:
        axes = [axes]
      for ax, v in zip(axes, map_vars):
        final_step = ds[v].isel(train_step=-1)
        final_step.plot(ax=ax)
        ax.set_title(v.split('/', 1)[1])
      plt.suptitle(f'{task_name} Maps (Final Step)')
      plt.tight_layout()
      plt.savefig(os.path.join(output_dir, f'{task_name}_maps.png'))
      plt.close()


def main(_):
  n_levels = 5
  batch_size = FLAGS.batch_size if FLAGS.batch_size > 0 else None

  lon_lat_grid = coordinates.LonLatGrid.TL31()
  ylm_grid = coordinates.SphericalHarmonicGrid.TL31()

  devices = jax.devices()
  # Single device mesh
  spmd_mesh = jax.sharding.Mesh(devices, ('data',))
  mesh = parallelism.Mesh(spmd_mesh=spmd_mesh)

  ylm_map = spherical_harmonics.FixedYlmMapping(
      lon_lat_grid=lon_lat_grid,
      ylm_grid=ylm_grid,
      mesh=mesh,
      partition_schema_key=None,
  )
  rngs = nnx.Rngs(0)

  # Tasks
  levels = coordinates.SigmaLevels.equidistant(n_levels)
  tasks = {
      'column_integrator': spherical_geometry_tasks.ColumnIntegratorTask(
          ylm_map=ylm_map,
          levels=levels,
          batch_size=batch_size,
          rngs=rngs,
      ),
      'helmholtz': spherical_geometry_tasks.HelmholtzDecompositionTask(
          ylm_map=ylm_map,
          batch_size=batch_size,
          rngs=rngs,
      ),
      'advection': spherical_geometry_tasks.LagrangianAdvectionTask(
          ylm_map=ylm_map,
          batch_size=batch_size,
          rngs=rngs,
      ),
      'diffusion': spherical_geometry_tasks.MaskedDiffusionTask(
          ylm_map=ylm_map,
          nan_mask=cx.field(
              jnp.zeros(lon_lat_grid.shape, dtype=jnp.bool_), lon_lat_grid
          ),
          batch_size=batch_size,
          rngs=rngs,
      ),
  }

  dataset_dict = {}

  for name, task in tasks.items():
    print(f'Running task: {name}')

    rng = jax.random.key(123)
    _, history = synthetic_tasks.run_training(
        task=task,
        model_factory=default_model_factory,
        optimizer_def=optax.adam(1e-3),
        train_steps=FLAGS.train_steps,
        eval_every=FLAGS.eval_every,
        n_eval_batches=1,
        rng=rng,
    )

    print(f'Task {name} finished.')
    if 'train_loss' in history:
      print(f"Final loss: {history['train_loss'].data[-1]}")

    for metric_name, field in history.items():
      var_name = f'{name}/{metric_name}'
      dataset_dict[var_name] = field.to_xarray()

  # Assemble the dataset and optionally serialize it
  ds = xr.Dataset(dataset_dict)

  os.makedirs(FLAGS.output_dir, exist_ok=True)
  ds_path = os.path.join(FLAGS.output_dir, 'synthetic_tasks_results.nc')
  ds.to_netcdf(ds_path)
  print(f'Dataset saved to {ds_path}')

  # Generate the report
  generate_report(ds, FLAGS.output_dir)


if __name__ == '__main__':
  app.run(main)
