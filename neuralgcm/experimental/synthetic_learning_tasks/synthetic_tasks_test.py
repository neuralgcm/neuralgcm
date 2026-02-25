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
"""Tests for synthetic_tasks and spherical_geometry_tasks."""

from absl.testing import absltest
from absl.testing import parameterized
import coordax as cx
from flax import nnx
import jax
import jax.numpy as jnp
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import spherical_harmonics
from neuralgcm.experimental.core import transforms
from neuralgcm.experimental.metrics import deterministic_losses
from neuralgcm.experimental.metrics import evaluators
from neuralgcm.experimental.synthetic_learning_tasks import spherical_geometry_tasks
from neuralgcm.experimental.synthetic_learning_tasks import synthetic_tasks
import optax


class SimpleTask(synthetic_tasks.SyntheticTask):

  @property
  def batch_axis(self):
    return self.b

  @property
  def input_shapes(self):
    return {'u': cx.shape_struct_field(self.b, self.x)}

  @property
  def target_split_axes(self):
    return {'u': cx.Scalar()}

  def __init__(self):
    self.b = cx.SizedAxis('b', 4)
    self.x = cx.SizedAxis('x', 5)

  def sample_batch(self, rng):
    u = cx.field(
        jax.random.normal(rng, (4, 5)) + 10.0,
        cx.coords.compose(self.b, self.x),
    )
    return {'u': u}, {'u': u}


class DummyModel(nnx.Module):

  def __init__(self, input_shapes, target_split_axes):
    self.scaler = transforms.StreamNorm.for_inputs_struct(
        input_shapes, independent_axes=(), update_stats=False
    )
    self.param = nnx.Param(jnp.ones((1,)))

  def __call__(self, inputs):
    return self.scaler(inputs)


class SyntheticTasksTest(parameterized.TestCase):

  @parameterized.parameters(
      {'batch_size': 4},
      {'batch_size': None},
  )
  def test_column_integrator_task(self, batch_size):
    grid = coordinates.LonLatGrid.TL31()
    ylm_grid = coordinates.SphericalHarmonicGrid.TL31()
    levels = coordinates.SigmaLevels.equidistant(5)
    b = cx.SizedAxis('b', batch_size) if batch_size else cx.Scalar()
    mesh = parallelism.Mesh(None)
    ylm_map = spherical_harmonics.FixedYlmMapping(
        lon_lat_grid=grid,
        ylm_grid=ylm_grid,
        mesh=mesh,
        partition_schema_key=None,
    )
    task = spherical_geometry_tasks.ColumnIntegratorTask(
        ylm_map=ylm_map,
        levels=levels,
        batch_size=batch_size,
        rngs=nnx.Rngs(0),
    )

    rng = jax.random.key(0)
    inputs_dict, targets_dict = task.sample_batch(rng)
    inputs = inputs_dict['features']
    targets = targets_dict['integral']
    surface_pressure = inputs_dict['surface_pressure']

    self.assertEqual(inputs.coordinate, cx.coords.compose(b, levels, grid))
    self.assertEqual(targets.coordinate, cx.coords.compose(b, grid))
    self.assertEqual(surface_pressure.coordinate, cx.coords.compose(b, grid))

  @parameterized.parameters(
      {'batch_size': 4},
      {'batch_size': None},
  )
  def test_helmholtz_decomposition_task(self, batch_size):
    grid = coordinates.LonLatGrid.TL31()
    ylm_grid = coordinates.SphericalHarmonicGrid.TL31()
    b = cx.SizedAxis('b', batch_size) if batch_size else cx.Scalar()
    mesh = parallelism.Mesh(None)
    ylm_map = spherical_harmonics.FixedYlmMapping(
        lon_lat_grid=grid,
        ylm_grid=ylm_grid,
        mesh=mesh,
        partition_schema_key=None,
    )
    task = spherical_geometry_tasks.HelmholtzDecompositionTask(
        ylm_map=ylm_map,
        batch_size=batch_size,
        rngs=nnx.Rngs(0),
    )
    rng = jax.random.key(0)
    inputs_dict, targets_dict = task.sample_batch(rng)
    u = inputs_dict['u']
    v = inputs_dict['v']
    vort = targets_dict['vorticity']
    div = targets_dict['divergence']

    expected_coord = cx.coords.compose(b, grid)
    self.assertEqual(u.coordinate, expected_coord)
    self.assertEqual(v.coordinate, expected_coord)
    self.assertEqual(vort.coordinate, expected_coord)
    self.assertEqual(div.coordinate, expected_coord)

  @parameterized.parameters(
      {'batch_size': 4},
      {'batch_size': None},
  )
  def test_lagrangian_advection_task(self, batch_size):
    grid = coordinates.LonLatGrid.TL31()
    ylm_grid = coordinates.SphericalHarmonicGrid.TL31()
    b = cx.SizedAxis('b', batch_size) if batch_size else cx.Scalar()
    mesh = parallelism.Mesh(None)
    ylm_map = spherical_harmonics.FixedYlmMapping(
        lon_lat_grid=grid,
        ylm_grid=ylm_grid,
        mesh=mesh,
        partition_schema_key=None,
    )
    task = spherical_geometry_tasks.LagrangianAdvectionTask(
        ylm_map=ylm_map,
        batch_size=batch_size,
        rngs=nnx.Rngs(0),
    )
    rng = jax.random.key(0)
    inputs_dict, targets_dict = task.sample_batch(rng)
    u = inputs_dict['u']
    v = inputs_dict['v']
    tracer = inputs_dict['tracer']
    target_tracer = targets_dict['tracer']

    expected_coord = cx.coords.compose(b, grid)
    self.assertEqual(u.coordinate, expected_coord)
    self.assertEqual(v.coordinate, expected_coord)
    self.assertEqual(tracer.coordinate, expected_coord)
    self.assertEqual(target_tracer.coordinate, expected_coord)

  @parameterized.parameters(
      {'batch_size': 4},
      {'batch_size': None},
  )
  def test_masked_diffusion_task(self, batch_size):
    grid = coordinates.LonLatGrid.TL31()
    ylm_grid = coordinates.SphericalHarmonicGrid.TL31()
    b = cx.SizedAxis('b', batch_size) if batch_size else cx.Scalar()
    mesh = parallelism.Mesh(None)
    ylm_map = spherical_harmonics.FixedYlmMapping(
        lon_lat_grid=grid,
        ylm_grid=ylm_grid,
        mesh=mesh,
        partition_schema_key=None,
    )
    nan_mask = cx.field(jnp.zeros(grid.shape, dtype=jnp.bool_), grid)

    task = spherical_geometry_tasks.MaskedDiffusionTask(
        ylm_map=ylm_map,
        nan_mask=nan_mask,
        batch_size=batch_size,
        rngs=nnx.Rngs(0),
    )
    rng = jax.random.key(0)
    inputs_dict, targets_dict = task.sample_batch(rng)
    u = inputs_dict['u']
    mask = inputs_dict['nan_mask']
    target_u = targets_dict['u']

    expected_coord = cx.coords.compose(b, grid)
    self.assertEqual(u.coordinate, expected_coord)
    self.assertEqual(mask.coordinate, expected_coord)
    self.assertEqual(target_u.coordinate, expected_coord)


  def test_run_training_calibration(self):
    task = SimpleTask()
    model_factory = lambda shapes, axes: DummyModel(shapes, axes)
    optimizer_def = optax.adam(1e-3)

    loss_evaluator = evaluators.Evaluator(
        metrics={'mse': deterministic_losses.MSE()},
        aggregators=evaluators.aggregation.Aggregator(
            dims_to_reduce=('b', 'x')
        ),
    )

    rng = jax.random.key(0)

    # Run with calibration
    model, _ = synthetic_tasks.run_training(
        task=task,
        model_factory=model_factory,
        optimizer_def=optimizer_def,
        train_steps=1,
        eval_every=1,
        n_eval_batches=1,
        rng=rng,
        loss_evaluator=loss_evaluator,
        calibration_steps=5,
    )

    means = model.scaler.stream_norm.means.get_value()
    mean_val = means['u'].data

    # Assert that mean is non-zero in calibrated model (since inputs have mean ~10)
    self.assertTrue(
        jnp.all(mean_val > 5.0), f'Expected mean > 5, got {mean_val}'
    )


if __name__ == '__main__':
  absltest.main()
