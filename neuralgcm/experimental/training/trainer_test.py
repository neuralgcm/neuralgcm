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

import os
import shutil
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import coordax as cx
from flax import nnx
import jax
import jax.numpy as jnp
import jax_datetime as jdt
from neuralgcm.experimental.core import api
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import data_specs
from neuralgcm.experimental.core import transforms
from neuralgcm.experimental.core import xarray_utils
from neuralgcm.experimental.metrics import aggregation
from neuralgcm.experimental.metrics import deterministic_metrics
from neuralgcm.experimental.metrics import evaluators
from neuralgcm.experimental.metrics import probabilistic_losses
from neuralgcm.experimental.toy_model_examples import lorenz96
from neuralgcm.experimental.training import data_loading
from neuralgcm.experimental.training import trainer
import numpy as np
import optax
import pandas as pd


class TestLorenz96(lorenz96.Lorenz96):
  def __post_init__(self):
    super().__post_init__()
    # Add a dummy variable to ensure non_params is not empty
    self.dummy_non_param = nnx.Variable(jnp.array(0.0))


class TestLorenz96WithTwoScales(lorenz96.Lorenz96WithTwoScales):
  def __post_init__(self):
    super().__post_init__()
    # Add a dummy variable to ensure non_params is not empty
    self.dummy_non_param = nnx.Variable(jnp.array(0.0))
    # Add a dummy param to ensure params is not empty
    self.dummy_param = nnx.Param(jnp.array(0.0))


class InMemorySaver(trainer.OnlineMetricsSaver):
  """Online metrics saver that stores metrics in memory for testing."""

  def __init__(self):
    self.metrics = []

  def save(self, step, metrics):
    self.metrics.append((step, metrics))


def _construct_spec(supported_spec, timedelta):
  """Construct concrete data specs from inputs spec."""
  # Assumes all supported_spec are CoordSpec with "any" timedelta.
  def _set_timedelta(c):
    assert isinstance(c, data_specs.CoordSpec)
    c = c.coord
    return cx.coords.compose(
        *[timedelta if isinstance(ax, coordinates.TimeDelta) else ax
          for ax in c.axes]
    )

  return jax.tree.map(
      _set_timedelta,
      supported_spec,
      is_leaf=lambda x: isinstance(x, data_specs.CoordSpec)
  )


class TrainerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.test_dir)
    super().tearDown()

  def create_model_and_data(self, multiscale: bool = False):
    k = cx.LabeledAxis('k', np.arange(8))
    t0 = jdt.Datetime.from_isoformat('2000-01-01')
    dummy_dt = coordinates.TimeDelta(np.timedelta64(0, 's') * np.arange(1))
    x_init = jnp.zeros(k.sizes['k'])
    if multiscale:
      j = cx.LabeledAxis('j', np.arange(4))
      kj = cx.coords.compose(k, j)
      model = TestLorenz96WithTwoScales(k, j)
      y_init = jnp.zeros((k.sizes['k'], j.sizes['j']))
      inputs = {
          'slow': {
              'x': cx.field(x_init[None, ...], dummy_dt, k),
              'time': cx.field(t0[None, ...], dummy_dt),
          },
          'fast': {'y': cx.field(y_init[None, ...], dummy_dt, kj)}
      }
    else:
      kj = None  # make pytype happy.
      model = TestLorenz96(k_axis=k, dt=0.01, forcing=nnx.Param(10.0))
      x_init = jnp.zeros(k.sizes['k'])
      inputs = {
          'slow': {
              'x': cx.field(x_init[None, ...], dummy_dt, k),
              'time': cx.field(t0[None, ...], dummy_dt),
          }
      }

    inference_model = api.InferenceModel.from_model_api(model)
    state = inference_model.assimilate(inputs)

    trajectories = {}
    dt_slow = model.timestep * 4
    n_steps = 400
    _, trajectory = api.unroll_from_advance(
        inference_model,
        initial_state=state,
        timedelta=dt_slow,
        steps=n_steps,
        query={'slow': {'x': k, 'time': cx.Scalar()}},
    )
    trajectories.update(trajectory)
    if multiscale:
      assert isinstance(kj, cx.Coordinate)
      dt_fast = model.timestep
      _, trajectory_fast = api.unroll_from_advance(
          inference_model,
          initial_state=state,
          timedelta=dt_fast,
          steps=n_steps * 4,
          query={'fast': {'y': kj, 'time': cx.Scalar()}}
      )
      trajectories.update(trajectory_fast)

    ds = xarray_utils.nested_fields_to_xarray(trajectories)
    # Post-process to match expected dataset structure (timedelta -> time).
    start_time = pd.Timestamp('2000-01-01')
    all_data = {}
    for key in ds:
      ds[key] = ds[key].swap_dims({'timedelta': 'time'}).set_coords('time')
      ds[key].coords['time'] = start_time + ds[key].coords['timedelta']
      all_data[key] = ds[key]
    return model, all_data

  def _create_data_loader(self, all_data):
    spmd_mesh = trainer.create_spmd_mesh(1, 1, 1, 1)
    training_mesh = trainer.create_training_mesh(spmd_mesh, {'physics': {}})
    return data_loading.DataLoader(
        all_data=all_data,
        parallelism_mesh=training_mesh,
        loading_partition_schema='physics',
    )

  def _create_evaluators(self, dims_to_reduce):
    aggregator = aggregation.Aggregator(
        dims_to_reduce=dims_to_reduce,
        weight_by=[],
    )
    mse = deterministic_metrics.MSE()
    eval_metrics = evaluators.Evaluator(
        metrics={'mse': mse},
        aggregators={'mse': aggregator}
    )
    crps = probabilistic_losses.CRPS()
    loss = evaluators.Evaluator(
        metrics={'crps': crps},
        aggregators={'crps': aggregator}
    )
    return eval_metrics, loss

  def _create_queries_specs(self, model):
    remove_timedelta = lambda c: cx.coords.compose(
        *[ax for ax in c.axes if not isinstance(ax, coordinates.TimeDelta)]
    )
    queries_specs = {}
    for data_key, k_spec in model.inputs_spec.items():
      queries_specs[data_key] = {
          k: remove_timedelta(v.coord) for k, v in k_spec.items() if k != 'time'
      }
    return queries_specs

  def test_trainer(self):
    model, all_data = self.create_model_and_data(multiscale=False)
    queries_specs = self._create_queries_specs(model)
    data_loader = self._create_data_loader(all_data)
    eval_metrics, loss = self._create_evaluators(
        dims_to_reduce=('ensemble', 'batch', 'k')
    )
    dt_slow = model.timestep * 4

    optimizer = optax.adam(1e-3)
    opt_config = trainer.OptimizationConfig(optimizer, ema_num_steps=10)

    train_stages = []
    for steps in [2, 5]:
      timedelta = coordinates.TimeDelta(np.arange(steps) * dt_slow)
      inputs_spec = _construct_spec(model.inputs_spec, timedelta)
      dyn_spec = _construct_spec(model.dynamic_inputs_spec, timedelta)
      train_stages.append(
          trainer.TrainStage(
              duration=steps,
              inputs_spec=inputs_spec,
              dynamic_inputs_spec=dyn_spec,
              queries_spec=queries_specs,
              loss=loss,
              time_sample_offset=dt_slow,
              batch_size_per_device=1,
              shuffle_buffer_size=0,
              train_time_slice=None,
              eval_time_slice=None,
          )
      )
    train_schedule = trainer.TrainSchedule(stages=train_stages)

    eval_timedelta = coordinates.TimeDelta(np.arange(10) * dt_slow)
    eval_inputs_spec = _construct_spec(model.inputs_spec, eval_timedelta)
    eval_dyn_spec = _construct_spec(model.dynamic_inputs_spec, eval_timedelta)
    eval_schedule = trainer.EvalSchedule(stages=[
        trainer.EvalSchema(
            cadence=2,
            inputs_spec=eval_inputs_spec,
            dynamic_inputs_spec=eval_dyn_spec,
            queries_spec=queries_specs,
            metrics_evaluator=eval_metrics,
            loss_evaluator=loss,
            time_sample_offset=dt_slow,
            batch_size_per_device=1,
            num_batches=1,
            train_time_slice=None,
            eval_time_slice=None,
        )
    ])

    checkpoint_config = trainer.CheckpointConfig(
        save_interval_steps=2,
        keep_every_n_steps=10,
        model_config_str='{}',
        metadata={}
    )

    auto_restart = trainer.AutoRestartConfig()
    remat = trainer.RematConfig()
    process_obs = transforms.Identity()
    metrics_saver = InMemorySaver()
    rollout_trainer = trainer.RolloutTrainer(
        experiment_dir=self.test_dir,
        model=model,
        data_loader=data_loader,
        loss=loss,
        eval_metrics=eval_metrics,
        process_observations=process_obs,
        pretraining=None,
        train_schedule=train_schedule,
        eval_schedule=eval_schedule,
        optimization_config=opt_config,
        initial_checkpoint=None,
        checkpoint_config=checkpoint_config,
        auto_restart_config=auto_restart,
        remat_config=remat,
        ensemble_axis=cx.SizedAxis('ensemble', 2),
        online_metrics_saver=metrics_saver,
    )
    rollout_trainer.run_training()

    self.assertTrue(metrics_saver.metrics)
    self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'checkpoints')))

  def test_trainer_nested(self):
    model, all_data = self.create_model_and_data(multiscale=True)
    queries_specs = self._create_queries_specs(model)
    data_loader = self._create_data_loader(all_data)
    eval_metrics, loss = self._create_evaluators(
        dims_to_reduce=('ensemble', 'batch', 'k', 'j')
    )

    optimizer = optax.adam(1e-3)
    opt_config = trainer.OptimizationConfig(optimizer, ema_num_steps=10)

    # Create train stage with nested data specs
    train_stages = []
    # 2 slow steps = 5 fast steps (0, 1, 2, 3, 4) if dt_slow=4*dt_fast
    slow_steps = 2
    dt_slow = model.timestep * 4
    dt_fast = model.timestep

    fast_steps = (slow_steps - 1) * 4 + 1
    timedelta_slow = coordinates.TimeDelta(np.arange(slow_steps) * dt_slow)
    timedelta_fast = coordinates.TimeDelta(np.arange(fast_steps) * dt_fast)

    # inputs_spec must match model structure
    inputs_spec = {
        'slow': _construct_spec(model.inputs_spec['slow'], timedelta_slow),
        'fast': _construct_spec(model.inputs_spec['fast'], timedelta_fast),
    }
    # Dynamic inputs are empty for this model, so we don't need to set them.
    dyn_spec = {}

    train_stages.append(
        trainer.TrainStage(
            duration=5,
            inputs_spec=inputs_spec,
            dynamic_inputs_spec=dyn_spec,
            queries_spec=queries_specs,
            loss=loss,
            time_sample_offset=dt_slow,
            batch_size_per_device=1,
            shuffle_buffer_size=0,
            train_time_slice=None,
            eval_time_slice=None,
        )
    )
    train_schedule = trainer.TrainSchedule(stages=train_stages)

    eval_slow_steps = 4
    eval_fast_steps = (eval_slow_steps - 1) * 4 + 1
    eval_td_slow = coordinates.TimeDelta(np.arange(eval_slow_steps) * dt_slow)
    eval_td_fast = coordinates.TimeDelta(np.arange(eval_fast_steps) * dt_fast)
    eval_inputs_spec = {
        'slow': _construct_spec(model.inputs_spec['slow'], eval_td_slow),
        'fast': _construct_spec(model.inputs_spec['fast'], eval_td_fast),
    }
    eval_dyn_spec = {}
    eval_schedule = trainer.EvalSchedule(stages=[
        trainer.EvalSchema(
            cadence=2,
            inputs_spec=eval_inputs_spec,
            dynamic_inputs_spec=eval_dyn_spec,
            queries_spec=queries_specs,
            metrics_evaluator=eval_metrics,
            loss_evaluator=loss,
            time_sample_offset=dt_slow,
            batch_size_per_device=1,
            num_batches=1,
            train_time_slice=None,
            eval_time_slice=None,
        )
    ])

    checkpoint_config = trainer.CheckpointConfig(
        save_interval_steps=2,
        keep_every_n_steps=10,
        model_config_str='{}',
        metadata={}
    )

    auto_restart = trainer.AutoRestartConfig()
    remat = trainer.RematConfig()
    process_obs = transforms.Identity()
    metrics_saver = InMemorySaver()
    rollout_trainer = trainer.RolloutTrainer(
        experiment_dir=self.test_dir,
        model=model,
        data_loader=data_loader,
        loss=loss,
        eval_metrics=eval_metrics,
        process_observations=process_obs,
        pretraining=None,
        train_schedule=train_schedule,
        eval_schedule=eval_schedule,
        optimization_config=opt_config,
        initial_checkpoint=None,
        checkpoint_config=checkpoint_config,
        auto_restart_config=auto_restart,
        remat_config=remat,
        ensemble_axis=cx.SizedAxis('ensemble', 2),
        online_metrics_saver=metrics_saver,
    )
    rollout_trainer.run_training()

    self.assertTrue(metrics_saver.metrics)
    self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'checkpoints')))


if __name__ == '__main__':
  absltest.main()
