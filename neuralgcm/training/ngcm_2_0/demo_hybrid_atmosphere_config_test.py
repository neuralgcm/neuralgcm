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

import itertools
import os

from absl.testing import absltest
import coordax as cx
import fiddle as fdl
from fiddle.experimental import serialization
from flax import nnx
import jax
import jax_datetime as jdt
from neuralgcm.training.ngcm_2_0 import demo_hybrid_atmosphere_config
import numpy as np
from terrax.core import coordinates
from terrax.core import data_specs
from terrax.core import parallelism
from terrax.core import pytree_utils
from terrax.core import transforms
from terrax.core import typing
from terrax.training import data_loading


def _abstract_batch_from_spec(
    spec: dict[str, dict[str, cx.Coordinate]],
    batch: cx.SizedAxis,
    timedelta: coordinates.TimeDelta,
) -> dict[str, dict[str, cx.Field]]:
  """Constructs abstract shape-only fields matching a stage's data spec."""
  b_td = cx.coords.compose(batch, timedelta)

  def _with_batch_and_td(coord: cx.Coordinate) -> cx.Field:
    td_axis = coord.axes[coord.dims.index('timedelta')]
    coord = cx.coords.replace_axes(coord, td_axis, timedelta)
    return cx.shape_struct_field(cx.coords.compose(batch, coord))

  batch_data = jax.tree.map(_with_batch_and_td, spec, is_leaf=cx.is_coord)
  time_field = cx.field(
      jdt.to_datetime('2000-01-01T00')  # pyrefly: ignore[bad-argument-type]
  ).broadcast_like(b_td)
  time_struct = pytree_utils.shape_structure(time_field)
  flat, empty = pytree_utils.flatten_dict(batch_data)
  flat = {k: time_struct if k.endswith('&time') else v for k, v in flat.items()}
  return pytree_utils.unflatten_dict(flat, empty)


class DemoHybridAtmosphereConfigTest(absltest.TestCase):

  def test_config_contract_and_rollout_shapes(self):
    data_dir = self.create_tempdir().full_path
    trainer_cfg = demo_hybrid_atmosphere_config.get_config.as_buildable(
        pressure_era5_path=os.path.join(data_dir, 'pressure_era5.zarr'),
        surface_era5_path=os.path.join(data_dir, 'surface_era5.zarr'),
        precip_era5_path=os.path.join(data_dir, 'precip_era5.zarr'),
        ensemble_shards=1,
    )

    # 1. Standalone, model, and full trainer configs round-trip through JSON.
    standalone_model_cfg = (
        demo_hybrid_atmosphere_config.construct_model.as_buildable()
    )
    self.assertIsNotNone(
        serialization.load_json(serialization.dump_json(standalone_model_cfg))
    )
    unsharded_model_cfg = parallelism.update_mesh_properties(
        trainer_cfg.model, mesh=None
    )
    self.assertIsNotNone(
        serialization.load_json(serialization.dump_json(unsharded_model_cfg))
    )
    self.assertIsNotNone(
        serialization.load_json(serialization.dump_json(trainer_cfg))
    )

    # 2. Build core training components and validate data-spec contracts.
    model = fdl.build(trainer_cfg.model)
    train_schedule = fdl.build(trainer_cfg.train_schedule)
    eval_schedule = fdl.build(trainer_cfg.eval_schedule)
    process = fdl.build(trainer_cfg.process_observations)
    self.assertIsNotNone(fdl.build(trainer_cfg.optimization_config))
    self.assertEqual(sum(s.duration for s in train_schedule.stages), 25_000)

    stage = train_schedule.stages[0]
    data_specs.validate_inputs(stage.inputs_spec, model.inputs_spec)
    data_specs.validate_inputs(
        stage.dynamic_inputs_spec, model.dynamic_inputs_spec
    )

    # 3. Trace assimilate -> unroll -> process -> loss & evals via eval_shape.
    batch = cx.SizedAxis('batch', 1)
    ensemble = cx.SizedAxis('ensemble', 2)
    outer_steps = 2
    td = coordinates.TimeDelta(model.timestep * np.arange(outer_steps))
    in_data = _abstract_batch_from_spec(stage.inputs_spec, batch, td)
    dyn_data = _abstract_batch_from_spec(stage.dynamic_inputs_spec, batch, td)
    proc_def, proc_state = nnx.split(process)

    def _unroll(model, queries):
      proc = nnx.merge(proc_def, proc_state, copy=True)

      def _step(m):
        obs = m.observe(queries)
        m.advance()
        return proc(obs)

      in_axes = nnx.StateAxes({typing.SimulationVariable: nnx.Carry, ...: None})
      scanned = nnx.scan(_step, length=outer_steps, in_axes=in_axes, out_axes=0)
      return cx.tag(scanned(model), td)

    def _predict(model, in_data, dyn_data):
      b_model = model.to_vectorized({
          typing.SimulationVariable: batch,
          typing.DynamicInput: batch,
      })
      b_model.update_dynamic_inputs(dyn_data)
      eb_model = b_model.to_vectorized({typing.SimulationVariable: ensemble})
      eb_model.assimilate(in_data)
      in_t0 = transforms.NestedTransform(transforms.Isel({'timedelta': 0}))(
          in_data
      )
      queries = data_specs.construct_query(in_t0, stage.queries_spec)
      eb = cx.coords.compose(ensemble, batch)
      return eb_model.run_vectorized(_unroll, eb, queries)

    def _targets(in_data):
      proc = nnx.merge(proc_def, proc_state, copy=True)
      return proc(data_loading.select_targets(in_data, stage.queries_spec))

    predictions_struct = nnx.eval_shape(_predict, model, in_data, dyn_data)
    targets_struct = nnx.eval_shape(_targets, in_data)
    self.assertCountEqual(
        predictions_struct.keys(),
        [
            'era5:pressure',
            'modal_era5:pressure',
            'era5:surface',
            'modal_era5:surface',
            'era5:precip',
            'modal_era5:precip',
        ],
    )

    loss_struct = nnx.eval_shape(
        stage.loss.evaluate_total, predictions_struct, targets_struct
    )
    self.assertEqual(loss_struct.shape, ())

    evals = eval_schedule.stages[0].metrics_evaluator
    assert evals is not None
    evals_struct = nnx.eval_shape(
        evals.evaluate_metrics, predictions_struct, targets_struct
    )
    expected_eval_keys = {
        f'{metric}_P{level}_{lead}h'
        for metric, level, lead in itertools.product(
            ('rmse', 'crps'), (200, 500, 850), (12, 24, 48, 72, 168)
        )
    }
    self.assertSetEqual(
        set(evals_struct['era5:pressure'].keys()), expected_eval_keys
    )


if __name__ == '__main__':
  jax.config.update('jax_traceback_filtering', 'off')
  absltest.main()
