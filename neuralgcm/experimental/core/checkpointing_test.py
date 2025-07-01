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

"""Tests for checkpointing routines."""

import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import chex
import coordax as cx
from fiddle.experimental import auto_config
from flax import nnx
import jax
from neuralgcm.experimental.core import api
from neuralgcm.experimental.core import checkpointing
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import random_processes
import orbax.checkpoint as ocp


class MockForecastSystem(api.ForecastSystem):
  """Mock ForecastSystem for testing."""

  def __init__(self, grid: cx.Coordinate, rngs: nnx.Rngs):
    super().__init__()
    self.process = random_processes.UniformUncorrelated(
        coords=grid,
        minval=0.0,
        maxval=1.0,
        rngs=rngs,
    )
    self.linear = nnx.Linear(in_features=1, out_features=1, rngs=rngs)

  def advance_prognostics(self, *args, **kwargs):
    ...

  def assimilate_prognostics(self, *args, **kwargs):
    ...

  def observe_from_prognostics(self, *args, **kwargs):
    ...


@auto_config.auto_config
def build_model():
  rngs = nnx.Rngs(0)
  grid = coordinates.LonLatGrid.T21()
  model = MockForecastSystem(grid, rngs)
  return model


@absltest.skipThisClass('Re-enable once fiddle is updated on github.')
class CheckpointingTest(parameterized.TestCase):

  def test_save_and_load_roundtrip(self):
    model_cfg = build_model.as_buildable()
    model = api.ForecastSystem.from_fiddle_config(model_cfg)
    model.update_metadata('fiddle_config', model_cfg)
    with tempfile.TemporaryDirectory() as path:
      path = ocp.test_utils.erase_and_create_empty(path)
      checkpointing.save_checkpoint(model, path / 'checkpoint')
      restored_model = checkpointing.load_model(path / 'checkpoint')
    restored_model_params = nnx.state(restored_model)
    expected_model_params = nnx.state(model)
    chex.assert_trees_all_equal(restored_model_params, expected_model_params)


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
