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

import os.path

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


class MockModel(api.Model):
  """Mock Model for testing."""

  def __init__(self, grid: cx.Coordinate, rngs: nnx.Rngs):
    super().__init__()
    self.process = random_processes.UniformUncorrelated(
        minval=0.0,
        maxval=1.0,
        coord=grid,
        rngs=rngs,
    )
    self.linear = nnx.Linear(in_features=1, out_features=1, rngs=rngs)

  def advance(self, *args, **kwargs):
    ...

  def assimilate(self, *args, **kwargs):
    ...

  def observe(self, *args, **kwargs):
    ...

  @property
  def timestep(self):
    ...


@auto_config.auto_config
def build_model():
  rngs = nnx.Rngs(0)
  grid = coordinates.LonLatGrid.T21()
  model = MockModel(grid, rngs)
  return model


class CheckpointingTest(parameterized.TestCase):

  def test_save_and_load_roundtrip(self):
    path = os.path.join(self.create_tempdir(), 'checkpoint')
    model_cfg = build_model.as_buildable()
    model = api.Model.from_fiddle_config(model_cfg)
    checkpointing.save_checkpoint(model, path)
    restored_model = checkpointing.load_model_checkpoint(path)
    restored_model_params = nnx.state(restored_model)
    expected_model_params = nnx.state(model)
    chex.assert_trees_all_equal(restored_model_params, expected_model_params)
    self.assertEqual(model_cfg, restored_model.fiddle_config)


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
