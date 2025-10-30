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

"""Tests that transforms produce outputs with expected structure."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import coordax as cx
from flax import nnx
import jax
import jax_datetime as jdt
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import random_processes
from neuralgcm.experimental.toy_model_examples import lorenz96
import numpy as np


def _add_dummy_timedelta(
    inputs: dict[str, dict[str, cx.Field]],
) -> dict[str, dict[str, cx.Field]]:
  """Adds a dummy timedelta dimension to all fields in a pytree."""
  t_del = coordinates.TimeDelta(np.zeros(1) * np.timedelta64(1, 'h'))
  add_td = lambda f: f.broadcast_like(
      cx.compose_coordinates(t_del, f.coordinate)
  )
  return jax.tree.map(add_td, inputs, is_leaf=cx.is_field)


def _make_x_l96_inputs(
    k: cx.Coordinate, t0: jdt.Datetime
) -> dict[str, dict[str, cx.Field]]:
  """Helper to generate 'slow' inputs for Lorenz96 models."""
  dt = coordinates.TimeDelta(np.zeros(1) * np.timedelta64(1, 'h'))
  x_init = abs(10 * np.sin(np.linspace(0, 13 * 2 * np.pi, k.sizes['k'])))
  inputs = {
      'slow': {
          'x': cx.wrap(x_init[None, :], dt, k),
          'time': cx.wrap(t0[None], dt),
      }
  }
  return inputs


def _make_y_l96_inputs(
    k: cx.Coordinate, j: cx.Coordinate, t0: jdt.Datetime
) -> dict[str, dict[str, cx.Field]]:
  """Helper to generate 'fast' inputs for Lorenz96 models."""
  dt = coordinates.TimeDelta(np.zeros(1) * np.timedelta64(1, 'h'))
  kj = cx.compose_coordinates(k, j)
  rng = np.random.RandomState(0)
  inputs = {
      'fast': {
          'y': cx.wrap(rng.uniform(-0.5, 0.5, dt.shape + kj.shape), dt, kj),
          'time': cx.wrap(t0[None], dt),
      }
  }
  return inputs


def _get_coordinates(
    observed: dict[str, dict[str, cx.Field]],
) -> dict[str, dict[str, cx.Coordinate]]:
  """Helper to get coordinates of observed fields."""
  return jax.tree.map(cx.get_coordinate, observed, is_leaf=cx.is_field)


class Lorenz96Test(parameterized.TestCase):
  """Tests that lorenz96 models methods produce expected outputs."""

  def setUp(self):
    super().setUp()
    self.k = cx.LabeledAxis('k', np.arange(36))
    self.j = cx.LabeledAxis('j', np.arange(8))
    self.t0 = jdt.Datetime.from_isoformat('2000-01-01')

  def test_lorenz96_with_two_scales(self):
    """Tests Lorenz96WithTwoScales model methods produce expected outputs."""
    k, j, t0 = self.k, self.j, self.t0
    kj = cx.compose_coordinates(k, j)
    l96_model = lorenz96.Lorenz96WithTwoScales(k_axis=k, j_axis=j)
    self.assertEqual(l96_model.x.coordinate, k)
    self.assertEqual(l96_model.y.coordinate, kj)

    inputs = {**_make_x_l96_inputs(k, t0), **_make_y_l96_inputs(k, j, t0)}
    l96_model.assimilate(inputs)
    self.assertEqual(l96_model.x.coordinate, k)
    self.assertEqual(l96_model.y.coordinate, kj)

    l96_model.advance()
    self.assertEqual(l96_model.x.coordinate, k)
    self.assertEqual(l96_model.y.coordinate, kj)

    queries = {'slow': {'x': k}, 'fast': {'y': kj}}
    observed = l96_model.observe(queries)
    chex.assert_trees_all_equal(_get_coordinates(observed), queries)
    chex.assert_tree_all_finite(observed)

  def test_lorenz96_with_param(self):
    """Tests Lorenz96 with parameterization produce expected outputs."""
    k, t0 = self.k, self.t0
    random_process = random_processes.NormalUncorrelated(
        mean=0.0, std=1.0, coords=k, rngs=nnx.Rngs(0)
    )
    param = lorenz96.StochasticLinearParameterization(
        random_process=random_process
    )
    l96_model = lorenz96.Lorenz96(k_axis=k, parameterizations=[param])
    self.assertEqual(l96_model.x.coordinate, k)

    inputs = _make_x_l96_inputs(k, t0)
    key = cx.wrap(jax.random.key(0))
    l96_model.initialize_random_processes(key)
    l96_model.assimilate(inputs)
    self.assertEqual(l96_model.x.coordinate, k)
    l96_model.advance()
    self.assertEqual(l96_model.x.coordinate, k)

    queries = {'slow': {'x': k}}
    observed = l96_model.observe(queries)
    chex.assert_trees_all_equal(_get_coordinates(observed), queries)

  def test_lorenz96_fast_mode(self):
    """Tests Lorenz96FastMode model methods produce expected outputs."""
    k, j, t0 = self.k, self.j, self.t0
    kj = cx.compose_coordinates(k, j)
    l96_model = lorenz96.Lorenz96FastMode(k_axis=k, j_axis=j)
    self.assertEqual(l96_model.y.coordinate, kj)

    inputs = _make_y_l96_inputs(k, j, t0)
    l96_model.assimilate(inputs)
    self.assertEqual(l96_model.y.coordinate, kj)
    l96_model.advance()
    self.assertEqual(l96_model.y.coordinate, kj)

    queries = {'fast': {'y': kj}}
    observed = l96_model.observe(queries)
    chex.assert_trees_all_equal(_get_coordinates(observed), queries)


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
