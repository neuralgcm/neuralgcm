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

from absl.testing import absltest
from absl.testing import parameterized
import coordax as cx
from fiddle.experimental import auto_config
from flax import nnx
import jax
import jax.numpy as jnp
import jax_datetime as jdt
from neuralgcm.experimental.core import api
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import dynamic_io
from neuralgcm.experimental.core import module_utils
from neuralgcm.experimental.core import nnx_compat
from neuralgcm.experimental.core import observation_operators
from neuralgcm.experimental.core import random_processes
from neuralgcm.experimental.core import typing
import numpy as np


@nnx_compat.dataclass
class MockModel(api.Model):
  """A mock model for testing purposes."""

  x: cx.Coordinate
  steady_increment: nnx.Param[jax.Array] = nnx.Param(0.0)
  modulation_factor: dynamic_io.DynamicInputSlice | None = None
  random_increment: random_processes.NormalUncorrelated | None = None
  data_key: str = 'prognostics'
  prognostic_var_key: str = 'population'
  dynamic_modulation_key: str = 'modulation'

  def __post_init__(self):
    self.prognostics = typing.Prognostic({
        'time': cx.wrap(jdt.Datetime.from_isoformat('2000-01-01')),
        self.prognostic_var_key: cx.wrap(np.zeros(self.x.shape), self.x),
    })

  @module_utils.ensure_unchanged_state_structure
  def assimilate(self, observations: typing.Observation) -> None:
    data = observations[self.data_key]
    self.prognostics.value = {
        'time': data['time'],
        self.prognostic_var_key: data[self.prognostic_var_key],
    }

  @module_utils.ensure_unchanged_state_structure
  def advance(self) -> None:
    time = self.prognostics.value['time']
    population = self.prognostics.value[self.prognostic_var_key]
    modulation = 1.0
    if self.modulation_factor is not None:
      modulation = self.modulation_factor(time)[self.dynamic_modulation_key]
    noise = 0.0
    if self.random_increment is not None:
      noise = self.random_increment.state_values(coords=self.x)
      self.random_increment.advance()
    next_population = population * modulation + self.steady_increment + noise
    self.prognostics.value = {
        'time': time + self.timestep,
        self.prognostic_var_key: next_population,
    }

  @module_utils.ensure_unchanged_state_structure
  def observe(self, query: typing.Query) -> typing.Observation:
    prognostic = self.prognostics.value
    operators = {
        self.data_key: observation_operators.DataObservationOperator(prognostic)
    }
    result = {}
    for k, q in query.items():
      result[k] = operators[k].observe(prognostic, q)
    return result

  @property
  def timestep(self):
    return np.timedelta64(1, 'h')


@auto_config.auto_config
def construct_mock_model() -> MockModel:
  """Constructs a mock model for testing purposes."""
  x = cx.SizedAxis('x', 3)
  modulation_factor = dynamic_io.DynamicInputSlice(
      keys_to_coords={'modulation': x},
      observation_key='modulation',
  )
  random_increment = random_processes.NormalUncorrelated(
      mean=0.0,
      std=1.0,
      coords=x,
      rngs=nnx.Rngs(0),
  )
  return MockModel(
      x=x,
      modulation_factor=modulation_factor,
      random_increment=random_increment,
  )


def run_assimilate(model: api.Model, inputs):
  model.assimilate(inputs)


def run_advance(model: api.Model):
  model.advance()


def run_observe(model: api.Model, query: typing.Query):
  return model.observe(query)


def initialize_random_processes(model: api.Model, rng: cx.Field):
  model.initialize_random_processes(rng)


def update_dynamic(model: api.Model, dynamic_inputs):
  model.update_dynamic_inputs(dynamic_inputs)


class ModelApiTest(parameterized.TestCase):
  """Tests Model API methods."""

  def setUp(self):
    super().setUp()
    self.model = construct_mock_model()
    self.t0 = jdt.Datetime.from_isoformat('2000-01-01')
    self.x = self.model.x
    self.timedelta = coordinates.TimeDelta(
        np.arange(3) * np.timedelta64(1, 'h')
    )
    as_jnp = lambda x: jax.tree.map(jnp.asarray, x)
    self.inputs = as_jnp({
        'prognostics': {
            'population': cx.wrap(np.ones(self.x.shape), self.x),
            'time': cx.wrap(self.t0),
        }
    })
    self.dynamic_inputs = as_jnp({
        'modulation': {
            'modulation': cx.wrap(
                np.ones(self.timedelta.shape + self.x.shape),
                self.timedelta,
                self.x,
            ),
            'time': cx.wrap(self.t0 + self.timedelta.deltas, self.timedelta),
        },
    })
    self.batch_axis = cx.SizedAxis('batch', 5)
    self.ensemble_axis = cx.SizedAxis('ensemble', 3)
    self.batched_inputs = jax.tree.map(
        lambda x: x.broadcast_like(
            cx.compose_coordinates(self.batch_axis, x.coordinate)
        ),
        self.inputs,
        is_leaf=cx.is_field,
    )
    self.rng = cx.wrap(jax.random.key(0), cx.Scalar())
    self.ensemble_rng = cx.wrap(
        jax.random.split(jax.random.key(0), self.ensemble_axis.size),
        self.ensemble_axis,
    )
    self.query = {'prognostics': {'population': self.x}}

  def test_api_methods(self):
    """Tests assimilate, advance, and observe methods."""
    with self.subTest('update_dynamic_inputs'):
      update_dynamic(self.model, self.dynamic_inputs)

    with self.subTest('initialize_random_processes'):
      initialize_random_processes(self.model, self.rng)

    with self.subTest('assimilate'):
      run_assimilate(self.model, self.inputs)
      np.testing.assert_allclose(
          self.model.prognostics.value['population'].data,
          self.inputs['prognostics']['population'].data,
      )
    with self.subTest('advance'):
      initial_population = self.model.prognostics.value['population'].data
      run_advance(self.model)
      final_population = self.model.prognostics.value['population'].data
      self.assertFalse(np.all(initial_population == final_population))

    with self.subTest('observe'):
      obs = run_observe(self.model, self.query)
      model_population = self.model.prognostics.value['population'].data
      np.testing.assert_allclose(
          obs['prognostics']['population'].data,
          model_population,
      )
      self.assertEqual(obs['prognostics']['population'].coordinate, self.x)

  def test_batch_vectorized_api_methods(self):
    """Tests vectorized assimilate, advance, and observe methods."""
    update_dynamic(self.model, self.dynamic_inputs)
    initialize_random_processes(self.model, self.rng)
    self.model.vectorize({typing.Prognostic: self.batch_axis})
    batch_model_axes = self.model.model_state_axes(self.batch_axis)
    self.model.untag_state(self.batch_axis)
    vmap_run_assimilate = nnx.vmap(
        run_assimilate, in_axes=(batch_model_axes, 0)
    )
    vmap_run_advance = nnx.vmap(run_advance, in_axes=(batch_model_axes,))
    vmap_run_observe = nnx.vmap(run_observe, in_axes=(batch_model_axes, None))
    vmap_run_assimilate(
        self.model, cx.untag(self.batched_inputs, self.batch_axis)
    )
    vmap_run_advance(self.model)
    obs = cx.tag(vmap_run_observe(self.model, self.query), self.batch_axis)
    self.assertEqual(
        obs['prognostics']['population'].coordinate,
        cx.compose_coordinates(self.batch_axis, self.x),
    )

  def test_ensemble_vectorized_api_methods(self):
    """Tests vectorized assimilate, advance, and observe methods."""
    update_dynamic(self.model, self.dynamic_inputs)
    self.model.vectorize({
        typing.Prognostic: self.ensemble_axis,
        typing.Randomness: self.ensemble_axis,
    })
    ensemble_model_axes = self.model.model_state_axes(self.ensemble_axis)
    vmap_initialize_random_processes = nnx.vmap(
        initialize_random_processes, in_axes=(ensemble_model_axes, 0)
    )
    vmap_run_assimilate = nnx.vmap(
        run_assimilate, in_axes=(ensemble_model_axes, None)
    )
    vmap_run_advance = nnx.vmap(run_advance, in_axes=(ensemble_model_axes,))
    vmap_run_observe = nnx.vmap(
        run_observe, in_axes=(ensemble_model_axes, None)
    )
    self.model.untag_state(self.ensemble_axis)
    vmap_initialize_random_processes(
        self.model, cx.untag(self.ensemble_rng, self.ensemble_axis)
    )
    vmap_run_assimilate(self.model, self.inputs)
    vmap_run_advance(self.model)
    obs = cx.tag(vmap_run_observe(self.model, self.query), self.ensemble_axis)
    self.assertEqual(
        obs['prognostics']['population'].coordinate,
        cx.compose_coordinates(self.ensemble_axis, self.x),
    )

  def test_raises_on_pytree_state_change(self):
    pass


class InferenceModelApiTest(parameterized.TestCase):
  """Tests InferenceModel API."""

  def setUp(self):
    super().setUp()
    self.model_config = construct_mock_model.as_buildable()
    self.inference_model = api.InferenceModel.from_model_api(
        api.Model.from_fiddle_config(self.model_config)
    )
    self.t0 = jdt.Datetime.from_isoformat('2000-01-01')
    self.x = self.inference_model._dummy_model().x
    self.timedelta = coordinates.TimeDelta(
        np.arange(3) * np.timedelta64(1, 'h')
    )
    self.inputs = {
        'prognostics': {
            'population': cx.wrap(np.ones(self.x.shape), self.x),
            'time': cx.wrap(self.t0),
        }
    }
    self.dynamic_inputs = {
        'modulation': {
            'modulation': cx.wrap(
                np.ones(self.timedelta.shape + self.x.shape),
                self.timedelta,
                self.x,
            ),
            'time': cx.wrap(self.t0 + self.timedelta.deltas, self.timedelta),
        },
    }
    self.query = {'prognostics': {'population': self.x}}
    self.rng = cx.wrap(jax.random.key(0))

  def test_api_methods(self):
    """Tests assimilate, advance, and observe methods."""
    state = self.inference_model.assimilate(
        self.inputs, self.dynamic_inputs, self.rng
    )
    state = self.inference_model.advance(state, self.dynamic_inputs)
    obs = self.inference_model.observe(state, self.query)
    self.assertEqual(obs['prognostics']['population'].coordinate, self.x)

  def test_model_is_immutable(self):
    """Tests that the InferenceModel itself is immutable."""
    state = self.inference_model.assimilate(
        self.inputs, self.dynamic_inputs, self.rng
    )
    new_state = self.inference_model.advance(state, self.dynamic_inputs)
    self.assertIsNot(state, new_state)
    pop_before = state.prognostics['prognostics']['population'].data
    pop_after = new_state.prognostics['prognostics']['population'].data
    self.assertFalse(np.all(pop_before == pop_after))
    # check that original state is unchanged
    self.assertTrue(
        np.all(
            state.prognostics['prognostics']['population'].data
            == self.inputs['prognostics']['population'].data
        )
    )
    second_new_state = self.inference_model.advance(state, self.dynamic_inputs)
    self.assertTrue(
        np.all(
            new_state.prognostics['prognostics']['population'].data
            == second_new_state.prognostics['prognostics']['population'].data
        )
    )

  def test_inference_functions(self):
    """Tests unroll_from_advance and forecast_steps."""
    state = self.inference_model.assimilate(
        self.inputs, self.dynamic_inputs, self.rng
    )
    _, trajectory = api.unroll_from_advance(
        self.inference_model,
        initial_state=state,
        timedelta=self.inference_model.timestep,
        steps=5,
        query=self.query,
        dynamic_inputs=self.dynamic_inputs,
    )
    self.assertEqual(
        trajectory['prognostics']['population'].shape,
        (5,) + self.x.shape,
    )


if __name__ == '__main__':
  jax.config.update('jax_traceback_filtering', 'off')
  absltest.main()
