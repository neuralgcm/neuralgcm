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

import functools

from typing import Any
from absl.testing import absltest
from absl.testing import parameterized
import chex
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
from neuralgcm.experimental.core import observation_operators
from neuralgcm.experimental.core import random_processes
from neuralgcm.experimental.core import transforms as trxt
from neuralgcm.experimental.core import typing
import numpy as np


map_fields = functools.partial(jax.tree.map, is_leaf=cx.is_field)


@nnx.dataclass
class MockModel(api.Model):
  """A mock model for testing purposes."""

  x: cx.Coordinate
  steady_increment: nnx.Param[jax.Array] = nnx.data(
      default_factory=lambda: nnx.Param(0.0)
  )
  modulation_factor: dynamic_io.DynamicInputSlice | None = None
  random_increment: random_processes.NormalUncorrelated | None = None
  data_key: str = 'prognostics'
  prognostic_var_key: str = 'population'
  dynamic_modulation_key: str = 'modulation'
  operators: dict[str, Any] = nnx.data(default_factory=dict)

  def __post_init__(self):
    self.prognostics = typing.Prognostic({
        'time': cx.field(jdt.Datetime.from_isoformat('2000-01-01')),
        self.prognostic_var_key: cx.field(np.zeros(self.x.shape), self.x),
    })

  @module_utils.ensure_unchanged_state_structure
  def assimilate(self, observations: typing.Observation) -> None:
    data = observations[self.data_key]
    self.prognostics.set_value({
        'time': data['time'],
        self.prognostic_var_key: data[self.prognostic_var_key],
    })

  @module_utils.ensure_unchanged_state_structure
  def advance(self) -> None:
    time = self.prognostics.get_value()['time']
    population = self.prognostics.get_value()[self.prognostic_var_key]
    modulation = 1.0
    if self.modulation_factor is not None:
      modulation = self.modulation_factor(time)[self.dynamic_modulation_key]
    noise = 0.0
    if self.random_increment is not None:
      noise = self.random_increment.state_values(coord=self.x)
      self.random_increment.advance()
    next_population = population * modulation + self.steady_increment + noise
    self.prognostics.set_value({
        'time': time + self.timestep,
        self.prognostic_var_key: next_population,
    })

  @module_utils.ensure_unchanged_state_structure
  def observe(self, query: typing.Query) -> typing.Observation:
    prognostic = self.prognostics.get_value()
    result = {}
    operators = {
        self.data_key: observation_operators.DataObservationOperator(prognostic)
    } | self.operators
    for k, q in query.items():
      if k not in operators:
        raise ValueError(f'No operator for query key: {k}')
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
      coord=x,
      rngs=nnx.Rngs(0),
  )
  # Adding demo operators with different output types for testing purposes.
  # x_at_0: Selects values at x=0 coordinate.
  x_at_0 = trxt.Sequential([trxt.SelectKeys('population'), trxt.Isel({'x': 0})])
  # below_threshold: Selects values where population < thresholds.
  get_population_thresholds = trxt.Sequential([
      trxt.SelectKeys('thresholds'),
      trxt.RenameKeys({'thresholds': 'population'}),
  ])
  below_threshold = trxt.EntrywiseBinaryOp(
      op='less_than',
      operand_transform=get_population_thresholds,
      inputs_transform=trxt.SelectKeys('population'),
  )

  operators = {
      'x_at_0': observation_operators.TransformObservationOperator(x_at_0),
      'below_threshold': observation_operators.TransformObservationOperator(
          transform=below_threshold,
          requested_fields_from_query=('thresholds',),
      ),
  }
  return MockModel(
      x=x,
      modulation_factor=modulation_factor,
      random_increment=random_increment,
      operators=operators or {},
  )


class ModelApiTest(parameterized.TestCase):
  """Tests Model API methods."""

  def setUp(self):
    super().setUp()
    self.model = construct_mock_model()
    t0 = jdt.Datetime.from_isoformat('2000-01-01')
    self.x = self.model.x
    self.timedelta = coordinates.TimeDelta(
        np.arange(3) * np.timedelta64(1, 'h')
    )
    as_jnp = lambda x: jax.tree.map(jnp.asarray, x)
    self.inputs = as_jnp({
        'prognostics': {
            'population': cx.field(np.ones(self.x.shape), self.x),
            'time': cx.field(t0),
        }
    })
    self.dynamic_inputs = as_jnp({
        'modulation': {
            'modulation': cx.field(
                np.ones(self.timedelta.shape + self.x.shape),
                self.timedelta,
                self.x,
            ),
            'time': cx.field(t0 + self.timedelta.deltas, self.timedelta),
        },
    })
    self.batch_axis = cx.SizedAxis('batch', 5)
    self.ensemble_axis = cx.SizedAxis('ensemble', 3)
    replicate_batch_axis = lambda x: x.broadcast_like(
        cx.coords.compose(self.batch_axis, x.coordinate)
    )
    self.batched_inputs = map_fields(replicate_batch_axis, self.inputs)
    self.rng = cx.field(jax.random.key(0), cx.Scalar())
    self.ensemble_rng = cx.field(
        jax.random.split(jax.random.key(0), self.ensemble_axis.size),
        self.ensemble_axis,
    )
    self.query = {'prognostics': {'population': self.x}}

  def test_api_methods(self):
    """Tests assimilate, advance, and observe methods."""
    with self.subTest('update_dynamic_inputs'):
      self.model.update_dynamic_inputs(self.dynamic_inputs)

    with self.subTest('initialize_random_processes'):
      self.model.initialize_random_processes(self.rng)

    with self.subTest('assimilate'):
      self.model.assimilate(self.inputs)
      np.testing.assert_allclose(
          self.model.prognostics.get_value()['population'].data,
          self.inputs['prognostics']['population'].data,
      )
    with self.subTest('advance'):
      initial_population = self.model.prognostics.get_value()['population'].data
      self.model.advance()
      final_population = self.model.prognostics.get_value()['population'].data
      self.assertFalse(np.all(initial_population == final_population))

    with self.subTest('observe'):
      obs = self.model.observe(self.query)
      model_population = self.model.prognostics.get_value()['population'].data
      np.testing.assert_allclose(
          obs['prognostics']['population'].data,
          model_population,
      )
      self.assertEqual(obs['prognostics']['population'].coordinate, self.x)

  def test_batch_vectorized_api_methods(self):
    """Tests vectorized assimilate, advance, and observe methods."""
    self.model.update_dynamic_inputs(self.dynamic_inputs)
    self.model.initialize_random_processes(self.rng)
    v_model = self.model.to_vectorized({typing.Prognostic: self.batch_axis})
    v_model.assimilate(self.batched_inputs)
    v_model.advance()
    obs = v_model.observe(self.query)
    self.assertEqual(
        obs['prognostics']['population'].coordinate,
        cx.coords.compose(self.batch_axis, self.x),
    )

  def test_ensemble_vectorized_api_methods(self):
    """Tests vectorized assimilate, advance, and observe methods."""
    self.model.update_dynamic_inputs(self.dynamic_inputs)
    v_model = self.model.to_vectorized({
        typing.Prognostic: self.ensemble_axis,
        typing.Randomness: self.ensemble_axis,
    })
    ensemble_inputs = jax.tree.map(
        lambda x: jnp.stack([x] * self.ensemble_axis.size), self.inputs
    )
    # could also call assimilate prior to vectorization for the same effect.
    ensemble_inputs = cx.tag(ensemble_inputs, self.ensemble_axis)
    v_model.initialize_random_processes(self.ensemble_rng)
    v_model.assimilate(ensemble_inputs)
    v_model.advance()
    obs = v_model.observe(self.query)
    self.assertEqual(
        obs['prognostics']['population'].coordinate,
        cx.coords.compose(self.ensemble_axis, self.x),
    )

  def test_raises_on_pytree_state_change(self):
    """Tests that ensure_unchanged_state_structure raise."""
    with self.subTest('assimilate_with_batched_input'):
      model = construct_mock_model()
      with self.assertRaisesRegex(
          ValueError,
          'change in the pytree structure detected while running "assimilate"',
      ):
        model.assimilate(self.batched_inputs)

    with self.subTest('advance_with_partial_vectorization'):
      model = construct_mock_model()
      model.update_dynamic_inputs(self.dynamic_inputs)
      model.initialize_random_processes(self.rng)
      module_utils.vectorize_module(
          model.random_increment, {typing.Randomness: self.batch_axis}
      )
      with self.assertRaisesRegex(
          ValueError,
          'change in the pytree structure detected while running "advance"',
      ):
        model.advance()


class InferenceModelApiTest(parameterized.TestCase):
  """Tests InferenceModel API."""

  def setUp(self):
    super().setUp()
    self.model_config = construct_mock_model.as_buildable()
    self.inference_model = api.InferenceModel.from_model_api(
        api.Model.from_fiddle_config(self.model_config)
    )
    t0 = jdt.Datetime.from_isoformat('2000-01-01')
    self.x = cx.SizedAxis('x', 3)
    self.timedelta = coordinates.TimeDelta(
        np.arange(3) * np.timedelta64(1, 'h')
    )
    self.inputs = {
        'prognostics': {
            'population': cx.field(np.ones(self.x.shape), self.x),
            'time': cx.field(t0),
        }
    }
    self.dynamic_inputs = {
        'modulation': {
            'modulation': cx.field(
                np.ones(self.timedelta.shape + self.x.shape),
                self.timedelta,
                self.x,
            ),
            'time': cx.field(t0 + self.timedelta.deltas, self.timedelta),
        },
    }
    self.query = {'prognostics': {'population': self.x}}
    self.rng = cx.field(jax.random.key(0))

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
    np.testing.assert_allclose(
        state.prognostics['prognostics']['population'].data,
        self.inputs['prognostics']['population'].data,
    )
    second_new_state = self.inference_model.advance(state, self.dynamic_inputs)
    np.testing.assert_allclose(
        new_state.prognostics['prognostics']['population'].data,
        second_new_state.prognostics['prognostics']['population'].data,
    )

  def test_raises_on_insufficient_state(self):
    """Tests that the API raises on insufficient state."""
    with self.assertRaisesRegex(
        TypeError,
        'JAX raised TypeError.* This often indicates uninitialized state',
    ):
      self.inference_model.assimilate(self.inputs, self.dynamic_inputs)


class InferenceHelpersTest(parameterized.TestCase):
  """Tests inference helper functions."""

  def setUp(self):
    super().setUp()
    t0 = jdt.Datetime.from_isoformat('2000-01-01')
    x = cx.SizedAxis('x', 3)
    self.x = x
    self.model_config = construct_mock_model.as_buildable()
    self.model = api.InferenceModel.from_model_api(
        api.Model.from_fiddle_config(self.model_config)
    )
    self.inputs = {
        'prognostics': {
            'population': cx.field(np.ones(x.shape), x),
            'time': cx.field(t0),
        }
    }
    td = coordinates.TimeDelta(np.arange(3) * np.timedelta64(1, 'h'))
    self.dynamic_inputs = {
        'modulation': {
            'modulation': cx.field(np.ones(td.shape + x.shape), td, x),
            'time': cx.field(t0 + td.deltas, td),
        },
    }
    self.rng = cx.field(jax.random.key(0))

  @parameterized.parameters(
      (False, False),
      (True, False),
      (False, True),
      (True, True),
  )
  def test_unroll_functions(self, prepend_init, trim_last):
    """Tests unroll_from_advance and forecast_steps."""
    expected_steps = 5
    if prepend_init:
      expected_steps += 1
    if trim_last:
      expected_steps -= 1

    start_idx = 0 if prepend_init else 1
    expected_td = coordinates.TimeDelta(
        np.arange(start_idx, start_idx + expected_steps) * self.model.timestep
    )
    expected_coord = cx.coords.compose(expected_td, self.x)

    with self.subTest('unroll_from_advance'):
      state = self.model.assimilate(self.inputs, self.dynamic_inputs, self.rng)
      _, trajectory = api.unroll_from_advance(
          self.model,
          initial_state=state,
          timedelta=self.model.timestep,
          steps=5,
          queries={'prognostics': {'population': self.x}},
          dynamic_inputs=self.dynamic_inputs,
          prepend_init=prepend_init,
          trim_last=trim_last,
      )
      self.assertEqual(
          trajectory['prognostics']['population'].coordinate, expected_coord
      )

    with self.subTest('forecast_steps'):
      _, trajectory = api.forecast_steps(
          self.model,
          inputs=self.inputs,
          timedelta=self.model.timestep,
          steps=5,
          queries={'prognostics': {'population': self.x}},
          dynamic_inputs=self.dynamic_inputs,
          rng=self.rng,
          prepend_init=prepend_init,
          trim_last=trim_last,
      )
      self.assertEqual(
          trajectory['prognostics']['population'].coordinate, expected_coord
      )

  def test_unroll_from_advance_with_nested_timedeltas(self):
    """Tests unroll_from_advance with nested timedeltas."""
    state = self.model.assimilate(self.inputs, self.dynamic_inputs, self.rng)
    inner_dt = self.model.timestep
    inner_query = {'prognostics': {'population': self.x}}
    outer_dt = self.model.timestep * 2
    outer_query = {'x_at_0': {'population': cx.Scalar()}}
    timedeltas = (inner_dt, outer_dt)
    queries = (inner_query, outer_query)

    _, trajectory = api.unroll_from_advance(
        self.model,
        initial_state=state,
        timedelta=timedeltas,
        steps=5,
        queries=queries,
        dynamic_inputs=self.dynamic_inputs,
    )
    # Build expected coordinates.
    prog_td = coordinates.TimeDelta(np.arange(1, 11) * inner_dt)
    x_at_0_td = coordinates.TimeDelta(np.arange(1, 6) * outer_dt)
    expected_coords = {
        'prognostics': {'population': cx.coords.compose(prog_td, self.x)},
        'x_at_0': {'population': x_at_0_td},
    }

    actual_coords = map_fields(cx.get_coordinate, trajectory)
    self.assertEqual(actual_coords, expected_coords)

    with self.subTest('raises_on_unsorted_timedeltas'):
      wrong_order_timedeltas = (outer_dt, inner_dt)
      with self.assertRaises(ValueError):
        api.unroll_from_advance(
            self.model,
            initial_state=state,
            timedelta=wrong_order_timedeltas,
            steps=5,
            queries=(outer_query, inner_query),
            dynamic_inputs=self.dynamic_inputs,
        )

  def test_unroll_from_advance_with_field_query(self):
    """Tests unroll_from_advance with queries containing fields."""
    state = self.model.assimilate(self.inputs, self.dynamic_inputs, self.rng)
    inner_dt = self.model.timestep * 2
    outer_dt = self.model.timestep * 4
    inner_query = {'x_at_0': {'population': cx.Scalar()}}
    outer_query = {
        'below_threshold': {
            'population': self.x,
            'thresholds': cx.field(np.ones(self.x.shape) * 0.5, self.x),
        }
    }

    timedeltas = (self.model.timestep, inner_dt, outer_dt)
    queries = ({}, inner_query, outer_query)

    _, trajectory = api.unroll_from_advance(
        self.model,
        initial_state=state,
        timedelta=timedeltas,
        steps=5,
        queries=queries,
        dynamic_inputs=self.dynamic_inputs,
    )

    # Build expected coordinates.
    td1 = coordinates.TimeDelta(np.arange(1, 11) * inner_dt)
    td2 = coordinates.TimeDelta(np.arange(1, 6) * outer_dt)

    expected_coords = {
        'x_at_0': {'population': td1},
        'below_threshold': {
            'population': cx.coords.compose(td2, self.x),
            'thresholds': cx.coords.compose(td2, self.x),
        },
    }

    actual_coords = map_fields(cx.get_coordinate, trajectory)
    self.assertEqual(actual_coords, expected_coords)

  def test_unroll_from_advance_with_time_varying_field_query(self):
    """Tests unroll_from_advance with queries containing time-varying fields."""
    state = self.model.assimilate(self.inputs, self.dynamic_inputs, self.rng)

    # Thresholds vary in time.
    td_outer = coordinates.TimeDelta(np.arange(1, 6) * self.model.timestep * 2)
    thresholds = cx.field(
        np.arange(td_outer.shape[0])[:, None] + np.zeros((1,) + self.x.shape),
        *(td_outer, self.x),
    )

    outer_query = {
        'below_threshold': {'population': self.x, 'thresholds': thresholds}
    }

    timedelta = (self.model.timestep, self.model.timestep * 2)
    queries = ({}, outer_query)

    _, trajectory = api.unroll_from_advance(
        self.model,
        initial_state=state,
        timedelta=timedelta,
        steps=5,
        queries=queries,
        dynamic_inputs=self.dynamic_inputs,
    )

    expected_coords = {
        'below_threshold': {
            'population': cx.coords.compose(td_outer, self.x),
            'thresholds': cx.coords.compose(td_outer, self.x),
        },
    }
    actual_coords = map_fields(cx.get_coordinate, trajectory)
    self.assertEqual(actual_coords, expected_coords)

  def test_unroll_for_template_returns_expected_coordinates(self):
    """Tests unroll_for_template returns expected coordinates."""
    state = self.model.assimilate(self.inputs, self.dynamic_inputs, self.rng)
    t2 = coordinates.TimeDelta(np.arange(1, 3) * np.timedelta64(2, 'h'))

    _, predictions = api.unroll_for_template(
        self.model,
        initial_state=state,
        template={'prognostics': {'population': cx.coords.compose(t2, self.x)}},
        dynamic_inputs=self.dynamic_inputs,
    )
    self.assertEqual(
        predictions['prognostics']['population'].coordinate,
        cx.coords.compose(t2, self.x),
    )

  def test_unroll_for_template_with_time_varying_fields(self):
    """Tests unroll_for_template with template containing time-varying fields."""
    state = self.model.assimilate(self.inputs, self.dynamic_inputs, self.rng)

    td_outer = coordinates.TimeDelta(np.arange(1, 3) * self.model.timestep * 2)
    thresholds = cx.field(
        np.ones(td_outer.shape + self.x.shape) * 0.5, td_outer, self.x
    )

    template = {
        'below_threshold': {
            'population': cx.coords.compose(td_outer, self.x),
            'thresholds': thresholds,
        }
    }

    _, predictions = api.unroll_for_template(
        self.model,
        initial_state=state,
        template=template,
        dynamic_inputs=self.dynamic_inputs,
    )

    expected_coords = {
        'below_threshold': {
            'population': cx.coords.compose(td_outer, self.x),
            'thresholds': cx.coords.compose(td_outer, self.x),
        },
    }
    actual_coords = map_fields(cx.get_coordinate, predictions)
    self.assertEqual(actual_coords, expected_coords)

  def test_unroll_for_template_reconstruction(self):
    """Tests that unroll_for_template can reconstruct trajectory."""
    state = self.model.assimilate(self.inputs, self.dynamic_inputs, self.rng)

    # Generate a trajectory with unroll_from_advance
    _, expected_trajectory = api.unroll_from_advance(
        self.model,
        initial_state=state,
        timedelta=self.model.timestep,
        steps=5,
        queries={'prognostics': {'population': self.x}},
        dynamic_inputs=self.dynamic_inputs,
    )

    template = map_fields(cx.get_coordinate, expected_trajectory)

    _, predictions = api.unroll_for_template(
        self.model,
        initial_state=state,
        template=template,
        dynamic_inputs=self.dynamic_inputs,
    )

    chex.assert_trees_all_close(predictions, expected_trajectory)

  def test_unroll_for_template_reconstruction_nested(self):
    """Tests that unroll_for_template can reconstruct nested trajectory."""
    state = self.model.assimilate(self.inputs, self.dynamic_inputs, self.rng)

    inner_dt = self.model.timestep
    outer_dt = self.model.timestep * 2
    queries = (
        {'prognostics': {'population': self.x}},
        {'x_at_0': {'population': cx.Scalar()}},
    )
    timedeltas = (inner_dt, outer_dt)

    _, expected_unroll = api.unroll_from_advance(
        self.model,
        initial_state=state,
        timedelta=timedeltas,
        steps=5,
        queries=queries,
        dynamic_inputs=self.dynamic_inputs,
    )

    template = map_fields(cx.get_coordinate, expected_unroll)
    _, actual_unroll = api.unroll_for_template(
        self.model,
        initial_state=state,
        template=template,
        dynamic_inputs=self.dynamic_inputs,
    )
    chex.assert_trees_all_close(actual_unroll, expected_unroll, atol=1e-5)


if __name__ == '__main__':
  jax.config.update('jax_traceback_filtering', 'off')
  absltest.main()
