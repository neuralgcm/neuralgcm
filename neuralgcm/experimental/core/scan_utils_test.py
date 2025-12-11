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

import dataclasses
import functools

from absl.testing import absltest
from absl.testing import parameterized
import chex
import coordax as cx
from flax import nnx
import jax
import jax.numpy as jnp
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import nnx_compat
from neuralgcm.experimental.core import pytree_utils
from neuralgcm.experimental.core import scan_utils
from neuralgcm.experimental.core import typing
import numpy as np


@nnx_compat.dataclass
class MockStepper(nnx.Module):
  """Mock stepper module."""

  p: typing.Prognostic = dataclasses.field(
      default_factory=lambda: typing.Prognostic(
          {'a': jnp.zeros(()), 'b': jnp.zeros(()), 'x': jnp.zeros(())}
      )
  )

  def advance(self):
    self.p.set_value({k: v + 1 for k, v in self.p.get_value().items()})

  def observe(self, queries):
    result = {}
    for q_key, query in queries.items():
      if q_key not in result:
        result[q_key] = {}
      result[q_key] |= {k: self.p.get_value().get(k) for k in query}
    return result


def _merge_nested(a, b):
  a_flat, _ = pytree_utils.flatten_dict(a)
  b_flat, _ = pytree_utils.flatten_dict(b)
  return pytree_utils.unflatten_dict(a_flat | b_flat)


def recursive_nested_observe_scan(
    fn, scan_steps, scan_queries, in_axes, out_axes
):
  """Applies scan recursively, calling observe with queries at each level."""
  if not scan_steps:
    return fn

  steps, current_level_queries = scan_steps[0], scan_queries[0]

  def _inner_fn(model):
    inner_obs = fn(model)
    current_obs = model.observe(current_level_queries)
    return _merge_nested(inner_obs, current_obs)

  in_fn = nnx.scan(_inner_fn, length=steps, in_axes=in_axes, out_axes=out_axes)
  return recursive_nested_observe_scan(
      in_fn, scan_steps[1:], scan_queries[1:], in_axes, out_axes
  )


def range_from_one(n: int) -> np.ndarray:
  return np.arange(1, n + 1)


class ScanSpecsUtilsTest(parameterized.TestCase):
  """Tests nested_scan_specs and nested_scan_steps utilities."""

  @parameterized.parameters(
      np.timedelta64(1, 'h'),
      np.timedelta64(3, 'h'),
  )
  def test_nested_scan_specs_unique_timedelta_step(self, dt: np.timedelta64):
    x = cx.SizedAxis('x', 4)
    grid = coordinates.LonLatGrid.TL31()
    data_dt = np.timedelta64(6, 'h')
    timedelta = coordinates.TimeDelta(range_from_one(4) * data_dt)
    with_td = lambda c: cx.compose_coordinates(timedelta, c)
    inputs_spec = {'data': {'u': with_td(x), 'v': with_td(grid)}}
    actual_scan_specs = scan_utils.nested_scan_specs(inputs_spec, dt)
    # empty dict is the `dt` scan step.
    expected_scan_specs = (
        {},
        {'data': {'u': with_td(x), 'v': with_td(grid)}},
    )
    self.assertEqual(actual_scan_specs, expected_scan_specs)
    actual_scan_steps = scan_utils.nested_scan_steps(inputs_spec, dt)
    expected_scan_steps = (int(data_dt // dt), 4)
    self.assertEqual(actual_scan_steps, expected_scan_steps)
    with self.subTest('dict_input'):
      actual_scan_specs = scan_utils.nested_scan_specs(inputs_spec['data'], dt)
      expected_scan_specs = ({}, {'u': with_td(x), 'v': with_td(grid)})
      self.assertEqual(actual_scan_specs, expected_scan_specs)

  def test_nested_scan_specs_timedelta_equal_dt(self):
    x = cx.SizedAxis('x', 4)
    data_dt = np.timedelta64(6, 'h')
    timedelta = coordinates.TimeDelta(range_from_one(6) * data_dt)
    with_td = lambda c: cx.compose_coordinates(timedelta, c)
    inputs_spec = {'data': {'u': with_td(x)}}
    actual_scan_specs = scan_utils.nested_scan_specs(inputs_spec)
    expected_scan_specs = ({'data': {'u': with_td(x)}},)
    self.assertEqual(actual_scan_specs, expected_scan_specs)
    actual_scan_steps = scan_utils.nested_scan_steps(inputs_spec)
    expected_scan_steps = (6,)
    self.assertEqual(actual_scan_steps, expected_scan_steps)
    with self.subTest('dict_input'):
      actual_scan_specs = scan_utils.nested_scan_specs(inputs_spec['data'])
      expected_scan_specs = ({'u': with_td(x)},)
      self.assertEqual(actual_scan_specs, expected_scan_specs)
      actual_scan_steps = scan_utils.nested_scan_steps(inputs_spec['data'])
      self.assertEqual(actual_scan_steps, expected_scan_steps)

  def test_nested_scan_specs_with_multiple_timedeltas(self):
    dt = np.timedelta64(1, 'h')
    x, grid = cx.SizedAxis('x', 4), coordinates.LonLatGrid.TL31()
    data_dt = np.timedelta64(6, 'h')
    long_factor = 4
    data_long_dt = long_factor * data_dt
    s_steps = 4 * 5  # in data_dt == 5 days.
    l_steps = s_steps // long_factor
    short_td = coordinates.TimeDelta(range_from_one(s_steps) * data_dt)
    long_td = coordinates.TimeDelta(range_from_one(l_steps) * data_long_dt)
    with_s_td = lambda c: cx.compose_coordinates(short_td, c)
    with_l_td = lambda c: cx.compose_coordinates(long_td, c)
    inputs_spec = {'data': {'u': with_s_td(x), 'sst': with_l_td(grid)}}
    actual_scan_specs = scan_utils.nested_scan_specs(inputs_spec, dt)
    expected_scan_specs = (
        {},  # 1hr.
        {'data': {'u': with_s_td(x)}},  # 6hr.
        {'data': {'sst': with_l_td(grid)}},  # 24hr.
    )
    self.assertEqual(actual_scan_specs, expected_scan_specs)
    actual_scan_steps = scan_utils.nested_scan_steps(inputs_spec, dt)
    expected_scan_steps = (int(data_dt // dt), int(data_long_dt // data_dt), 5)
    self.assertEqual(actual_scan_steps, expected_scan_steps)
    with self.subTest('dict_input'):
      actual_scan_specs = scan_utils.nested_scan_specs(inputs_spec['data'], dt)
      expected_scan_specs = (
          {},  # 1hr.
          {'u': with_s_td(x)},  # 6hr.
          {'sst': with_l_td(grid)},  # 24hr.
      )
      self.assertEqual(actual_scan_specs, expected_scan_specs)
      actual_scan_steps = scan_utils.nested_scan_steps(inputs_spec['data'], dt)
      self.assertEqual(actual_scan_steps, expected_scan_steps)

  def test_nested_scan_specs_raises_on_incompatible_dt(self):
    x = cx.SizedAxis('x', 4)
    dt = np.timedelta64(2, 'h')
    data_dt = np.timedelta64(3, 'h')
    timedelta = coordinates.TimeDelta(range_from_one(4) * data_dt)
    with_td = lambda c: cx.compose_coordinates(timedelta, c)
    inputs_spec = {'data': {'u': with_td(x)}}
    with self.assertRaises(ValueError):
      scan_utils.nested_scan_specs(inputs_spec, dt)
    with self.subTest('dict_input'):
      with self.assertRaises(ValueError):
        scan_utils.nested_scan_specs(inputs_spec['data'], dt)

  def test_nested_scan_specs_raises_on_incongruent_timedeltas(self):
    dt = np.timedelta64(3, 'h')
    x, grid = cx.SizedAxis('x', 4), coordinates.LonLatGrid.TL31()
    data_dt = np.timedelta64(6, 'h')
    data_long_dt = np.timedelta64(15, 'h')  # not divisible by data_dt
    small_timedelta = coordinates.TimeDelta(range_from_one(5) * data_dt)
    large_timedelta = coordinates.TimeDelta(range_from_one(2) * data_long_dt)
    with_s_td = lambda c: cx.compose_coordinates(small_timedelta, c)
    with_l_td = lambda c: cx.compose_coordinates(large_timedelta, c)
    inputs_spec = {'data': {'u': with_s_td(x), 'sst': with_l_td(grid)}}
    with self.assertRaises(ValueError):
      scan_utils.nested_scan_specs(inputs_spec, dt)
    with self.subTest('dict_input'):
      with self.assertRaises(ValueError):
        scan_utils.nested_scan_specs(inputs_spec['data'], dt)

  def test_nested_scan_specs_raises_on_unequal_timedeltas(self):
    dt = np.timedelta64(3, 'h')
    x, grid = cx.SizedAxis('x', 4), coordinates.LonLatGrid.TL31()
    data_dt = np.timedelta64(6, 'h')
    data_long_dt = np.timedelta64(12, 'h')
    small_timedelta = coordinates.TimeDelta(range_from_one(4) * data_dt)
    large_timedelta = coordinates.TimeDelta(range_from_one(3) * data_long_dt)
    with_s_td = lambda c: cx.compose_coordinates(small_timedelta, c)
    with_l_td = lambda c: cx.compose_coordinates(large_timedelta, c)
    inputs_spec = {'data': {'u': with_s_td(x), 'sst': with_l_td(grid)}}
    with self.assertRaises(ValueError):
      scan_utils.nested_scan_specs(inputs_spec, dt)
    with self.assertRaises(ValueError):
      scan_utils.nested_scan_steps(inputs_spec, dt)
    with self.subTest('dict_input'):
      with self.assertRaises(ValueError):
        scan_utils.nested_scan_specs(inputs_spec['data'], dt)
      with self.assertRaises(ValueError):
        scan_utils.nested_scan_steps(inputs_spec['data'], dt)


class NestDataForScansUtilsTest(parameterized.TestCase):
  """Tests that nest_data_for_scans correctly formats data for nested scans."""

  def test_nest_data_single_scan(self):
    dt = np.timedelta64(1, 'h')
    x = cx.SizedAxis('x', 4)
    data_dt = np.timedelta64(6, 'h')
    timedelta = coordinates.TimeDelta(range_from_one(4) * data_dt)
    with_td = lambda c: cx.compose_coordinates(timedelta, c)
    ones_like = lambda c: cx.wrap(np.ones(c.shape), c)
    inputs = {'data': {'u': ones_like(with_td(x))}}
    with self.subTest('smaller_dt'):
      nested_data = scan_utils.nest_data_for_scans(inputs, dt)
      expected_nested_data = (
          {},
          cx.untag(inputs, timedelta),
      )
      chex.assert_trees_all_equal(nested_data, expected_nested_data)

    with self.subTest('dt_equals_data_dt'):
      nested_data = scan_utils.nest_data_for_scans(inputs)
      expected_nested_data = (cx.untag(inputs, timedelta),)
      chex.assert_trees_all_equal(nested_data, expected_nested_data)
    with self.subTest('dict_input'):
      with self.subTest('smaller_dt'):
        nested_data = scan_utils.nest_data_for_scans(inputs['data'], dt)
        expected_nested_data = ({}, cx.untag(inputs['data'], timedelta))
        chex.assert_trees_all_equal(nested_data, expected_nested_data)

  def test_nest_data_for_scans_with_multiple_timedeltas(self):
    dt = np.timedelta64(1, 'h')
    x = cx.SizedAxis('x', 4)
    a_dt = np.timedelta64(6, 'h')
    b_dt = np.timedelta64(12, 'h')
    c_dt = np.timedelta64(24, 'h')
    full_td = np.timedelta64(24 * 5, 'h')
    a_td = coordinates.TimeDelta(range_from_one(full_td / a_dt) * a_dt)
    b_td = coordinates.TimeDelta(range_from_one(full_td / b_dt) * b_dt)
    c_td = coordinates.TimeDelta(range_from_one(full_td / c_dt) * c_dt)
    ones_like = lambda *cs: cx.wrap(
        np.ones(sum([c.shape for c in cs], start=())), *cs
    )
    inputs = {
        'data': {
            'a': ones_like(a_td, x),
            'b': ones_like(b_td, x),
            'c': ones_like(c_td, x),
        }
    }

    actual_nested_data = scan_utils.nest_data_for_scans(inputs, dt)
    dummy_a_nest = cx.DummyAxis(None, int(b_dt / a_dt))
    dummy_b_nest = cx.DummyAxis(None, int(c_dt / b_dt))
    dummy_c_nest = cx.DummyAxis(None, int(full_td / c_dt))
    expected_nested_data = (
        {},  # 1hr.
        {'data': {'a': ones_like(dummy_c_nest, dummy_b_nest, dummy_a_nest, x)}},
        {'data': {'b': ones_like(dummy_c_nest, dummy_b_nest, x)}},
        {'data': {'c': ones_like(dummy_c_nest, x)}},  # 24hr.
    )
    chex.assert_trees_all_equal(actual_nested_data, expected_nested_data)
    with self.subTest('dict_input'):
      actual = scan_utils.nest_data_for_scans(inputs['data'], dt)
      expected = tuple(x.get('data', {}) for x in expected_nested_data)
      chex.assert_trees_all_equal(actual, expected)


class NestedScanExampleTest(parameterized.TestCase):
  """Tests the use of utilities in setup of end-to-end nested scan."""

  def test_nested_scan_end_to_end(self):
    dt = np.timedelta64(1, 'h')
    data_a_dt = np.timedelta64(6, 'h')  # 6 inner dt steps.
    data_b_dt = np.timedelta64(24, 'h')  # 4 a_dt steps.
    data_x_dt = np.timedelta64(48, 'h')  # 2 b_dt steps.

    total_time = np.timedelta64(240, 'h')  # 5 x_dt steps.

    def make_time_axis(step_dt):
      steps = int(total_time // step_dt)
      return coordinates.TimeDelta(range_from_one(steps) * step_dt)

    a_td = make_time_axis(data_a_dt)
    b_td = make_time_axis(data_b_dt)
    x_td = make_time_axis(data_x_dt)
    inputs_spec = {
        'snapshots': {
            'a': cx.compose_coordinates(a_td),
            'b': cx.compose_coordinates(b_td),
        },
        'averaged': {'x': cx.compose_coordinates(x_td)},
    }

    scan_steps = scan_utils.nested_scan_steps(inputs_spec, dt)
    self.assertEqual(scan_steps, (6, 4, 2, 5))
    nested_scan_specs = scan_utils.nested_scan_specs(inputs_spec, dt)
    is_coord = lambda c: isinstance(c, cx.Coordinate)
    remove_timedelta = lambda c: cx.compose_coordinates(
        *[ax for ax in c.axes if not isinstance(ax, coordinates.TimeDelta)]
    )
    # In training experiment queries do not contain timedeltas to begin with.
    nested_queries_specs = jax.tree.map(
        remove_timedelta, nested_scan_specs, is_leaf=is_coord
    )
    model = MockStepper()
    model_state_scan_axes = nnx.StateAxes(
        {typing.SimulationVariable: nnx.Carry, ...: None}
    )

    def model_step(model):
      model.advance()
      return {}  # innermost step that returns no observations.

    nested_scan_obs_fn = recursive_nested_observe_scan(
        model_step,
        scan_steps,
        nested_queries_specs,
        in_axes=model_state_scan_axes,
        out_axes=0,
    )
    outputs = nested_scan_obs_fn(model)

    # Outputs contain `arange` with time granularity specified by inputs_spec.
    expected_fields = {
        'snapshots': {
            'a': cx.wrap(range_from_one(240 // 6) * 6, a_td),
            'b': cx.wrap(range_from_one(240 // 24) * 24, b_td),
        },
        'averaged': {'x': cx.wrap(range_from_one(240 // 48) * 48, x_td)},
    }
    expected_flat_data = jax.tree.map(
        lambda x: x.data, expected_fields, is_leaf=cx.is_field
    )
    outputs_flat_data = jax.tree.map(lambda x: x.ravel(), outputs)
    chex.assert_trees_all_equal(outputs_flat_data, expected_flat_data)

    # calling nest_data_for_scans does the reverse of splitting flat fields into
    # separate components for each nesting level.
    expected_nested = scan_utils.nest_data_for_scans(expected_fields, dt)
    merged_expected_nested = functools.reduce(_merge_nested, expected_nested)
    merged_expected_nested = jax.tree.map(
        lambda x: x.data, merged_expected_nested, is_leaf=cx.is_field
    )
    chex.assert_trees_all_equal(outputs, merged_expected_nested)


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
