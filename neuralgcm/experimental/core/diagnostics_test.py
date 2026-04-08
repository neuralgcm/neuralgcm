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

"""Tests for diagnostics modules and diagnostics API."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import coordax as cx
from flax import nnx
import jax
import jax.numpy as jnp
import jax_datetime as jdt
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import diagnostics
from neuralgcm.experimental.core import feature_transforms
from neuralgcm.experimental.core import module_utils
from neuralgcm.experimental.core import observation_operators
from neuralgcm.experimental.core import spherical_harmonics
from neuralgcm.experimental.core import transforms
import numpy as np


class MockMethod(nnx.Module):
  """Mock method to which diagnostics are attached for testing."""

  def custom_add_one_to_increasing(self, inputs):
    inputs['increasing'] += 1.0
    return inputs

  def pass_through(self, inputs):
    return inputs

  def __call__(self, inputs):
    result = {k: v for k, v in inputs.items()}
    # call twice to make it distinct from custom_add_one_to_increasing
    result = self.custom_add_one_to_increasing(result)
    result = self.custom_add_one_to_increasing(result)
    return result


class DiagnosticsTest(parameterized.TestCase):

  def test_cumulative_diagnostic(self):
    x_coord, y_coord = cx.SizedAxis('x', 3), cx.SizedAxis('y', 5)
    inputs = {
        'fixed': cx.field(jnp.arange(3.0), x_coord),
        'increasing': cx.field(jnp.zeros(5), y_coord),
    }
    extract = lambda x, *args, **kwargs: x
    d_coords = {'fixed': x_coord, 'increasing': y_coord}
    diagnostic = diagnostics.CumulativeDiagnostic(extract, d_coords)
    module = MockMethod()
    module_with_diagnostic = module_utils.with_callback(module, diagnostic)

    with self.subTest('does_not_change_outputs'):
      no_diagnostic_output = module(inputs)
      output = module_with_diagnostic(inputs)
      chex.assert_trees_all_equal(output, no_diagnostic_output)

    with self.subTest('produces_expected_values'):
      diagnostic.reset_diagnostic_state()
      n_steps = 10
      output = inputs
      for _ in range(n_steps):
        output = module_with_diagnostic(output)
      # `fixed` is unchanged, its cumulative is n_steps * arange(3)
      fixed_sum = jnp.arange(3.0) * n_steps
      # `increasing` is incremented by 2 at each step. Diagnostic sees outputs
      # 2, 4, ..., 20. The sum is 2 * (1 + 2 + ... + 10) = 2*(1+10)*10/2 = 110.
      increasing_sum = 2 * n_steps * (n_steps + 1) / 2
      expected_cumulatives = {
          'fixed': cx.field(fixed_sum, x_coord),
          'increasing': cx.field(jnp.ones(5) * increasing_sum, y_coord),
      }
      actual_cumulatives = diagnostic.diagnostic_values()
      chex.assert_trees_all_close(actual_cumulatives, expected_cumulatives)

    with self.subTest('resets_values'):
      diagnostic.reset_diagnostic_state()
      actual_cumulatives = diagnostic.diagnostic_values()
      expected_zeros = {
          'fixed': cx.field(jnp.zeros(3), x_coord),
          'increasing': cx.field(jnp.zeros(5), y_coord),
      }
      chex.assert_trees_all_close(actual_cumulatives, expected_zeros)

  def test_cumulative_diagnostic_callback_on_custom_method(self):
    x_coord, y_coord = cx.SizedAxis('x', 3), cx.SizedAxis('y', 5)
    inputs = {
        'fixed': cx.field(jnp.arange(3.0), x_coord),
        'increasing': cx.field(jnp.zeros(5), y_coord),
    }
    extract = lambda x, *args, **kwargs: x | {'count': 1}
    d_coords = {'fixed': x_coord, 'increasing': y_coord, 'count': cx.Scalar()}
    diagnostic = diagnostics.CumulativeDiagnostic(extract, d_coords)
    module = MockMethod()
    module_with_diagnostic = module_utils.with_callback(
        module, diagnostic, method_name='custom_add_one_to_increasing'
    )
    n_steps = 10
    output = inputs
    for _ in range(n_steps):
      output = module_with_diagnostic(output)
    # `fixed` is seen by diagnostic 2*n_steps times.
    n_calls = 2 * n_steps
    fixed_sum = jnp.arange(3.0) * n_calls
    # `increasing` is incremented by 1 two times per step. The diagnostic is
    # on the custom method, so it sees values 1, 2, 3, 4, ... 2*n_steps.
    # The sum is (2*n_steps) * (2*n_steps+1)/2 = 10 * 21 = 210.
    increasing_sum = n_calls * (n_calls + 1) / 2
    expected_cumulatives = {
        'fixed': cx.field(fixed_sum, x_coord),
        'increasing': cx.field(jnp.ones(5) * increasing_sum, y_coord),
        'count': cx.field(n_calls, cx.Scalar()),
    }
    actual_cumulatives = diagnostic.diagnostic_values()
    chex.assert_trees_all_close(actual_cumulatives, expected_cumulatives)

  def test_instant_diagnostic(self):
    x_coord = cx.LabeledAxis('x', np.arange(7))
    y_coord = cx.LabeledAxis('y', np.arange(5))
    inputs = {
        'fixed': cx.field(jnp.arange(7.0), x_coord),
        'increasing': cx.field(jnp.zeros(5), y_coord),
    }
    extract = lambda x, *args, **kwargs: x
    d_coords = {'fixed': x_coord, 'increasing': y_coord}
    diagnostic = diagnostics.InstantDiagnostic(extract, d_coords)
    module = MockMethod()
    module_with_diagnostic = module_utils.with_callback(module, diagnostic)

    with self.subTest('does_not_change_outputs'):
      no_diagnostic_output = module(inputs)
      output = module_with_diagnostic(inputs.copy())
      chex.assert_trees_all_equal(output, no_diagnostic_output)

    with self.subTest('produces_expected_values'):
      diagnostic.reset_diagnostic_state()
      n_steps = 10
      output = inputs.copy()
      for _ in range(n_steps):
        output = module_with_diagnostic(output)
      # `fixed` is unchanged.
      fixed_final = jnp.arange(7.0)
      # `increasing` is incremented by 2 at each step for 10 steps.
      increasing_final = jnp.ones(5) * 2 * n_steps
      expected_final = {
          'fixed': cx.field(fixed_final, x_coord),
          'increasing': cx.field(increasing_final, y_coord),
      }
      actual_final = diagnostic.diagnostic_values()
      chex.assert_trees_all_close(expected_final, actual_final)

    with self.subTest('resets_values'):
      diagnostic.reset_diagnostic_state()
      actual_cumulatives = diagnostic.diagnostic_values()
      expected_zeros = {
          'fixed': cx.field(jnp.zeros(7), x_coord),
          'increasing': cx.field(jnp.zeros(5), y_coord),
      }
      chex.assert_trees_all_close(actual_cumulatives, expected_zeros)

  def test_interval_diagnostic(self):
    x_coord, y_coord = cx.SizedAxis('x', 3), cx.SizedAxis('y', 5)
    extract = lambda x, *args, **kwargs: x
    d_coords = {'fixed': x_coord, 'increasing': y_coord}
    interval = np.timedelta64(6, 'h')
    resolution = np.timedelta64(1, 'h')
    default_timedelta = np.timedelta64(1, 'h')
    diagnostic = diagnostics.IntervalDiagnostic(
        extract,
        d_coords,
        interval=interval,
        resolution=resolution,
        default_timedelta=default_timedelta,
        include_dt_offset=True,
    )
    module = MockMethod()
    module_with_diagnostic = module_utils.with_callback(module, diagnostic)
    # call `advance_clock` on every module.__call__.
    module_with_diagnostic = module_utils.with_callback(
        module_with_diagnostic, (diagnostic, 'advance_clock')
    )
    inputs = {
        'fixed': cx.field(jnp.arange(3.0), x_coord),
        'increasing': cx.field(jnp.zeros(5), y_coord),
    }
    with self.subTest('does_not_change_outputs'):
      no_diagnostic_output = module(inputs)
      output = module_with_diagnostic(inputs.copy())
      chex.assert_trees_all_close(output, no_diagnostic_output)

    with self.subTest('produces_expected_values'):
      diagnostic.reset_diagnostic_state()
      n_steps = 20
      output = inputs.copy()
      for _ in range(n_steps):
        output = module_with_diagnostic(output)

      actual_interval_cumulatives = diagnostic.diagnostic_values()
      # After 20 steps (1h each), with freq=1h and periods=6, we expect
      # accumulation over 6h interval, which covers steps 15-20.
      # `increasing` is incremented by 2.0 at each step. So at step i, its value
      # is 2*i. Sum for steps 15-20 is 2*(15+16+17+18+19+20) = 210.
      expected_increasing = jnp.ones(5) * 210.0
      # `fixed` is constant arange(3). Sum over 6 steps is 6 * arange(3).
      expected_fixed = jnp.arange(3.0) * 6.0
      expected_values = {
          'fixed': cx.field(expected_fixed, x_coord),
          'increasing': cx.field(expected_increasing, y_coord),
      }
      chex.assert_trees_all_close(
          actual_interval_cumulatives['fixed'], expected_values['fixed']
      )
      chex.assert_trees_all_close(
          actual_interval_cumulatives['increasing'],
          expected_values['increasing'],
      )
      self.assertEqual(
          actual_interval_cumulatives['timedelta_since_sub_interval'].data,
          jdt.to_timedelta(np.timedelta64(0, 'h')),
      )

    with self.subTest('resets_values'):
      diagnostic.reset_diagnostic_state()
      actual_interval_cumulatives = diagnostic.diagnostic_values()
      expected_zeros = {
          'fixed': cx.field(jnp.zeros(3), x_coord),
          'increasing': cx.field(jnp.zeros(5), y_coord),
      }
      chex.assert_trees_all_close(
          actual_interval_cumulatives['fixed'], expected_zeros['fixed']
      )
      chex.assert_trees_all_close(
          actual_interval_cumulatives['increasing'],
          expected_zeros['increasing'],
      )

  def test_interval_with_multiple_intervals(self):
    y_coord = cx.SizedAxis('y', 5)
    diagnostic = diagnostics.IntervalDiagnostic(
        extract=lambda x, *args, **kwargs: {'inc': x['increasing']},
        extract_coords={'inc': y_coord},
        interval={'4h': np.timedelta64(4, 'h'), '2h': np.timedelta64(2, 'h')},
        resolution=np.timedelta64(2, 'h'),
        default_timedelta=np.timedelta64(1, 'h'),
    )
    module = MockMethod()
    module_with_diagnostic = module_utils.with_callback(module, diagnostic)
    module_with_diagnostic = module_utils.with_callback(
        module_with_diagnostic, (diagnostic, 'advance_clock')
    )
    inputs = {'increasing': cx.field(jnp.zeros(5), y_coord)}

    output = inputs.copy()
    for _ in range(8):
      output = module_with_diagnostic(output)

    # `increasing` increments by 2 per step. The value at step i is 2*i.
    # Resolution is 2h, so updates happen at steps 2, 4, 6, 8.
    # The 4h interval covers steps 5, 6, 7, 8.
    # Sum over steps 5, 6, 7, 8 is 10 + 12 + 14 + 16 = 52.
    # The 2h interval covers steps 7, 8.
    # Sum over steps 7, 8 is 14 + 16 = 30.

    expected_increasing_4h = cx.field(jnp.ones(5) * 52.0, y_coord)
    expected_increasing_2h = cx.field(jnp.ones(5) * 30.0, y_coord)

    actual = diagnostic.diagnostic_values()
    cx.testing.assert_fields_allclose(actual['inc_4h'], expected_increasing_4h)
    cx.testing.assert_fields_allclose(actual['inc_2h'], expected_increasing_2h)

  def test_interval_diagnostic_with_explicit_timedelta_in_advance_clock(self):
    y_coord = cx.SizedAxis('y', 5)
    extract = lambda x, *args, **kwargs: {'increasing': x['increasing']}
    d_coords = {'increasing': y_coord}
    diagnostic = diagnostics.IntervalDiagnostic(
        extract,
        d_coords,
        interval=np.timedelta64(12, 'h'),
        resolution=np.timedelta64(4, 'h'),
    )
    module = MockMethod()
    module_with_diagnostic = module_utils.with_callback(module, diagnostic)
    module_with_diagnostic = module_utils.with_callback(
        module_with_diagnostic,
        (diagnostic, 'advance_clock'),
        method_name='pass_through',
    )
    state = {'increasing': cx.field(jnp.zeros(5), y_coord)}
    explicit_timedelta = jdt.to_timedelta(np.timedelta64(1, 'h'))
    for _ in range(16):
      state = module_with_diagnostic(state)
      _ = module_with_diagnostic.pass_through({'timedelta': explicit_timedelta})

    diag_values = diagnostic.diagnostic_values()
    # In 16 steps we are 4 steps past the start of the sequence, hence backwards
    # looking interval should contain sum over 16 termms - sum over 4 terms.
    # 2 * (1 + 16) * 16 / 2 - 2 * (1 + 2 + 3 + 4) = 252.
    expected_increasing = jnp.ones(5) * 252
    chex.assert_trees_all_close(
        diag_values['increasing'],
        cx.field(expected_increasing, y_coord),
        atol=1e-6,
    )
    # advancing a full sub-interval at a time also works:
    timedelta_4hr = jdt.to_timedelta(np.timedelta64(4, 'h'))
    _ = module_with_diagnostic.pass_through({'timedelta': timedelta_4hr})
    diag_values = diagnostic.diagnostic_values()
    expected_increasing = jnp.ones(5) * (252 - 10 - 12 - 14 - 16)
    chex.assert_trees_all_close(
        diag_values['increasing'],
        cx.field(expected_increasing, y_coord),
        atol=1e-6,
    )

  def test_interval_diagnostic_values_returns_complete_intervals(self):
    y_coord = cx.SizedAxis('y', 5)
    extract = lambda x, *args, **kwargs: {'increasing': x['increasing']}
    d_coords = {'increasing': y_coord}
    diagnostic = diagnostics.IntervalDiagnostic(
        extract,
        d_coords,
        interval=np.timedelta64(5, 's'),
        resolution=np.timedelta64(5, 's'),
        default_timedelta=np.timedelta64(1, 's'),
        include_dt_offset=True,
    )
    module = MockMethod()
    module_with_diagnostic = module_utils.with_callback(module, diagnostic)
    module_with_diagnostic = module_utils.with_callback(
        module_with_diagnostic, (diagnostic, 'advance_clock')
    )
    output = {'increasing': cx.field(jnp.zeros(5), y_coord)}
    for _ in range(5):
      output = module_with_diagnostic(output)
    diag_values_after_5_steps = diagnostic.diagnostic_values()
    # `increasing` values for 5 steps: 2, 4, 6, 8, 10 sum to 30.
    expected_increasing = jnp.ones(5) * 30.0
    chex.assert_trees_all_close(
        diag_values_after_5_steps['increasing'],
        cx.field(expected_increasing, y_coord),
    )

    for _ in range(3):
      output = module_with_diagnostic(output)
    diag_values = diagnostic.diagnostic_values()
    # After performing 3 more steps, we are 3s into the next sub-interval.
    # Diagnostic values are expected to return values from a completed
    # sub-intervals, so we expect the same value as after 5 steps.
    chex.assert_trees_all_close(
        diag_values['increasing'], cx.field(expected_increasing, y_coord)
    )
    # We can check that the timedelta_since_sub_interval is as expected.
    self.assertEqual(
        diag_values['timedelta_since_sub_interval'].data,
        jdt.to_timedelta(np.timedelta64(3, 's')),
    )

  def test_interval_diagnostic_nnx_jit_compatible(self):
    """Tests that IntervalDiagnostic works with nnx.jit."""
    y_coord = cx.SizedAxis('y', 1)
    extract = lambda x, *args, **kwargs: {'increasing': x['increasing']}
    d_coords = {'increasing': y_coord}
    diagnostic = diagnostics.IntervalDiagnostic(
        extract,
        d_coords,
        interval=np.timedelta64(1, 's'),
        resolution=np.timedelta64(1, 's'),
        default_timedelta=np.timedelta64(1, 's'),
    )
    module = MockMethod()
    module_with_diagnostic = module_utils.with_callback(module, diagnostic)
    module_with_diagnostic = module_utils.with_callback(
        module_with_diagnostic, (diagnostic, 'advance_clock')
    )
    inputs = {'increasing': cx.field(jnp.zeros(1), y_coord)}

    @nnx.jit
    def run_step(model, inputs):
      return model(inputs)

    # test that jitted call runs without errors and updates state
    for _ in range(3):
      inputs = run_step(module_with_diagnostic, inputs)

    diag_values = diagnostic.diagnostic_values()
    # After 3 steps, `increasing` is 6 and diagnostic values reflect
    # accumulation over last step.
    chex.assert_trees_all_close(
        diag_values['increasing'],
        cx.field(jnp.ones(1) * 6.0, y_coord),
    )

  def test_interval_diagnostic_resolution_not_int_seconds_raises_error(self):
    extract = lambda x, *args, **kwargs: x
    d_coords = {'x': cx.Scalar()}
    with self.assertRaisesRegex(
        ValueError, 'resolution must be an integer number of seconds'
    ):
      diagnostics.IntervalDiagnostic(
          extract,
          d_coords,
          interval=np.timedelta64(3, 's'),
          resolution=np.timedelta64(1500, 'ms'),
      )

  def test_interval_diagnostic_interval_not_multiple_of_resolution_raises_error(
      self,
  ):
    extract = lambda x, *args, **kwargs: x
    d_coords = {'x': cx.Scalar()}
    with self.assertRaisesRegex(ValueError, 'must be a multiple of'):
      diagnostics.IntervalDiagnostic(
          extract,
          d_coords,
          interval=np.timedelta64(3, 's'),
          resolution=np.timedelta64(2, 's'),
      )

  def test_time_offset_diagnostic(self):
    x_coord, y_coord = cx.SizedAxis('x', 3), cx.SizedAxis('y', 5)
    diagnostic = diagnostics.TimeOffsetDiagnostic(
        extract=lambda x, *args, **kwargs: x,
        extract_coords={'fixed': x_coord, 'increasing': y_coord},
        offset={
            '4h_ago': np.timedelta64(4, 'h'),
            '2h_ago': np.timedelta64(2, 'h'),
        },
        resolution=np.timedelta64(2, 'h'),
        default_timedelta=np.timedelta64(1, 'h'),
        include_dt_offset=True,
    )
    module = MockMethod()
    module_with_diagnostic = module_utils.with_callback(module, diagnostic)
    module_with_diagnostic = module_utils.with_callback(
        module_with_diagnostic, (diagnostic, 'advance_clock')
    )
    inputs = {
        'fixed': cx.field(jnp.arange(3.0), x_coord),
        'increasing': cx.field(jnp.zeros(5), y_coord),
    }

    with self.subTest('produces_expected_values'):
      diagnostic.reset_diagnostic_state()
      output = inputs.copy()
      for _ in range(10):
        output = module_with_diagnostic(output)

      actual = diagnostic.diagnostic_values()
      # The resolution is 2h, so updates happen at steps 2, 4, 6, 8, 10.
      # Tracked "increasing" values from 4h ago and 2h ago are: 6*2=12, 8*2=16.
      expected_increasing_4h = cx.field(jnp.ones(5) * 12.0, y_coord)
      expected_increasing_2h = cx.field(jnp.ones(5) * 16.0, y_coord)
      expected_fixed = cx.field(jnp.arange(3.0), x_coord)  # remains unchanged.

      cx.testing.assert_fields_allclose(actual['fixed_2h_ago'], expected_fixed)
      cx.testing.assert_fields_allclose(actual['fixed_4h_ago'], expected_fixed)
      cx.testing.assert_fields_allclose(
          actual['increasing_2h_ago'], expected_increasing_2h
      )
      cx.testing.assert_fields_allclose(
          actual['increasing_4h_ago'], expected_increasing_4h
      )
      self.assertEqual(
          actual['timedelta_since_sub_interval'].data,
          jdt.to_timedelta(np.timedelta64(0, 'h')),
      )

    with self.subTest('resets_values'):
      diagnostic.reset_diagnostic_state()
      actual = diagnostic.diagnostic_values()
      expected_zeros = {
          'fixed_2h_ago': cx.field(jnp.zeros(3), x_coord),
          'fixed_4h_ago': cx.field(jnp.zeros(3), x_coord),
          'increasing_2h_ago': cx.field(jnp.zeros(5), y_coord),
          'increasing_4h_ago': cx.field(jnp.zeros(5), y_coord),
          'timedelta_since_sub_interval': cx.field(jdt.Timedelta(0)),
      }
      chex.assert_trees_all_close(actual, expected_zeros)

  def test_chained_interval_and_time_offset(self):
    y_coord = cx.SizedAxis('y', 5)
    interval_diag = diagnostics.IntervalDiagnostic(
        extract=lambda x, *args, **kwargs: {'increasing': x['increasing']},
        extract_coords={'increasing': y_coord},
        interval=np.timedelta64(4, 'h'),
        resolution=np.timedelta64(2, 'h'),
        default_timedelta=np.timedelta64(1, 'h'),
    )
    time_offset_diag = diagnostics.TimeOffsetDiagnostic(
        extract=diagnostics.ExtractTransformedOutputs(
            feature_transforms.DiagnosticValueFeatures(interval_diag)
        ),
        extract_coords={'increasing': y_coord},
        offset={'2h_ago': np.timedelta64(2, 'h')},
        resolution=np.timedelta64(2, 'h'),
        default_timedelta=np.timedelta64(1, 'h'),
    )

    module = MockMethod()
    # The goal is to get offset interval, so we ensure that interval diagnostics
    # is computed first (i.e. first callback).
    module = module_utils.with_callback(module, interval_diag)
    module = module_utils.with_callback(
        module, (interval_diag, 'advance_clock')
    )
    module = module_utils.with_callback(module, time_offset_diag)
    module = module_utils.with_callback(
        module, (time_offset_diag, 'advance_clock')
    )

    inputs = {'increasing': cx.field(jnp.zeros(5), y_coord)}

    output = inputs.copy()
    for _ in range(8):
      output = module(output)

    # At t=8h with -2h offset and 4h interval, we expect values from (2-6].
    # Steps 3, 4, 5, 6 have values: 6, 8, 10, 12.
    # Sum over steps 3, 4, 5, 6 is 6 + 8 + 10 + 12 = 36.
    expected = cx.field(jnp.ones(5) * 36.0, y_coord)

    actual = time_offset_diag.diagnostic_values()
    cx.testing.assert_fields_allclose(actual['increasing_2h_ago'], expected)

  def test_time_offset_with_explicit_timedelta_in_advance_clock(self):
    y_coord = cx.SizedAxis('y', 5)
    diagnostic = diagnostics.TimeOffsetDiagnostic(
        extract=lambda x, *args, **kwargs: {'increasing': x['increasing']},
        extract_coords={'increasing': y_coord},
        offset=np.timedelta64(8, 'h'),
        resolution=np.timedelta64(4, 'h'),
    )
    module = MockMethod()
    module_with_diagnostic = module_utils.with_callback(module, diagnostic)
    module_with_diagnostic = module_utils.with_callback(
        module_with_diagnostic,
        (diagnostic, 'advance_clock'),
        method_name='pass_through',
    )
    state = {'increasing': cx.field(jnp.zeros(5), y_coord)}
    explicit_timedelta = jdt.to_timedelta(np.timedelta64(1, 'h'))
    for _ in range(12):
      state = module_with_diagnostic(state)
      _ = module_with_diagnostic.pass_through({'timedelta': explicit_timedelta})

    actual = diagnostic.diagnostic_values()
    # 12 steps of 1h. Resolution is 4h, so updates at steps 4, 8, 12.
    # offset is 8h (which means we check the value from step 4).
    # Values at steps 4 is 8.0.
    expected = cx.field(jnp.ones(5) * 8.0, y_coord)
    cx.testing.assert_fields_allclose(actual['increasing'], expected)

  def test_time_offset_values_returns_complete_intervals(self):
    y_coord = cx.SizedAxis('y', 5)
    diagnostic = diagnostics.TimeOffsetDiagnostic(
        extract=lambda x, *args, **kwargs: {'increasing': x['increasing']},
        extract_coords={'increasing': y_coord},
        offset={'10s_ago': np.timedelta64(10, 's')},
        resolution=np.timedelta64(5, 's'),
        default_timedelta=np.timedelta64(1, 's'),
        include_dt_offset=True,
    )
    module = MockMethod()
    module_with_diagnostic = module_utils.with_callback(module, diagnostic)
    module_with_diagnostic = module_utils.with_callback(
        module_with_diagnostic, (diagnostic, 'advance_clock')
    )
    output = {'increasing': cx.field(jnp.zeros(5), y_coord)}

    for _ in range(15):
      output = module_with_diagnostic(output)

    values = diagnostic.diagnostic_values()
    # After 15s, the value from (15-10)=5 is 10 (2 per step)
    expected = cx.field(jnp.ones(5) * 10.0, y_coord)
    cx.testing.assert_fields_allclose(values['increasing_10s_ago'], expected)

    for _ in range(3):
      output = module_with_diagnostic(output)

    values = diagnostic.diagnostic_values()
    # At 18s, past values haven't shifted. It still returns the value from 5s.
    # Which represents 13s ago (10s + 3s).
    cx.testing.assert_fields_allclose(values['increasing_10s_ago'], expected)
    self.assertEqual(
        values['timedelta_since_sub_interval'].data,
        jdt.to_timedelta(np.timedelta64(3, 's')),
    )

  def test_time_offset_nnx_jit_compatible(self):
    y_coord = cx.SizedAxis('y', 1)
    diagnostic = diagnostics.TimeOffsetDiagnostic(
        extract=lambda x, *args, **kwargs: {'increasing': x['increasing']},
        extract_coords={'increasing': y_coord},
        offset=np.timedelta64(2, 's'),
        resolution=np.timedelta64(1, 's'),
        default_timedelta=np.timedelta64(1, 's'),
    )
    module = MockMethod()
    module_with_diagnostic = module_utils.with_callback(module, diagnostic)
    module_with_diagnostic = module_utils.with_callback(
        module_with_diagnostic, (diagnostic, 'advance_clock')
    )
    inputs = {'increasing': cx.field(jnp.zeros(1), y_coord)}

    @nnx.jit
    def run_step(model, inputs):
      return model(inputs)

    for _ in range(3):
      inputs = run_step(module_with_diagnostic, inputs)

    actual = diagnostic.diagnostic_values()
    # offset 2s, current time 3s. So we want value at 1s. Value is 2.0.
    expected = cx.field(jnp.array([2.0]), y_coord)
    cx.testing.assert_fields_allclose(actual['increasing'], expected)

  def test_time_offset_resolution_not_int_seconds_raises_error(self):
    with self.assertRaisesRegex(
        ValueError, 'resolution must be an integer number of seconds'
    ):
      diagnostics.TimeOffsetDiagnostic(
          extract=lambda x, *args, **kwargs: x,
          extract_coords={'x': cx.Scalar()},
          offset=np.timedelta64(3, 's'),
          resolution=np.timedelta64(1500, 'ms'),
      )

  def test_time_offset_offset_not_multiple_of_resolution_raises_error(self):
    with self.assertRaisesRegex(ValueError, 'must be a multiple of'):
      diagnostics.TimeOffsetDiagnostic(
          extract=lambda x, *args, **kwargs: x,
          extract_coords={'x': cx.Scalar()},
          offset=np.timedelta64(3, 's'),
          resolution=np.timedelta64(2, 's'),
      )


class ExtractModulesTest(parameterized.TestCase):

  def test_extract_transformed_outputs(self):

    grid = coordinates.LonLatGrid.T21()
    ylm_grid = coordinates.SphericalHarmonicGrid.T21()
    ylm_map = spherical_harmonics.FixedYlmMapping(grid, ylm_grid)

    to_nodal = transforms.ToNodal(ylm_map)
    extract = diagnostics.ExtractTransformedOutputs(to_nodal)

    inputs = {'u': cx.field(jnp.zeros(ylm_grid.shape), ylm_grid)}
    actual = extract(inputs)

    expected_grid = grid
    self.assertEqual(actual['u'].coordinate, expected_grid)

  def test_extract_fixed_query_observations(self):
    pressure_coord = cx.LabeledAxis('pressure', np.arange(5))
    field_data = cx.field(np.arange(5), pressure_coord)
    obs_op = observation_operators.DataObservationOperator({'a': field_data})

    query_coord = cx.LabeledAxis('pressure', np.array([0, 2, 4]))
    query = {'a': query_coord}

    extract = diagnostics.ExtractFixedQueryObservations(obs_op, query)
    prognostics = {}
    actual = extract(result={}, prognostics=prognostics)

    expected = {'a': cx.field(np.array([0, 2, 4]), query_coord)}
    chex.assert_trees_all_equal(actual, expected)


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
