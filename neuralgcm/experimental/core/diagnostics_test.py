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
from flax import nnx
import jax
import jax.numpy as jnp
from neuralgcm.experimental import coordax as cx
from neuralgcm.experimental.core import diagnostics
from neuralgcm.experimental.core import module_utils
import numpy as np


class MockMethod(nnx.Module):
  """Mock method to which diagnostics are attached for testing."""

  def custom_add_half_to_y(self, inputs):
    inputs['y'] += 0.5
    return inputs

  def __call__(self, inputs):
    result = {k: v for k, v in inputs.items()}
    result = self.custom_add_half_to_y(result)
    result = self.custom_add_half_to_y(result)
    return result


class DiagnosticsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.x_coord = cx.LabeledAxis('x', np.arange(7))
    self.y_coord = cx.LabeledAxis('y', np.arange(5))
    self.inputs = {
        'x': cx.wrap(jnp.arange(7), self.x_coord),
        'y': cx.wrap(jnp.zeros(5), self.y_coord),
    }

  def test_cumulative(self):
    extract = lambda x, *args, **kwargs: x  # examine all outputs;
    extract_coords = {'x': self.x_coord, 'y': self.y_coord}
    diagnostic = diagnostics.CumulativeDiagnostic(extract, extract_coords)
    module = MockMethod()
    no_diagnostic_output = module(self.inputs)
    module_with_diagnostic = module_utils.with_callback(module, diagnostic)
    with self.subTest('output_unchanged'):
      output = module_with_diagnostic(self.inputs)
      chex.assert_trees_all_equal(output, no_diagnostic_output)

    with self.subTest('correct_cumulatives'):
      n_steps = 10
      for _ in range(n_steps - 1):
        output = module_with_diagnostic(output)
      x_sum = 1.0 * np.arange(7) * n_steps
      y_sum = n_steps * np.ones(5) * (n_steps + 1) / 2
      expected_cumulatives = {
          'x': cx.wrap(x_sum, self.x_coord),
          'y': cx.wrap(y_sum, self.y_coord),
      }
      actual_cumulatives = diagnostic.format_diagnostics(None)
      chex.assert_trees_all_close(actual_cumulatives, expected_cumulatives)

  def test_cumulative_on_custom_method(self):
    extract = lambda x, *args, **kwargs: x  # examine all outputs;
    extract_coords = {'x': self.x_coord, 'y': self.y_coord}
    diagnostic = diagnostics.CumulativeDiagnostic(extract, extract_coords)
    module = MockMethod()
    module_with_diagnostic = module_utils.with_callback(
        module, diagnostic, method_name='custom_add_half_to_y'
    )
    n_steps = 10
    output = self.inputs
    for _ in range(n_steps):
      output = module_with_diagnostic(output)
    x_sum = 1.0 * np.arange(7) * n_steps * 2  # 2 calls to custom_add_half_to_y
    y_sum = (2 * n_steps) * np.ones(5) * (n_steps + 0.5) / 2
    expected_cumulatives = {
        'x': cx.wrap(x_sum, self.x_coord),
        'y': cx.wrap(y_sum, self.y_coord),
    }
    actual_cumulatives = diagnostic.format_diagnostics(None)
    chex.assert_trees_all_close(actual_cumulatives, expected_cumulatives)

  def test_instant(self):
    extract = lambda x, *args, **kwargs: x  # examine all outputs;
    extract_coords = {'x': self.x_coord, 'y': self.y_coord}
    diagnostic = diagnostics.InstantDiagnostic(extract, extract_coords)
    module = MockMethod()
    no_diagnostic_output = module(self.inputs)
    module_with_diagnostic = module_utils.with_callback(module, diagnostic)
    with self.subTest('output_unchanged'):
      output = module_with_diagnostic(self.inputs)
      chex.assert_trees_all_equal(output, no_diagnostic_output)

    with self.subTest('correct_instants'):
      n_steps = 10
      for _ in range(n_steps - 1):
        output = module_with_diagnostic(output)
      x_final = 1.0 * np.arange(7)  # unchanged
      y_final = np.ones(5) * n_steps
      expected_final = {
          'x': cx.wrap(x_final, self.x_coord),
          'y': cx.wrap(y_final, self.y_coord),
      }
      actual_final = diagnostic.format_diagnostics(None)
      chex.assert_trees_all_close(expected_final, actual_final)

  def test_interval(self):
    extract = lambda x, *args, **kwargs: x  # examine all outputs;
    extract_coords = {'x': self.x_coord, 'y': self.y_coord}
    diagnostic = diagnostics.IntervalDiagnostic(
        extract, extract_coords, interval_length=3
    )
    module = MockMethod()
    no_diagnostic_output = module(self.inputs)
    module_with_diagnostic = module_utils.with_callback(module, diagnostic)
    module_with_diagnostic = module_utils.with_callback(
        module_with_diagnostic, (diagnostic, 'next_interval')
    )
    with self.subTest('output_unchanged'):
      output = module_with_diagnostic(self.inputs)
      chex.assert_trees_all_equal(output, no_diagnostic_output)

    with self.subTest('correct_interval'):
      n_steps = 10
      for _ in range(n_steps - 1):
        output = module_with_diagnostic(output)
      x_sum_last_2 = 1.0 * np.arange(7) * (3 - 1)
      y_sum_last_2 = (3 - 1) * np.ones(5) * (n_steps + n_steps - 1) / 2
      expected_final = {
          'x': cx.wrap(x_sum_last_2, self.x_coord),
          'y': cx.wrap(y_sum_last_2, self.y_coord),
      }
      actual_final = diagnostic.format_diagnostics(None)
      chex.assert_trees_all_close(expected_final, actual_final)

if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
