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
import jax
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.metrics import weighting
import numpy as np


class WeightingTest(parameterized.TestCase):

  def test_grid_area_weighting_with_lon_lat_grid(self):
    grid = coordinates.LonLatGrid.T21()
    field = cx.field(np.ones(grid.shape), grid)
    area_weighting = weighting.GridAreaWeighting()
    weights = area_weighting.weights(field)
    self.assertEqual(weights.shape, grid.shape)
    # weights should be independent of longitude.
    np.testing.assert_allclose(weights.data[0, :], weights.data[10, :])

  def test_grid_area_weighting_with_spherical_harmonic_grid(self):
    grid = coordinates.SphericalHarmonicGrid.T21()
    field = cx.field(np.ones(grid.shape), grid)
    area_weighting = weighting.GridAreaWeighting()
    weights = area_weighting.weights(field)
    self.assertEqual(weights.shape, grid.shape)
    # weights should be the mask for YLM grid
    np.testing.assert_allclose(weights.data, grid.fields['mask'].data)

  def test_constant_weighting(self):
    dim = cx.SizedAxis('spatial', 5)
    extra = cx.SizedAxis('extra', 2)
    field = cx.field(np.zeros(dim.shape + extra.shape), dim, extra)
    constant_weights_data = np.arange(5, dtype=np.float32)
    constant_weights = cx.field(constant_weights_data, dim)
    constant_weighting = weighting.ConstantWeighting(constant=constant_weights)
    weights = constant_weighting.weights(field)
    np.testing.assert_allclose(weights.data, constant_weights_data)

  def test_clip_weighting(self):
    dim = cx.SizedAxis('spatial', 5)
    field = cx.field(np.ones(dim.shape), dim)
    constant_weights_data = np.array([-1.0, 0.5, 1.0, 1.5, 2.0])
    constant_weights = cx.field(constant_weights_data, dim)
    base_weighting = weighting.ConstantWeighting(constant=constant_weights)
    clip_weighting = weighting.ClipWeighting(
        weighting=base_weighting, min_val=0.0, max_val=1.0
    )
    weights = clip_weighting.weights(field)
    expected_weights = np.array([0.0, 0.5, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(weights.data, expected_weights)

  def test_per_variable_weighting(self):
    dim = cx.SizedAxis('spatial', 2)
    field = cx.field(np.ones(dim.shape), dim)
    weighting_x = weighting.ConstantWeighting(constant=cx.field(2.0))
    weighting_y = weighting.ConstantWeighting(constant=cx.field(3.0))
    default_weighting = weighting.ConstantWeighting(constant=cx.field(0.5))
    per_var_weighting = weighting.PerVariableWeighting(
        weightings_by_name={'x': weighting_x, 'y': weighting_y},
        default_weighting=default_weighting,
    )
    weights_x = per_var_weighting.weights(field, 'x')
    np.testing.assert_allclose(weights_x.data, 2.0)
    weights_y = per_var_weighting.weights(field, 'y')
    np.testing.assert_allclose(weights_y.data, 3.0)
    weights_z = per_var_weighting.weights(field, 'z')
    np.testing.assert_allclose(weights_z.data, 0.5)

  def test_per_variable_weighting_from_constants(self):
    dim = cx.SizedAxis('spatial', 2)
    field_x = cx.field(np.ones(dim.shape), dim)
    variable_weights = {'x': 2.0, 'z': 3.14}
    per_var_weighting = weighting.PerVariableWeighting.from_constants(
        variable_weights=variable_weights
    )
    weights_x = per_var_weighting.weights(field_x, 'x')
    np.testing.assert_allclose(weights_x.data, 2.0)
    weights_z = per_var_weighting.weights(field_x, 'z')
    np.testing.assert_allclose(weights_z.data, 3.14)

  def test_coordinate_mask_weighting(self):
    time_coord = coordinates.TimeDelta(
        np.array([0, 6, 12, 18]) * np.timedelta64(1, 'h')
    )
    field = cx.field(np.ones(time_coord.shape), time_coord)
    mask_deltas = np.array([6, 18]) * np.timedelta64(1, 'h')
    mask_coord = coordinates.TimeDelta(mask_deltas)
    mask_weighting = weighting.CoordinateMaskWeighting(mask_coord=mask_coord)
    weights = mask_weighting.weights(field)
    expected_weights = np.array([1.0, 0.0, 1.0, 0.0])
    np.testing.assert_allclose(weights.data, expected_weights)

  def test_coordinate_mask_weighting_with_context(self):
    x = cx.SizedAxis('x', 4)
    field = cx.field(np.ones(x.shape), x)
    mask_deltas = np.array([6, 18]) * np.timedelta64(1, 'h')
    mask_coord = coordinates.TimeDelta(mask_deltas)
    mask_weighting = weighting.CoordinateMaskWeighting(mask_coord=mask_coord)

    context_time_match = {'timedelta': cx.field(np.timedelta64(6, 'h'))}
    weights_match = mask_weighting.weights(field, context=context_time_match)
    np.testing.assert_allclose(weights_match.data, 0.0)

    context_time_no_match = {'timedelta': cx.field(np.timedelta64(3, 'h'))}
    weights_no_match = mask_weighting.weights(
        field, context=context_time_no_match
    )
    np.testing.assert_allclose(weights_no_match.data, 1.0)


if __name__ == '__main__':
  jax.config.update('jax_traceback_filtering', 'off')
  absltest.main()
