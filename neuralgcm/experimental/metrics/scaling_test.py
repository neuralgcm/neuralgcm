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

"""Tests for scaling."""

from absl.testing import absltest
from absl.testing import parameterized
import coordax as cx
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.metrics import scaling
import numpy as np


class CoordinateMaskScalerTest(parameterized.TestCase):

  def test_coordinate_mask_scaler(self):
    time_coord = coordinates.TimeDelta(
        np.array([0, 6, 12, 18]) * np.timedelta64(1, 'h')
    )
    field = cx.wrap(np.ones(time_coord.shape), time_coord)
    mask_deltas = np.array([6, 18]) * np.timedelta64(1, 'h')
    mask_coord = coordinates.TimeDelta(mask_deltas)
    mask_scaler = scaling.CoordinateMaskScaler(
        mask_coord=mask_coord, masked_value=0.0, unmasked_value=5.0
    )
    scales = mask_scaler.scales(field)
    expected_scales = np.array([5.0, 0.0, 5.0, 0.0])
    np.testing.assert_allclose(scales.data, expected_scales)

  def test_coordinate_mask_scaler_with_context(self):
    x = cx.SizedAxis('x', 4)
    field = cx.wrap(np.ones(x.shape), x)
    mask_deltas = np.array([6, 18]) * np.timedelta64(1, 'h')
    mask_coord = coordinates.TimeDelta(mask_deltas)
    mask_scaler = scaling.CoordinateMaskScaler(mask_coord=mask_coord)

    context_time_match = {'timedelta': cx.wrap(np.timedelta64(6, 'h'))}
    scales_match = mask_scaler.scales(field, context=context_time_match)
    np.testing.assert_allclose(scales_match.data, 0.0)

    context_time_no_match = {'timedelta': cx.wrap(np.timedelta64(3, 'h'))}
    scales_no_match = mask_scaler.scales(field, context=context_time_no_match)
    np.testing.assert_allclose(scales_no_match.data, 1.0)

  def test_missing_dimension_raises_error(self):
    """Tests that a missing dimension raises a ValueError."""
    time_coord = coordinates.TimeDelta(
        np.array([0, 6, 12, 18]) * np.timedelta64(1, 'h')
    )
    field = cx.wrap(np.ones(time_coord.shape), time_coord)
    mask_coord = cx.SizedAxis('nondim', 2)
    with self.assertRaisesRegex(
        ValueError, "Coordinate for 'nondim' not found"
    ):
      scaler = scaling.CoordinateMaskScaler(
          mask_coord=mask_coord, skip_missing=False
      )
      scaler.scales(field)


if __name__ == '__main__':
  absltest.main()
