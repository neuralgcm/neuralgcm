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


class MaskedScalerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.coord = coordinates.TimeDelta(
        np.array([np.timedelta64(i, 'h') for i in range(10)])
    )
    self.field = cx.wrap(np.zeros(self.coord.shape, np.float32), self.coord)

  @parameterized.named_parameters(
      dict(
          testcase_name='mask_from_start',
          n=3,
          position='start',
          expected=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
      ),
      dict(
          testcase_name='mask_from_end',
          n=2,
          position='end',
          expected=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
      ),
      dict(
          testcase_name='mask_zero_elements',
          n=0,
          position='start',
          expected=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
      ),
      dict(
          testcase_name='mask_all_elements',
          n=10,
          position='start',
          expected=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      ),
  )
  def test_scales_returns_correct_mask(self, n, position, expected):
    """Tests that the scaler produces the correct mask array."""
    scaler = scaling.MaskedScaler(
        coord_name='timedelta', n_to_mask=n, mask_position=position
    )
    result = scaler.scales(self.field)
    np.testing.assert_allclose(
        result.data, np.array(expected, dtype=np.float32)
    )
    self.assertEqual(result.coordinate, self.coord)

  def test_invalid_position_raises_error(self):
    """Tests that an invalid position raises a ValueError."""
    with self.assertRaisesRegex(
        ValueError, "`mask_position` must be either 'start' or 'end'."
    ):
      scaler = scaling.MaskedScaler(
          coord_name='timedelta', n_to_mask=1, mask_position='middle'
      )
      scaler.scales(self.field)

  def test_missing_dimension_raises_error(self):
    """Tests that a missing dimension raises a ValueError."""
    with self.assertRaisesRegex(
        ValueError, "Dimension 'nondim' not found in field."
    ):
      scaler = scaling.MaskedScaler(coord_name='nondim', n_to_mask=1)
      scaler.scales(self.field)


if __name__ == '__main__':
  absltest.main()
