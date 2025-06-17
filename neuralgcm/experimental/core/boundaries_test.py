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

from absl.testing import absltest
from absl.testing import parameterized
import jax
from neuralgcm.experimental import coordax as cx
from neuralgcm.experimental.core import boundaries
from neuralgcm.experimental.core import coordinates
import numpy as np


class BoundariesTest(parameterized.TestCase):
  """Tests that boundary conditions correctly pad and trim boundary cells."""

  def test_lon_lat_boundary_values(self):
    """Tests that boundary ghost cells contain expected values."""
    bc = boundaries.LonLatBoundary()
    grid = coordinates.LonLatGrid.T21()  # [64, 32].
    grid_sizes = np.prod(grid.shape)
    # reshape to lat,lon and transpose to make range increment along longitude.
    f = cx.wrap(np.arange(grid_sizes).reshape(grid.shape[::-1]).T, grid)
    pad_sizes = {'longitude': (1, 1), 'latitude': (2, 2)}
    padded_f = bc.pad(f, pad_sizes)

    self.assertEqual(padded_f.shape, (66, 36))  # 64 + 1 + 1, 32 + 2 + 2.
    np.testing.assert_array_equal(
        padded_f.data[0, 2:-2], f.data[-1, :], err_msg='left_lon_padding'
    )
    np.testing.assert_array_equal(
        padded_f.data[-1, 2:-2], f.data[0, :], err_msg='right_lon_padding'
    )
    np.testing.assert_array_equal(
        padded_f.data[1:-1, -3], f.data[:, -1], err_msg='preserved_top_boundary'
    )
    np.testing.assert_array_equal(
        padded_f.data[1:-1, 2], f.data[:, 0], err_msg='preserved_bot_boundary'
    )
    # latitude boundary should be padded with values on the opposite side of the
    # pole. For 64 longitude points the diameter opposite side of 0 will be 32,
    # with 33, ... being to the left of it (clockwise direction) that pairs
    # with the following neighbors of `0` (i.e. `1`, `2`, ...).
    np.testing.assert_array_equal(
        padded_f.data[1:-1, 1],
        np.roll(f.data[:, 0], 32),
        err_msg='lat_boundary_nearest',
    )
    np.testing.assert_array_equal(
        padded_f.data[1:-1, 0],
        np.roll(f.data[:, 1], 32),
        err_msg='lat_boundary_next_nearest',
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='odd_padding',
          grid=coordinates.LonLatGrid.TL31(),
          pad_sizes={'longitude': (1, 1), 'latitude': (1, 1)},
          expected_padded_shape=(66, 34),
      ),
      dict(
          testcase_name='even_padding',
          grid=coordinates.LonLatGrid.T21(),
          pad_sizes={'longitude': (2, 2), 'latitude': (2, 2)},
          expected_padded_shape=(68, 36),
      ),
      dict(
          testcase_name='mixed_padding',
          grid=coordinates.LonLatGrid.T21(),
          pad_sizes={'longitude': (1, 4), 'latitude': (2, 3)},
          expected_padded_shape=(69, 37),
      ),
  )
  def test_lon_lat_boundary_pad_and_trim(
      self,
      grid: coordinates.LonLatGrid,
      pad_sizes: dict[str, tuple[int, int]],
      expected_padded_shape: tuple[int, ...],
  ):
    bc = boundaries.LonLatBoundary()
    rng = jax.random.PRNGKey(0)
    data = jax.random.normal(rng, grid.shape)

    with self.subTest('on_field_inputs'):
      f = cx.wrap(data, grid)
      padded = bc.pad(f, pad_sizes)
      self.assertEqual(padded.shape, expected_padded_shape)
      trimmed = bc.trim(padded)
      self.assertEqual(trimmed.shape, grid.shape)
      np.testing.assert_array_equal(trimmed.data, f.data)

    with self.subTest('on_array_inputs'):
      pad_width = tuple(pad_sizes[d] for d in grid.dims)
      padded = bc.pad_array(data, pad_width)
      self.assertEqual(padded.shape, expected_padded_shape)
      trimmed = bc.trim_array(padded, pad_width)
      self.assertEqual(trimmed.shape, data.shape)
      np.testing.assert_array_equal(trimmed, data)


if __name__ == '__main__':
  jax.config.update('jax_traceback_filtering', 'off')
  absltest.main()
