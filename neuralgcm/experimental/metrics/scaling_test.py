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
import jax.numpy as jnp
import jax_datetime as jdt
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.metrics import scaling
import numpy as np


class CoordinateMaskScalerTest(parameterized.TestCase):

  def test_coordinate_mask_scaler(self):
    time_coord = coordinates.TimeDelta(
        np.array([0, 6, 12, 18]) * np.timedelta64(1, 'h')
    )
    field = cx.field(np.ones(time_coord.shape), time_coord)
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
    field = cx.field(np.ones(x.shape), x)
    mask_deltas = np.array([6, 18]) * np.timedelta64(1, 'h')
    mask_coord = coordinates.TimeDelta(mask_deltas)
    mask_scaler = scaling.CoordinateMaskScaler(mask_coord=mask_coord)

    context_time_match = {'timedelta': cx.field(np.timedelta64(6, 'h'))}
    scales_match = mask_scaler.scales(field, context=context_time_match)
    np.testing.assert_allclose(scales_match.data, 0.0)

    context_time_no_match = {'timedelta': cx.field(np.timedelta64(3, 'h'))}
    scales_no_match = mask_scaler.scales(field, context=context_time_no_match)
    np.testing.assert_allclose(scales_no_match.data, 1.0)

  def test_missing_dimension_raises_error(self):
    """Tests that a missing dimension raises a ValueError."""
    time_coord = coordinates.TimeDelta(
        np.array([0, 6, 12, 18]) * np.timedelta64(1, 'h')
    )
    field = cx.field(np.ones(time_coord.shape), time_coord)
    mask_coord = cx.SizedAxis('nondim', 2)
    with self.assertRaisesRegex(
        ValueError, "Coordinate for 'nondim' not found"
    ):
      scaler = scaling.CoordinateMaskScaler(
          mask_coord=mask_coord, skip_missing=False
      )
      scaler.scales(field)


class GeneralizedLeadTimeScalerTest(parameterized.TestCase):

  def test_normalization(self):
    time_coord = coordinates.TimeDelta(
        np.array([0, 3, 8, 15]) * np.timedelta64(1, 'h')
    )
    field = cx.field(np.ones(time_coord.shape), time_coord)
    scaler = scaling.GeneralizedLeadTimeScaler(base_squared_error_in_hours=1.0)
    scales = scaler.scales(field)
    np.testing.assert_allclose(scales.data.mean(), 1.0, atol=1e-6)

  def test_normalization_with_weights_power(self):
    time_coord = coordinates.TimeDelta(
        np.array([0, 6, 12, 24]) * np.timedelta64(1, 'h')
    )
    field = cx.field(np.ones(time_coord.shape), time_coord)
    scaler = scaling.GeneralizedLeadTimeScaler(
        base_squared_error_in_hours=4.0, weights_power=2.0
    )
    scales = scaler.scales(field)
    np.testing.assert_allclose(scales.data.mean(), 1.0, atol=1e-6)

  def test_normalization_in_context(self):
    x = cx.SizedAxis('x', 1)
    field = cx.field(np.ones(x.shape), x)
    scaler = scaling.GeneralizedLeadTimeScaler(base_squared_error_in_hours=7.0)
    step_delta = jdt.Timedelta.from_timedelta64(np.timedelta64(6, 'h'))
    n_steps = 4
    scaled_weights = []
    for i in range(n_steps):
      ctx = {
          'timedelta': cx.field(step_delta * i),
          'times': cx.field(step_delta * jnp.arange(n_steps)),
      }
      scaled_weights.append(scaler.scales(field, context=ctx).data)
    scaled_weights = np.array(scaled_weights)
    np.testing.assert_allclose(scaled_weights.mean(), 1.0, atol=1e-6)

  def test_asymptotic_normalization(self):
    time_coord = coordinates.TimeDelta(
        np.array([0, 6, 12, 18]) * np.timedelta64(1, 'h')
    )
    field = cx.field(np.ones(time_coord.shape), time_coord)

    # Case 1: max_t = 18. T_asymp = 18. Ratio = 1.
    # asymptotic_norm = 0.5.
    # expected_scale = (1 + 0.5 * 1) / (1 + 1) = 0.75
    scaler = scaling.GeneralizedLeadTimeScaler(
        base_squared_error_in_hours=1.0,
        asymptotic_squared_error_in_hours=240.0,
        asymptotic_norm=0.5,
        norm_transition_timescale_in_hours=18.0,
    )
    scales = scaler.scales(field)
    np.testing.assert_allclose(scales.data.mean(), 0.75, atol=1e-6)

    # Case 2: max_t >> T_asymp
    # max_t = 1800, T_asymp = 18. Ratio = 100.
    # asymptotic_norm = 0.5.
    # expected ~ (1 + 50) / 101 ~ 0.505
    time_coord_long = coordinates.TimeDelta(
        np.linspace(0, 1800, 100) * np.timedelta64(1, 'h')
    )
    field_long = cx.field(np.ones(time_coord_long.shape), time_coord_long)
    scales_long = scaler.scales(field_long)
    expected_long = (1 + 0.5 * (1800 / 18)) / (1 + 1800 / 18)
    np.testing.assert_allclose(
        scales_long.data.mean(), expected_long, atol=1e-6
    )

    # Case 3: max_t << T_asymp
    # max_t = 0.18. Ratio = 0.01.
    # expected ~ (1 + 0.005) / 1.01 ~ 0.995
    time_coord_short = coordinates.TimeDelta(
        np.array([0, 6]) * np.timedelta64(1, 'm')
    )
    field_short = cx.field(np.ones(time_coord_short.shape), time_coord_short)
    scales_short = scaler.scales(field_short)
    expected_short = (1 + 0.5 * (0.1 / 18)) / (1 + 0.1 / 18)
    np.testing.assert_allclose(
        scales_short.data.mean(), expected_short, atol=1e-6
    )

  def test_asymptotic_normalization_with_power(self):
    time_coord = coordinates.TimeDelta(
        np.array([0, 6, 12, 18]) * np.timedelta64(1, 'h')
    )
    field = cx.field(np.ones(time_coord.shape), time_coord)
    scaler = scaling.GeneralizedLeadTimeScaler(
        base_squared_error_in_hours=1.0,
        asymptotic_norm=0.5,
        norm_transition_power=2.0,
        norm_transition_timescale_in_hours=18.0,
    )
    scales = scaler.scales(field)
    # Ratio = (18/18)**2 = 1.
    # expected = (1 + 0.5 * 1) / (1 + 1) = 0.75
    np.testing.assert_allclose(scales.data.mean(), 0.75, atol=1e-6)

    # Check with max_t = 36. Ratio = (36/18)**2 = 4.
    # expected = (1 + 0.5 * 4) / (1 + 4) = 3 / 5 = 0.6
    time_coord_long = coordinates.TimeDelta(
        np.array([0, 36]) * np.timedelta64(1, 'h')
    )
    field_long = cx.field(np.ones(time_coord_long.shape), time_coord_long)
    scales_long = scaler.scales(field_long)
    np.testing.assert_allclose(scales_long.data.mean(), 0.6, atol=1e-6)

  def test_asymptotic_normalization_raises_error_when_missing_timescale(self):
    with self.assertRaisesRegex(
        ValueError, 'norm_transition_timescale_in_hours'
    ):
      f = cx.field(np.zeros(1), coordinates.TimeDelta([np.timedelta64(1, 'h')]))
      scaling.GeneralizedLeadTimeScaler(
          base_squared_error_in_hours=1.0, asymptotic_norm=0.5
      ).scales(f)


if __name__ == '__main__':
  absltest.main()
