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
from neuralgcm.experimental import xreader
from neuralgcm.experimental.training import data_loading
import numpy as np
import pandas as pd


class GetDatetimeForecastStartsTest(absltest.TestCase):

  def test_parity(self):
    candidates = pd.date_range(start='2020-01-01', end='2020-04-30', freq='12h')

    out = data_loading._get_datetime_forecast_starts(20, candidates)
    self.assertLen(out, 20)
    self.assertEqual(set(candidates).intersection(out), set(out))

    # Since the candidates are nice and regular, we should have exactly 10
    # samples at 0h and 12h
    out = pd.DatetimeIndex(out)
    self.assertLen(out[out.hour == 0], 10)
    self.assertLen(out[out.hour == 12], 10)

  def test_disjoint_candidates(self):
    # Create disjoint time ranges with 8-hour separation between items
    ranges = [
        pd.date_range(start='2020-01-01', end='2020-01-30', freq='8h'),
        pd.date_range(start='2020-06-01', end='2020-06-30', freq='8h'),
        pd.date_range(start='2020-10-01', end='2020-10-30', freq='8h'),
    ]
    candidates = pd.DatetimeIndex(pd.concat([pd.Series(r) for r in ranges]))
    out = data_loading._get_datetime_forecast_starts(6, candidates)
    self.assertLen(out, 6)
    self.assertEqual(set(candidates).intersection(out), set(out))

    # should have 2 candidates from each of the months
    expected = np.array(
        ['2020-01', '2020-01', '2020-06', '2020-06', '2020-10', '2020-10'],
        dtype='datetime64[M]',
    )
    np.testing.assert_array_equal(out.astype('datetime64[M]'), expected)


class GetSampleOriginsTest(absltest.TestCase):

  def test_whole_range(self):
    stencil = xreader.TimeStencil(
        start='0h', stop='6h', step='1h', closed='both'
    )
    time_axis = pd.date_range(start='2020-01-01', end='2020-01-10', freq='1h')

    out = data_loading._get_sample_origins(
        time_axis=time_axis,
        time_slices=None,
        stencil=stencil,
        stride_between_windows=1,
    )

    # Everything in time_axis other than the end should be a valid origin.
    expected = time_axis[:-6]

    np.testing.assert_array_equal(out, expected)

  def test_stride(self):
    stencil = xreader.TimeStencil(
        start='0h', stop='6h', step='1h', closed='both'
    )
    time_axis = pd.date_range(start='2020-01-01', end='2020-01-06', freq='1h')

    out = data_loading._get_sample_origins(
        time_axis=time_axis,
        time_slices=None,
        stencil=stencil,
        stride_between_windows=24,
    )

    # Everything in time_axis other than the end should be a valid origin.
    expected = np.array(
        ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05'],
        dtype='datetime64[ns]',
    )

    np.testing.assert_array_equal(out, expected)

  def test_one_small_subset(self):
    stencil = xreader.TimeStencil(
        start='0h', stop='6h', step='1h', closed='both'
    )
    time_axis = pd.date_range(start='2020-01-01', end='2020-01-10', freq='1h')

    out = data_loading._get_sample_origins(
        time_axis=time_axis,
        time_slices=('2020-01-03T00:00', '2020-01-03T10:00'),
        stencil=stencil,
        stride_between_windows=1,
    )

    expected = pd.date_range('2020-01-03T00:00', '2020-01-03T10:00', freq='1h')[
        :-6
    ]

    np.testing.assert_array_equal(out, expected)

  def test_one_subset(self):
    stencil = xreader.TimeStencil(
        start='0h', stop='6h', step='1h', closed='both'
    )
    time_axis = pd.date_range(start='2020-01-01', end='2020-01-10', freq='1h')

    out = data_loading._get_sample_origins(
        time_axis=time_axis,
        time_slices=('2020-01-03', '2020-01-05'),  # includes whole final day
        stencil=stencil,
        stride_between_windows=1,
    )

    expected = pd.date_range('2020-01-03 00:00', '2020-01-05 23:00', freq='1h')[
        :-6
    ]

    np.testing.assert_array_equal(out, expected)

  def test_multiple_subsets(self):
    stencil = xreader.TimeStencil(
        start='0h', stop='6h', step='1h', closed='both'
    )
    time_axis = pd.date_range(start='2020-01-01', end='2020-03-31', freq='1h')

    out = data_loading._get_sample_origins(
        time_axis=time_axis,
        time_slices=[
            ('2020-01-03', '2020-01-05'),
            ('2020-02-03', '2020-02-05'),
            ('2020-03-03', '2020-03-05'),
        ],
        stencil=stencil,
        stride_between_windows=1,
    )

    ranges = [
        pd.date_range('2020-01-03 00:00', '2020-01-05 23:00', freq='1h')[:-6],
        pd.date_range('2020-02-03 00:00', '2020-02-05 23:00', freq='1h')[:-6],
        pd.date_range('2020-03-03 00:00', '2020-03-05 23:00', freq='1h')[:-6],
    ]
    expected = pd.DatetimeIndex(pd.concat([pd.Series(r) for r in ranges]))

    np.testing.assert_array_equal(out, expected)


if __name__ == '__main__':
  absltest.main()
