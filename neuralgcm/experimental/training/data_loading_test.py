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
import coordax as cx
import jax.numpy as jnp
from neuralgcm.experimental import xreader
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import data_specs
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


class SelTimedeltaTest(absltest.TestCase):

  def _make_field(self, hours):
    deltas = np.array(hours, dtype='timedelta64[h]')
    td_axis = coordinates.TimeDelta(deltas)
    # Create data matching the length of the time axis
    data = jnp.arange(len(hours))
    return cx.field(data, td_axis)

  def test_select_all_values(self):
    field = self._make_field([-1, 0, 1])
    # Select all using None slice
    result = data_loading.sel_timedelta_fields(
        {'f': field}, values=slice(None, None)
    )
    self.assertEqual(
        result['f'].axes['timedelta'].deltas.tolist(),
        [np.timedelta64(h, 'h') for h in [-1, 0, 1]],
    )

  def test_select_range_subset(self):
    field = self._make_field([-2, -1, 0, 1, 2])
    # Select range [-1, 1]. Note that this implementation is inclusive for the
    # stop value if it exists in the array because of
    # searchsorted(side='right').
    result = data_loading.sel_timedelta_fields(
        {'f': field},
        values=slice(np.timedelta64(-1, 'h'), np.timedelta64(1, 'h')),
    )
    self.assertEqual(
        result['f'].axes['timedelta'].deltas.tolist(),
        [np.timedelta64(h, 'h') for h in [-1, 0, 1]],
    )

  def test_select_empty_range(self):
    field = self._make_field([-1, 0, 1])
    # Select range [2, 3] -> should be empty
    result = data_loading.sel_timedelta_fields(
        {'f': field},
        values=slice(np.timedelta64(2, 'h'), np.timedelta64(3, 'h')),
    )
    self.assertEmpty(result['f'].axes['timedelta'].deltas)
    self.assertEmpty(result['f'].data)

  def test_select_single_value(self):
    field = self._make_field([-1, 0, 1])
    result = data_loading.sel_timedelta_fields(
        {'f': field}, values=np.timedelta64(0, 'h')
    )
    self.assertEqual(
        result['f'].axes['timedelta'].deltas.tolist(), [np.timedelta64(0, 'h')]
    )

  def test_select_missing_value_raises_key_error(self):
    field = self._make_field([-1, 0, 1])
    with self.assertRaisesRegex(KeyError, 'Value .* not found'):
      data_loading.sel_timedelta_fields(
          {'f': field}, values=np.timedelta64(2, 'h')
      )


class SelTimedeltaCoordsTest(absltest.TestCase):

  def test_filters_timedelta_coords(self):
    deltas = np.array([-1, 0, 1], dtype='timedelta64[h]')
    td_coord = coordinates.TimeDelta(deltas)
    x = cx.SizedAxis('x', 5)
    combined_coord = cx.coords.compose(td_coord, x)

    with self.subTest('single_value'):
      result = data_loading.sel_timedelta_coords(
          combined_coord, values=np.timedelta64(0, 'h')
      )
      actual_td = cx.coords.extract(result, coordinates.TimeDelta)
      expected = coordinates.TimeDelta(np.array([0], dtype='timedelta64[h]'))
      self.assertEqual(actual_td, expected)
      actual_x = cx.coords.extract(result, cx.SizedAxis)
      self.assertEqual(actual_x, x)

    with self.subTest('slice_value'):
      result = data_loading.sel_timedelta_coords(
          combined_coord, values=slice(np.timedelta64(0, 'h'), None)
      )
      actual_td = cx.coords.extract(result, coordinates.TimeDelta)
      expected = coordinates.TimeDelta(np.array([0, 1], dtype='timedelta64[h]'))
      self.assertEqual(actual_td, expected)
      actual_x = cx.coords.extract(result, cx.SizedAxis)
      self.assertEqual(actual_x, x)


class InferStencilsTest(absltest.TestCase):

  def _make_spec(self, hours):
    deltas = np.array(hours, dtype='timedelta64[h]')
    return coordinates.TimeDelta(deltas)

  def test_single_dataset_single_variable(self):
    spec = {'ds_a': {'var_a': self._make_spec([0, 6, 12])}}
    stencils = data_loading.infer_stencils(spec)
    expected = xreader.TimeStencil(
        start='0h', stop='12h', step='6h', closed='both'
    )
    self.assertEqual(stencils['ds_a'], expected)

  def test_multiple_datasets(self):
    spec = {
        'ds_a': {'var_a': self._make_spec([0, 24])},
        'ds_b': {'var_b': self._make_spec([0, 1])},
    }
    stencils = data_loading.infer_stencils(spec)
    self.assertEqual(
        stencils['ds_a'],
        xreader.TimeStencil(start='0h', stop='24h', step='24h', closed='both'),
    )
    self.assertEqual(
        stencils['ds_b'],
        xreader.TimeStencil(start='0h', stop='1h', step='1h', closed='both'),
    )

  def test_consistent_variables_in_dataset(self):
    coord = self._make_spec([0, 6])
    spec = {'ds_a': {'var_a': coord, 'var_b': coord}}
    stencils = data_loading.infer_stencils(spec)
    expected = xreader.TimeStencil(
        start='0h', stop='6h', step='6h', closed='both'
    )
    self.assertEqual(stencils['ds_a'], expected)

  def test_inconsistent_variables_raises_error(self):
    spec = {
        'ds_a': {
            'var_a': self._make_spec([0, 6]),
            'var_b': self._make_spec([0, 12]),
        }
    }
    with self.assertRaisesRegex(
        ValueError, 'Expected exactly 1 unique stencil'
    ):
      data_loading.infer_stencils(spec)

  def test_non_uniform_steps_raises_error(self):
    spec = {'ds_a': {'var_a': self._make_spec([0, 6, 10])}}
    with self.assertRaisesRegex(
        ValueError,
        'TimeDelta must be uniformly spaced to convert to TimeStencil',
    ):
      data_loading.infer_stencils(spec)

  def test_missing_entire_dataset_with_optional_specs(self):
    specs = {'ds_a': {'var_a': data_specs.OptionalSpec(cx.Scalar())}}
    all_data = {}
    result = data_loading.filter_missing_optional(specs, all_data)
    self.assertEqual(result, {})

  def test_empty_timedelta_raises_error(self):
    spec = {'ds_a': {'var_a': self._make_spec([])}}
    with self.assertRaisesRegex(ValueError, 'TimeDelta must be of size >= 2'):
      data_loading.infer_stencils(spec)


if __name__ == '__main__':
  absltest.main()
