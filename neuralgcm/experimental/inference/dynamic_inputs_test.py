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
from neuralgcm.experimental.inference import dynamic_inputs as dynamic_inputs_lib
import numpy as np
import xarray


def create_climatology(dayofyear=None, data=None):
  if dayofyear is None:
    dayofyear = 1 + np.arange(366)
  if data is None:
    data = np.arange(dayofyear.size)
  return xarray.Dataset(
      {'foo': ('dayofyear', data)}, coords={'dayofyear': dayofyear}
  )


class EmptyDynamicInputsTest(absltest.TestCase):

  def test_forecast(self):
    t0 = np.datetime64('2025-01-01T00')
    forecast = dynamic_inputs_lib.EmptyDynamicInputs().get_forecast(t0)
    actual = forecast.get_data(
        lead_start=np.timedelta64(0, 'h'), lead_stop=np.timedelta64(60, 'h')
    )
    expected = {}
    self.assertEqual(expected, actual)


class PersistenceTest(absltest.TestCase):

  def test_forecast(self):
    t0 = np.datetime64('2025-01-01T00')
    initial_ds = xarray.Dataset(
        {'foo': ('time', np.array([10.0]))},
        coords={'time': np.array([t0])},
    )
    forecast = dynamic_inputs_lib.Persistence(
        full_data=initial_ds,
        climatology=None,
        update_freq=np.timedelta64(12, 'h'),
    ).get_forecast(t0)

    actual = forecast.get_data(
        lead_start=np.timedelta64(0, 'h'), lead_stop=np.timedelta64(60, 'h')
    )
    expected_deltas = np.arange(5) * np.timedelta64(12, 'h')
    expected = xarray.Dataset(
        {'foo': ('time', np.array([10.0] * 5))},
        coords={'time': t0 + expected_deltas},
    )
    xarray.testing.assert_equal(expected, actual)

  def test_missing_full_data(self):
    with self.assertRaisesRegex(TypeError, 'full_data is required'):
      dynamic_inputs_lib.Persistence(
          full_data=None, climatology=None, update_freq=np.timedelta64(1, 'h')
      )


class PrescribedTest(absltest.TestCase):

  def test_forecast(self):
    t0 = np.datetime64('2025-01-01T00')
    times = t0 + np.arange(10) * np.timedelta64(12, 'h')
    full_ds = xarray.Dataset(
        {'foo': ('time', np.arange(10.0))},
        coords={'time': times},
    )
    dynamic_inputs = dynamic_inputs_lib.Prescribed(
        full_data=full_ds,
        climatology=None,
        update_freq=np.timedelta64(12, 'h'),
    )

    forecast = dynamic_inputs.get_forecast(np.datetime64('2025-01-01'))
    actual = forecast.get_data(np.timedelta64(0, 'D'), np.timedelta64(2, 'D'))
    expected = xarray.Dataset(
        {'foo': ('time', np.array([0.0, 1.0, 2.0, 3.0]))},
        coords={
            'time': np.array(
                ['2025-01-01', '2025-01-01T12', '2025-01-02', '2025-01-02T12'],
                dtype=np.datetime64,
            )
        },
    )
    xarray.testing.assert_equal(expected, actual)

    # non-default init_time
    forecast = dynamic_inputs.get_forecast(np.datetime64('2025-01-03'))
    actual = forecast.get_data(np.timedelta64(0, 'D'), np.timedelta64(1, 'D'))
    expected = xarray.Dataset(
        {'foo': ('time', np.array([4.0, 5.0]))},
        coords={
            'time': np.array(
                ['2025-01-03', '2025-01-03T12'], dtype=np.datetime64
            )
        },
    )
    xarray.testing.assert_equal(expected, actual)

  def test_missing_full_data(self):
    with self.assertRaisesRegex(TypeError, 'full_data is required'):
      dynamic_inputs_lib.Prescribed(
          full_data=None, climatology=None, update_freq=np.timedelta64(1, 'h')
      )


class ClimatologyTest(absltest.TestCase):

  def test_forecast(self):
    t0 = np.datetime64('2025-01-01T00')
    forecast = dynamic_inputs_lib.Climatology(
        full_data=None,
        climatology=create_climatology(data=np.arange(366.0)),
        update_freq=np.timedelta64(12, 'h'),
    ).get_forecast(t0)

    actual = forecast.get_data(
        lead_start=np.timedelta64(0, 'h'), lead_stop=np.timedelta64(60, 'h')
    )
    # dayofyear starts from 1, so indices are 0, 0, 1, 1, 2
    expected_data = np.array([0, 0, 1, 1, 2])
    expected_deltas = np.arange(5) * np.timedelta64(12, 'h')
    expected = xarray.Dataset(
        {'foo': ('time', expected_data)}, coords={'time': t0 + expected_deltas}
    )
    xarray.testing.assert_equal(expected, actual)

  def test_missing_climatology(self):
    with self.assertRaisesRegex(TypeError, 'climatology is required'):
      dynamic_inputs_lib.Climatology(
          full_data=xarray.Dataset(),
          climatology=None,
          update_freq=np.timedelta64(1, 'h'),
      )


class AnomalyPersistenceTest(absltest.TestCase):

  def test_forecast(self):
    t0 = np.datetime64('2025-01-01T00')
    forecast = dynamic_inputs_lib.AnomalyPersistence(
        full_data=xarray.Dataset(
            {'foo': ('time', np.array([10.0]))},
            coords={'time': np.array([t0])},
        ),
        climatology=create_climatology(data=np.arange(366.0)),
        update_freq=np.timedelta64(12, 'h'),
    ).get_forecast(t0)

    with self.subTest('short_forecast'):
      actual = forecast.get_data(
          lead_start=np.timedelta64(0, 'h'), lead_stop=np.timedelta64(60, 'h')
      )
      # initial anomaly: 10.0 - 0.0 = 10.0
      # dayofyear indices: 0, 0, 1, 1, 2
      # climatology values: 0.0, 0.0, 1.0, 1.0, 2.0
      expected_data = 10.0 + np.array([0.0, 0.0, 1.0, 1.0, 2.0])
      expected_deltas = np.arange(5) * np.timedelta64(12, 'h')
      expected = xarray.Dataset(
          {'foo': ('time', expected_data)},
          coords={'time': t0 + expected_deltas},
      )
      xarray.testing.assert_equal(expected, actual)

    with self.subTest('long_forecast'):
      forecast = dynamic_inputs_lib.AnomalyPersistence(
          full_data=xarray.Dataset(
              {'foo': ('time', np.array([10.0]))},
              coords={'time': np.array([t0])},
          ),
          climatology=create_climatology(data=np.arange(366.0)),
          update_freq=np.timedelta64(24, 'h'),
      ).get_forecast(t0)
      actual = forecast.get_data(
          lead_start=np.timedelta64(0, 'h'),
          lead_stop=np.timedelta64(1000 * 24, 'h'),
      )
      days = np.arange(1000)
      expected_data = 10.0 + days % 365
      expected_deltas = np.arange(1000) * np.timedelta64(24, 'h')
      expected = xarray.Dataset(
          {'foo': ('time', expected_data)},
          coords={'time': t0 + expected_deltas},
      )
      xarray.testing.assert_allclose(expected, actual)

  def test_missing_full_data(self):
    with self.assertRaisesRegex(TypeError, 'full_data is required'):
      dynamic_inputs_lib.AnomalyPersistence(
          full_data=None,
          climatology=create_climatology(data=np.arange(366.0)),
          update_freq=np.timedelta64(1, 'h'),
      )

  def test_missing_climatology(self):
    with self.assertRaisesRegex(TypeError, 'climatology is required'):
      dynamic_inputs_lib.AnomalyPersistence(
          full_data=xarray.Dataset(),
          climatology=None,
          update_freq=np.timedelta64(1, 'h'),
      )


class InvalidClimatologyTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='_missing_dayofyear_coord',
          climatology=xarray.Dataset({'foo': ('time', np.arange(366))}),
          error_regex="climatology does not have a 'dayofyear' coordinate",
      ),
      dict(
          testcase_name='_incomplete_dayofyear',
          climatology=create_climatology(dayofyear=np.arange(365)),
          error_regex='dayofyear coordinate must include 1 through 366',
      ),
      dict(
          testcase_name='_extra_dayofyear',
          climatology=create_climatology(dayofyear=np.arange(367)),
          error_regex='dayofyear coordinate must include 1 through 366',
      ),
      dict(
          testcase_name='_dayofyear_not_starting_at_1',
          climatology=create_climatology(dayofyear=np.arange(366)),
          error_regex='dayofyear coordinate must include 1 through 366',
      ),
  )
  def test_invalid(self, climatology, error_regex):
    test_classes = [
        dynamic_inputs_lib.Climatology,
        dynamic_inputs_lib.AnomalyPersistence,
    ]
    for cls in test_classes:
      with self.subTest(cls.__name__):
        with self.assertRaisesRegex(ValueError, error_regex):
          cls(
              full_data=xarray.Dataset(),
              climatology=climatology,
              update_freq=np.timedelta64(1, 'h'),
          )


if __name__ == '__main__':
  absltest.main()
