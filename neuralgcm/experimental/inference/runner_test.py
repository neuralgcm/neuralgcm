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

import dataclasses
import functools
import operator
import os
import pathlib

from absl.testing import absltest
from absl.testing import parameterized
import coordax as cx
from flax import nnx
import jax
from neuralgcm.experimental.core import api
from neuralgcm.experimental.core import dynamic_io
from neuralgcm.experimental.core import random_processes
from neuralgcm.experimental.core import typing
from neuralgcm.experimental.core import xarray_utils
from neuralgcm.experimental.inference import dynamic_inputs as dynamic_inputs_lib
from neuralgcm.experimental.inference import runner as runnerlib
import numpy as np
import xarray


@dataclasses.dataclass
class MockForecastSystem(api.ForecastSystem):
  required_input_specs: dict[str, dict[str, cx.Coordinate]]
  required_dynamic_input_specs: dict[str, dict[str, cx.Coordinate]]
  dynamic_input_slice: dynamic_io.DynamicInputSlice
  assimilation_noise: random_processes.RandomProcessModule | None = None

  @property
  def timestep(self) -> np.timedelta64:
    return np.timedelta64(1, 'h')

  def assimilate_prognostics(
      self,
      observations: typing.Observation,
      dynamic_inputs: typing.Observation | None = None,
      rng: typing.PRNGKeyArray | None = None,
      initial_state: typing.ModelState | None = None,
  ) -> typing.Prognostics:
    state = jax.tree.map(
        # TODO(shoyer): create a .isel() method on Field?
        cx.cmap(lambda x: x[-1]),
        cx.untag(observations['state'], 'timedelta'),
        is_leaf=cx.is_field,
    )
    if self.assimilation_noise is not None:
      noise = self.assimilation_noise.state_values()
      state['foo'] = state['foo'] + noise
    return state

  def advance_prognostics(
      self, prognostics: typing.Prognostics
  ) -> typing.Prognostics:
    prognostics = prognostics.copy()
    time = prognostics.pop('time')
    sliced_inputs = self.dynamic_input_slice(time)
    next_prognostics = jax.tree.map(
        operator.add, prognostics, sliced_inputs, is_leaf=cx.is_field
    )
    next_prognostics['time'] = time + self.timestep
    return next_prognostics

  def observe_from_prognostics(
      self, prognostics: typing.Prognostics, query: typing.Query
  ) -> typing.Observation:
    return {
        'state': {k: v for k, v in prognostics.items() if k in query['state']}
    }

  # TODO(shoyer): Move these methods upstream onto the base api.ForecastSystem?

  def inputs_from_xarray(
      self,
      nested_data: dict[str, xarray.Dataset],
  ) -> dict[str, dict[str, cx.Field]]:
    nested_data = xarray_utils.ensure_timedelta_axis(nested_data)
    return xarray_utils.read_fields_from_xarray(
        nested_data, self.required_input_specs
    )

  def dynamic_inputs_from_xarray(
      self,
      nested_data: dict[str, xarray.Dataset],
  ) -> dict[str, dict[str, cx.Field]]:
    if not self.required_dynamic_input_specs:
      return {}
    nested_data = xarray_utils.ensure_timedelta_axis(nested_data)
    return xarray_utils.read_fields_from_xarray(
        nested_data, self.required_dynamic_input_specs
    )

  def data_to_xarray(
      self, data: dict[str, dict[str, cx.Field]]
  ) -> dict[str, xarray.Dataset]:
    return {k: xarray_utils.fields_to_xarray(v) for k, v in data.items()}


class RunnerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='deterministic', ensemble_size=None),
      dict(testcase_name='ensemble', ensemble_size=2),
  )
  def test_inference_runner(self, ensemble_size):
    if ensemble_size is not None:
      assimilation_noise = random_processes.UniformUncorrelated(
          coords=cx.Scalar(), minval=-0.1, maxval=0.1, rngs=nnx.Rngs(0)
      )
    else:
      assimilation_noise = None
    model = MockForecastSystem(
        required_input_specs={
            'state': {
                'foo': cx.Scalar(),
                'bar': cx.LabeledAxis('x', np.array([0.1, 0.2, 0.3])),
                'time': cx.Scalar(),
            }
        },
        required_dynamic_input_specs={
            'data': {
                'foo': cx.Scalar(),
                'bar': cx.Scalar(),
                'time': cx.Scalar(),
            }
        },
        dynamic_input_slice=dynamic_io.DynamicInputSlice(
            keys_to_coords={'foo': cx.Scalar(), 'bar': cx.Scalar()},
            observation_key='data',
        ),
        assimilation_noise=assimilation_noise,
    )
    init_times = np.array(
        [np.datetime64('2025-01-01'), np.datetime64('2025-01-02')]
    )
    inputs = {
        'state': xarray.Dataset(
            {
                'foo': (('time',), np.array([0.0, 10.0])),
                'bar': (('time', 'x'), np.array(2 * [[1.0, 2.0, 3.0]])),
            },
            coords={'time': init_times, 'x': np.array([0.1, 0.2, 0.3])},
        )
    }
    delta = xarray.Dataset({'foo': 1.0, 'bar': 2.0})
    dynamic_inputs = dynamic_inputs_lib.Persistence(
        full_data={'data': delta.expand_dims(time=init_times)},
        climatology=None,
        update_freq=np.timedelta64(6, 'h'),
    )
    output_path = self.create_tempdir().full_path
    output_query = {
        'state': {
            'foo': cx.Scalar(),
            'bar': cx.LabeledAxis('x', np.array([0.1, 0.2, 0.3])),
        }
    }
    output_chunks = {'lead_time': 4, 'init_time': 1}
    if ensemble_size is not None:
      output_chunks['realization'] = 1
    runner = runnerlib.InferenceRunner(
        model=model,
        inputs=inputs,
        dynamic_inputs=dynamic_inputs,
        init_times=init_times,
        ensemble_size=ensemble_size,
        output_path=output_path,
        output_query=output_query,
        output_freq=np.timedelta64(6, 'h'),
        output_duration=np.timedelta64(48, 'h'),
        output_chunks=output_chunks,
        write_duration=np.timedelta64(24, 'h'),
        unroll_duration=np.timedelta64(12, 'h'),
        checkpoint_duration=np.timedelta64(24, 'h'),
    )
    runner.setup()

    ensemble_count = 1 if ensemble_size is None else ensemble_size
    self.assertEqual(runner.task_count, len(init_times) * ensemble_count)

    expected_lead_times = np.arange(0, 48, 6) * np.timedelta64(1, 'h')
    nans = functools.partial(np.full, fill_value=np.nan)
    coords = {
        'init_time': init_times.astype('datetime64[ns]'),
        'lead_time': expected_lead_times.astype('timedelta64[ns]'),
    }
    dims = ('init_time', 'lead_time')
    shape = (len(init_times), len(expected_lead_times))
    if ensemble_size is not None:
      coords['realization'] = np.arange(ensemble_size)
      dims = ('realization',) + dims
      shape = (ensemble_size,) + shape

    root_node = xarray.Dataset(coords=coords)
    child_node = xarray.Dataset(
        {
            'foo': (dims, nans(shape)),
            'bar': (dims + ('x',), nans(shape + (3,))),
        },
        coords={'x': np.array([0.1, 0.2, 0.3])},
    )
    expected = xarray.DataTree.from_dict({'/': root_node, '/state': child_node})
    actual = xarray.open_datatree(output_path, engine='zarr')
    xarray.testing.assert_equal(actual, expected)

    for task_id in range(runner.task_count):
      runner.run(task_id)

    h = np.array([0.0, 6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0])
    expected_foo = np.stack([h, 10 + h])
    expected_bar = np.stack(
        2 * [np.stack([2 * h + 1, 2 * h + 2, 2 * h + 3], axis=1)]
    )
    if ensemble_size is not None:
      expected_foo = np.stack([expected_foo] * ensemble_size, axis=0)
      expected_bar = np.stack([expected_bar] * ensemble_size, axis=0)

    child_node = xarray.Dataset(
        {
            'foo': (dims, expected_foo),
            'bar': (dims + ('x',), expected_bar),
        },
        coords={'x': np.array([0.1, 0.2, 0.3])},
    )
    expected = xarray.DataTree.from_dict({'/': root_node, '/state': child_node})
    actual = xarray.open_datatree(output_path, engine='zarr')
    # round() removes initialization noise, which is between -0.1 and 0.1
    xarray.testing.assert_equal(actual.round(), expected)

    if ensemble_size is not None:
      # different ensemble members have different RNGs
      actual_foo = actual.state.foo.isel(init_time=0)
      first_realization = actual_foo.sel(realization=0)
      second_realization = actual_foo.sel(realization=1)
      self.assertFalse(
          np.allclose(first_realization, second_realization, atol=0.001),
          msg=f'{first_realization=}, {second_realization=}',
      )
      # different initialization also have different RNGs
      second_init = actual.state.foo.isel(init_time=1).sel(realization=0)
      self.assertFalse(
          np.allclose(first_realization, second_init, atol=0.001),
          msg=f'{first_realization=}, {second_init=}',
      )


class AtomicWriteTest(absltest.TestCase):

  def test_successful(self):
    path = pathlib.Path(self.create_tempdir()) / 'data'
    with runnerlib._atomic_write(path) as f:
      f.write(b'abc')
    self.assertTrue(path.exists())
    self.assertEqual(path.read_bytes(), b'abc')
    self.assertEqual(os.listdir(path.parent), ['data'])  # no temp files

  def test_incomplete(self):
    path = pathlib.Path(self.create_tempdir()) / 'data'
    with self.assertRaises(RuntimeError):
      with runnerlib._atomic_write(path) as f:
        f.write(b'a')
        raise RuntimeError
    self.assertFalse(path.exists())
    self.assertEqual(os.listdir(path.parent), [])  # no temp files


if __name__ == '__main__':
  absltest.main()
