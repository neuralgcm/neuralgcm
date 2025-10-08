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

import functools
import operator
import os
import pathlib

from absl.testing import absltest
from absl.testing import parameterized
import coordax as cx
from fiddle.experimental import auto_config
from flax import nnx
import jax
from neuralgcm.experimental.core import api
from neuralgcm.experimental.core import dynamic_io
from neuralgcm.experimental.core import nnx_compat
from neuralgcm.experimental.core import random_processes
from neuralgcm.experimental.core import typing
from neuralgcm.experimental.inference import dynamic_inputs as dynamic_inputs_lib
from neuralgcm.experimental.inference import runner as runnerlib
import numpy as np
import xarray


@nnx_compat.dataclass
class MockModel(api.Model):
  """A mock Model for testing."""

  input_specs: dict[str, dict[str, cx.Coordinate]]
  dynamic_input_specs: dict[str, dict[str, cx.Coordinate]]
  dynamic_input_slice: dynamic_io.DynamicInputSlice
  assimilation_noise: random_processes.RandomProcessModule | None

  def __post_init__(self):
    self.prognostics = typing.Prognostic({
        k: cx.wrap(np.zeros(c.shape), c)
        for k, c in self.required_input_specs['state'].items()
    })

  @property
  def required_input_specs(self) -> dict[str, dict[str, cx.Coordinate]]:
    return self.input_specs

  @property
  def required_dynamic_input_specs(
      self,
  ) -> dict[str, dict[str, cx.Coordinate]]:
    return self.dynamic_input_specs

  @property
  def timestep(self) -> np.timedelta64:
    return np.timedelta64(1, 'h')

  def assimilate(self, observations: typing.Observation) -> None:
    state = jax.tree.map(
        # TODO(shoyer): create a .isel() method on Field?
        cx.cmap(lambda x: x[-1]),
        cx.untag(observations['state'], 'timedelta'),
        is_leaf=cx.is_field,
    )
    if self.assimilation_noise is not None:
      noise = self.assimilation_noise.state_values()
      state['foo'] = state['foo'] + noise
    self.prognostics.value = state

  def advance(self) -> None:
    prognostics = self.prognostics.value
    time = prognostics.pop('time')
    sliced_inputs = self.dynamic_input_slice(time)
    next_prognostics = jax.tree.map(
        operator.add, prognostics, sliced_inputs, is_leaf=cx.is_field
    )
    next_prognostics['time'] = time + self.timestep
    self.prognostics.value = next_prognostics

  def observe(self, query: typing.Query) -> typing.Observation:
    prognostics = self.prognostics.value
    return {
        'state': {k: v for k, v in prognostics.items() if k in query['state']}
    }


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

    @auto_config.auto_config
    def construct_model() -> MockModel:
      return MockModel(
          input_specs={
              'state': {
                  'foo': cx.Scalar(),
                  'bar': cx.LabeledAxis('x', np.array([0.1, 0.2, 0.3])),
                  'time': cx.Scalar(),
              }
          },
          dynamic_input_specs={
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

    module_model = construct_model()
    model = api.InferenceModel.from_model_api(
        module_model, construct_model.as_buildable()
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
    zarr_chunks = {'lead_time': 4, 'init_time': 1}
    if ensemble_size is not None:
      zarr_chunks['realization'] = 1
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
        zarr_chunks=zarr_chunks,
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
