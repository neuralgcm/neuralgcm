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

from absl.testing import absltest
import coordax as cx
import jax
from neuralgcm.experimental.core import api
from neuralgcm.experimental.core import typing
from neuralgcm.experimental.core import xarray_utils
from neuralgcm.experimental.inference import runner as runnerlib
import numpy as np
import xarray


@dataclasses.dataclass
class MockForecastSystem(api.ForecastSystem):
  required_input_specs: dict[str, dict[str, cx.Coordinate]]

  @property
  def timestep(self) -> np.timedelta64:
    return np.timedelta64(1, "h")

  def assimilate_prognostics(
      self,
      observations: typing.Observation,
      dynamic_inputs: typing.Observation | None = None,
      rng: typing.PRNGKeyArray | None = None,
      initial_state: typing.ModelState | None = None,
  ) -> typing.Prognostics:
    return jax.tree.map(
        # TODO(shoyer): create a .isel() method on Field?
        cx.cmap(lambda x: x[-1]),
        cx.untag(observations["state"], "timedelta"),
        is_leaf=cx.is_field,
    )

  def advance_prognostics(
      self, prognostics: typing.Prognostics
  ) -> typing.Prognostics:
    return jax.tree.map(lambda x: x + 1, prognostics)

  def observe_from_prognostics(
      self,
      prognostics: typing.Prognostics,
      query: typing.Query,
  ) -> typing.Observation:
    del query  # ignored for now
    return {"state": prognostics}

  # TODO(shoyer): Move these methods upstream onto the base api.ForecastSystem?

  def observations_from_xarray(
      self,
      nested_data: dict[str, xarray.Dataset],
  ) -> dict[str, dict[str, cx.Field]]:
    nested_data = xarray_utils.ensure_timedelta_axis(nested_data)
    return xarray_utils.read_fields_from_xarray(
        nested_data, self.required_input_specs
    )

  def data_to_xarray(
      self,
      data: dict[str, dict[str, cx.Field]],
  ) -> dict[str, xarray.Dataset]:
    return {k: xarray_utils.fields_to_xarray(v) for k, v in data.items()}


class RunnerTest(absltest.TestCase):

  def test_inference_runner_setup(self):
    output_path = self.create_tempdir().full_path
    init_times = np.array(
        [np.datetime64("2025-01-01"), np.datetime64("2025-01-02")]
    )
    runner = runnerlib.InferenceRunner(
        model=MockForecastSystem(
            required_input_specs={
                "state": {
                    "foo": cx.Scalar(),
                    "bar": cx.LabeledAxis("x", np.array([0.1, 0.2, 0.3])),
                }
            }
        ),
        inputs={
            "state": xarray.Dataset(
                {
                    "foo": (("time",), np.array([0.0, 10.0])),
                    "bar": (("time", "x"), np.array(2 * [[1.0, 2.0, 3.0]])),
                },
                coords={"time": init_times, "x": np.array([0.1, 0.2, 0.3])},
            )
        },
        dynamic_inputs={},
        init_times=init_times,
        ensemble_size=None,
        output_path=output_path,
        output_query={
            "state": {
                "foo": cx.Scalar(),
                "bar": cx.LabeledAxis("x", np.array([0.1, 0.2, 0.3])),
            }
        },
        output_freq=np.timedelta64(6, "h"),
        output_duration=np.timedelta64(1, "D"),
        output_chunks={"lead_time": 4, "init_time": 1},
    )
    runner.setup()
    self.assertEqual(runner.task_count, 2)

    expected_lead_times = np.arange(0, 24, 6) * np.timedelta64(1, "h")
    nans = functools.partial(np.full, fill_value=np.nan)
    root_node = xarray.Dataset(
        coords={
            "init_time": init_times.astype("datetime64[ns]"),
            "lead_time": expected_lead_times.astype("timedelta64[ns]"),
        },
    )
    child_node = xarray.Dataset(
        {
            "foo": (("init_time", "lead_time"), nans((2, 4))),
            "bar": (("init_time", "lead_time", "x"), nans((2, 4, 3))),
        },
        coords={"x": np.array([0.1, 0.2, 0.3])},
    )
    expected = xarray.DataTree.from_dict({"/": root_node, "/state": child_node})
    actual = xarray.open_datatree(output_path, engine="zarr")
    xarray.testing.assert_equal(actual, expected)

    for task_id in range(runner.task_count):
      runner.run(task_id)

    h = np.array([0.0, 6.0, 12.0, 18.0])
    expected_foo = np.stack([h, 10 + h])
    expected_bar = np.stack(2 * [np.stack([h + 1, h + 2, h + 3], axis=1)])
    child_node = xarray.Dataset(
        {
            "foo": (("init_time", "lead_time"), expected_foo),
            "bar": (("init_time", "lead_time", "x"), expected_bar),
        },
        coords={"x": np.array([0.1, 0.2, 0.3])},
    )
    expected = xarray.DataTree.from_dict({"/": root_node, "/state": child_node})
    actual = xarray.open_datatree(output_path, engine="zarr")
    xarray.testing.assert_equal(actual, expected)


if __name__ == "__main__":
  absltest.main()
