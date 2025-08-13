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
import jax
from neuralgcm.experimental.core import api
from neuralgcm.experimental.core import typing
from neuralgcm.experimental.inference import runner as runnerlib
import numpy as np
import xarray


class MockForecastSystem(api.ForecastSystem):

  def assimilate_prognostics(
      self,
      observations: typing.Observation,
      dynamic_inputs: typing.Observation | None = None,
      rng: typing.PRNGKeyArray | None = None,
      initial_state: typing.ModelState | None = None,
  ) -> typing.Prognostics:
    return observations["state"]

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


class RunnerTest(absltest.TestCase):

  def test_inference_runner_setup(self):
    output_path = self.create_tempdir().full_path
    init_times = np.arange(
        np.datetime64("2025-01-01"),
        np.datetime64("2025-01-04"),
        np.timedelta64(1, "D"),
    )
    runner = runnerlib.InferenceRunner(
        model=MockForecastSystem(),
        inputs={
            "state": xarray.Dataset(
                {"foo": 0.0, "bar": ("x", np.array([1.0, 2.0, 3.0]))},
                coords={"x": np.array([0.1, 0.2, 0.3])},
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
        output_duration=np.timedelta64(10, "D"),
        output_chunks={"lead_time": 4, "init_time": 1},
    )
    runner.setup()

    tree = xarray.open_datatree(output_path, engine="zarr")
    actual_coords = tree["state"].coords.to_dataset()
    expected_lead_times = np.arange(0, 240, 6) * np.timedelta64(1, "h")
    expected_coords = xarray.Dataset(
        coords={
            "init_time": init_times.astype("datetime64[ns]"),
            "lead_time": expected_lead_times.astype("timedelta64[ns]"),
            "x": np.array([0.1, 0.2, 0.3]),
        }
    )
    xarray.testing.assert_equal(actual_coords, expected_coords)


if __name__ == "__main__":
  absltest.main()
