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
import jax
import neuralgcm
import numpy as np
import xarray

jax.config.update('jax_threefry_partitionable', False)


def _assert_allclose(actual, desired, *, err_msg=None, range_rtol=1e-5):
  span = desired.max() - desired.min()
  np.testing.assert_allclose(
      actual / span, desired / span, rtol=0, atol=range_rtol, err_msg=err_msg
  )


def load_tl63_stochastic_model():
  return neuralgcm.PressureLevelModel.from_checkpoint(
      neuralgcm.demo.load_checkpoint_tl63_stochastic()
  )


class APITest(absltest.TestCase):

  def assertDictEqual(self, actual, desired, *, err_msg=None):
    self.assertEqual(sorted(actual.keys()), sorted(desired.keys()))
    for key in actual:
      x = actual[key]
      y = desired[key]
      err_msg2 = key if err_msg is None else f'{err_msg}/{key}'
      if isinstance(y, dict):
        self.assertDictAllclose(x, y, err_msg=err_msg2)
      else:
        np.testing.assert_array_equal(x, y, err_msg=err_msg2)

  def assertDictAllclose(
      self, actual, desired, *, err_msg=None, range_rtol=1e-5
  ):
    self.assertEqual(sorted(actual.keys()), sorted(desired.keys()))
    for key in actual:
      x = actual[key]
      y = desired[key]
      err_msg2 = key if err_msg is None else f'{err_msg}/{key}'
      if isinstance(y, dict):
        self.assertDictAllclose(x, y, err_msg=err_msg2, range_rtol=range_rtol)
      else:
        _assert_allclose(x, y, err_msg=err_msg2, range_rtol=range_rtol)

  def test_model_properties(self):
    model = load_tl63_stochastic_model()

    self.assertEqual(model.timestep, np.timedelta64(1, 'h'))

    expected_inputs = [
        'geopotential',
        'specific_humidity',
        'temperature',
        'u_component_of_wind',
        'v_component_of_wind',
        'specific_cloud_ice_water_content',
        'specific_cloud_liquid_water_content',
    ]
    self.assertEqual(model.input_variables, expected_inputs)

    expected_forcings = [
        'sea_ice_cover',
        'sea_surface_temperature',
    ]
    self.assertEqual(model.forcing_variables, expected_forcings)

    self.assertEqual(model.data_coords.nodal_shape, (37, 128, 64))

    self.assertEqual(model.model_coords.nodal_shape, (32, 128, 64))

  def test_to_and_from_nondim_units(self):
    model = load_tl63_stochastic_model()

    nondim = model.to_nondim_units(1.0, 'meters')
    self.assertAlmostEqual(nondim, 1 / 6.37122e6)  # Earth radius units

    roundtripped = model.from_nondim_units(nondim, 'meters')
    self.assertAlmostEqual(roundtripped, 1.0)

  def test_sim_time_utilities(self):
    model = load_tl63_stochastic_model()
    origin = model.datetime64_to_sim_time(np.datetime64('1979-01-01T00:00:00'))
    self.assertEqual(origin, 0.0)

    original = np.datetime64('1959-01-01T00:00:00')
    roundtripped = model.sim_time_to_datetime64(
        model.datetime64_to_sim_time(original)
    )
    self.assertEqual(original, roundtripped)

  def test_from_xarray(self):
    model = load_tl63_stochastic_model()
    ds = neuralgcm.demo.load_data(model.data_coords)

    expected_inputs = {k: ds[k].values for k in model.input_variables}
    expected_inputs['sim_time'] = np.array([-92034.607104])

    expected_forcings = {
        k: ds[k].values[:, np.newaxis, :, :] for k in model.forcing_variables
    }
    expected_forcings['sim_time'] = np.array([-92034.607104])

    inputs = model.inputs_from_xarray(ds)
    self.assertDictEqual(inputs, expected_inputs)

    forcings = model.forcings_from_xarray(ds)
    self.assertDictEqual(forcings, expected_forcings)

    inputs, forcings = model.data_from_xarray(ds)
    self.assertDictEqual(inputs, expected_inputs)
    self.assertDictEqual(forcings, expected_forcings)

  def test_stochastic_model_basics(self):
    timesteps = 3
    dt = np.timedelta64(1, 'h')

    model = load_tl63_stochastic_model()
    ds_in = neuralgcm.demo.load_data(model.data_coords)

    data_in, forcings_in = model.data_from_xarray(ds_in.isel(time=0))
    persistence_forcings = model.forcings_from_xarray(ds_in.head(time=1))

    # run model
    encoded = model.encode(data_in, forcings_in, rng_key=jax.random.key(0))
    _, data_out = model.unroll(
        encoded,
        persistence_forcings,
        steps=timesteps,
        timedelta=dt,
        start_with_input=True,
    )

    # convert to xarray
    t0 = ds_in.time.values[0]
    times = np.arange(t0, t0 + timesteps * dt, dt)
    ds_out = model.data_to_xarray(data_out, times=times)

    # validate decoded data
    actual = ds_out.head(time=1)

    sim_time = model.datetime64_to_sim_time(ds_in.time.data)
    expected = ds_in.drop_vars(
        ['sea_surface_temperature', 'sea_ice_cover']
    ).assign(sim_time=('time', sim_time))

    # check matching variable shapes
    xarray.testing.assert_allclose(actual, expected, atol=1e6)

    # check that round-tripping the initial condition is approximately correct
    typical_relative_error = abs(actual - expected).median() / expected.std()
    tolerance = xarray.Dataset({
        'u_component_of_wind': 0.04,
        'v_component_of_wind': 0.08,
        'temperature': 0.02,
        'geopotential': 0.0005,
        'specific_humidity': 0.003,
        'specific_cloud_liquid_water_content': 0.12,
        'specific_cloud_ice_water_content': 0.15,
    })
    self.assertTrue(
        (typical_relative_error < tolerance).to_array().values.all(),
        msg=f'typical relative error is too large:\n{typical_relative_error}',
    )

    # test decode()
    decoded = model.decode(encoded, forcings_in)
    expected = jax.tree.map(lambda x: x[0, ...], data_out)
    self.assertDictAllclose(decoded, expected)

    # test advance()
    self.assertEqual(model.timestep, dt)
    decoded2 = model.decode(model.advance(encoded, forcings_in), forcings_in)
    expected2 = jax.tree.map(lambda x: x[1, ...], data_out)
    self.assertDictAllclose(decoded2, expected2)

    # TODO(shoyer): verify RNG key works correctly
    # TODO(shoyer): verify RNG key is optional for deterministic models

  def test_encoded_state(self):
    model = load_tl63_stochastic_model()
    ds_in = neuralgcm.demo.load_data(model.data_coords)
    data_in, forcings_in = model.data_from_xarray(ds_in.isel(time=0))

    encoded = model.encode(data_in, forcings_in, rng_key=jax.random.key(0))

    ds_advanced = model.data_to_xarray(
        {
            'vorticity': encoded.state.vorticity,
            'divergence': encoded.state.divergence,
        },
        times=None,
        decoded=False,
    )

    dims = ('level', 'longitudinal_mode', 'total_wavenumber')
    z, x, y = model.model_coords.modal_shape
    levels = np.linspace(1 - 1/32, 1 / 32, 32)
    longitudinal_modes = np.array([(i//2) * (-1)**i for i in range(128)])
    total_wavenumbers = np.arange(65)
    expected = xarray.Dataset(
        {
            'vorticity': (dims, np.zeros((z, x, y))),
            'divergence': (dims, np.zeros((z, x, y))),
        },
        coords={
            'level': levels,
            'longitudinal_mode': longitudinal_modes,
            'total_wavenumber': total_wavenumbers,
        },
    )
    # verify expected shpaes
    xarray.testing.assert_allclose(ds_advanced, expected, atol=1e6)


if __name__ == '__main__':
  absltest.main()
