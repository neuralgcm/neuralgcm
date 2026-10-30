# Copyright 2026 Google LLC
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

"""Tests for coupled ocean-atmosphere and ocean-land-atmosphere models."""

from absl.testing import absltest
from absl.testing import parameterized
import coordax as cx
from flax import nnx
import jax.numpy as jnp
import jax_datetime as jdt
from neuralgcm.base import operator_utils
from neuralgcm.coupled import models
from neuralgcm.hybrid import atmosphere as hybrid_atmosphere
from neuralgcm.ml import learned_update
import numpy as np
from terrax.core import api
from terrax.core import coordinates
from terrax.core import data_specs
from terrax.core import diagnostics
from terrax.core import observation_operators
from terrax.core import orographies
from terrax.core import spatial_filters
from terrax.core import spherical_harmonics
from terrax.core import transforms
from terrax.core import typing
from terrax.core import units
from terrax.couplers import generic

OCEAN_KEY = 'era5:ocean_surface'
LAND_KEY = 'era5:land_surface'
ATM_SURFACE_KEY = 'era5:surface'


class _ZeroPhysicsWithSurfaceDiagnostic(nnx.Module):
  """Zero-tendency physics that can read surface couplers for testing."""

  def __init__(
      self,
      ocean_coupler: models.DataCoupler | None = None,
      land_coupler: models.DataCoupler | None = None,
  ):
    self.ocean_coupler = ocean_coupler
    self.land_coupler = land_coupler

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    time = inputs['time']
    offset = 0.0
    if self.ocean_coupler is not None:
      ocean_fields = self.ocean_coupler(time.data)
      offset = offset + jnp.mean(ocean_fields['sea_surface_temperature'].data)
    if self.land_coupler is not None:
      land_fields = self.land_coupler(time.data)
      offset = offset + jnp.mean(land_fields['volumetric_soil_water'].data)
    return {
        k: 0.0 * v + (0.0 * offset if k == 'temperature' else 0.0)
        for k, v in inputs.items()
        if k != 'time'
    }


class _CouplerDrivenUpdate(nnx.Module):
  """Sets prognostic fields from an atmosphere coupler or increments them."""

  def __init__(
      self,
      var_names: tuple[str, ...],
      coupler: generic.InstantDataCoupler | None = None,
      coupler_var: str = 'temperature_variation',
  ):
    self.var_names = var_names
    self.coupler = coupler
    self.coupler_var = coupler_var

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    time = inputs['time']
    if self.coupler is not None:
      coupled_val = self.coupler(time.data)[self.coupler_var]
      return {k: inputs[k] + coupled_val for k in self.var_names}
    return {k: inputs[k] + 1.0 for k in self.var_names}


class _WindowedCouplerUpdate(nnx.Module):
  """Increments `var_name` by the change of a coupled field over `window`.

  The coupled field is read at the start and the end of the window that begins
  at the `time` of the inputs. Other prognostic variables are held fixed.
  """

  def __init__(
      self,
      coupler: generic.MultiTimeDataCoupler,
      window: np.timedelta64,
      var_name: str = 'sea_surface_temperature',
      coupler_var: str = 'temperature_variation',
  ):
    self.coupler = coupler
    self.window = window
    self.var_name = var_name
    self.coupler_var = coupler_var

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    start = inputs['time']
    end = start + cx.field(jdt.to_timedelta(self.window))
    start_value = self.coupler(start)[self.coupler_var]
    end_value = self.coupler(end)[self.coupler_var]
    outputs = {k: v for k, v in inputs.items() if k != 'time'}
    outputs[self.var_name] = outputs[self.var_name] + end_value - start_value
    return outputs


class _ExtractField(nnx.Module):
  """Extracts `key` from `prognostics` under `out_name`, for testing."""

  def __init__(self, key: str, out_name: str):
    self.key = key
    self.out_name = out_name

  def __call__(self, inputs, *, prognostics):
    del inputs
    return {self.out_name: prognostics[self.key]}


class _AtmosphereSurfaceOperator(observation_operators.ObservationOperatorABC):
  """Observes nodal `temperature_variation` and `time` from atmosphere state."""

  def __init__(
      self,
      ylm_map: spherical_harmonics.FixedYlmMapping,
      levels: coordinates.SigmaLevels,
  ):
    self.ylm_map = ylm_map
    self.levels = levels

  def observe(
      self,
      inputs: dict[str, cx.Field],
      query: typing.Query,
  ) -> dict[str, cx.Field]:
    temp_modal = inputs['temperature'].isel({self.levels.dims[0]: -1})
    temp_nodal = self.ylm_map.to_nodal(temp_modal)
    fields = {
        'temperature_variation': temp_nodal,
        'time': inputs['time'],
    }
    return observation_operators.DataObservationOperator(fields).observe(
        {}, query
    )


class _ElapsedHoursOperator(observation_operators.ObservationOperatorABC):
  """Observes hours elapsed since `start` as `temperature_variation`."""

  def __init__(self, grid: coordinates.LonLatGrid, start: np.datetime64):
    self.grid = grid
    self.start = start

  def observe(
      self,
      inputs: dict[str, cx.Field],
      query: typing.Query,
  ) -> dict[str, cx.Field]:
    time = inputs['time']
    hours = (time.data - jdt.to_datetime(self.start)) / np.timedelta64(1, 'h')
    fields = {
        'temperature_variation': cx.field(
            jnp.full(self.grid.shape, hours), self.grid
        ),
        'time': time,
    }
    return observation_operators.DataObservationOperator(fields).observe(
        {}, query
    )


class CoupledModelsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.sim_units = units.SI_UNITS
    self.grid = coordinates.LonLatGrid.T21()
    self.ylm_grid = coordinates.SphericalHarmonicGrid.T21()
    self.levels = coordinates.SigmaLevels.equidistant(4)
    self.ylm_map = spherical_harmonics.FixedYlmMapping(
        self.grid, self.ylm_grid, radius=self.sim_units.radius
    )
    self.init_time = jdt.to_datetime('2020-01-01T00:00')

  def _make_atmosphere(
      self,
      model_timestep: np.timedelta64 = np.timedelta64(6, 'h'),
      ocean_coupler: models.DataCoupler | None = None,
      land_coupler: models.DataCoupler | None = None,
      tracer_names: tuple[str, ...] = ('specific_humidity',),
      step_end_diagnostics: (
          dict[str, diagnostics.DiagnosticModule] | None
      ) = None,
      surface_operator: (
          observation_operators.ObservationOperatorABC | None
      ) = None,
  ) -> hybrid_atmosphere.LearnedParameterizationSpectralDycore:
    orography = orographies.ModalOrography(
        ylm_map=self.ylm_map, rngs=nnx.Rngs(0)
    )
    dt = self.sim_units.nondimensionalize_timedelta64(model_timestep)
    modal_filter = spatial_filters.ExponentialModalFilter.from_timescale(
        ylm_map=self.ylm_map,
        dt=dt,
        timescale='2 hours',
        order=4,
        cutoff=0.0,
        sim_units=self.sim_units,
    )
    if surface_operator is None:
      surface_operator = _AtmosphereSurfaceOperator(self.ylm_map, self.levels)
    operators = {
        'prognostics': operator_utils.primitive_equations_state_operator(
            self.ylm_map
        ),
        ATM_SURFACE_KEY: surface_operator,
    }
    return hybrid_atmosphere.LearnedParameterizationSpectralDycore(
        model_ylm_map=self.ylm_map,
        model_levels=self.levels,
        model_orography=orography,
        model_timestep=model_timestep,
        modal_filter=modal_filter,
        learned_physics=_ZeroPhysicsWithSurfaceDiagnostic(
            ocean_coupler=ocean_coupler, land_coupler=land_coupler
        ),
        tracer_names=tracer_names,
        sim_units=self.sim_units,
        operators=operators,
        step_end_diagnostics=step_end_diagnostics or {},
    )

  def _make_ocean(
      self,
      model_timestep: np.timedelta64 = np.timedelta64(6, 'h'),
      update_timestep: np.timedelta64 = np.timedelta64(6, 'h'),
      atm_coupler: generic.InstantDataCoupler | None = None,
      update_transform: typing.Transform | None = None,
      step_end_diagnostics: (
          dict[str, diagnostics.DiagnosticModule] | None
      ) = None,
      sim_units: units.SimUnits | None = None,
      units_mapping: dict[str, str] | None = None,
  ) -> learned_update.LearnedUpdateModel:
    var_names = ('sea_surface_temperature', 'sea_ice_cover')
    if update_transform is None:
      update_transform = _CouplerDrivenUpdate(var_names, coupler=atm_coupler)
    coords = {k: self.grid for k in var_names}
    var_to_keys = {k: OCEAN_KEY for k in (*var_names, 'time')}
    op = observation_operators.TransformObservationOperator(
        transforms.Identity()
    )
    return learned_update.LearnedUpdateModel(
        prognostics_coords=coords,
        update_transform=update_transform,
        var_to_dataset_keys=var_to_keys,
        model_timestep=model_timestep,
        update_timestep=update_timestep,
        predict_state=True,
        sim_units=sim_units or self.sim_units,
        units_mapping=units_mapping,
        prognostics_data_key='ocean_prognostics',
        operators={OCEAN_KEY: op},
        step_end_diagnostics=step_end_diagnostics or {},
    )

  def _make_land(
      self,
      model_timestep: np.timedelta64 = np.timedelta64(6, 'h'),
      update_timestep: np.timedelta64 = np.timedelta64(6, 'h'),
      atm_coupler: generic.InstantDataCoupler | None = None,
      step_end_diagnostics: (
          dict[str, diagnostics.DiagnosticModule] | None
      ) = None,
  ) -> learned_update.LearnedUpdateModel:
    var_names = ('volumetric_soil_water',)
    coords = {k: self.grid for k in var_names}
    var_to_keys = {k: LAND_KEY for k in (*var_names, 'time')}
    op = observation_operators.TransformObservationOperator(
        transforms.Identity()
    )
    return learned_update.LearnedUpdateModel(
        prognostics_coords=coords,
        update_transform=_CouplerDrivenUpdate(var_names, coupler=atm_coupler),
        var_to_dataset_keys=var_to_keys,
        model_timestep=model_timestep,
        update_timestep=update_timestep,
        predict_state=True,
        sim_units=self.sim_units,
        prognostics_data_key='land_prognostics',
        operators={LAND_KEY: op},
        step_end_diagnostics=step_end_diagnostics or {},
    )

  def _initial_inputs(
      self, include_land: bool = False
  ) -> dict[str, dict[str, cx.Field]]:
    modal_vol = cx.coords.compose(self.levels, self.ylm_grid)
    zeros_modal = cx.field(jnp.zeros(modal_vol.shape), modal_vol)
    atm_state = {
        'divergence': zeros_modal,
        'vorticity': zeros_modal,
        'temperature': zeros_modal,
        'log_surface_pressure': cx.field(
            jnp.zeros(self.ylm_grid.shape), self.ylm_grid
        ),
        'specific_humidity': zeros_modal,
        'time': cx.field(self.init_time),
    }
    ocean_data = {
        'sea_surface_temperature': cx.field(
            jnp.full(self.grid.shape, 290.0), self.grid
        ),
        'sea_ice_cover': cx.field(jnp.full(self.grid.shape, 0.25), self.grid),
        'time': cx.field(self.init_time),
    }
    inputs = {
        'nondim_atmosphere_prognostics': atm_state,
        OCEAN_KEY: ocean_data,
    }
    if include_land:
      inputs[LAND_KEY] = {
          'volumetric_soil_water': cx.field(
              jnp.full(self.grid.shape, 0.4), self.grid
          ),
          'time': cx.field(self.init_time),
      }
    return inputs

  def test_defaults_and_inputs_spec(self):
    ocean = self._make_ocean()
    atmosphere = self._make_atmosphere()
    coupled = models.OceanAtmosphere(
        ocean=ocean,
        atmosphere=atmosphere,
    )
    self.assertEqual(coupled.timestep, np.timedelta64(6, 'h'))
    # Units default to those of the components.
    self.assertEqual(coupled.sim_units, atmosphere.sim_units)
    self.assertEqual(coupled.units_mapping, atmosphere.units_mapping)
    self.assertIsInstance(
        coupled.nondim_transform, transforms.Nondimensionalize
    )
    self.assertIsInstance(coupled.redim_transform, transforms.Redimensionalize)
    spec = coupled.inputs_spec
    self.assertIn(OCEAN_KEY, spec)
    self.assertIn('ocean_prognostics', spec)
    self.assertIn('atmosphere_prognostics', spec)
    data_specs.validate_inputs(self._initial_inputs(), spec)

  def test_raises_on_mismatched_timesteps(self):
    ocean = self._make_ocean(
        model_timestep=np.timedelta64(12, 'h'),
        update_timestep=np.timedelta64(12, 'h'),
    )
    atmosphere = self._make_atmosphere(model_timestep=np.timedelta64(6, 'h'))
    with self.assertRaisesRegex(ValueError, 'same timestep'):
      models.OceanAtmosphere(ocean=ocean, atmosphere=atmosphere)

    land = self._make_land(
        model_timestep=np.timedelta64(12, 'h'),
        update_timestep=np.timedelta64(12, 'h'),
    )
    with self.assertRaisesRegex(ValueError, 'same timestep'):
      models.OceanLandAtmosphere(
          ocean=self._make_ocean(),
          land=land,
          atmosphere=atmosphere,
      )

  def test_two_way_instant_coupling_ocean_atmosphere(self):
    ocean_to_atm = generic.InstantDataCoupler(
        field_coords={
            'sea_surface_temperature': self.grid,
            'sea_ice_cover': self.grid,
        },
        var_to_data_keys={
            'sea_surface_temperature': OCEAN_KEY,
            'sea_ice_cover': OCEAN_KEY,
        },
    )
    atm_to_ocean = generic.InstantDataCoupler(
        field_coords={'temperature_variation': self.grid},
        var_to_data_keys={'temperature_variation': ATM_SURFACE_KEY},
    )
    ocean = self._make_ocean(atm_coupler=atm_to_ocean)
    atmosphere = self._make_atmosphere(ocean_coupler=ocean_to_atm)
    coupled = models.OceanAtmosphere(
        ocean=ocean,
        atmosphere=atmosphere,
        ocean_to_atmosphere_coupler=ocean_to_atm,
        atmosphere_to_ocean_coupler=atm_to_ocean,
    )
    inputs = self._initial_inputs()
    coupled.assimilate(inputs)

    # Couplers are populated with nondimensionalized values after assimilate.
    expected_nondim_sst = coupled.nondim_transform(inputs[OCEAN_KEY])[
        'sea_surface_temperature'
    ]
    cx.testing.assert_fields_allclose(
        ocean_to_atm(self.init_time)['sea_surface_temperature'],
        expected_nondim_sst,
    )
    cx.testing.assert_fields_allclose(
        atm_to_ocean(self.init_time)['temperature_variation'],
        cx.field(jnp.zeros(self.grid.shape), self.grid),
    )

    coupled.advance()
    observed = coupled.observe({
        OCEAN_KEY: {
            'sea_surface_temperature': self.grid,
            'sea_ice_cover': self.grid,
        },
        ATM_SURFACE_KEY: {'temperature_variation': self.grid},
    })
    cx.testing.assert_fields_allclose(
        observed[OCEAN_KEY]['sea_surface_temperature'],
        inputs[OCEAN_KEY]['sea_surface_temperature'],
        rtol=1e-5,
    )
    cx.testing.assert_fields_allclose(
        observed[OCEAN_KEY]['sea_ice_cover'],
        inputs[OCEAN_KEY]['sea_ice_cover'],
        rtol=1e-5,
    )

  def test_substepped_ocean_with_multi_time_coupler(self):
    dt = np.timedelta64(6, 'h')
    ocean_dt = np.timedelta64(12, 'h')
    # The ocean updates every 2 steps with `time` set to the start of its update
    # interval, after `atm_to_ocean` is updated with the snapshot at its end.
    # Hence 3 time slices are needed to access both ends of the interval.
    atm_to_ocean = generic.MultiTimeDataCoupler(
        field_coords={'temperature_variation': self.grid},
        var_to_data_keys={'temperature_variation': ATM_SURFACE_KEY},
        time_data_key=ATM_SURFACE_KEY,
        n_time_slices=3,
    )
    ocean_to_atm = generic.InstantDataCoupler(
        field_coords={'sea_surface_temperature': self.grid},
        var_to_data_keys={'sea_surface_temperature': OCEAN_KEY},
    )
    ocean = self._make_ocean(
        model_timestep=dt,
        update_timestep=ocean_dt,
        update_transform=_WindowedCouplerUpdate(atm_to_ocean, window=ocean_dt),
    )
    atmosphere = self._make_atmosphere(
        model_timestep=dt,
        ocean_coupler=ocean_to_atm,
        surface_operator=_ElapsedHoursOperator(
            self.grid, self.init_time.to_datetime64()
        ),
    )
    coupled = models.OceanAtmosphere(
        ocean=ocean,
        atmosphere=atmosphere,
        ocean_to_atmosphere_coupler=ocean_to_atm,
        atmosphere_to_ocean_coupler=atm_to_ocean,
    )
    coupled.assimilate(self._initial_inputs())
    initial_sst = ocean.prognostics.get_value()['sea_surface_temperature']

    # First 6h step: ocean holds prognostics fixed; atm_to_ocean records t=6h.
    coupled.advance()
    cx.testing.assert_fields_allclose(
        ocean.prognostics.get_value()['sea_surface_temperature'], initial_sst
    )
    # Second 6h step (12h total): ocean applies its update, incrementing SST by
    # the 12 hours elapsed between the ends of its update interval (in SI units
    # nondimensionalization of `temperature_variation` is the identity).
    coupled.advance()
    expected_sst = initial_sst + 12.0
    cx.testing.assert_fields_allclose(
        ocean.prognostics.get_value()['sea_surface_temperature'], expected_sst
    )
    cx.testing.assert_fields_allclose(
        ocean_to_atm(self.init_time)['sea_surface_temperature'], expected_sst
    )
    expected_times = np.array([
        (self.init_time + jdt.to_timedelta(i * dt)).to_datetime64()
        for i in range(3)
    ])
    np.testing.assert_array_equal(
        atm_to_ocean.times.get_value().data.to_datetime64(), expected_times
    )

  def test_ocean_land_atmosphere_coupling_and_diagnostics(self):
    ocean_to_atm = generic.InstantDataCoupler(
        field_coords={'sea_surface_temperature': self.grid},
        var_to_data_keys={'sea_surface_temperature': OCEAN_KEY},
    )
    land_to_atm = generic.InstantDataCoupler(
        field_coords={'volumetric_soil_water': self.grid},
        var_to_data_keys={'volumetric_soil_water': LAND_KEY},
    )
    atm_to_ocean = generic.InstantDataCoupler(
        field_coords={'temperature_variation': self.grid},
        var_to_data_keys={'temperature_variation': ATM_SURFACE_KEY},
    )
    atm_to_land = generic.InstantDataCoupler(
        field_coords={'temperature_variation': self.grid},
        var_to_data_keys={'temperature_variation': ATM_SURFACE_KEY},
    )
    ocean_diag = diagnostics.InstantDiagnostic(
        extract=_ExtractField(
            'sea_surface_temperature', 'sea_surface_temperature'
        ),
        extract_coords={'sea_surface_temperature': self.grid},
    )
    land_diag = diagnostics.InstantDiagnostic(
        extract=_ExtractField('volumetric_soil_water', 'volumetric_soil_water'),
        extract_coords={'volumetric_soil_water': self.grid},
    )
    ocean = self._make_ocean(
        atm_coupler=atm_to_ocean, step_end_diagnostics={'sst': ocean_diag}
    )
    land = self._make_land(
        atm_coupler=atm_to_land, step_end_diagnostics={'sw': land_diag}
    )
    atmosphere = self._make_atmosphere(
        ocean_coupler=ocean_to_atm, land_coupler=land_to_atm
    )
    # Top-level operator observes collective prognostics across components.
    top_op = observation_operators.TransformObservationOperator(
        transforms.Identity()
    )
    coupled = models.OceanLandAtmosphere(
        ocean=ocean,
        land=land,
        atmosphere=atmosphere,
        ocean_to_atmosphere_coupler=ocean_to_atm,
        land_to_atmosphere_coupler=land_to_atm,
        atmosphere_to_ocean_coupler=atm_to_ocean,
        atmosphere_to_land_coupler=atm_to_land,
        operators={'custom_top': top_op},
        remat_advance=True,
    )
    inputs = self._initial_inputs(include_land=True)
    data_specs.validate_inputs(inputs, coupled.inputs_spec)
    coupled.assimilate(inputs)
    coupled.advance()

    prognostics = coupled.prognostic_values()
    self.assertContainsSubset(
        {
            'sea_surface_temperature',
            'volumetric_soil_water',
            'temperature',
            'time',
        },
        prognostics.keys(),
    )

    observed = coupled.observe({
        OCEAN_KEY: {'sea_surface_temperature': self.grid},
        LAND_KEY: {'volumetric_soil_water': self.grid},
        'diagnostics': {
            'sea_surface_temperature': self.grid,
            'volumetric_soil_water': self.grid,
        },
        'custom_top': {
            'sea_surface_temperature': self.grid,
            'volumetric_soil_water': self.grid,
        },
    })
    cx.testing.assert_fields_allclose(
        observed['diagnostics']['sea_surface_temperature'],
        observed[OCEAN_KEY]['sea_surface_temperature'],
        rtol=1e-5,
    )
    cx.testing.assert_fields_allclose(
        observed['diagnostics']['volumetric_soil_water'],
        observed[LAND_KEY]['volumetric_soil_water'],
        rtol=1e-5,
    )
    cx.testing.assert_fields_allclose(
        observed['custom_top']['sea_surface_temperature'],
        observed[OCEAN_KEY]['sea_surface_temperature'],
        rtol=1e-5,
    )
    cx.testing.assert_fields_allclose(
        observed['custom_top']['volumetric_soil_water'],
        observed[LAND_KEY]['volumetric_soil_water'],
        rtol=1e-5,
    )

  def test_inference_model_and_vectorized_model(self):
    ocean_to_atm = generic.InstantDataCoupler(
        field_coords={'sea_surface_temperature': self.grid},
        var_to_data_keys={'sea_surface_temperature': OCEAN_KEY},
    )
    ocean = self._make_ocean()
    atmosphere = self._make_atmosphere(ocean_coupler=ocean_to_atm)
    coupled = models.OceanAtmosphere(
        ocean=ocean,
        atmosphere=atmosphere,
        ocean_to_atmosphere_coupler=ocean_to_atm,
    )
    inputs = self._initial_inputs()

    # Verify compatibility with InferenceModel.from_model_api.
    inference_model = api.InferenceModel.from_model_api(coupled)
    state = inference_model.assimilate(inputs)
    state = inference_model.advance(state)
    obs = inference_model.observe(
        state, {OCEAN_KEY: {'sea_surface_temperature': self.grid}}
    )
    self.assertEqual(
        obs[OCEAN_KEY]['sea_surface_temperature'].coordinate, self.grid
    )

    # Verify compatibility with VectorizedModel.
    batch = cx.SizedAxis('ensemble', 2)
    vec_model = coupled.to_vectorized(
        {(typing.Prognostic, typing.Coupling): batch}
    )
    batched_inputs = {
        ds_key: {
            k: v.broadcast_like(cx.coords.compose(batch, v.coordinate))
            for k, v in ds.items()
        }
        for ds_key, ds in inputs.items()
    }
    vec_model.assimilate(batched_inputs)
    vec_model.advance()
    vec_obs = vec_model.observe(
        {OCEAN_KEY: {'sea_surface_temperature': self.grid}}
    )
    self.assertEqual(
        vec_obs[OCEAN_KEY]['sea_surface_temperature'].coordinate,
        cx.coords.compose(batch, self.grid),
    )

  def test_nondim_coupler_queries_and_sim_units_validation(self):
    ocean_to_atm = generic.InstantDataCoupler(
        field_coords={'sea_surface_temperature': self.grid},
        var_to_data_keys={'sea_surface_temperature': f'nondim_{OCEAN_KEY}'},
    )
    atm_to_ocean = generic.InstantDataCoupler(
        field_coords={'temperature_variation': self.grid},
        var_to_data_keys={'temperature_variation': f'nondim_{ATM_SURFACE_KEY}'},
    )
    ocean = self._make_ocean(atm_coupler=atm_to_ocean)
    atmosphere = self._make_atmosphere(ocean_coupler=ocean_to_atm)
    coupled = models.OceanAtmosphere(
        ocean=ocean,
        atmosphere=atmosphere,
        ocean_to_atmosphere_coupler=ocean_to_atm,
        atmosphere_to_ocean_coupler=atm_to_ocean,
    )
    inputs = self._initial_inputs()
    coupled.assimilate(inputs)
    coupled.advance()

    # Coupler values match nondimensional observations directly.
    nondim_obs = coupled.observe({
        f'nondim_{OCEAN_KEY}': {'sea_surface_temperature': self.grid},
        f'nondim_{ATM_SURFACE_KEY}': {'temperature_variation': self.grid},
    })
    t = ocean.time.get_value().data
    cx.testing.assert_fields_allclose(
        ocean_to_atm(t)['sea_surface_temperature'],
        nondim_obs[f'nondim_{OCEAN_KEY}']['sea_surface_temperature'],
        rtol=1e-5,
    )
    cx.testing.assert_fields_allclose(
        atm_to_ocean(t)['temperature_variation'],
        nondim_obs[f'nondim_{ATM_SURFACE_KEY}']['temperature_variation'],
        rtol=1e-5,
    )

    # Mismatched `sim_units` with `nondim_` coupler keys raises ValueError.
    ocean_alt_units = self._make_ocean(sim_units=units.DEFAULT_UNITS)
    with self.assertRaisesRegex(ValueError, 'requires matching `sim_units`'):
      models.OceanAtmosphere(
          ocean=ocean_alt_units,
          atmosphere=atmosphere,
          ocean_to_atmosphere_coupler=ocean_to_atm,
      )
    # Same for `time_data_key` of `MultiTimeDataCoupler`.
    multi_time_atm_to_ocean = generic.MultiTimeDataCoupler(
        field_coords={'temperature_variation': self.grid},
        var_to_data_keys={'temperature_variation': ATM_SURFACE_KEY},
        time_data_key=f'nondim_{ATM_SURFACE_KEY}',
        n_time_slices=2,
    )
    with self.assertRaisesRegex(ValueError, 'requires matching `sim_units`'):
      models.OceanAtmosphere(
          ocean=ocean_alt_units,
          atmosphere=atmosphere,
          atmosphere_to_ocean_coupler=multi_time_atm_to_ocean,
      )

  def test_dimensional_coupling_across_distinct_sim_units(self):
    ocean_to_atm = generic.InstantDataCoupler(
        field_coords={'sea_surface_temperature': self.grid},
        var_to_data_keys={'sea_surface_temperature': OCEAN_KEY},
    )
    atm_to_ocean = generic.InstantDataCoupler(
        field_coords={'temperature_variation': self.grid},
        var_to_data_keys={'temperature_variation': ATM_SURFACE_KEY},
    )
    ocean_units_mapping = {
        'sea_surface_temperature': 'degK',
        'sea_ice_cover': 'dimensionless',
        'temperature_variation': 'millikelvin',
    }
    ocean_nondim = transforms.Nondimensionalize(
        units.DEFAULT_UNITS, inputs_to_units_mapping=ocean_units_mapping
    )
    ocean = self._make_ocean(
        atm_coupler=atm_to_ocean,
        sim_units=units.DEFAULT_UNITS,
        units_mapping=ocean_units_mapping,
    )
    atmosphere = self._make_atmosphere(ocean_coupler=ocean_to_atm)
    # Units mappings conflict on `temperature_variation`, so the coupled model
    # needs an explicit `units_mapping`.
    with self.assertRaisesRegex(ValueError, 'Conflicting units'):
      models.OceanAtmosphere(ocean=ocean, atmosphere=atmosphere)
    coupled = models.OceanAtmosphere(
        ocean=ocean,
        atmosphere=atmosphere,
        ocean_to_atmosphere_coupler=ocean_to_atm,
        atmosphere_to_ocean_coupler=atm_to_ocean,
        units_mapping=atmosphere.units_mapping,
    )
    inputs = self._initial_inputs()
    # Non-zero temperature so that unit conversion factors are exercised.
    atm_state = inputs['nondim_atmosphere_prognostics']
    temperature = atm_state['temperature']
    atm_state['temperature'] = cx.field(
        temperature.data.at[..., 0, 0].set(250.0), temperature.coordinate
    )
    coupled.assimilate(inputs)

    # `atm_to_ocean` converts nondimensional `temperature_variation` directly
    # from `atmosphere.sim_units` to `ocean.sim_units` via `units_passage`
    # (where scalar unit prefixes like `'kelvin'` vs `'millikelvin'` cancel),
    # rather than redimensionalizing with `atmosphere.observe` (in kelvin) and
    # re-nondimensionalizing with `ocean_nondim` (which expects millikelvin).
    dim_temp = atmosphere.observe(
        {ATM_SURFACE_KEY: {'temperature_variation': self.grid}}
    )[ATM_SURFACE_KEY]
    to_ocean_units = transforms.Nondimensionalize(
        units.DEFAULT_UNITS, {'temperature_variation': 'kelvin'}
    )
    expected_ocean_temp = to_ocean_units(dim_temp)['temperature_variation']
    t = ocean.time.get_value().data
    coupled_temp = atm_to_ocean(t)['temperature_variation']
    cx.testing.assert_fields_allclose(
        coupled_temp, expected_ocean_temp, rtol=1e-5
    )
    millikelvin_temp = ocean_nondim(dim_temp)['temperature_variation']
    self.assertFalse(np.allclose(coupled_temp.data, millikelvin_temp.data))


if __name__ == '__main__':
  absltest.main()
