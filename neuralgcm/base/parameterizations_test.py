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

"""Tests for ModalParameterization and spectral parameterization helpers."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import coordax as cx
from flax import nnx
import jax.numpy as jnp
import jax_datetime as jdt
from neuralgcm.base import parameterizations
from neuralgcm.hybrid import atmosphere as hybrid_atmosphere
import numpy as np
from terrax.atmosphere import equations as atm_equations
from terrax.atmosphere import transforms as atm_transforms
from terrax.core import coordinates
from terrax.core import learned_transforms
from terrax.core import orographies
from terrax.core import parallelism
from terrax.core import spatial_filters
from terrax.core import spherical_harmonics
from terrax.core import standard_layers
from terrax.core import time_integrators
from terrax.core import towers
from terrax.core import transforms
from terrax.core import units


def _make_ylm_map(
    mesh: parallelism.Mesh,
) -> spherical_harmonics.FixedYlmMapping:
  return spherical_harmonics.FixedYlmMapping(
      lon_lat_grid=coordinates.LonLatGrid.T21(),
      ylm_grid=coordinates.SphericalHarmonicGrid.T21(),
      mesh=mesh,
      partition_schema_key=None,
      radius=units.SI_UNITS.radius,
  )


def _make_tower_factory() -> towers.ForwardTowerFactory:
  net_factory = functools.partial(
      standard_layers.Mlp.uniform, hidden_size=8, hidden_layers=1
  )
  return functools.partial(
      towers.ForwardTower.build_using_factories,
      inputs_in_dims=('d',),
      out_dims=('d',),
      neural_net_factory=net_factory,
  )


class _ConstantNodalTendencies(transforms.TransformABC):
  """Returns constant nodal tendencies on `coords` for testing."""

  def __init__(self, coords: dict[str, cx.Coordinate], value: float = 1.0):
    self.coords = coords
    self.value = value

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    del inputs
    return {
        k: cx.field(jnp.full(c.shape, self.value, dtype=jnp.float32), c)
        for k, c in self.coords.items()
    }


class ParameterizationsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mesh = parallelism.Mesh()
    self.ylm_map = _make_ylm_map(self.mesh)
    self.grid = self.ylm_map.lon_lat_grid
    self.ylm_grid = self.ylm_map.ylm_grid
    self.levels = coordinates.SigmaLevels.equidistant(4)
    self.modal_volume = cx.coords.compose(self.levels, self.ylm_grid)
    self.nodal_volume = cx.coords.compose(self.levels, self.grid)

  def _sample_prognostics(
      self,
      tracer_names: tuple[str, ...] = ('specific_humidity',),
      nodal_tracers: tuple[str, ...] = (),
  ) -> dict[str, cx.Field]:
    ones = lambda c: cx.field(jnp.ones(c.shape, dtype=jnp.float32), c)
    prognostics = {
        'divergence': ones(self.modal_volume),
        'vorticity': ones(self.modal_volume),
        'temperature': ones(self.modal_volume),
        'log_surface_pressure': ones(self.ylm_grid),
        'time': cx.field(jdt.to_datetime('2020-01-01T00:00')),
    }
    for k in tracer_names:
      coord = self.nodal_volume if k in nodal_tracers else self.modal_volume
      prognostics[k] = ones(coord)
    return prognostics

  def test_div_curl_target_split_axes_with_div_curl_or_uv(self):
    from_div_curl = parameterizations.div_curl_target_split_axes(
        self.levels,
        volume_field_names=('divergence', 'vorticity', 'temperature'),
        surface_field_names=('log_surface_pressure',),
    )
    from_uv = parameterizations.div_curl_target_split_axes(
        self.levels,
        volume_field_names=(
            'u_component_of_wind',
            'v_component_of_wind',
            'temperature',
        ),
        surface_field_names=('log_surface_pressure',),
    )
    expected = {
        'u_component_of_wind': self.levels,
        'v_component_of_wind': self.levels,
        'temperature': self.levels,
        'log_surface_pressure': cx.Scalar(),
    }
    self.assertEqual(from_div_curl, expected)
    self.assertEqual(from_uv, expected)

    direct_div_curl = parameterizations.div_curl_target_split_axes(
        self.levels,
        volume_field_names=('divergence', 'vorticity', 'temperature'),
        surface_field_names=('log_surface_pressure',),
        div_curl_from_uv=False,
    )
    self.assertEqual(
        direct_div_curl,
        {
            'divergence': self.levels,
            'vorticity': self.levels,
            'temperature': self.levels,
            'log_surface_pressure': cx.Scalar(),
        },
    )

  def test_div_curl_target_split_axes_validation(self):
    with self.assertRaisesRegex(
        ValueError, "must contain both 'divergence' and 'vorticity'"
    ):
      parameterizations.div_curl_target_split_axes(
          self.levels, volume_field_names=('divergence', 'temperature')
      )
    with self.assertRaisesRegex(
        ValueError, "must contain both 'u_component_of_wind'"
    ):
      parameterizations.div_curl_target_split_axes(
          self.levels,
          volume_field_names=('u_component_of_wind', 'temperature'),
      )
    with self.assertRaisesRegex(
        ValueError, 'cannot contain both divergence/vorticity and'
    ):
      parameterizations.div_curl_target_split_axes(
          self.levels,
          volume_field_names=(
              'divergence',
              'vorticity',
              'u_component_of_wind',
              'v_component_of_wind',
          ),
      )
    with self.assertRaisesRegex(ValueError, 'when `div_curl_from_uv=False`'):
      parameterizations.div_curl_target_split_axes(
          self.levels,
          volume_field_names=('u_component_of_wind', 'v_component_of_wind'),
          div_curl_from_uv=False,
      )
    with self.assertRaisesRegex(
        ValueError, 'Overlapping volume and surface field names'
    ):
      parameterizations.div_curl_target_split_axes(
          self.levels,
          volume_field_names=('temperature',),
          surface_field_names=('temperature',),
      )

  def test_primitive_equation_state_shapes(self):
    shapes = parameterizations.primitive_equation_state_shapes(
        self.ylm_map,
        self.levels,
        tracer_names=('specific_humidity', 'cloud_water'),
        nodal_tracers=('cloud_water',),
    )
    self.assertEqual(shapes['divergence'].coordinate, self.modal_volume)
    self.assertEqual(shapes['vorticity'].coordinate, self.modal_volume)
    self.assertEqual(shapes['temperature'].coordinate, self.modal_volume)
    self.assertEqual(shapes['specific_humidity'].coordinate, self.modal_volume)
    self.assertEqual(shapes['cloud_water'].coordinate, self.nodal_volume)
    self.assertEqual(shapes['log_surface_pressure'].coordinate, self.ylm_grid)
    self.assertIn('time', shapes)

    with self.assertRaisesRegex(ValueError, 'must be a subset of'):
      parameterizations.primitive_equation_state_shapes(
          self.ylm_map,
          self.levels,
          tracer_names=('specific_humidity',),
          nodal_tracers=('unknown_tracer',),
      )

  @parameterized.parameters(True, False)
  def test_build_using_factories_and_output_shapes(
      self, div_curl_from_uv: bool
  ):
    features_tx = transforms.Sequential([
        transforms.SelectKeys(
            ('divergence', 'vorticity', 'temperature', 'specific_humidity')
        ),
        atm_transforms.PrognosticsWithVelocityAndGrads(ylm_map=self.ylm_map),
    ])
    transform_factory = functools.partial(
        learned_transforms.ForwardTowerTransform.build_using_factories,
        tower_factory=_make_tower_factory(),
        concat_dims=(self.levels,),
        mesh=self.mesh,
    )
    modal_filter = spatial_filters.ExponentialModalFilter(
        ylm_map=self.ylm_map, attenuation=2.0, order=2
    )
    wind_keys = (
        ('u_component_of_wind', 'v_component_of_wind')
        if div_curl_from_uv
        else ('divergence', 'vorticity')
    )
    scale_keys = (
        *wind_keys,
        'temperature',
        'specific_humidity',
        'log_surface_pressure',
    )
    tendency_scale = transforms.StandardizeFields(
        shifts={k: cx.field(0.0) for k in scale_keys},
        scales={k: cx.field(0.5) for k in scale_keys},
        from_normalized=True,
    )
    param = parameterizations.ModalParameterization.build_using_factories(
        ylm_map=self.ylm_map,
        levels=self.levels,
        transform_factory=transform_factory,
        modal_filter=modal_filter,
        features_transform=features_tx,
        tendency_transform=tendency_scale,
        div_curl_from_uv=div_curl_from_uv,
        rngs=nnx.Rngs(0),
    )
    prognostics = self._sample_prognostics()
    tendencies = param(prognostics)
    expected_keys = {
        'divergence',
        'vorticity',
        'temperature',
        'specific_humidity',
        'log_surface_pressure',
    }
    self.assertEqual(set(tendencies), expected_keys)
    for k in expected_keys:
      self.assertEqual(tendencies[k].coordinate, prognostics[k].coordinate)

    state_shapes = parameterizations.primitive_equation_state_shapes(
        self.ylm_map, self.levels
    )
    out_shapes = param.output_shapes(state_shapes)
    self.assertEqual(set(out_shapes), expected_keys)
    for k in expected_keys:
      self.assertEqual(out_shapes[k].coordinate, prognostics[k].coordinate)

  def test_nodal_tracers_stay_nodal_and_unfiltered_and_missing_zero_filled(
      self,
  ):
    nodal_coords = {
        'u_component_of_wind': self.nodal_volume,
        'v_component_of_wind': self.nodal_volume,
        'temperature': self.nodal_volume,
        'cloud_water': self.nodal_volume,
    }
    nodal_tendency_transform = _ConstantNodalTendencies(nodal_coords, value=2.5)
    # High attenuation filter that strongly damps non-zero wavenumbers.
    modal_filter = spatial_filters.ExponentialModalFilter(
        ylm_map=self.ylm_map, attenuation=16.0, order=1
    )
    param = parameterizations.ModalParameterization(
        ylm_map=self.ylm_map,
        nodal_tendency_transform=nodal_tendency_transform,
        modal_filter=modal_filter,
    )
    prognostics = self._sample_prognostics(
        tracer_names=('specific_humidity', 'cloud_water'),
        nodal_tracers=('cloud_water',),
    )
    tendencies = param(prognostics)

    # `cloud_water` is a nodal tracer in `prognostics`: it stays on
    # `nodal_volume` and is not modified by `to_modal` or `modal_filter`.
    expected_cloud = cx.field(
        jnp.full(self.nodal_volume.shape, 2.5, dtype=jnp.float32),
        self.nodal_volume,
    )
    cx.testing.assert_fields_equal(tendencies['cloud_water'], expected_cloud)

    # `specific_humidity` and `log_surface_pressure` were not predicted by
    # `nodal_tendency_transform` and are filled with zero `cx.Field`s on their
    # prognostic coordinates.
    cx.testing.assert_fields_equal(
        tendencies['specific_humidity'],
        cx.cmap(jnp.zeros_like)(prognostics['specific_humidity']),
    )
    cx.testing.assert_fields_equal(
        tendencies['log_surface_pressure'],
        cx.cmap(jnp.zeros_like)(prognostics['log_surface_pressure']),
    )
    self.assertEqual(tendencies['divergence'].coordinate, self.modal_volume)
    self.assertEqual(tendencies['vorticity'].coordinate, self.modal_volume)
    self.assertEqual(tendencies['temperature'].coordinate, self.modal_volume)

  def test_integrates_with_hybrid_atmosphere_and_nodal_tracers(self):
    sim_units = units.SI_UNITS
    dt = np.timedelta64(1, 'h')
    modal_filter = spatial_filters.ExponentialModalFilter.from_timescale(
        ylm_map=self.ylm_map,
        dt=dt,
        timescale='2 hours',
        order=2,
        sim_units=sim_units,
    )
    # Nodal features from both modal prognostics and nodal `specific_humidity`.
    modal_state_ft = transforms.Sequential([
        transforms.SelectKeys(('divergence', 'vorticity', 'temperature')),
        atm_transforms.PrognosticsWithVelocityAndGrads(ylm_map=self.ylm_map),
    ])
    features_tx = transforms.Merge({
        'modal_state': modal_state_ft,
        'nodal_humidity': transforms.SelectKeys(('specific_humidity',)),
    })
    transform_factory = functools.partial(
        learned_transforms.ForwardTowerTransform.build_using_factories,
        tower_factory=_make_tower_factory(),
        concat_dims=(self.levels,),
        inputs_transform=features_tx,
        mesh=self.mesh,
    )
    learned_physics = (
        parameterizations.ModalParameterization.build_using_factories(
            ylm_map=self.ylm_map,
            levels=self.levels,
            transform_factory=transform_factory,
            modal_filter=modal_filter,
            surface_field_names=(),
            nodal_tracers=('specific_humidity',),
            rngs=nnx.Rngs(0),
        )
    )
    model = hybrid_atmosphere.LearnedParameterizationSpectralDycore(
        model_ylm_map=self.ylm_map,
        model_levels=self.levels,
        model_orography=orographies.ModalOrography(
            ylm_map=self.ylm_map, rngs=nnx.Rngs(0)
        ),
        model_timestep=dt,
        modal_filter=modal_filter,
        learned_physics=learned_physics,
        tracer_names=('specific_humidity',),
        nodal_tracers=('specific_humidity',),
        dycore_equation_cls=atm_equations.SemiLagrangianPrimitiveEquations,
        time_integrator_cls=time_integrators.SemiLagrangianCrankNicolsonRK2,
        mesh=self.mesh,
    )
    prognostics = self._sample_prognostics(
        tracer_names=('specific_humidity',),
        nodal_tracers=('specific_humidity',),
    )
    # Scale initial state down so primitive equation step stays well-behaved.
    init_state = {
        k: 1e-4 * v if k != 'time' else v for k, v in prognostics.items()
    }
    init_state['temperature'] = cx.field(
        jnp.zeros(self.modal_volume.shape, dtype=jnp.float32),
        self.modal_volume,
    )
    model.assimilate({'nondim_atmosphere_prognostics': init_state})
    model.advance()
    observed = model.observe(
        {'atmosphere_prognostics': {'specific_humidity': self.nodal_volume}}
    )
    self.assertEqual(
        observed['atmosphere_prognostics']['specific_humidity'].coordinate,
        self.nodal_volume,
    )

  def test_validation_errors(self):
    prognostics = self._sample_prognostics()

    only_u = parameterizations.ModalParameterization(
        ylm_map=self.ylm_map,
        nodal_tendency_transform=_ConstantNodalTendencies(
            {'u_component_of_wind': self.nodal_volume}
        ),
    )
    with self.assertRaisesRegex(
        ValueError,
        "Both 'u_component_of_wind' and 'v_component_of_wind' must be present",
    ):
      only_u(prognostics)

    both_uv_and_div = parameterizations.ModalParameterization(
        ylm_map=self.ylm_map,
        nodal_tendency_transform=_ConstantNodalTendencies({
            'u_component_of_wind': self.nodal_volume,
            'v_component_of_wind': self.nodal_volume,
            'divergence': self.nodal_volume,
        }),
    )
    with self.assertRaisesRegex(
        ValueError,
        "Tendencies cannot contain 'divergence' or 'vorticity' when "
        '`div_curl_from_uv=True`',
    ):
      both_uv_and_div(prognostics)

    uv_when_direct_div_curl = parameterizations.ModalParameterization(
        ylm_map=self.ylm_map,
        nodal_tendency_transform=_ConstantNodalTendencies({
            'u_component_of_wind': self.nodal_volume,
            'v_component_of_wind': self.nodal_volume,
        }),
        div_curl_from_uv=False,
    )
    with self.assertRaisesRegex(ValueError, 'when `div_curl_from_uv=False`'):
      uv_when_direct_div_curl(prognostics)

    unknown_key = parameterizations.ModalParameterization(
        ylm_map=self.ylm_map,
        nodal_tendency_transform=_ConstantNodalTendencies(
            {'nonexistent_var': self.nodal_volume}
        ),
    )
    with self.assertRaisesRegex(
        ValueError, "Unexpected tendency keys \\['nonexistent_var'\\]"
    ):
      unknown_key(prognostics)

    bad_shapes = parameterizations.primitive_equation_state_shapes(
        self.ylm_map, self.levels, tracer_names=('specific_humidity',)
    )
    with self.assertRaisesRegex(
        ValueError, "Nodal tracer 'specific_humidity' in `input_state_shapes`"
    ):
      parameterizations.ModalParameterization.build_using_factories(
          ylm_map=self.ylm_map,
          levels=self.levels,
          transform_factory=lambda **kw: _ConstantNodalTendencies({}),
          nodal_tracers=('specific_humidity',),
          input_state_shapes=bad_shapes,
          rngs=nnx.Rngs(0),
      )


if __name__ == '__main__':
  absltest.main()
