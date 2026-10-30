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

"""Tests for observation operator and inputs_spec dispatch utilities."""

from absl.testing import absltest
import coordax as cx
from flax import nnx
import jax.numpy as jnp
import jax_datetime as jdt
from neuralgcm.base import model
from neuralgcm.base import operator_utils
from neuralgcm.base import units_utils
import numpy as np
from terrax.core import coordinates
from terrax.core import diagnostics
from terrax.core import observation_operators
from terrax.core import spherical_harmonics
from terrax.core import transforms
from terrax.core import typing
from terrax.core import units
from terrax.couplers import generic


@nnx.dataclass
class _DummyComponent(model.Model):
  """Minimal `model.Model` recording `observe_nondim` calls for testing."""

  fields_by_key: dict[str, dict[str, cx.Field]] = nnx.data(default_factory=dict)
  prognostics: dict[str, cx.Field] = nnx.data(default_factory=dict)
  diag_module: diagnostics.DiagnosticModule | None = nnx.data(default=None)
  observe_calls: list[tuple[str, ...]] = nnx.static(
      default_factory=list, init=False
  )

  def __post_init__(self):
    self.operators = {
        k: observation_operators.DataObservationOperator(v)
        for k, v in self.fields_by_key.items()
    }
    super().__post_init__()

  @property
  def timestep(self) -> np.timedelta64:
    return np.timedelta64(1, 'h')

  def assimilate(self, inputs: typing.Observation) -> None:
    pass

  def advance(self) -> None:
    pass

  def prognostic_values(self) -> dict[str, cx.Field]:
    return dict(self.prognostics)

  def observe_nondim(self, queries: typing.Queries) -> typing.Observation:
    self.observe_calls.append(tuple(queries.keys()))
    return super().observe_nondim(queries)


@nnx.dataclass
class _DummyComposite(model.Model):
  """Composite `model.Model` dispatching to `components` for testing."""

  components: dict[str, model.Model] = nnx.data(default_factory=dict)
  extra_prognostics: dict[str, cx.Field] = nnx.data(default_factory=dict)
  diagnostics: dict[str, cx.Field] = nnx.data(default_factory=dict)
  num_diagnostic_values_calls: int = nnx.static(default=0, init=False)

  @property
  def timestep(self) -> np.timedelta64:
    return np.timedelta64(1, 'h')

  def assimilate(self, inputs: typing.Observation) -> None:
    pass

  def advance(self) -> None:
    pass

  def prognostic_values(self) -> dict[str, cx.Field]:
    merged = operator_utils.merge_prognostics(
        *self.components.values(), sim_units=self.sim_units
    )
    return merged | self.extra_prognostics

  def diagnostic_values(self) -> dict[str, cx.Field]:
    self.num_diagnostic_values_calls += 1
    return super().diagnostic_values() | self.diagnostics

  def observe_nondim(self, queries: typing.Queries) -> typing.Observation:
    return operator_utils.dispatch_observe_nondim(
        queries, composite=self, components=self.components
    )


class OperatorUtilsTest(absltest.TestCase):

  def test_merge_inputs_specs(self):
    t21 = coordinates.LonLatGrid.T21()
    t42 = coordinates.LonLatGrid.T42()
    spec_1 = {'ds1': {'u': t21, 'time': cx.Scalar()}}
    spec_2 = {
        'ds1': {'v': t21, 'time': cx.Scalar()},
        'ds2': {'sst': t21},
    }
    merged = operator_utils.merge_inputs_specs(spec_1, spec_2)
    self.assertEqual(
        merged,
        {
            'ds1': {'u': t21, 'v': t21, 'time': cx.Scalar()},
            'ds2': {'sst': t21},
        },
    )
    # Input dicts are not mutated in place.
    self.assertEqual(set(spec_1['ds1']), {'u', 'time'})

    with self.assertRaisesRegex(ValueError, 'Conflicting input specs'):
      operator_utils.merge_inputs_specs(spec_1, {'ds1': {'u': t42}})

  def test_merge_prognostics(self):
    ax = cx.SizedAxis('x', 2)
    t = cx.field(jdt.to_datetime('2020-01-01T00:00'))
    f1 = cx.field(jnp.array([1.0, 2.0]), ax)
    f2 = cx.field(jnp.array([3.0, 4.0]), ax)
    c1 = _DummyComponent(prognostics={'sst': f1, 'time': t})
    c2 = _DummyComponent(prognostics={'u': f2, 'time': t})
    merged = operator_utils.merge_prognostics(c1, c2)
    self.assertEqual(set(merged), {'sst', 'u', 'time'})
    cx.testing.assert_fields_equal(merged['sst'], f1)
    cx.testing.assert_fields_equal(merged['u'], f2)

    c_conflict = _DummyComponent(prognostics={'sst': f2})
    with self.assertRaisesRegex(
        ValueError, "Overlapping prognostic variable 'sst'"
    ):
      operator_utils.merge_prognostics(c1, c_conflict)

  def test_get_all_operators_collects_nested_and_top_level(self):
    ax = cx.SizedAxis('x', 2)
    f1 = cx.field(jnp.array([1.0, 2.0]), ax)
    f2 = cx.field(jnp.array([3.0, 4.0]), ax)
    f_top = cx.field(jnp.array([5.0, 6.0]), ax)
    ocean = _DummyComponent(fields_by_key={'ocean_ds': {'sst': f1}})
    atmosphere = _DummyComponent(
        fields_by_key={'atm_ds': {'u': f2}, 'shared': {'a': f1}}
    )
    aux = _DummyComponent(fields_by_key={'snotel': {'swe': f1}})

    @nnx.dataclass(kw_only=True)
    class _CompositeModel(_DummyComponent):
      ocean: model.Model
      atmosphere: model.Model
      aux_models: dict[str, model.Model] = nnx.data(default_factory=dict)

    composite = _CompositeModel(
        fields_by_key={'shared': {'a': f_top}},
        ocean=ocean,
        atmosphere=atmosphere,
        aux_models={'snotel': aux},
    )
    all_ops = operator_utils.get_all_operators(composite)
    self.assertEqual(
        set(all_ops.keys()), {'ocean_ds', 'atm_ds', 'snotel', 'shared'}
    )
    # Top-level operator overrides nested component operator for 'shared'.
    self.assertIs(all_ops['shared'], composite.operators['shared'])

  def test_build_coupler_query_instant_and_multi_time(self):
    grid = coordinates.LonLatGrid.T21()
    self.assertEqual(operator_utils.build_coupler_query(None), {})

    instant = generic.InstantDataCoupler(
        field_coords={'sea_surface_temperature': grid},
        var_to_data_keys={'sea_surface_temperature': 'era5:ocean'},
    )
    multi_time = generic.MultiTimeDataCoupler(
        field_coords={'2m_temperature': grid},
        var_to_data_keys={'2m_temperature': 'era5:surface'},
        time_data_key='era5:clock',
        n_time_slices=4,
    )
    query = operator_utils.build_coupler_query(instant, None, multi_time)
    self.assertEqual(
        query,
        {
            'era5:ocean': {'sea_surface_temperature': grid},
            'era5:surface': {'2m_temperature': grid},
            'era5:clock': {'time': cx.Scalar()},
        },
    )

  def test_build_coupler_query_raises_on_conflicting_coordinates(self):
    t21 = coordinates.LonLatGrid.T21()
    t42 = coordinates.LonLatGrid.T42()
    c1 = generic.InstantDataCoupler(
        field_coords={'2m_temperature': t21},
        var_to_data_keys={'2m_temperature': 'era5:surface'},
    )
    c2 = generic.InstantDataCoupler(
        field_coords={'2m_temperature': t42},
        var_to_data_keys={'2m_temperature': 'era5:surface'},
    )
    with self.assertRaisesRegex(ValueError, 'Conflicting coordinates'):
      operator_utils.build_coupler_query(c1, c2)

  def test_dispatch_observe_nondim_batches_and_uses_composite_state(self):
    ax = cx.SizedAxis('x', 3)
    f_ocean = cx.field(jnp.array([1.0, 2.0, 3.0]), ax)
    f_atm_1 = cx.field(jnp.array([4.0, 5.0, 6.0]), ax)
    f_atm_2 = cx.field(jnp.array([7.0, 8.0, 9.0]), ax)
    f_marker = cx.field(jnp.array([0.5, 0.5, 0.5]), ax)
    f_diag = cx.field(jnp.array([1.5, 2.5, 3.5]), ax)
    units_mapping = {
        'sst': 'kelvin',
        'u': 'meter / second',
        'v': 'meter / second',
        'marker': 'dimensionless',
        'mean_sst': 'millikelvin',
    }
    ocean = _DummyComponent(
        fields_by_key={'ocean_ds': {'sst': f_ocean}},
        prognostics={'sst': f_ocean},
        units_mapping=units_mapping,
    )
    atmosphere = _DummyComponent(
        fields_by_key={'atm_1': {'u': f_atm_1}, 'atm_2': {'v': f_atm_2}},
        prognostics={'u': f_atm_1, 'v': f_atm_2},
        units_mapping=units_mapping,
    )
    composite = _DummyComposite(
        components={'ocean': ocean, 'atmosphere': atmosphere},
        extra_prognostics={'marker': f_marker},
        diagnostics={'mean_sst': f_diag},
        # Top-level operator observes the composite's own `prognostic_values()`.
        operators={
            'top_ds': observation_operators.TransformObservationOperator(
                transforms.Identity()
            )
        },
        units_mapping=units_mapping,
    )

    queries = {
        'atm_2': {'v': ax},
        'ocean_ds': {'sst': ax},
        'top_ds': {'sst': ax, 'u': ax, 'marker': ax},
        'atm_1': {'u': ax},
    }
    observed = composite.observe(queries)
    self.assertEqual(list(observed.keys()), list(queries.keys()))
    self.assertEqual(ocean.observe_calls, [('ocean_ds',)])
    # Both atmosphere keys are batched into a single `observe_nondim` call.
    self.assertEqual(atmosphere.observe_calls, [('atm_2', 'atm_1')])
    self.assertEqual(composite.num_diagnostic_values_calls, 0)
    # With SI units and SI unit names, all outputs are returned as-is.
    self.assertIs(observed['ocean_ds']['sst'], f_ocean)
    self.assertIs(observed['atm_1']['u'], f_atm_1)
    self.assertIs(observed['atm_2']['v'], f_atm_2)
    self.assertIs(observed['top_ds']['sst'], f_ocean)
    self.assertIs(observed['top_ds']['u'], f_atm_1)
    self.assertIs(observed['top_ds']['marker'], f_marker)

    # Dimensional and nondimensional queries of the same key are evaluated as
    # two groups; `mean_sst` is redimensionalized to millikelvin.
    observed = composite.observe({
        'diagnostics': {'mean_sst': ax},
        'nondim_diagnostics': {'mean_sst': ax},
        'nondim_ocean_ds': {'sst': ax},
    })
    self.assertEqual(composite.num_diagnostic_values_calls, 2)
    cx.testing.assert_fields_allclose(
        observed['diagnostics']['mean_sst'], 1000.0 * f_diag
    )
    self.assertIs(observed['nondim_diagnostics']['mean_sst'], f_diag)
    self.assertIs(observed['nondim_ocean_ds']['sst'], f_ocean)

  def test_dispatch_observe_nondim_converts_component_units(self):
    ax = cx.SizedAxis('x', 3)
    sst_kelvin = np.array([280.0, 285.0, 290.0])
    kelvin_nondim = units_utils.unit_factor(units.DEFAULT_UNITS, 'kelvin')
    f_ocean = cx.field(jnp.asarray(sst_kelvin * kelvin_nondim), ax)
    ocean_diag = diagnostics.InstantDiagnostic(
        extract=lambda *args, **kwargs: {'sst_diag': f_ocean},
        extract_coords={'sst_diag': ax},
    )
    ocean_diag({})
    ocean = _DummyComponent(
        fields_by_key={'ocean_ds': {'sst': f_ocean}},
        prognostics={'sst': f_ocean},
        diag_module=ocean_diag,
        sim_units=units.DEFAULT_UNITS,
        units_mapping={'sst': 'kelvin', 'sst_diag': 'kelvin'},
    )
    composite = _DummyComposite(
        components={'ocean': ocean},
        sim_units=units.SI_UNITS,
        units_mapping={'sst': 'kelvin', 'sst_diag': 'kelvin'},
    )
    observed = composite.observe({
        'ocean_ds': {'sst': ax},
        'nondim_ocean_ds': {'sst': ax},
        'diagnostics': {'sst_diag': ax},
        'nondim_diagnostics': {'sst_diag': ax},
    })
    # Physical units are defined by the ocean's mapping; nondimensional values
    # are expressed in the composite's (SI) `sim_units` rather than the ocean's.
    np.testing.assert_allclose(
        observed['ocean_ds']['sst'].data, sst_kelvin, rtol=1e-6
    )
    np.testing.assert_allclose(
        observed['nondim_ocean_ds']['sst'].data, sst_kelvin, rtol=1e-6
    )
    np.testing.assert_allclose(
        observed['diagnostics']['sst_diag'].data, sst_kelvin, rtol=1e-6
    )
    np.testing.assert_allclose(
        observed['nondim_diagnostics']['sst_diag'].data, sst_kelvin, rtol=1e-6
    )
    np.testing.assert_allclose(
        composite.prognostic_values()['sst'].data, sst_kelvin, rtol=1e-6
    )
    cx.testing.assert_fields_allclose(
        ocean.observe({'nondim_ocean_ds': {'sst': ax}})['nondim_ocean_ds'][
            'sst'
        ],
        f_ocean,
    )
    # Field entries of queries are converted into the component's units and
    # round-trip back unchanged.
    provided = cx.field(jnp.asarray(sst_kelvin + 1.0), ax)
    observed = composite.observe({'ocean_ds': {'sst': provided}})
    cx.testing.assert_fields_allclose(
        observed['ocean_ds']['sst'], provided, rtol=1e-6
    )

  def test_dispatch_observe_nondim_raises_on_overlapping_or_unknown_keys(
      self,
  ):
    ax = cx.SizedAxis('x', 2)
    f = cx.field(jnp.ones(2), ax)
    ocean = _DummyComponent(fields_by_key={'shared': {'a': f}})
    atmosphere = _DummyComponent(
        fields_by_key={'shared': {'a': f}, 'atm_only': {'b': f}}
    )
    components = {'ocean': ocean, 'atmosphere': atmosphere}
    units_mapping = {'a': 'dimensionless', 'b': 'dimensionless'}
    composite = _DummyComposite(
        components=components, units_mapping=units_mapping
    )
    with self.assertRaisesRegex(ValueError, 'multiple model components'):
      composite.observe({'shared': {'a': ax}})

    # Top-level operator overrides an overlapping component key.
    override_op = observation_operators.DataObservationOperator({'a': 2.0 * f})
    composite = _DummyComposite(
        components=components,
        operators={'shared': override_op},
        units_mapping=units_mapping,
    )
    observed = composite.observe({'shared': {'a': ax}})
    cx.testing.assert_fields_allclose(observed['shared']['a'], 2.0 * f)

    with self.assertRaisesRegex(
        ValueError, "No observation operator for key 'missing'"
    ):
      composite.observe({'missing': {'a': ax}})

  def test_primitive_equations_state_operator(self):
    grid = coordinates.LonLatGrid.T21()
    ylm_grid = coordinates.SphericalHarmonicGrid.T21()
    levels = coordinates.SigmaLevels.equidistant(4)
    ylm_map = spherical_harmonics.FixedYlmMapping(
        grid, ylm_grid, radius=units.SI_UNITS.radius
    )
    modal_vol = cx.coords.compose(levels, ylm_grid)
    nodal_vol = cx.coords.compose(levels, grid)
    zeros_modal = cx.field(jnp.zeros(modal_vol.shape), modal_vol)
    time = cx.field(jdt.to_datetime('2000-01-01T00:00'))
    prognostics = {
        'divergence': zeros_modal,
        'vorticity': zeros_modal,
        'temperature': zeros_modal,
        'log_surface_pressure': cx.field(jnp.zeros(ylm_grid.shape), ylm_grid),
        'specific_humidity': cx.field(jnp.ones(nodal_vol.shape), nodal_vol),
        'time': time,
    }
    op = operator_utils.primitive_equations_state_operator(ylm_map)
    query = {
        'u_component_of_wind': nodal_vol,
        'v_component_of_wind': nodal_vol,
        'divergence': nodal_vol,
        'divergence_ylm': modal_vol,
        'vorticity_ylm': modal_vol,
        'temperature': nodal_vol,
        'temperature_ylm': modal_vol,
        'surface_pressure': grid,
        'log_surface_pressure_ylm': ylm_grid,
        'specific_humidity': nodal_vol,
        'time': cx.Scalar(),
    }
    observed = op.observe(prognostics, query)
    self.assertEqual(set(observed.keys()), set(query.keys()))
    cx.testing.assert_fields_allclose(
        observed['divergence_ylm'], zeros_modal
    )
    cx.testing.assert_fields_allclose(
        observed['surface_pressure'],
        cx.field(jnp.ones(grid.shape), grid),
    )
    # `time` is passed through, e.g. for querying by `MultiTimeDataCoupler`.
    self.assertEqual(
        observed['time'].data.to_datetime64(), time.data.to_datetime64()
    )
    # Nodal fields and `time` do not get a `'_ylm'` suffix.
    with self.assertRaises(ValueError):
      op.observe(prognostics, {'specific_humidity_ylm': nodal_vol})

if __name__ == '__main__':
  absltest.main()
