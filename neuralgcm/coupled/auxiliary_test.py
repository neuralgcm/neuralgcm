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
"""Tests for auxiliary models.

Note: Branched from `third_party/py/terrax/auxiliary/models_test.py` while
iterating on the `neuralgcm.base.model.Model` integration.
"""

from absl.testing import absltest
from absl.testing import parameterized
import coordax as cx
from flax import nnx
from jax import config  # pylint: disable=g-importing-member
import jax.numpy as jnp
import jax_datetime as jdt
from neuralgcm.base import model as base_model_lib
from neuralgcm.coupled import auxiliary as models
from neuralgcm.coupled import models as coupled_models
import numpy as np
from terrax.core import coordinates
from terrax.core import data_specs
from terrax.core import diagnostics as diagnostics_lib
from terrax.core import observation_operators
from terrax.core import transforms
from terrax.core import typing
from terrax.core import units

config.parse_flags_with_absl()


class LabeledStateModelTest(parameterized.TestCase):

  def test_basic_stage_and_advance(self):
    s_axis = cx.LabeledAxis('station', ['A', 'B'])
    model = models.LabeledStateModel(
        prognostic_coords={'total': s_axis},
        data_key='state',
        model_timestep=np.timedelta64(1, 'h'),
    )
    x = cx.field(np.array([2.5, 4.0]), s_axis)
    model.assimilate({'state': {'total': x}})
    state = model.prognostics.get_value()
    np.testing.assert_allclose(state['total'].data, [2.5, 4.0])

    update = cx.field(np.array([0.5, -1.0]), s_axis)
    model.stage_updates({'total': update})
    model.advance()

    state = model.prognostics.get_value()
    np.testing.assert_allclose(state['total'].data, [3.0, 3.0])

  def test_update_every(self):
    s_axis = cx.LabeledAxis('station', ['A', 'B'])
    model = models.LabeledStateModel(
        prognostic_coords={'total': s_axis},
        data_key='state',
        model_timestep=np.timedelta64(1, 'h'),
        update_every=np.timedelta64(3, 'h'),
    )
    x = cx.field(np.array([1.0, 2.0]), s_axis)
    model.assimilate({'state': {'total': x}})

    update = cx.field(np.array([10.0, 10.0]), s_axis)
    model.stage_updates({'total': update})

    for _ in range(2):
      model.advance()
      state = model.prognostics.get_value()
      np.testing.assert_allclose(state['total'].data, [1.0, 2.0])

    model.advance()
    state = model.prognostics.get_value()
    np.testing.assert_allclose(state['total'].data, [11.0, 12.0])

  def test_accumulate_updates(self):
    s_axis = cx.LabeledAxis('station', ['A', 'B'])
    model = models.LabeledStateModel(
        prognostic_coords={'total': s_axis},
        data_key='state',
        model_timestep=np.timedelta64(1, 'h'),
        update_every=np.timedelta64(3, 'h'),
        collect_update_method='accumulate',
    )
    x = cx.field(np.array([0.0, 0.0]), s_axis)
    model.assimilate({'state': {'total': x}})

    update = cx.field(np.array([1.0, 2.0]), s_axis)
    for _ in range(3):
      model.stage_updates({'total': update})
      model.advance()

    state = model.prognostics.get_value()
    np.testing.assert_allclose(state['total'].data, [3.0, 6.0])
    staged = model._staged_updates.get_value()
    np.testing.assert_allclose(staged['total'].data, [0.0, 0.0])

  def test_observe(self):
    s_axis = cx.LabeledAxis('station', ['A', 'B'])
    model = models.LabeledStateModel(
        prognostic_coords={'total': s_axis},
        data_key='state',
        model_timestep=np.timedelta64(1, 'h'),
    )
    x = cx.field(np.array([3.0, 5.0]), s_axis)
    model.assimilate({'state': {'total': x}})

    with self.subTest('full_state'):
      query = {'state': {'total': s_axis}}
      obs = model.observe(query)
      np.testing.assert_allclose(obs['state']['total'].data, [3.0, 5.0])

    with self.subTest('partial_state'):
      a_s_axis = cx.LabeledAxis('station', ['A'])
      query = {'state': {'total': a_s_axis}}
      obs = model.observe(query)
      np.testing.assert_allclose(obs['state']['total'].data, [3.0])

  def test_missing_assimilation_input_raises(self):
    s_axis = cx.LabeledAxis('station', ['A', 'B'])
    model = models.LabeledStateModel(
        prognostic_coords={'total': s_axis},
        data_key='state',
        model_timestep=np.timedelta64(1, 'h'),
    )
    with self.assertRaisesRegex(
        ValueError, 'missing required prognostic variable'
    ):
      model.assimilate({'state': {}})

  def test_invalid_update_every_raises(self):
    s_axis = cx.LabeledAxis('station', ['A', 'B'])
    with self.assertRaisesRegex(ValueError, 'positive integer multiple'):
      models.LabeledStateModel(
          prognostic_coords={'total': s_axis},
          data_key='state',
          model_timestep=np.timedelta64(1, 'h'),
          update_every=np.timedelta64(90, 'm'),
      )

  def test_apply_update_method_set(self):
    s_axis = cx.LabeledAxis('station', ['A', 'B'])
    model = models.LabeledStateModel(
        prognostic_coords={'total': s_axis},
        data_key='state',
        model_timestep=np.timedelta64(1, 'h'),
        apply_update_method='set',
    )
    x = cx.field(np.array([10.0, 20.0]), s_axis)
    model.assimilate({'state': {'total': x}})

    replacement = cx.field(np.array([99.0, 88.0]), s_axis)
    model.stage_updates({'total': replacement})
    model.advance()
    state = model.prognostics.get_value()
    cx.testing.assert_fields_allclose(state['total'], replacement)

  def test_update_mapping(self):
    s_axis = cx.LabeledAxis('station', ['A', 'B'])
    model = models.LabeledStateModel(
        prognostic_coords={'state_var': s_axis},
        data_key='state',
        model_timestep=np.timedelta64(1, 'h'),
        collect_update_method='accumulate',
        update_mapping={'update_var': 'state_var'},
    )
    x = cx.field(np.array([1.0, 2.0]), s_axis)
    model.assimilate({'state': {'state_var': x}})

    update = cx.field(np.array([3.0, 4.0]), s_axis)
    model.stage_updates({'update_var': update})
    model.advance()

    state = model.prognostics.get_value()
    np.testing.assert_allclose(state['state_var'].data, [4.0, 6.0])

  def test_diagnostics_merged(self):
    s_axis = cx.LabeledAxis('station', ['A', 'B'])

    extractor = diagnostics_lib.ExtractTransformedOutputs(
        transform=transforms.SelectKeys('state_var')
    )
    diagnostic = diagnostics_lib.TimeOffsetDiagnostic(
        extract=extractor,
        extract_coords={'state_var': s_axis},
        offset={'minus_24h': np.timedelta64(24, 'h')},
        resolution=np.timedelta64(24, 'h'),
        output_mapping={'state_var_minus_24h': 'condition_var'},
    )

    model = models.LabeledStateModel(
        prognostic_coords={'state_var': s_axis},
        data_key='state',
        model_timestep=np.timedelta64(24, 'h'),
        diagnostics={'t_minus_24h': diagnostic},
    )
    x = cx.field(np.array([1.0, 2.0]), s_axis)
    model.assimilate({'state': {'state_var': x}})

    # Advance two steps to fully propagate time t state into the lag buffer
    model.advance()
    model.advance()

    obs = model.observe(
        {'state': {'state_var': s_axis, 'condition_var': s_axis}}
    )
    np.testing.assert_allclose(obs['state']['condition_var'].data, [1.0, 2.0])

  def test_nondim_and_redim_units_and_nondim_queries(self):
    s_axis = cx.LabeledAxis('station', ['A', 'B'])
    sim_units = units.DEFAULT_UNITS
    model = models.LabeledStateModel(
        prognostic_coords={'snow_water_equivalent': s_axis},
        data_key='snotel',
        model_timestep=np.timedelta64(1, 'h'),
        sim_units=sim_units,
    )
    swe_meters = cx.field(np.array([1.5, 3.0]), s_axis)
    model.assimilate({'snotel': {'snow_water_equivalent': swe_meters}})

    expected_nondim = sim_units.nondimensionalize(
        typing.Quantity('meter') * np.array([1.5, 3.0])
    )
    np.testing.assert_allclose(
        model.prognostic_values()['snow_water_equivalent'].data,
        expected_nondim,
    )

    # Dimensional query returns meters; `nondim_`-prefixed query returns nondim.
    obs = model.observe({
        'snotel': {'snow_water_equivalent': s_axis},
        'nondim_snotel': {'snow_water_equivalent': s_axis},
    })
    np.testing.assert_allclose(
        obs['snotel']['snow_water_equivalent'].data, [1.5, 3.0], rtol=1e-5
    )
    np.testing.assert_allclose(
        obs['nondim_snotel']['snow_water_equivalent'].data, expected_nondim
    )

    # Assimilating from `nondim_snotel` bypasses `nondim_transform`.
    model.assimilate(
        {'nondim_snotel': {'snow_water_equivalent': swe_meters}}
    )
    np.testing.assert_allclose(
        model.prognostic_values()['snow_water_equivalent'].data, [1.5, 3.0]
    )

  def test_prognostics_data_key_and_zero_timedelta_selection(self):
    s_axis = cx.LabeledAxis('station', ['A', 'B'])
    td_axis = coordinates.TimeDelta(
        np.array([-24, 0, 24], dtype='timedelta64[h]')
    )
    model = models.LabeledStateModel(
        prognostic_coords={'snow_water_equivalent': s_axis},
        data_key='snotel',
        model_timestep=np.timedelta64(1, 'h'),
        prognostics_data_key='snotel_prognostics',
    )
    swe = cx.field(
        np.array([[1.0, 2.0], [10.0, 20.0], [100.0, 200.0]]),
        cx.coords.compose(td_axis, s_axis),
    )
    model.assimilate({'snotel_prognostics': {'snow_water_equivalent': swe}})
    np.testing.assert_allclose(
        model.prognostic_values()['snow_water_equivalent'].data, [10.0, 20.0]
    )
    self.assertIn('snotel_prognostics', model.inputs_spec)
    self.assertIn('nondim_snotel_prognostics', model.inputs_spec)

  def test_per_variable_update_methods_and_unupdated_set_preservation(self):
    s_axis = cx.LabeledAxis('station', ['A', 'B'])
    model = models.LabeledStateModel(
        prognostic_coords={
            'snow_water_equivalent': s_axis,
            '2m_temperature': s_axis,
            'time_offset': s_axis,
        },
        data_key='stations',
        model_timestep=np.timedelta64(1, 'h'),
        update_every=np.timedelta64(2, 'h'),
        collect_update_method={
            'snow_water_equivalent': 'accumulate',
            '2m_temperature': 'set',
        },
        apply_update_method={
            'snow_water_equivalent': 'add',
            '2m_temperature': 'set',
            'time_offset': 'set',
        },
    )
    model.assimilate({
        'stations': {
            'snow_water_equivalent': cx.field(np.array([1.0, 2.0]), s_axis),
            '2m_temperature': cx.field(np.array([280.0, 290.0]), s_axis),
            'time_offset': cx.field(np.array([0.25, -0.5]), s_axis),
        }
    })
    for step in range(2):
      model.stage_updates({
          'snow_water_equivalent': cx.field(np.array([0.5, 1.0]), s_axis),
          '2m_temperature': cx.field(
              np.array([281.0 + step, 291.0 + step]), s_axis
          ),
      })
      model.advance()
    state = model.prognostic_values()
    np.testing.assert_allclose(
        state['snow_water_equivalent'].data, [2.0, 4.0]
    )
    np.testing.assert_allclose(state['2m_temperature'].data, [282.0, 292.0])
    # `time_offset` is in `'set'` mode but was not in `stage_updates`, so it
    # preserves its assimilated value rather than being overwritten with 0.
    np.testing.assert_allclose(state['time_offset'].data, [0.25, -0.5])

  def test_instant_diagnostic_tracks_lag_state_at_assimilate_and_advance(
      self,
  ):
    s_axis = cx.LabeledAxis('station', ['A', 'B'])
    lag_diag = diagnostics_lib.InstantDiagnostic(
        extract=diagnostics_lib.ExtractTransformedOutputs(
            transform=transforms.Sequential([
                transforms.SelectKeys(['2m_temperature', 'time_offset']),
                transforms.RenameKeys({
                    '2m_temperature': '2m_temperature_1h_lag',
                    'time_offset': 'time_offset_1h_lag',
                }),
            ])
        ),
        extract_coords={
            '2m_temperature_1h_lag': s_axis,
            'time_offset_1h_lag': s_axis,
        },
    )
    model = models.LabeledStateModel(
        prognostic_coords={'2m_temperature': s_axis, 'time_offset': s_axis},
        data_key='metar',
        model_timestep=np.timedelta64(1, 'h'),
        apply_update_method='set',
        diagnostics={'lag_1h': lag_diag},
    )
    t0 = cx.field(np.array([280.0, 290.0]), s_axis)
    offset0 = cx.field(np.array([0.1, -0.2]), s_axis)
    model.assimilate({'metar': {'2m_temperature': t0, 'time_offset': offset0}})

    # Immediately after `assimilate`, `*_1h_lag` diagnostics match initial
    # state.
    obs_0 = model.observe({
        'metar': {
            '2m_temperature': s_axis,
            '2m_temperature_1h_lag': s_axis,
            'time_offset_1h_lag': s_axis,
        }
    })
    np.testing.assert_allclose(
        obs_0['metar']['2m_temperature_1h_lag'].data, [280.0, 290.0]
    )
    np.testing.assert_allclose(
        obs_0['metar']['time_offset_1h_lag'].data, [0.1, -0.2]
    )

    # Stage new temperature and advance; `2m_temperature` updates while
    # `2m_temperature_1h_lag` retains the pre-update temperature.
    t1 = cx.field(np.array([285.0, 295.0]), s_axis)
    model.stage_updates({'2m_temperature': t1})
    model.advance()
    obs_1 = model.observe({
        'metar': {
            '2m_temperature': s_axis,
            '2m_temperature_1h_lag': s_axis,
            'time_offset_1h_lag': s_axis,
        },
        'diagnostics': {'2m_temperature_1h_lag': s_axis},
    })
    np.testing.assert_allclose(
        obs_1['metar']['2m_temperature'].data, [285.0, 295.0]
    )
    np.testing.assert_allclose(
        obs_1['metar']['2m_temperature_1h_lag'].data, [280.0, 290.0]
    )
    np.testing.assert_allclose(
        obs_1['diagnostics']['2m_temperature_1h_lag'].data, [280.0, 290.0]
    )


def _get_station_rates(coord: cx.LabeledAxis) -> jnp.ndarray:
  """Computes decay rates dynamically based on station label indices."""
  rates = []
  for tick in coord.ticks:
    station_id = int(tick[1:])
    rates.append(-0.1 * ((1 / 5) ** (station_id - 1)))
  return jnp.array(rates)


class _DecreaseToZeroModel(base_model_lib.Model):
  """Base model where delta observation requires total state as a Field."""

  def prognostic_values(self) -> dict[str, cx.Field]:
    return {}

  def assimilate(self, inputs):
    pass

  def advance(self):
    pass

  def observe(self, queries):
    results = {}
    for op_key, q in queries.items():
      if op_key in ('state', 'nondim_state'):
        obs = {}
        delta_q_coord, delta_is_aux = typing.unwrap_auxiliary(q.get('delta'))
        total, total_is_aux = typing.unwrap_auxiliary(q.get('total'))

        if delta_q_coord is not None:
          if not cx.is_field(total):
            raise ValueError('Model requires total to be a Field in query.')
          rates = cx.field(_get_station_rates(delta_q_coord), delta_q_coord)
          delta = cx.cmap(jnp.where)(total > 0, rates, 0.0)

          if not delta_is_aux:
            obs['delta'] = delta
          if not total_is_aux:
            obs['total'] = total
        results[op_key] = obs
    return results

  @property
  def timestep(self):
    return np.timedelta64(1, 'h')

  @property
  def inputs_spec(self):
    return {}


@nnx.dataclass
class _OperatorDrivenModel(base_model_lib.Model):
  """Minimal `Model` that evaluates `self.operators` with unit conversions."""

  prognostics: dict[str, cx.Field] = nnx.data(default_factory=dict)
  model_timestep: np.timedelta64 = nnx.static(
      default=np.timedelta64(1, 'h')
  )

  def prognostic_values(self) -> dict[str, cx.Field]:
    return dict(self.prognostics)

  def assimilate(self, inputs: typing.Observation) -> None:
    pass

  def advance(self) -> None:
    pass

  def observe(self, queries: typing.Queries) -> typing.Observation:
    outputs = {}
    for k, q in queries.items():
      op_key = (
          k.removeprefix('nondim_')
          if k not in self.operators and k.startswith('nondim_')
          else k
      )
      obs = self.operators[op_key].observe(self.prognostic_values(), q)
      if self.redim_transform is not None and not k.startswith('nondim_'):
        obs = self.redim_transform(obs)
      outputs[k] = obs
    return outputs

  @property
  def timestep(self) -> np.timedelta64:
    return self.model_timestep

  @property
  def inputs_spec(self):
    return {}


class WithObservedStateTest(parameterized.TestCase):

  def _make_wrapped_model(self):
    s_axis = cx.LabeledAxis('station', ['S1', 'S2'])
    base_model = _DecreaseToZeroModel()
    obs_model = models.LabeledStateModel(
        prognostic_coords={'total': s_axis},
        data_key='state',
        model_timestep=np.timedelta64(1, 'h'),
    )
    wrapper = models.WithObservedState(
        base_model=base_model,
        observation_models={'state': obs_model},
        coupling_query={},
        observation_query_forwarding={'state': ('total',)},
    )
    return wrapper, s_axis

  def test_base_model_requires_field_in_query(self):
    base_model = _DecreaseToZeroModel()
    s_axis = cx.LabeledAxis('station', ['S1', 'S2'])

    query_invalid = {'state': {'delta': s_axis, 'total': s_axis}}
    with self.assertRaisesRegex(ValueError, 'Field in query'):
      base_model.observe(query_invalid)

    total_field = cx.field(np.array([1.0, 0.0]), s_axis)
    query_valid = {'state': {'delta': s_axis, 'total': total_field}}
    obs = base_model.observe(query_valid)
    np.testing.assert_allclose(obs['state']['delta'].data, [-0.1, 0.0])
    np.testing.assert_allclose(obs['state']['total'].data, [1.0, 0.0])

  def test_wrapper_forwards_query_and_enables_coordinate_query(self):
    wrapper, s_axis = self._make_wrapped_model()
    x = cx.field(np.array([1.0, 0.0]), s_axis)
    wrapper.assimilate({'state': {'total': x}})

    query = {'state': {'delta': s_axis, 'total': s_axis}}
    obs = wrapper.observe(query)
    self.assertIn('state', obs)
    np.testing.assert_allclose(obs['state']['delta'].data, [-0.1, 0.0])
    np.testing.assert_allclose(obs['state']['total'].data, [1.0, 0.0])

  def test_wrapper_observes_subset_of_stations(self):
    wrapper, s_axis = self._make_wrapped_model()
    x = cx.field(np.array([1.0, 0.5]), s_axis)
    wrapper.assimilate({'state': {'total': x}})

    # Query with only station S2.
    s_subset = cx.LabeledAxis('station', ['S2'])
    query = {'state': {'delta': s_subset, 'total': s_subset}}
    obs = wrapper.observe(query)
    self.assertIn('state', obs)
    np.testing.assert_allclose(obs['state']['delta'].data, [-0.02])
    np.testing.assert_allclose(obs['state']['total'].data, [0.5])

  def test_observe_preserves_auxiliary_tags(self):
    wrapper, s_axis = self._make_wrapped_model()
    x = cx.field(np.array([1.0, 0.0]), s_axis)
    wrapper.assimilate({'state': {'total': x}})
    query = {'state': {'delta': s_axis, 'total': typing.Auxiliary(s_axis)}}
    obs = wrapper.observe(query)
    self.assertIn('state', obs)
    self.assertIn('delta', obs['state'])
    self.assertNotIn('total', obs['state'])
    np.testing.assert_allclose(obs['state']['delta'].data, [-0.1, 0.0])

  def test_observe_missing_forwarded_key_raises(self):
    wrapper, s_axis = self._make_wrapped_model()
    query_missing = {'state': {'delta': s_axis}}
    with self.assertRaisesRegex(ValueError, 'requires key'):
      wrapper.observe(query_missing)

  @parameterized.named_parameters(
      # Data for 'total' is forwarded and used by the base model as is.
      dict(testcase_name='present', is_present=True, expected=[0.0, -0.02]),
      # Missing data is requested, i.e. resolved by the observation model.
      dict(testcase_name='missing', is_present=False, expected=[-0.1, 0.0]),
  )
  def test_optional_forwarded_entry_falls_back_to_observation_model(
      self, is_present, expected
  ):
    wrapper, s_axis = self._make_wrapped_model()
    x = cx.field(np.array([1.0, 0.0]), s_axis)
    wrapper.assimilate({'state': {'total': x}})
    optional_total = data_specs.OptionalSpec(
        data_specs.FieldInQuerySpec(s_axis)
    )
    queries_spec = {
        'state': {'delta': s_axis, 'total': typing.Auxiliary(optional_total)}
    }
    data = {'state': {'total': cx.field(np.array([0.0, 1.0]), s_axis)}}
    queries = data_specs.construct_query(
        data if is_present else {}, queries_spec
    )
    obs = wrapper.observe(queries)
    self.assertEqual(set(obs['state']), {'delta'})
    expected_delta = cx.field(np.array(expected), s_axis)
    cx.testing.assert_fields_allclose(obs['state']['delta'], expected_delta)

  def test_dual_coupling_and_observation_query_forwarding(self):
    s_axis = cx.LabeledAxis('station', ['S1', 'S2'])
    base_model = _DecreaseToZeroModel()
    obs_model = models.LabeledStateModel(
        prognostic_coords={'prog_total': s_axis, 'lag_total': s_axis},
        data_key='state',
        model_timestep=np.timedelta64(1, 'h'),
        update_mapping={'delta': 'prog_total'},
    )
    wrapper = models.WithObservedState(
        base_model=base_model,
        observation_models={'state': obs_model},
        coupling_query={
            'state': {'delta': s_axis, 'total': typing.Auxiliary(s_axis)}
        },
        observation_query_forwarding={'state': {'total': 'lag_total'}},
        coupling_query_forwarding={'state': {'total': 'prog_total'}},
    )
    x_prog = cx.field(np.array([10.0, 5.0]), s_axis)
    x_lag = cx.field(np.array([100.0, 50.0]), s_axis)
    wrapper.assimilate({'state': {'prog_total': x_prog, 'lag_total': x_lag}})

    query = {'state': {'delta': s_axis, 'total': s_axis}}
    obs = wrapper.observe(query)
    np.testing.assert_allclose(obs['state']['delta'].data, [-0.1, -0.02])
    np.testing.assert_allclose(obs['state']['total'].data, [100.0, 50.0])

    wrapper.advance()
    state = wrapper.observation_models['state'].prognostics.get_value()
    np.testing.assert_allclose(state['prog_total'].data, [9.9, 4.98])

  def test_direct_state_observation_shortcut(self):
    wrapper, s_axis = self._make_wrapped_model()
    x = cx.field(np.array([10.0, 20.0]), s_axis)
    wrapper.assimilate({'state': {'total': x}})
    query = {'state_state': {'total': s_axis}}
    obs = wrapper.observe(query)
    self.assertIn('state_state', obs)
    np.testing.assert_allclose(obs['state_state']['total'].data, [10.0, 20.0])

  def test_timestep_delegated_to_base(self):
    wrapper, _ = self._make_wrapped_model()
    self.assertEqual(wrapper.timestep, np.timedelta64(1, 'h'))

  def test_inputs_spec_merges_both_models(self):
    wrapper, _ = self._make_wrapped_model()
    spec = wrapper.inputs_spec
    self.assertIn('state', spec)
    self.assertIn('total', spec['state'])

  def test_nondim_coupling_and_observation_with_default_units(self):
    s_axis = cx.LabeledAxis('station', ['S1', 'S2'])
    sim_units = units.DEFAULT_UNITS
    one_meter_nondim = float(
        sim_units.nondimensionalize(typing.Quantity('1 meter'))
    )

    class _AddOneMeterTransform(transforms.PytreeTransformABC):

      def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
        prev = inputs['snow_water_equivalent_day_before_PST']
        diff = cx.field(
            jnp.full(prev.shape, one_meter_nondim), prev.coordinate
        )
        return inputs | {'snow_water_equivalent_daily_diff_PST': diff}

    snotel_op = observation_operators.TransformObservationOperator(
        _AddOneMeterTransform(),
        requested_fields_from_query=('snow_water_equivalent_day_before_PST',),
    )
    units_map = {
        'snow_water_equivalent_day_before_PST': 'meter',
        'snow_water_equivalent_daily_diff_PST': 'meter',
    }
    base_model = _OperatorDrivenModel(
        sim_units=sim_units,
        nondim_transform=transforms.Nondimensionalize(sim_units, units_map),
        redim_transform=transforms.Redimensionalize(sim_units, units_map),
        operators={'snotel': snotel_op},
    )
    obs_model = models.LabeledStateModel(
        prognostic_coords={'snow_water_equivalent_day_before_PST': s_axis},
        data_key='snotel',
        model_timestep=np.timedelta64(1, 'h'),
        update_mapping={
            'snow_water_equivalent_daily_diff_PST': (
                'snow_water_equivalent_day_before_PST'
            )
        },
        sim_units=sim_units,
    )
    wrapper = models.WithObservedState(
        base_model=base_model,
        observation_models={'snotel': obs_model},
        coupling_query={
            'nondim_snotel': {
                'snow_water_equivalent_daily_diff_PST': s_axis,
                'snow_water_equivalent_day_before_PST': typing.Auxiliary(
                    s_axis
                ),
            }
        },
        observation_query_forwarding={
            'snotel': ('snow_water_equivalent_day_before_PST',)
        },
        coupling_query_forwarding={
            'snotel': ('snow_water_equivalent_day_before_PST',)
        },
        sim_units=sim_units,
    )
    init_swe = cx.field(np.array([2.0, 5.0]), s_axis)
    wrapper.assimilate(
        {'snotel': {'snow_water_equivalent_day_before_PST': init_swe}}
    )
    wrapper.advance()

    obs = wrapper.observe({
        'snotel': {
            'snow_water_equivalent_daily_diff_PST': s_axis,
            'snow_water_equivalent_day_before_PST': s_axis,
        },
        'snotel_state': {'snow_water_equivalent_day_before_PST': s_axis},
        'nondim_snotel_state': {'snow_water_equivalent_day_before_PST': s_axis},
    })
    np.testing.assert_allclose(
        obs['snotel']['snow_water_equivalent_day_before_PST'].data,
        [3.0, 6.0],
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        obs['snotel']['snow_water_equivalent_daily_diff_PST'].data,
        [1.0, 1.0],
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        obs['snotel_state']['snow_water_equivalent_day_before_PST'].data,
        [3.0, 6.0],
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        obs['nondim_snotel_state']['snow_water_equivalent_day_before_PST'].data,
        np.array([3.0, 6.0]) * one_meter_nondim,
        rtol=1e-5,
    )

    # Mismatched `sim_units` on a `nondim_` coupling query raises `ValueError`.
    obs_model_si = models.LabeledStateModel(
        prognostic_coords={'snow_water_equivalent_day_before_PST': s_axis},
        data_key='snotel',
        model_timestep=np.timedelta64(1, 'h'),
        sim_units=units.SI_UNITS,
    )
    with self.assertRaisesRegex(ValueError, 'requires matching `sim_units`'):
      models.WithObservedState(
          base_model=base_model,
          observation_models={'snotel': obs_model_si},
          coupling_query={
              'nondim_snotel': {'snow_water_equivalent_daily_diff_PST': s_axis}
          },
      )

  def test_multi_station_snotel_delta_and_metar_value_with_time_offset(self):
    snotel_axis = cx.LabeledAxis('station', ['SN1', 'SN2'])
    metar_axis = cx.LabeledAxis('station', ['M1', 'M2'])

    class _SnotelTransform(transforms.PytreeTransformABC):

      def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
        # Daily diff depends on station elevation and previous SWE.
        elev = inputs['elevation']
        swe_prev = inputs['snow_water_equivalent_day_before_PST']
        diff = 0.001 * elev + 0.0 * swe_prev
        return inputs | {'snow_water_equivalent_daily_diff_PST': diff}

    class _MetarTransform(transforms.PytreeTransformABC):

      def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
        # New temperature shifts from `2m_temperature_1h_lag` by
        # `2.0 * (time_offset - time_offset_1h_lag)`.
        t_lag = inputs['2m_temperature_1h_lag']
        dt_offset = inputs['time_offset'] - inputs['time_offset_1h_lag']
        t_new = t_lag + 1.0 + 2.0 * dt_offset
        w_new = 0.01 * inputs['elevation']
        return inputs | {'2m_temperature': t_new, '10m_wind_speed': w_new}

    snotel_prescribed = transforms.PrescribedFields({
        'longitude': cx.field(np.array([-120.0, -110.0]), snotel_axis),
        'latitude': cx.field(np.array([40.0, 45.0]), snotel_axis),
        'elevation': cx.field(np.array([1000.0, 2000.0]), snotel_axis),
        'time_offset': cx.field(np.array([0.0, 0.0]), snotel_axis),
    })
    metar_fixed_offset = cx.field(np.array([0.25, -0.25]), metar_axis)
    metar_prescribed = transforms.PrescribedFields({
        'longitude': cx.field(np.array([-100.0, -90.0]), metar_axis),
        'latitude': cx.field(np.array([35.0, 38.0]), metar_axis),
        'elevation': cx.field(np.array([500.0, 800.0]), metar_axis),
        'time_offset': metar_fixed_offset,
    })

    snotel_op = observation_operators.TransformObservationOperator(
        _SnotelTransform(),
        requested_fields_from_query=(
            'longitude',
            'latitude',
            'elevation',
            'time_offset',
            'snow_water_equivalent_day_before_PST',
        ),
        fallback_transform=snotel_prescribed,
    )
    metar_op = observation_operators.TransformObservationOperator(
        _MetarTransform(),
        requested_fields_from_query=(
            'longitude',
            'latitude',
            'elevation',
            'time_offset',
            '2m_temperature_1h_lag',
            'time_offset_1h_lag',
        ),
        fallback_transform=metar_prescribed,
    )
    base_model = _OperatorDrivenModel(
        operators={'snotel': snotel_op, 'metar': metar_op}
    )

    snotel_state = models.LabeledStateModel(
        prognostic_coords={'snow_water_equivalent_day_before_PST': snotel_axis},
        data_key='snotel',
        model_timestep=np.timedelta64(1, 'h'),
        update_every=np.timedelta64(2, 'h'),
        collect_update_method='set',
        apply_update_method='add',
        update_mapping={
            'snow_water_equivalent_daily_diff_PST': (
                'snow_water_equivalent_day_before_PST'
            )
        },
    )
    metar_lag_diag = diagnostics_lib.InstantDiagnostic(
        extract=diagnostics_lib.ExtractTransformedOutputs(
            transform=transforms.Sequential([
                transforms.SelectKeys(['2m_temperature', 'time_offset']),
                transforms.RenameKeys({
                    '2m_temperature': '2m_temperature_1h_lag',
                    'time_offset': 'time_offset_1h_lag',
                }),
            ])
        ),
        extract_coords={
            '2m_temperature_1h_lag': metar_axis,
            'time_offset_1h_lag': metar_axis,
        },
    )
    metar_state = models.LabeledStateModel(
        prognostic_coords={
            '2m_temperature': metar_axis,
            '10m_wind_speed': metar_axis,
            'time_offset': metar_axis,
        },
        data_key='metar',
        model_timestep=np.timedelta64(1, 'h'),
        update_every=np.timedelta64(1, 'h'),
        collect_update_method='set',
        apply_update_method='set',
        diagnostics={'lag_1h': metar_lag_diag},
    )

    wrapper = models.WithObservedState(
        base_model=base_model,
        observation_models={'snotel': snotel_state, 'metar': metar_state},
        coupling_query={
            'nondim_snotel': {
                'snow_water_equivalent_daily_diff_PST': snotel_axis,
                'snow_water_equivalent_day_before_PST': typing.Auxiliary(
                    snotel_axis
                ),
            },
            'nondim_metar': {
                '2m_temperature': metar_axis,
                '10m_wind_speed': metar_axis,
                '2m_temperature_1h_lag': typing.Auxiliary(metar_axis),
                'time_offset_1h_lag': typing.Auxiliary(metar_axis),
            },
        },
        observation_query_forwarding={
            'snotel': ('snow_water_equivalent_day_before_PST',),
            'metar': ('2m_temperature_1h_lag', 'time_offset_1h_lag'),
        },
        coupling_query_forwarding={
            'snotel': ('snow_water_equivalent_day_before_PST',),
            'metar': {
                '2m_temperature_1h_lag': '2m_temperature',
                'time_offset_1h_lag': 'time_offset',
            },
        },
    )

    wrapper.assimilate({
        'snotel': {
            'snow_water_equivalent_day_before_PST': cx.field(
                np.array([10.0, 20.0]), snotel_axis
            )
        },
        'metar': {
            '2m_temperature': cx.field(np.array([280.0, 290.0]), metar_axis),
            '10m_wind_speed': cx.field(np.array([2.0, 3.0]), metar_axis),
            'time_offset': metar_fixed_offset,
        },
    })

    # Step 1: `metar` updates on every 1h step (t: +1.0 -> [281, 291]);
    # `snotel` only stages its update until step 2.
    wrapper.advance()
    np.testing.assert_allclose(
        snotel_state.prognostic_values()[
            'snow_water_equivalent_day_before_PST'
        ].data,
        [10.0, 20.0],
    )
    np.testing.assert_allclose(
        metar_state.prognostic_values()['2m_temperature'].data, [281.0, 291.0]
    )
    np.testing.assert_allclose(
        metar_state.prognostic_values()['10m_wind_speed'].data, [5.0, 8.0]
    )

    # Step 2: `snotel` applies its 2h update (+1.0, +2.0 -> [11, 22]);
    # `metar` updates again ([282, 292]) with `2m_temperature_1h_lag`=[281, 291]
    wrapper.advance()
    np.testing.assert_allclose(
        snotel_state.prognostic_values()[
            'snow_water_equivalent_day_before_PST'
        ].data,
        [11.0, 22.0],
    )
    np.testing.assert_allclose(
        metar_state.prognostic_values()['2m_temperature'].data, [282.0, 292.0]
    )

    # User queries `metar` at an arbitrary `time_offset` (shifted by +0.5 from
    # `metar_fixed_offset`), while `2m_temperature_1h_lag` ([281, 291]) and
    # `time_offset_1h_lag` ([0.25, -0.25]) are forwarded from `metar_state`.
    custom_offset = metar_fixed_offset + 0.5
    obs = wrapper.observe({
        'metar': {
            '2m_temperature': metar_axis,
            'time_offset': typing.Auxiliary(custom_offset),
            '2m_temperature_1h_lag': typing.Auxiliary(metar_axis),
            'time_offset_1h_lag': typing.Auxiliary(metar_axis),
        }
    })
    # Expected: t_before ([281, 291]) + 1.0 + 2.0 * 0.5 = [283.0, 293.0].
    np.testing.assert_allclose(
        obs['metar']['2m_temperature'].data, [283.0, 293.0]
    )

  def test_wrapping_coupled_ocean_atmosphere_model(self):
    s_axis = cx.LabeledAxis('station', ['S1', 'S2'])
    t = cx.field(jdt.to_datetime('2020-01-01T00:00'))
    ocean = _OperatorDrivenModel(
        prognostics={
            'sea_surface_temperature': cx.field(jnp.ones(2), s_axis),
            'time': t,
        },
        operators={
            'ocean_ds': observation_operators.TransformObservationOperator(
                transforms.Identity()
            )
        },
    )
    atmosphere = _OperatorDrivenModel(
        prognostics={
            '2m_temperature': cx.field(2.0 * jnp.ones(2), s_axis),
            'time': t,
        },
        operators={
            'snotel': observation_operators.TransformObservationOperator(
                transforms.Identity(),
                requested_fields_from_query=('snow_water_equivalent',),
            )
        },
    )
    coupled = coupled_models.OceanAtmosphere(
        ocean=ocean,
        atmosphere=atmosphere,
    )
    snotel_state = models.LabeledStateModel(
        prognostic_coords={'snow_water_equivalent': s_axis},
        data_key='snotel',
        model_timestep=np.timedelta64(1, 'h'),
    )
    wrapper = models.WithObservedState(
        base_model=coupled,
        observation_models={'snotel': snotel_state},
        coupling_query={},
        observation_query_forwarding={'snotel': ('snow_water_equivalent',)},
    )
    wrapper.assimilate({
        'snotel': {
            'snow_water_equivalent': cx.field(np.array([3.0, 4.0]), s_axis)
        }
    })
    self.assertEqual(
        set(wrapper.prognostic_values().keys()),
        {
            'sea_surface_temperature',
            '2m_temperature',
            'snow_water_equivalent',
            'time',
        },
    )
    obs = wrapper.observe({'snotel': {'snow_water_equivalent': s_axis}})
    np.testing.assert_allclose(
        obs['snotel']['snow_water_equivalent'].data, [3.0, 4.0]
    )


if __name__ == '__main__':
  config.update('jax_traceback_filtering', 'off')
  absltest.main()
