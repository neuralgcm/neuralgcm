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

"""Tests for the units contract of `neuralgcm.base.model.Model`."""

from absl.testing import absltest
import coordax as cx
from flax import nnx
import jax.numpy as jnp
from neuralgcm.base import model
import numpy as np
from terrax.core import observation_operators
from terrax.core import transforms
from terrax.core import typing
from terrax.core import units


class _AddOneMeterTransform(transforms.PytreeTransformABC):
  """Adds one meter (in `sim_units`) to `snow_depth` as `snow_depth_plus_1m`."""

  def __init__(self, sim_units: units.SimUnits):
    self.one_meter = float(sim_units.nondimensionalize(typing.Quantity('m')))

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    depth = inputs['snow_depth']
    return inputs | {'snow_depth_plus_1m': depth + self.one_meter}


@nnx.dataclass
class _StateModel(model.Model):
  """Minimal model holding nondimensional `prognostics`."""

  prognostics: dict[str, cx.Field] = nnx.data(default_factory=dict)

  @property
  def timestep(self) -> np.timedelta64:
    return np.timedelta64(1, 'h')

  def assimilate(self, inputs: typing.Observation) -> None:
    pass

  def advance(self) -> None:
    pass

  def prognostic_values(self) -> dict[str, cx.Field]:
    return dict(self.prognostics)


class ModelTest(absltest.TestCase):

  def test_defaults_and_derived_transforms(self):
    m = _StateModel()
    self.assertEqual(m.sim_units, units.SI_UNITS)
    self.assertIn('2m_temperature', m.units_mapping)
    self.assertIsInstance(m.nondim_transform, transforms.Nondimensionalize)
    self.assertIsInstance(m.redim_transform, transforms.Redimensionalize)
    self.assertEqual(m.redim_transform.inputs_to_units_mapping, m.units_mapping)

  def test_subclasses_cannot_override_observe(self):
    with self.assertRaisesRegex(TypeError, 'must implement `observe_nondim`'):

      class _Bad(_StateModel):  # pylint: disable=unused-variable

        def observe(self, queries):
          return {}

  def test_operator_keys_cannot_be_nondim_prefixed(self):
    with self.assertRaisesRegex(ValueError, "must not start with 'nondim_'"):
      _StateModel(
          operators={
              'nondim_ds': observation_operators.DataObservationOperator({})
          }
      )

  def test_observe_serves_dimensional_and_nondimensional_keys(self):
    ax = cx.SizedAxis('x', 2)
    sim_units = units.DEFAULT_UNITS
    one_meter = float(sim_units.nondimensionalize(typing.Quantity('m')))
    depth_meters = np.array([1.0, 2.0])
    depth = cx.field(jnp.asarray(depth_meters * one_meter), ax)
    station_id = cx.field(jnp.array([7, 8], jnp.int32), ax)
    m = _StateModel(
        prognostics={'snow_depth': depth, 'station_id': station_id},
        operators={
            'snotel': observation_operators.TransformObservationOperator(
                _AddOneMeterTransform(sim_units)
            )
        },
        sim_units=sim_units,
        units_mapping={
            'snow_depth': 'meter',
            'snow_depth_plus_1m': 'centimeter',
            'station_id': 'dimensionless',
        },
    )
    observed = m.observe({
        'snotel': {
            'snow_depth': ax,
            'snow_depth_plus_1m': ax,
            'station_id': ax,
        },
        'nondim_snotel': {'snow_depth': ax, 'station_id': ax},
    })
    self.assertEqual(list(observed), ['snotel', 'nondim_snotel'])
    np.testing.assert_allclose(
        observed['snotel']['snow_depth'].data, depth_meters, rtol=1e-6
    )
    # Output units follow `units_mapping` (centimeters here).
    np.testing.assert_allclose(
        observed['snotel']['snow_depth_plus_1m'].data,
        100.0 * (depth_meters + 1.0),
        rtol=1e-6,
    )
    # Dimensionless integer fields keep their dtype and identity.
    self.assertIs(observed['snotel']['station_id'], station_id)
    self.assertIs(observed['nondim_snotel']['snow_depth'], depth)
    self.assertIs(observed['nondim_snotel']['station_id'], station_id)

  def test_observe_nondimensionalizes_query_fields(self):
    ax = cx.SizedAxis('x', 2)
    sim_units = units.DEFAULT_UNITS
    m = _StateModel(
        operators={
            'snotel': observation_operators.TransformObservationOperator(
                _AddOneMeterTransform(sim_units),
                requested_fields_from_query=('snow_depth',),
            )
        },
        sim_units=sim_units,
        units_mapping={'snow_depth': 'meter', 'snow_depth_plus_1m': 'meter'},
    )
    provided_meters = cx.field(jnp.array([3.0, 4.0]), ax)
    # Under a dimensional key the provided field is in meters, is used by the
    # operator in `sim_units`, and is returned unchanged.
    observed = m.observe(
        {'snotel': {'snow_depth': provided_meters, 'snow_depth_plus_1m': ax}}
    )
    cx.testing.assert_fields_allclose(
        observed['snotel']['snow_depth'], provided_meters, rtol=1e-6
    )
    np.testing.assert_allclose(
        observed['snotel']['snow_depth_plus_1m'].data, [4.0, 5.0], rtol=1e-6
    )
    # `Auxiliary` fields are consumed but not returned.
    observed = m.observe({
        'snotel': {
            'snow_depth': typing.Auxiliary(provided_meters),
            'snow_depth_plus_1m': ax,
        }
    })
    self.assertEqual(list(observed['snotel']), ['snow_depth_plus_1m'])
    # Under a `nondim_` key the provided field is already nondimensional.
    one_meter = float(sim_units.nondimensionalize(typing.Quantity('m')))
    provided_nondim = provided_meters * one_meter
    observed = m.observe({
        'nondim_snotel': {
            'snow_depth': provided_nondim,
            'snow_depth_plus_1m': ax,
        }
    })
    self.assertIs(observed['nondim_snotel']['snow_depth'], provided_nondim)
    np.testing.assert_allclose(
        observed['nondim_snotel']['snow_depth_plus_1m'].data,
        np.array([4.0, 5.0]) * one_meter,
        rtol=1e-6,
    )

  def test_observe_raises_for_unknown_keys_and_missing_units(self):
    ax = cx.SizedAxis('x', 2)
    f = cx.field(jnp.array([1.0, 2.0]), ax)
    m = _StateModel(
        prognostics={'custom': f},
        operators={
            'ds': observation_operators.DataObservationOperator({'custom': f})
        },
        sim_units=units.DEFAULT_UNITS,
    )
    with self.assertRaisesRegex(ValueError, 'No observation operator for key'):
      m.observe({'other': {'custom': ax}})
    with self.assertRaisesRegex(ValueError, "Key 'custom' not found"):
      m.observe({'ds': {'custom': ax}})
    # Nondimensional queries need no units.
    observed = m.observe({'nondim_ds': {'custom': ax}})
    self.assertIs(observed['nondim_ds']['custom'], f)


if __name__ == '__main__':
  absltest.main()
