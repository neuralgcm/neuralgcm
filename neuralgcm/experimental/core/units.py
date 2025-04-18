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

"""Handling conversion dimensional <-> nondimensional quantities."""

from __future__ import annotations

import dataclasses

from neuralgcm.experimental.core import scales
from neuralgcm.experimental.core import typing
import numpy as np


# pylint: disable=invalid-name  # for standard physical constant names.

Numeric = typing.Numeric
Quantity = typing.Quantity


parse_units = scales.parse_units


@dataclasses.dataclass(frozen=True)
class SimUnits:
  """Provides units & time conversion methods and common constants."""

  radius: float
  angular_velocity: float
  gravity_acceleration: float
  ideal_gas_constant: float
  water_vapor_gas_constant: float
  water_vapor_isobaric_heat_capacity: float
  water_density: float
  kappa: float
  reference_datetime: np.datetime64
  scale: scales.Scale

  @property
  def R(self) -> float:
    """Alias for `ideal_gas_constant`."""
    return self.ideal_gas_constant

  @property
  def R_vapor(self) -> float:
    """Alias for `ideal_gas_constant`."""
    return self.water_vapor_gas_constant

  @property
  def g(self) -> float:
    """Alias for `gravity_acceleration`."""
    return self.gravity_acceleration

  @property
  def Cp(self) -> float:
    """Isobaric heat capacity."""
    return self.ideal_gas_constant / self.kappa

  @property
  def Cp_vapor(self) -> float:
    """Alias for `water_vapor_isobaric_heat_capacity`."""
    return self.water_vapor_isobaric_heat_capacity

  def nondimensionalize_timedelta64(self, timedelta: np.timedelta64) -> Numeric:
    """Non-dimensionalizes and rescales a numpy timedelta."""
    base_unit = 's'
    return self.scale.nondimensionalize(
        timedelta / np.timedelta64(1, base_unit) * typing.units(base_unit)
    )

  def dimensionalize_timedelta64(self, value: Numeric) -> np.timedelta64:
    """Rescales and casts the given non-dimensional value to timedelta64."""
    base_unit = 's'  # return value is rounded down to nearest base_unit
    dt = self.scale.dimensionalize(value, typing.units(base_unit)).m
    if isinstance(dt, np.ndarray):
      return dt.astype(f'timedelta64[{base_unit}]')
    else:
      return np.timedelta64(int(dt), base_unit)

  def datetime64_to_sim_time(self, datetime64: np.ndarray) -> typing.Numeric:
    """Converts a datetime64 array to sim_time."""
    return self.scale.nondimensionalize(
        ((datetime64 - self.reference_datetime) / np.timedelta64(1, 'h'))
        * typing.units.hour
    )

  def sim_time_to_datetime64(self, sim_time: np.ndarray) -> np.ndarray:
    """Converts a sim_time array to datetime64."""
    minutes = self.scale.dimensionalize(sim_time, typing.units.minute).magnitude
    delta = np.array(np.round(minutes).astype(int), 'timedelta64[m]')
    return self.reference_datetime + delta

  def nondimensionalize(
      self, quantity: typing.Quantity
  ) -> typing.Numeric:
    """Non-dimensionalizes and rescales `quantity`."""
    return self.scale.nondimensionalize(quantity)

  def dimensionalize(
      self,
      value: typing.Numeric,
      unit: typing.units.Unit,
      as_quantity: bool = True,
  ):
    """Rescales and adds units to the given non-dimensional value."""
    dimensionalized = self.scale.dimensionalize(value, unit)
    return dimensionalized if as_quantity else dimensionalized.m

  @classmethod
  def from_si(
      cls,
      radius_si: Quantity = scales.RADIUS,
      angular_velocity_si: Quantity = scales.ANGULAR_VELOCITY,
      gravity_acceleration_si: Quantity = scales.GRAVITY_ACCELERATION,
      ideal_gas_constant_si: Quantity = scales.IDEAL_GAS_CONSTANT,
      water_vapor_gas_constant_si: Quantity = scales.IDEAL_GAS_CONSTANT_H20,
      water_vapor_isobaric_heat_capacity_si: Quantity = scales.WATER_VAPOR_CP,
      water_density_si: Quantity = scales.WATER_DENSITY,
      kappa_si: Quantity = scales.KAPPA,
      reference_datetime: np.datetime64 = np.datetime64('1979-01-01T00:00:00'),
      scale: scales.Scale = scales.DEFAULT_SCALE,
  ) -> SimUnits:
    """Constructs `SimUnits` from SI constants."""
    return cls(
        scale.nondimensionalize(radius_si),
        scale.nondimensionalize(angular_velocity_si),
        scale.nondimensionalize(gravity_acceleration_si),
        scale.nondimensionalize(ideal_gas_constant_si),
        scale.nondimensionalize(water_vapor_gas_constant_si),
        scale.nondimensionalize(water_vapor_isobaric_heat_capacity_si),
        scale.nondimensionalize(water_density_si),
        scale.nondimensionalize(kappa_si),
        reference_datetime=reference_datetime,
        scale=scale,
    )


DEFAULT_UNITS = SimUnits.from_si(
    radius_si=scales.RADIUS,
    angular_velocity_si=scales.ANGULAR_VELOCITY,
    gravity_acceleration_si=scales.GRAVITY_ACCELERATION,
    ideal_gas_constant_si=scales.IDEAL_GAS_CONSTANT,
    water_vapor_gas_constant_si=scales.IDEAL_GAS_CONSTANT_H20,
    water_vapor_isobaric_heat_capacity_si=scales.WATER_VAPOR_CP,
    water_density_si=scales.WATER_DENSITY,
    kappa_si=scales.KAPPA,
    scale=scales.ATMOSPHERIC_SCALE,
    reference_datetime=np.datetime64('1979-01-01T00:00:00'),
)

ONE_MINUTE_DT_SCALE = scales.Scale(
    scales.RADIUS,
    typing.units('1 minute'),
    scales.MASS_OF_DRY_ATMOSPHERE,  # mass
    1 * typing.units.degK,
)

ONE_MINUTE_DT_UNITS = SimUnits.from_si(
    radius_si=scales.RADIUS,
    angular_velocity_si=scales.ANGULAR_VELOCITY,
    gravity_acceleration_si=scales.GRAVITY_ACCELERATION,
    ideal_gas_constant_si=scales.IDEAL_GAS_CONSTANT,
    water_vapor_gas_constant_si=scales.IDEAL_GAS_CONSTANT_H20,
    water_vapor_isobaric_heat_capacity_si=scales.WATER_VAPOR_CP,
    water_density_si=scales.WATER_DENSITY,
    kappa_si=scales.KAPPA,
    scale=ONE_MINUTE_DT_SCALE,
    reference_datetime=np.datetime64('1979-01-01T00:00:00'),
)


def maybe_nondimensionalize(
    x: typing.Numeric | typing.Quantity | str | None,
    sim_units: SimUnits,
) -> None | typing.Numeric:
  """Calls nondimensionalize on Quantity or str, otherwise passthrough."""
  if x is None:
    return None
  elif isinstance(x, (str, typing.Quantity)):
    nondim = sim_units.nondimensionalize(typing.Quantity(x))
    assert isinstance(nondim, typing.Numeric)  # make pytype happy.
    return nondim
  return x


def get_default_units() -> SimUnits:
  """units.DEFAULT_UNITS fn-wrapper for fiddle configs."""
  return DEFAULT_UNITS


def get_one_minute_dt_units() -> SimUnits:
  """units.DEFAULT_UNITS_RESCALED fn-wrapper for fiddle configs."""
  return ONE_MINUTE_DT_UNITS


def nondimensionalize_timedelta64(
    timedelta64: np.timedelta64, sim_units: SimUnits
) -> typing.Numeric:
  """sim_units.nondimensionalize_timedelta64 fn-wrapper for fiddle configs."""
  return sim_units.nondimensionalize_timedelta64(timedelta64)
