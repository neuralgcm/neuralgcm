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

"""Utilities for interfacing between unitful and non-dimensionalized values."""

from collections import abc
from typing import Iterator

import jax.numpy as jnp
import numpy as np
import pint

# Use the default "application" registry, because this is what pint always uses
# for unpickling. Otherwise, unpickled objects end up with incompatible
# registries.
units = pint.get_application_registry()

Quantity = units.Quantity
Unit = units.Unit
UnitsContainer = pint.util.UnitsContainer

Array = np.ndarray | jnp.ndarray
Numeric = Array | float | int

#
# Physical constants.
#

# The radius of the earth.
RADIUS = 6.37122e6 * units.m

# The rotation rate if the Earth in radians per second, often denoted Ω.
ANGULAR_VELOCITY = OMEGA = 7.292e-5 / units.s

# Acceleration due to gravity on Earth.
GRAVITY_ACCELERATION = 9.80616 * units.m / units.s**2

# Specific heat capacity at constant pressure.
ISOBARIC_HEAT_CAPACITY = 1004 * units.J / units.kilogram / units.degK

# Specific heat capacity of water vapor at constant pressure.
# value taken for T=275 from:
# https://www.engineeringtoolbox.com/water-vapor-d_979.html
WATER_VAPOR_CP = 1859 * units.J / units.kilogram / units.degK

# The mass of the dry atmosphere.
MASS_OF_DRY_ATMOSPHERE = 5.18e18 * units.kg

# The ratio of the ideal gas constant to the isobaric specific heat capacity of
# a diatomic ideal gas, often denoted κ. This value corresponds to a heat
# capacity ratio ɣ = 7 / 5. Note that this quantity is dimensionless, and this
# "unit" is included for consistency with other constants.
KAPPA = 2 / 7 * units.dimensionless

# The Latent Heat of Vaporization for water assuming T = 273.15 K
# Used to calculate enthalpy and MSE budgets as well as other moisture values
# https://glossary.ametsoc.org/wiki/Latent_heat
LATENT_HEAT_OF_VAPORIZATION = 2.501e6 * units.J / units.kilogram

# Ideas gas constant for dry air (di-atomic). For `KAPPA == 1004` this value is
# approximately 287.07.
IDEAL_GAS_CONSTANT = ISOBARIC_HEAT_CAPACITY * KAPPA

# Ideal gas constant for air with water vapor included
# Generally approximated to 461.
IDEAL_GAS_CONSTANT_H20 = 461.0 * units.J / units.kilogram / units.degK

# Freezing point of Temperature in Kelvin
# Used to convert between Kelvin and degrees Celsius
T_FREEZING = 273.15 * units.degK

# Pressure of one atmosphere (standard pressure)
# used to calculate potential temperature away from surface
REFERENCE_PRESSURE = 101325.0 * units.pascal

# Density of water in SI units
WATER_DENSITY = 997 * units.kg / units.m**3

#
# Code for defining scales and non-dimensionalizing quantities.
#


def parse_units(units_str: str) -> Quantity:
  if units_str in {'(0 - 1)', '%', '~'}:
    units_str = 'dimensionless'
  return units.parse_expression(units_str)


def _get_dimension(quantity: Quantity) -> str:
  """Asserts `quantity` has a single dimension and returns that dimension."""
  exponents = list(quantity.dimensionality.values())
  if len(quantity.dimensionality) != 1 or exponents[0] != 1:
    raise ValueError(
        'All scales must describe a single dimension;'
        f'got dimensionality {quantity.dimensionality}'
    )
  return str(quantity.dimensionality)


class Scale(abc.Mapping):
  """A `Scale` converts values to and from dimensionless quantities."""

  def __init__(self, *scales: Quantity):
    """Initializes a `Scale` from the given `Quantity`s.

    Example usage:

    ```python
    # The radius of the Earth, in meters.
    RADIUS = 6.37122e6 * units.m

    # The rotation rate of the earth, in radians per second.
    OMEGA = 7.292e-5 / units.sec

    scale = Scale(RADIUS, 1 / 2 / OMEGA)

    g = 9.8 units.m / units.s ** 2
    dimensionless_g = scale.nondimensionalize(g)
    ```

    Args:
      *scales: `pint.Quantity`s indicating scales. The dimensionality of the
        argument must be one of [current], [length], [luminosity], [mass],
        [printing_unit], [substance], [temperature], [time]. In particular, each
        scale must not have a 'compound' unit such as 'm / s'.
    """
    self._scales = dict()
    for quantity in scales:
      dimension = _get_dimension(quantity)
      if dimension in self._scales:
        raise ValueError(f'Got duplicate scales for dimension {dimension}.')
      self._scales[_get_dimension(quantity)] = quantity.to_base_units()

  def __getitem__(self, key: str) -> Quantity:
    return self._scales[key]

  def __iter__(self) -> Iterator[str]:
    return iter(self._scales)

  def __len__(self) -> int:
    return len(self._scales)

  def __repr__(self) -> str:
    return '\n'.join(
        f'{dimension}: {quantity}'
        for dimension, quantity in self._scales.items()
    )

  def _scaling_factor(
      self, dimensionality: pint.util.UnitsContainer
  ) -> Quantity:
    """Returns the value used to scale quantities of given dimensionality."""
    factor = Quantity(1)
    for dimension, exponent in dimensionality.items():
      quantity = self._scales.get(dimension)
      if quantity is None:
        raise ValueError(f'No scale has been set for {dimension}.')
      factor *= quantity**exponent
    assert factor.check(dimensionality)
    return factor

  def nondimensionalize(self, quantity: Quantity) -> Numeric:
    """Converts a `pint.Quantity` to a non-dimensional value.

    Args:
      quantity: a `pint.Quantity` to be converted to a dimensionless quantity.
        The items in `quantity.dimensionality` should be keys in this `Scale`.

    Returns:
      A dimensionless value corresponding to the rescaled `quantity`.
    """
    scaling_factor = self._scaling_factor(quantity.dimensionality)
    nondimensionalized = (quantity / scaling_factor).to(units.dimensionless)
    return nondimensionalized.magnitude

  def dimensionalize(self, value: Numeric, unit: Unit) -> Quantity:
    """Scales `value` by the appropriate factor and attached `unit`.

    Args:
      value: a dimensionless value representing a quantity corresponding to
        `unit`.
      unit: the desired units for the returned `Quantity`.

    Returns:
      A `Quantity` corresponding to `value` after rescaling.
    """
    scaling_factor = self._scaling_factor(unit.dimensionality)
    dimensionalized = value * scaling_factor
    return dimensionalized.to(unit)  # pytype: disable=attribute-error  # jax-ndarray


DEFAULT_SCALE = Scale(
    RADIUS,  # length
    1 / 2 / OMEGA,  # time
    1 * units.kilogram,  # mass
    1 * units.degK,
)  # temperature

SI_SCALE = Scale(
    1 * units.m,  # length
    1 * units.s,  # time
    1 * units.kg,  # mass
    1 * units.degK,  # temperature
)

ATMOSPHERIC_SCALE = Scale(
    RADIUS,  # length
    1 / 2 / OMEGA,  # time
    MASS_OF_DRY_ATMOSPHERE,  # mass
    1 * units.degK,
)  # temperature
