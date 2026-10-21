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

"""Base Model class for NeuralGCM forecast models."""

import abc

import coordax as cx
from flax import nnx
from terrax.core import api
from terrax.core import transforms
from terrax.core import typing
from terrax.core import units

# pylint: disable=g-classes-have-attributes


@nnx.dataclass
class Model(api.Model, abc.ABC):
  """Base class for stateful NeuralGCM models.

  Extends `terrax.core.api.Model` with unit conversion transforms (`sim_units`,
  `nondim_transform`, `redim_transform`), an `operators` dictionary of
  observation operators, and an abstract `prognostic_values()` method
  (paralleling `diagnostic_values()`) that returns the model's current
  nondimensional prognostic fields.

  Attributes:
    sim_units: Simulation units helper. Defaults to `units.SI_UNITS`.
    nondim_transform: Transform that nondimensionalizes inputs from physical
      units into the model's simulation units.
    redim_transform: Transform that redimensionalizes observations from the
      model's simulation units into physical units.
    operators: Observation operators keyed by dataset or operator name. By
      convention, `observe` redimensionalizes observations using
      `redim_transform`, unless the query key is prefixed with `'nondim_'`, in
      which case the operator for the unprefixed key returns nondimensional
      values.
  """

  sim_units: units.SimUnits = nnx.static(default=units.SI_UNITS, kw_only=True)
  nondim_transform: transforms.Nondimensionalize | None = nnx.data(
      default=None, kw_only=True
  )
  redim_transform: transforms.Redimensionalize | None = nnx.data(
      default=None, kw_only=True
  )
  operators: dict[str, typing.ObservationOperator] = nnx.data(
      default_factory=dict, kw_only=True
  )

  @abc.abstractmethod
  def prognostic_values(self) -> dict[str, cx.Field]:
    """Returns current nondimensional prognostic values of the model."""
    raise NotImplementedError()
