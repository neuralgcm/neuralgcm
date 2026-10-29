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
from collections.abc import Mapping

import coordax as cx
from flax import nnx
from neuralgcm.base import defaults
from neuralgcm.base import units_utils
from terrax.core import api
from terrax.core import observation_operators
from terrax.core import transforms
from terrax.core import typing
from terrax.core import units

# pylint: disable=g-classes-have-attributes

NONDIM_PREFIX = 'nondim_'


@nnx.dataclass
class Model(api.Model, abc.ABC):
  """Base class for stateful NeuralGCM models.

  Extends `terrax.core.api.Model` with a units contract, an `operators`
  dictionary of observation operators, and an abstract `prognostic_values()`
  method (paralleling `diagnostic_values()`) that returns the model's current
  nondimensional prognostic fields.

  Units contract: a model evolves nondimensional state in `sim_units` and
  interacts with the user in physical units defined by `units_mapping`, unless
  a dataset key is prefixed with `'nondim_'`:

  * `assimilate(inputs)` reads dataset `k` in physical units and `nondim_k` in
    nondimensional `sim_units`.
  * `observe(queries)` returns observations for query key `k` in physical units
    and for `nondim_k` in nondimensional `sim_units`. `Field` entries supplied
    in a query follow the same convention as the outputs of that key and are
    nondimensionalized before reaching observation operators.

  `observe` is implemented here in terms of `observe_nondim`, which operates
  entirely in nondimensional units, so subclasses implement `observe_nondim`
  (the default dispatches to `operators` and `'diagnostics'`) and must not
  override `observe`. Composite models re-express observations of their
  components in their own `sim_units` via `units_utils.units_passage`, so that
  the units of the outermost model define the interface.

  Attributes:
    sim_units: Simulation units helper. Defaults to `units.SI_UNITS` if `None`
      (composite models may default to the units of a primary component).
    units_mapping: Physical units of each variable that crosses the model
      boundary, keyed by variable name. Defaults to
      `defaults.DEFAULT_UNITS_MAPPING` if `None`.
    operators: Observation operators keyed by dataset or operator name,
      evaluated on nondimensional `prognostic_values()`. Keys must not start
      with `'nondim_'`.
  """

  sim_units: units.SimUnits | None = nnx.static(default=None, kw_only=True)
  units_mapping: Mapping[str, str] | None = nnx.static(
      default=None, kw_only=True
  )
  operators: dict[str, typing.ObservationOperator] = nnx.data(
      default_factory=dict, kw_only=True
  )

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    if 'observe' in cls.__dict__:
      raise TypeError(
          f'{cls.__name__} must implement `observe_nondim` instead of '
          'overriding `Model.observe`, which defines the units contract.'
      )

  def __post_init__(self):
    self._setup_units()

  def _setup_units(self) -> None:
    """Resolves `sim_units` and `units_mapping` defaults.

    Called from `__post_init__`; subclasses that adjust `units_mapping` or
    `sim_units` (e.g. adding tracers or defaulting to a component's units) do
    so before calling `super().__post_init__()`.
    """
    if self.sim_units is None:
      self.sim_units = units.SI_UNITS
    if self.units_mapping is None:
      self.units_mapping = defaults.DEFAULT_UNITS_MAPPING
    nondim_keys = [k for k in self.operators if k.startswith(NONDIM_PREFIX)]
    if nondim_keys:
      raise ValueError(
          f'Operator keys must not start with {NONDIM_PREFIX!r}, got '
          f'{nondim_keys}.'
      )

  @property
  def nondim_transform(self) -> transforms.Nondimensionalize:
    """Transform nondimensionalizing physical inputs into `sim_units`."""
    assert self.sim_units is not None and self.units_mapping is not None
    return transforms.Nondimensionalize(
        self.sim_units, inputs_to_units_mapping=dict(self.units_mapping)
    )

  @property
  def redim_transform(self) -> transforms.Redimensionalize:
    """Transform redimensionalizing `sim_units` values into physical units."""
    assert self.sim_units is not None and self.units_mapping is not None
    return transforms.Redimensionalize(
        self.sim_units, inputs_to_units_mapping=dict(self.units_mapping)
    )

  @abc.abstractmethod
  def prognostic_values(self) -> dict[str, cx.Field]:
    """Returns current nondimensional prognostic values of the model."""
    raise NotImplementedError()

  def observe_nondim(self, queries: typing.Queries) -> typing.Observation:
    """Returns nondimensional observations for nondimensional `queries`.

    Keys of `queries` carry no `'nondim_'` prefix and any `Field` entries are
    expressed in `sim_units`; outputs are expressed in `sim_units` as well. The
    default implementation evaluates `operators` on `prognostic_values()` and
    serves the `'diagnostics'` key from `diagnostic_values()`.

    Args:
      queries: Observation queries keyed by dataset or operator key.

    Returns:
      Nondimensional observations keyed like `queries`.
    """
    prognostics: dict[str, cx.Field] | None = None
    outputs: typing.Observation = {}
    for k, query in queries.items():
      if k in self.operators:
        if prognostics is None:
          prognostics = self.prognostic_values()
        outputs[k] = self.operators[k].observe(prognostics, query)
      elif k == 'diagnostics':
        diagnostics_operator = observation_operators.DataObservationOperator(
            self.diagnostic_values()
        )
        outputs[k] = diagnostics_operator.observe({}, query)
      else:
        raise ValueError(
            f'No observation operator for key {k!r}, available: '
            f'{sorted([*self.operators, "diagnostics"])}.'
        )
    return outputs

  def observe(self, queries: typing.Queries) -> typing.Observation:
    """Computes observations specified in `queries` from the model state.

    Query keys prefixed with `'nondim_'` are served nondimensionally in
    `sim_units`; all other keys are served in physical units per
    `units_mapping`, with `Field` entries of their queries nondimensionalized
    before evaluation. Each group of keys is evaluated with a single call to
    `observe_nondim`.

    Args:
      queries: mapping from query key to query, see `observe_nondim`.

    Returns:
      Observations keyed like `queries`, in physical units for keys without the
      `'nondim_'` prefix and in `sim_units` otherwise.
    """
    assert self.sim_units is not None and self.units_mapping is not None
    dim_queries: typing.Queries = {}
    nondim_queries: typing.Queries = {}
    for k, query in queries.items():
      if k.startswith(NONDIM_PREFIX):
        nondim_queries[k.removeprefix(NONDIM_PREFIX)] = query
      else:
        dim_queries[k] = units_utils.nondim_query(
            query, self.sim_units, self.units_mapping
        )
    outputs: typing.Observation = {}
    if dim_queries:
      for k, obs in self.observe_nondim(dim_queries).items():
        outputs[k] = units_utils.units_passage(
            obs, self.sim_units, None, self.units_mapping
        )
    if nondim_queries:
      for k, obs in self.observe_nondim(nondim_queries).items():
        outputs[NONDIM_PREFIX + k] = obs
    return {k: outputs[k] for k in queries}
