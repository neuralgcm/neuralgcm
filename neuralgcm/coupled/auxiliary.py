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
"""Auxiliary models that facilitate extension of model prognostics.

Note: Branched from `third_party/py/terrax/auxiliary/models.py` while iterating
on the `neuralgcm.base.model.Model` integration.
"""

import abc
import collections
from collections.abc import Mapping
from typing import Literal, TypeAlias

import coordax as cx
from flax import nnx
import jax.numpy as jnp
from neuralgcm.base import model
from neuralgcm.base import operator_utils
from neuralgcm.base import stepping
from neuralgcm.base import time_utils
from neuralgcm.base import units_utils
import numpy as np
from terrax.core import data_specs
from terrax.core import diagnostics as diagnostics_lib
from terrax.core import module_utils
from terrax.core import observation_operators
from terrax.core import typing

# pylint: disable=g-classes-have-attributes

CollectMethod: TypeAlias = Literal['set', 'accumulate']
ApplyMethod: TypeAlias = Literal['add', 'set']
# Forwarding rule for one observation operator key, see `WithObservedState`.
ForwardingRule: TypeAlias = tuple[str, ...] | dict[str, str]
_VALID_COLLECT_METHODS: frozenset[str] = frozenset({'set', 'accumulate'})
_VALID_APPLY_METHODS: frozenset[str] = frozenset({'add', 'set'})
_DIRECT_STATE_SUFFIX = '_state'


def _full_like(coord: cx.Coordinate, fill_value: float = 0.0) -> cx.Field:
  return cx.field(jnp.full(coord.shape, fill_value), coord)


def _as_mapping(rule: ForwardingRule) -> dict[str, str]:
  """Expands a tuple forwarding rule into the identity mapping."""
  if isinstance(rule, Mapping):
    mapping = dict(rule)
    if len(set(mapping.values())) != len(mapping):
      raise ValueError(
          'Forwarding rule maps multiple query keys to the same observation '
          f'variable: {mapping}.'
      )
    return mapping
  if isinstance(rule, str) or not isinstance(rule, (tuple, list)):
    raise TypeError(
        'Forwarding rule must be a tuple of names or a dict mapping names of '
        f'base-model query entries to observation variables, got {rule!r}.'
    )
  return {k: k for k in rule}


class StateModelABC(model.Model, abc.ABC):
  """Base class for models that track observation-anchored prognostic state.

  Extends `neuralgcm.base.model.Model` with `stage_updates` for receiving
  incremental nondimensional values from a coupling wrapper between advance
  steps.
  """

  data_key: str

  @abc.abstractmethod
  def stage_updates(self, updates: dict[str, cx.Field]) -> None:
    """Stages nondimensional updates to be applied at the next update."""


@nnx.dataclass
class LabeledStateModel(StateModelABC):
  """State model that resolves queries via coordinate labels.

  Maintains named prognostic fields in nondimensional `sim_units` and serves
  coordinate-based queries for `data_key` from the pool of prognostic and
  diagnostic values (via `DataObservationOperator`), following the units
  contract of `model.Model`: observations and assimilated inputs are in
  physical units per `units_mapping`, unless the key is prefixed with
  `'nondim_'`. Staged updates are applied every `update_every` interval;
  `assimilate` requires all prognostic keys.

  Attributes:
    prognostic_coords: Variable name -> coordinate for each tracked field.
    data_key: Key used to look up this model's data in observations.
    model_timestep: Duration of a single advance step.
    update_every: Interval between applying staged updates. Must be a positive
      integer multiple of `model_timestep`. Defaults to `model_timestep`.
    collect_update_method: How `stage_updates` accumulates values between update
      boundaries, either as a single method (`'set'` or `'accumulate'`) or a
      per-variable mapping. `'set'` (default) replaces staged values;
      `'accumulate'` adds to them.
    apply_update_method: How staged updates modify prognostics at update
      boundaries, either as a single method (`'add'` or `'set'`) or a
      per-variable mapping. `'add'` (default) adds staged values to current
      state; `'set'` replaces current state with staged values.
    update_mapping: Renames update keys to prognostic variable names.
    diagnostics: Named diagnostic modules whose outputs are merged into the
      observable state pool. Diagnostics are evaluated in `advance` prior to
      applying staged updates; `InstantDiagnostic` modules are also initialized
      during `assimilate` so pre-update lag covariates (e.g. `*_1h_lag`,
      `*_24h_lag`) are available at `t=0` and after every step.
    prognostics_data_key: Optional dataset key for direct prognostic state
      initialization in `assimilate`.
    prognostics: Nondimensional prognostic state fields, initialized to NaN. Set
      via `assimilate` and updated by `advance`.
    sim_units: Simulation units helper. Defaults to `units.SI_UNITS`.
    units_mapping: Physical units of the prognostic and diagnostic variables
      read by `assimilate` or served by dimensional queries. Defaults to
      `defaults.DEFAULT_UNITS_MAPPING`; keys outside the defaults require an
      explicit mapping.
  """

  prognostic_coords: dict[str, cx.Coordinate]
  data_key: str = nnx.static()
  model_timestep: np.timedelta64 = nnx.static()
  update_every: np.timedelta64 | None = nnx.static(default=None)
  collect_update_method: CollectMethod | Mapping[str, CollectMethod] = (
      nnx.static(default='set')
  )
  apply_update_method: ApplyMethod | Mapping[str, ApplyMethod] = nnx.static(
      default='add'
  )
  update_mapping: dict[str, str] = nnx.static(default_factory=dict)
  diagnostics: dict[str, diagnostics_lib.DiagnosticModule] = nnx.data(
      default_factory=dict
  )
  prognostics_data_key: str = nnx.static(default='', kw_only=True)
  prognostics: typing.Prognostic = nnx.data(init=False)
  _staged_updates: typing.Prognostic = nnx.data(init=False)
  _steps_done: typing.Prognostic = nnx.data(init=False)
  _steps_per_update: int = nnx.static(init=False)

  def __post_init__(self):
    if self.update_every is None:
      object.__setattr__(self, 'update_every', self.model_timestep)
    assert self.update_every is not None
    self._steps_per_update = time_utils.exact_timedelta_ratio(
        self.update_every,
        self.model_timestep,
        f'update_every ({self.update_every}) must be a positive '
        f'integer multiple of model_timestep ({self.model_timestep}).',
    )
    self._validate_update_methods()
    super().__post_init__()
    self.prognostics = typing.Prognostic({
        name: _full_like(coord, jnp.nan)
        for name, coord in self.prognostic_coords.items()
    })
    self._staged_updates = typing.Prognostic(self._make_zero_updates())
    self._steps_done = typing.Prognostic(
        cx.field(jnp.array(0, dtype=jnp.int32))
    )

  def _validate_update_methods(self) -> None:
    """Validates `update_mapping` and the collect/apply update methods."""
    unknown_targets = set(self.update_mapping.values()) - set(
        self.prognostic_coords
    )
    if unknown_targets:
      raise ValueError(
          f'update_mapping targets unknown prognostic keys '
          f'{sorted(unknown_targets)}, expected subset of '
          f'{sorted(self.prognostic_coords)}.'
      )
    if isinstance(self.collect_update_method, str):
      if self.collect_update_method not in _VALID_COLLECT_METHODS:
        raise ValueError(
            'Unsupported collect_update_method: '
            f'{self.collect_update_method!r}'
        )
    else:
      unknown = set(self.collect_update_method) - set(self.prognostic_coords)
      if unknown:
        raise ValueError(
            f'collect_update_method has unknown prognostic keys '
            f'{sorted(unknown)}, expected subset of '
            f'{sorted(self.prognostic_coords)}.'
        )
      for k, method in self.collect_update_method.items():
        if method not in _VALID_COLLECT_METHODS:
          raise ValueError(
              f'Unsupported collect_update_method {method!r} for {k!r}.'
          )

    if isinstance(self.apply_update_method, str):
      if self.apply_update_method not in _VALID_APPLY_METHODS:
        raise ValueError(
            f'Unsupported apply_update_method: {self.apply_update_method!r}'
        )
    else:
      unknown = set(self.apply_update_method) - set(self.prognostic_coords)
      if unknown:
        raise ValueError(
            f'apply_update_method has unknown prognostic keys '
            f'{sorted(unknown)}, expected subset of '
            f'{sorted(self.prognostic_coords)}.'
        )
      for k, method in self.apply_update_method.items():
        if method not in _VALID_APPLY_METHODS:
          raise ValueError(
              f'Unsupported apply_update_method {method!r} for {k!r}.'
          )

  def _collect_method_for(self, var_name: str) -> CollectMethod:
    if isinstance(self.collect_update_method, str):
      return self.collect_update_method
    return self.collect_update_method.get(var_name, 'set')

  def _apply_method_for(self, var_name: str) -> ApplyMethod:
    if isinstance(self.apply_update_method, str):
      return self.apply_update_method
    return self.apply_update_method.get(var_name, 'add')

  def _make_zero_updates(self) -> dict[str, cx.Field]:
    return {
        name: _full_like(coord)
        for name, coord in self.prognostic_coords.items()
    }

  def _default_staged_updates(
      self, state: dict[str, cx.Field]
  ) -> dict[str, cx.Field]:
    """Returns reset values for `_staged_updates` given current `state`."""
    zeros = self._make_zero_updates()
    result = {}
    for k in self.prognostic_coords:
      if (
          self._collect_method_for(k) == 'set'
          and self._apply_method_for(k) == 'set'
      ):
        result[k] = state[k]
      else:
        result[k] = zeros[k]
    return result

  def prognostic_values(self) -> dict[str, cx.Field]:
    """Returns current nondimensional prognostic values of the model."""
    return dict(self.prognostics.get_value())

  def stage_updates(self, updates: dict[str, cx.Field]) -> None:
    """Stages nondimensional updates to be applied at the next update."""
    in_keys = list(updates.keys())
    if self.update_mapping:
      updates = {self.update_mapping.get(k, k): v for k, v in updates.items()}
    if len(updates) != len(in_keys):
      raise ValueError(
          f'Multiple update keys in {in_keys} map to the same prognostic in '
          f'{list(updates.keys())}.'
      )
    if unknown := set(updates) - set(self.prognostic_coords):
      raise ValueError(
          f'Update keys {sorted(unknown)} are not in prognostic_coords '
          f'{sorted(self.prognostic_coords)}.'
      )
    current = self._staged_updates.get_value()
    new_staged = {}
    for k, cur_val in current.items():
      if k not in updates:
        new_staged[k] = cur_val
        continue
      method = self._collect_method_for(k)
      if method == 'set':
        new_staged[k] = updates[k]
      elif method == 'accumulate':
        new_staged[k] = cur_val + updates[k]
      else:
        raise ValueError(f'Unsupported collect_update_method: {method!r}')
    self._staged_updates.set_value(new_staged)

  def _resolve_assimilation_key(
      self, inputs: typing.Observation
  ) -> tuple[str, bool]:
    """Resolves the dataset key to initialize from in `assimilate`.

    `prognostics_data_key` (if set) takes precedence over `data_key`; either may
    be provided in `inputs` with or without the `'nondim_'` prefix.

    Args:
      inputs: Assimilation inputs keyed by dataset key.

    Returns:
      The dataset key found in `inputs` (defaulting to `data_key`) and whether
      its values are nondimensional.
    """
    candidates = [self.prognostics_data_key, self.data_key]
    for data_key in filter(None, candidates):
      nondim_key = model.NONDIM_PREFIX + data_key
      present = [k for k in (data_key, nondim_key) if k in inputs]
      if len(present) > 1:
        raise ValueError(
            f'Got ambiguous initialization from both {present}, expected '
            'at most one.'
        )
      if present:
        [key] = present
        return key, key == nondim_key
    return self.data_key, False

  @module_utils.ensure_unchanged_state_structure
  def assimilate(self, inputs: typing.Observation) -> None:
    """Resets prognostic state from `inputs`."""
    current_state = self.prognostics.get_value()
    source_key, is_nondim = self._resolve_assimilation_key(inputs)
    data_vars = inputs.get(source_key, {})
    new_state = {}
    for k in current_state:
      if k not in data_vars:
        raise ValueError(
            f'Assimilation inputs for data_key {source_key!r} are missing'
            f' required prognostic variable {k!r}. Available keys:'
            f' {list(data_vars.keys())}'
        )
      new_state[k] = time_utils.select_zero_timedelta(data_vars[k])
    if not is_nondim:
      new_state = self.nondim_transform(new_state)
    self.prognostics.set_value(new_state)
    self._steps_done.set_value(cx.field(jnp.zeros((), dtype=jnp.int32)))
    self._staged_updates.set_value(self._default_staged_updates(new_state))
    # TODO(dkochkov): Consider what diagnostics contract is broadly for t=0.
    for diagnostic in self.diagnostics.values():
      diagnostic.reset_diagnostic_state()
      if isinstance(diagnostic, diagnostics_lib.InstantDiagnostic):
        diagnostic(new_state, prognostics=new_state)

  @module_utils.ensure_unchanged_state_structure
  def advance(self) -> None:
    """Advances state by one timestep, applying staged updates at boundaries."""
    current_state = self.prognostics.get_value()
    stepping.update_diagnostics(
        self.diagnostics, current_state, prognostics=current_state
    )
    stepping.advance_diagnostic_clocks(self.model_timestep, self.diagnostics)

    next_step = self._steps_done.get_value() + 1
    is_update_step = next_step.data >= self._steps_per_update
    staged = self._staged_updates.get_value()
    next_state = {}
    for k, v in current_state.items():
      method = self._apply_method_for(k)
      if method == 'add':
        updated_val = v + staged[k]
      elif method == 'set':
        updated_val = staged[k]
      else:
        raise ValueError(f'Unsupported apply_update_method: {method!r}')
      next_state[k] = cx.cmap(jnp.where)(is_update_step, updated_val, v)
    self.prognostics.set_value(next_state)

    # Reset counter and staged updates when update is applied.
    self._steps_done.set_value(next_step % self._steps_per_update)
    default_updates = self._default_staged_updates(next_state)
    reset_staged = {
        k: cx.cmap(jnp.where)(is_update_step, default_updates[k], staged[k])
        for k in staged
    }
    self._staged_updates.set_value(reset_staged)

  def observe_nondim(self, queries: typing.Queries) -> typing.Observation:
    """Serves `data_key` from the prognostic and diagnostic state pool."""
    diagnostics = self.diagnostic_values()
    state_pool = self.prognostic_values() | diagnostics
    operators = {
        self.data_key: observation_operators.DataObservationOperator(
            state_pool
        ),
        'diagnostics': observation_operators.DataObservationOperator(
            diagnostics
        ),
    } | self.operators
    outputs = {}
    for k, query in queries.items():
      if k not in operators:
        raise ValueError(
            f'Query key {k!r} does not match data_key {self.data_key!r} or '
            f'available operators {sorted(operators)}.'
        )
      outputs[k] = operators[k].observe(state_pool, query)
    return outputs

  @property
  def timestep(self) -> np.timedelta64:
    return self.model_timestep

  @property
  def inputs_spec(
      self,
  ) -> dict[str, dict[str, cx.Coordinate | data_specs.CoordLikeSpec]]:
    specs: dict[str, cx.Coordinate | data_specs.CoordLikeSpec] = {
        k: data_specs.OptionalSpec(
            data_specs.CoordSpec.with_any_timedelta(
                v, optional_timedelta=True
            )
        )
        for k, v in self.prognostic_coords.items()
    }
    keys = [self.data_key, f'{model.NONDIM_PREFIX}{self.data_key}']
    if self.prognostics_data_key:
      keys.extend([
          self.prognostics_data_key,
          f'{model.NONDIM_PREFIX}{self.prognostics_data_key}',
      ])
    return {k: dict(specs) for k in keys}


@nnx.dataclass
class WithObservedState(model.Model):
  """Couples a base model with observation state models.

  Wraps `base_model` with independent `observation_models` that track
  observation-anchored prognostics (e.g. station measurements). All exchanges
  between the wrapper, `base_model` and `observation_models` are nondimensional
  and re-expressed in the receiving model's `sim_units` via
  `units_utils.units_passage`, which is a no-op when the models share a scale.
  The units of the wrapper define its interface, see `model.Model`.

  On `advance`, `base_model` is advanced first; `coupling_query` is then
  observed from it (enriched per `coupling_query_forwarding`) and the result is
  staged on the corresponding observation models, which are advanced last.

  On `observe`, queries keyed `{ds_key}_state` bypass the base model and are
  served directly by `observation_models[ds_key]`; top-level `operators` and
  `'diagnostics'` are served by the wrapper; all remaining queries are enriched
  per `observation_query_forwarding` and served by `base_model`.

  Forwarding rules (`observation_query_forwarding`, `coupling_query_forwarding`)
  are keyed by observation operator key `ds_key`, i.e. a key of
  `observation_models` that also names an operator of `base_model`. A rule
  specifies which entries of the base model's query for `ds_key` are resolved
  from `observation_models[ds_key]` and injected into that query as `Field`
  values before `base_model` observes it:

  * a `dict[str, str]` maps the name of an entry in the base model's query (key)
    to the name of the observation model variable providing its value (value),
    e.g. `{'2m_temperature_1h_lag': '2m_temperature'}`;
  * a `tuple[str, ...]` is shorthand for the identity mapping over those names,
    i.e. `('a', 'b')` is equivalent to `{'a': 'a', 'b': 'b'}`.

  Entries named in a rule must be present in the query. `Coordinate` entries
  are replaced by the observation model's state on that coordinate, whereas
  `Field` entries (e.g. data supplied via `FieldInQuerySpec`) are kept as
  caller-provided values; `Auxiliary` wrapping is preserved in both cases.

  Attributes:
    base_model: The primary base model to wrap.
    observation_models: Observation operator key -> observation state model.
      Keys must match observation operators of `base_model` (including those of
      nested component models).
    coupling_query: Queries observed from `base_model` after each `advance` to
      update observation models, keyed by observation operator key. As for
      `observe`, a `'nondim_'` prefix on a key only determines the units of
      `Field` entries in its query (nondimensional in `sim_units` with the
      prefix, physical per `units_mapping` without); exchanged values are
      always nondimensional.
    observation_query_forwarding: Forwarding rules applied in `observe`.
    coupling_query_forwarding: Forwarding rules applied in `advance` when
      observing `coupling_query`.
    sim_units: Simulation units of the wrapper. Defaults to
      `base_model.sim_units` if `None`.
    units_mapping: Physical units of variables observed from the wrapper.
      Defaults to the merged mappings of `base_model` and `observation_models`
      if `None`.
    remat_advance: Whether to wrap `advance` with `nnx.remat`.
  """

  base_model: model.Model
  observation_models: dict[str, StateModelABC]
  coupling_query: typing.Queries
  observation_query_forwarding: dict[str, ForwardingRule] = nnx.static(
      default_factory=dict
  )
  coupling_query_forwarding: dict[str, ForwardingRule] = nnx.static(
      default_factory=dict
  )
  remat_advance: bool = nnx.static(default=False, kw_only=True)

  def __post_init__(self):
    operator_utils.setup_composite_units(
        self, [self.base_model, *self.observation_models.values()]
    )
    super().__post_init__()
    assert self.sim_units is not None and self.units_mapping is not None
    for ds_key, obs_model in self.observation_models.items():
      if obs_model.timestep != self.base_model.timestep:
        raise ValueError(
            f'Observation model {ds_key!r} timestep ({obs_model.timestep}) '
            f'must match base_model timestep ({self.base_model.timestep}).'
        )
    base_operators = operator_utils.get_all_operators(self.base_model)
    if base_operators:
      for ds_key in self.observation_models:
        if ds_key not in base_operators:
          raise ValueError(
              f'Observation model key {ds_key!r} must match an observation'
              ' operator in base_model. Available keys:'
              f' {list(base_operators.keys())}'
          )
    for forwarding in (
        self.observation_query_forwarding,
        self.coupling_query_forwarding,
    ):
      for ds_key, rule in forwarding.items():
        if ds_key not in self.observation_models:
          raise ValueError(
              f'Forwarding key {ds_key!r} must match a key in '
              'observation_models.'
          )
        _as_mapping(rule)  # Validates the forwarding rule.
    # Normalize `coupling_query` to nondimensional queries keyed by `ds_key`.
    coupling_query: typing.Queries = {}
    for query_key, query in self.coupling_query.items():
      is_nondim = query_key.startswith(model.NONDIM_PREFIX)
      ds_key = query_key.removeprefix(model.NONDIM_PREFIX)
      if ds_key in coupling_query:
        raise ValueError(
            f'coupling_query has both dimensional and nondimensional entries '
            f'for {ds_key!r}.'
        )
      if ds_key not in self.observation_models:
        raise ValueError(
            f'Coupling query key {query_key!r} must match a key in '
            f'observation_models {list(self.observation_models.keys())}.'
        )
      if not is_nondim:
        query = units_utils.nondim_query(
            query, self.sim_units, self.units_mapping
        )
      coupling_query[ds_key] = query
    self.coupling_query = coupling_query

  def prognostic_values(self) -> dict[str, cx.Field]:
    """Returns merged prognostics, potentially prefixed with `{'ds_key}_'`."""
    assert self.sim_units is not None
    assert (
        self.base_model.sim_units is not None
        and self.base_model.units_mapping is not None
    )
    merged = units_utils.units_passage(
        self.base_model.prognostic_values(),
        self.base_model.sim_units,
        self.sim_units,
        self.base_model.units_mapping,
    )
    obs_prognostics: dict[str, dict[str, cx.Field]] = {}
    key_counts: collections.Counter[str] = collections.Counter(merged.keys())
    for ds_key, obs_model in self.observation_models.items():
      assert obs_model.sim_units is not None
      assert obs_model.units_mapping is not None  # make pytype happy.
      obs_vals = units_utils.units_passage(
          obs_model.prognostic_values(),
          obs_model.sim_units,
          self.sim_units,
          obs_model.units_mapping,
      )
      obs_prognostics[ds_key] = obs_vals
      key_counts.update(obs_vals.keys())
    for ds_key, obs_vals in obs_prognostics.items():
      for k, v in obs_vals.items():
        if k == 'time' and 'time' in merged:
          continue
        out_key = f'{ds_key}_{k}' if key_counts[k] > 1 else k
        merged[out_key] = v
    return merged

  def assimilate(self, inputs: typing.Observation) -> None:
    self.base_model.assimilate(inputs)
    for obs_model in self.observation_models.values():
      obs_model.assimilate(inputs)

  def advance(self) -> None:
    def step_fn(m: WithObservedState) -> None:
      m.base_model.advance()  # base model (t -> t+1).
      # pylint: disable=protected-access
      coupling_obs = m._observe_nondim_with_forwarding(
          m.coupling_query, m.coupling_query_forwarding
      )
      assert m.sim_units is not None and m.units_mapping is not None
      for ds_key, obs_model in m.observation_models.items():
        updates = units_utils.units_passage(
            coupling_obs.get(ds_key, {}),
            m.sim_units,
            obs_model.sim_units,
            m.units_mapping,
        )
        obs_model.stage_updates(updates)
        obs_model.advance()

    if self.remat_advance:
      step_fn = nnx.remat(step_fn)
    step_fn(self)

  def observe_nondim(self, queries: typing.Queries) -> typing.Observation:
    return self._observe_nondim_with_forwarding(
        queries, self.observation_query_forwarding
    )

  def _direct_state_key(self, key: str) -> str | None:
    """Returns `ds_key` if `key` is a direct state query `{ds_key}_state`."""
    if not key.endswith(_DIRECT_STATE_SUFFIX):
      return None
    ds_key = key.removesuffix(_DIRECT_STATE_SUFFIX)
    return ds_key if ds_key in self.observation_models else None

  def _observe_nondim_with_forwarding(
      self,
      queries: typing.Queries,
      forwarding_rules: Mapping[str, ForwardingRule],
  ) -> typing.Observation:
    """Resolves nondimensional `queries` in the wrapper's `sim_units`.

    1. Top-level `operators` and `'diagnostics'` are served by the wrapper.
    2. Direct state queries `{ds_key}_state` are served by
       `observation_models[ds_key]` (its `data_key` dataset).
    3. Remaining queries are enriched per `forwarding_rules`, see
       `_enrich_query`, and served by `base_model.observe_nondim`.

    Args:
      queries: Nondimensional queries in the wrapper's `sim_units`.
      forwarding_rules: Forwarding rules keyed by observation operator key.

    Returns:
      Nondimensional observations in the wrapper's `sim_units`, in the key
      order of `queries`.
    """
    assert self.sim_units is not None
    top_queries: typing.Queries = {}
    direct_queries: typing.Queries = {}
    base_queries: typing.Queries = {}
    for k, query in queries.items():
      if k in self.operators or k == 'diagnostics':
        top_queries[k] = query
      elif self._direct_state_key(k) is not None:
        direct_queries[k] = query
      else:
        base_queries[k] = query

    outputs: typing.Observation = {}
    if top_queries:
      outputs |= super().observe_nondim(top_queries)
    for k, query in direct_queries.items():
      ds_key = k.removesuffix(_DIRECT_STATE_SUFFIX)
      obs_model = self.observation_models[ds_key]
      assert obs_model.units_mapping is not None
      query = operator_utils.passage_query(query, self, obs_model)
      obs = obs_model.observe_nondim({obs_model.data_key: query})
      outputs[k] = units_utils.units_passage(
          obs[obs_model.data_key],
          obs_model.sim_units,
          self.sim_units,
          obs_model.units_mapping,
      )
    if base_queries:
      assert self.base_model.units_mapping is not None
      enriched = {
          k: self._enrich_query(k, query, forwarding_rules)
          for k, query in base_queries.items()
      }
      for k, obs in self.base_model.observe_nondim(enriched).items():
        outputs[k] = units_utils.units_passage(
            obs,
            self.base_model.sim_units,
            self.sim_units,
            self.base_model.units_mapping,
        )
    return {k: outputs[k] for k in queries}

  def _enrich_query(
      self,
      ds_key: str,
      query: typing.Query,
      forwarding_rules: Mapping[str, ForwardingRule],
  ) -> typing.Query:
    """Returns `query` for `base_model` with forwarded entries injected.

    `Field` entries are re-expressed in `base_model.sim_units`. For each entry
    named in `forwarding_rules[ds_key]` whose value is a `Coordinate`, the
    state of `observation_models[ds_key]` on that coordinate replaces the
    coordinate (preserving `Auxiliary` wrapping); `Field` entries named in the
    rule are caller-provided values and are kept.

    Args:
      ds_key: Observation operator key of `query`.
      query: Nondimensional query entries in the wrapper's `sim_units`.
      forwarding_rules: Forwarding rules keyed by observation operator key.

    Returns:
      Query entries for `base_model` in its `sim_units`.
    """
    base_query = dict(operator_utils.passage_query(
        query, self, self.base_model
    ))
    if ds_key not in forwarding_rules:
      return base_query
    obs_model = self.observation_models[ds_key]
    assert obs_model.units_mapping is not None
    coord_mapping: dict[str, str] = {}  # Query name -> obs model variable.
    obs_query: dict[str, cx.Coordinate] = {}
    for in_k, obs_k in _as_mapping(forwarding_rules[ds_key]).items():
      if in_k not in base_query:
        raise ValueError(
            f'Operator {ds_key!r} requires key {in_k!r} to be provided '
            f'for query forwarding. Got keys: {list(base_query.keys())}'
        )
      entry, _ = typing.unwrap_auxiliary(base_query[in_k])
      if not cx.is_field(entry):
        coord_mapping[in_k] = obs_k
        obs_query[obs_k] = entry
    if not obs_query:
      return base_query
    obs = obs_model.observe_nondim({obs_model.data_key: obs_query})
    resolved = obs[obs_model.data_key]
    # The passage is keyed by observation-model variables, so that errors about
    # missing units refer to entries of `obs_model.units_mapping`.
    fields = units_utils.units_passage(
        {obs_k: resolved[obs_k] for obs_k in coord_mapping.values()},
        obs_model.sim_units,
        self.base_model.sim_units,
        obs_model.units_mapping,
    )
    for in_k, obs_k in coord_mapping.items():
      is_aux = isinstance(base_query[in_k], typing.Auxiliary)
      field = fields[obs_k]
      base_query[in_k] = typing.Auxiliary(field) if is_aux else field
    return base_query

  @property
  def timestep(self) -> np.timedelta64:
    return self.base_model.timestep

  @property
  def inputs_spec(
      self,
  ) -> dict[str, dict[str, cx.Coordinate | data_specs.CoordLikeSpec]]:
    return operator_utils.merge_inputs_specs(
        self.base_model.inputs_spec,
        *(m.inputs_spec for m in self.observation_models.values()),
    )
