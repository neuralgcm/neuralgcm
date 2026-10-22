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
from collections.abc import Mapping
from typing import Literal, TypeAlias

import coordax as cx
from flax import nnx
import jax
import jax.numpy as jnp
from neuralgcm.base import defaults
from neuralgcm.base import model
from neuralgcm.base import stepping
from neuralgcm.base import time_utils
import numpy as np
from terrax.core import api
from terrax.core import data_specs
from terrax.core import diagnostics as diagnostics_lib
from terrax.core import module_utils
from terrax.core import observation_operators
from terrax.core import transforms
from terrax.core import typing

# pylint: disable=g-classes-have-attributes

CollectMethod: TypeAlias = Literal['set', 'accumulate']
ApplyMethod: TypeAlias = Literal['add', 'set']
_VALID_COLLECT_METHODS: frozenset[str] = frozenset({'set', 'accumulate'})
_VALID_APPLY_METHODS: frozenset[str] = frozenset({'add', 'set'})


def _full_like(coord: cx.Coordinate, fill_value: float = 0.0) -> cx.Field:
  return cx.field(jnp.full(coord.shape, fill_value), coord)


class StateModelABC(model.Model, abc.ABC):
  """Base class for models that track observation-anchored prognostic state.

  Extends `neuralgcm.base.model.Model` with `stage_updates` for receiving
  incremental nondimensional values from a coupling wrapper between advance
  steps.
  """

  data_key: str

  @abc.abstractmethod
  def stage_updates(self, updates: dict[str, cx.Field]) -> None:
    """Stages nondimensional updates to be applied at the next update boundary."""


@nnx.dataclass
class LabeledStateModel(StateModelABC):
  """State model that resolves queries via coordinate labels.

  Maintains named prognostic fields in nondimensional `sim_units` and uses
  `DataObservationOperator` to serve coordinate-based queries (applying
  `redim_transform` for dimensional query keys and returning nondimensional
  values for `'nondim_'`-prefixed query keys). Staged updates are applied every
  `update_every` interval; `assimilate` requires all prognostic keys.

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
    fallback_units = {
        k: 'dimensionless'
        for k in (*self.prognostic_coords, *self.update_mapping)
    }
    for diag in self.diagnostics.values():
      extract_coords = getattr(diag, 'extract_coords', ())
      offsets = getattr(diag, '_offsets', {})
      for k in extract_coords:
        fallback_units[k] = 'dimensionless'
        base_unit = defaults.DEFAULT_UNITS_MAPPING.get(k, 'dimensionless')
        for suffix in offsets:
          out_key = f'{k}_{suffix}' if suffix else k
          fallback_units[out_key] = base_unit
      for k in getattr(diag, 'output_mapping', {}).values():
        fallback_units[k] = 'dimensionless'
    units_mapping = fallback_units | defaults.DEFAULT_UNITS_MAPPING
    if self.nondim_transform is None:
      self.nondim_transform = transforms.Nondimensionalize(
          self.sim_units, inputs_to_units_mapping=units_mapping
      )
    if self.redim_transform is None:
      self.redim_transform = transforms.Redimensionalize(
          self.sim_units, inputs_to_units_mapping=units_mapping
      )
    self.prognostics = typing.Prognostic({
        name: _full_like(coord, jnp.nan)
        for name, coord in self.prognostic_coords.items()
    })
    self._staged_updates = typing.Prognostic(self._make_zero_updates())
    self._steps_done = typing.Prognostic(
        cx.field(jnp.array(0, dtype=jnp.int32))
    )

  def _validate_update_methods(self) -> None:
    """Validates `collect_update_method` and `apply_update_method`."""
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
    """Stages nondimensional updates to be applied at the next update boundary."""
    if self.update_mapping:
      updates = {self.update_mapping.get(k, k): v for k, v in updates.items()}
    current = self._staged_updates.get_value()
    new_staged = {}
    for k, cur_val in current.items():
      if k not in updates:
        new_staged[k] = cur_val
      elif self._collect_method_for(k) == 'set':
        new_staged[k] = updates[k]
      else:
        new_staged[k] = cur_val + updates[k]
    self._staged_updates.set_value(new_staged)

  def _resolve_assimilation_key(
      self, inputs: typing.Observation
  ) -> tuple[str, bool]:
    """Returns `(dataset_key_in_inputs, is_nondim)` for `assimilate`."""
    if self.prognostics_data_key:
      prog_keys = [
          k
          for k in (
              self.prognostics_data_key,
              f'nondim_{self.prognostics_data_key}',
          )
          if k in inputs
      ]
      if len(prog_keys) > 1:
        raise ValueError(
            f'Got ambiguous initialization from both {prog_keys}, expected '
            'at most one.'
        )
      if prog_keys:
        [key] = prog_keys
        return key, key.startswith('nondim_')

    data_keys = [
        k for k in (self.data_key, f'nondim_{self.data_key}') if k in inputs
    ]
    if len(data_keys) > 1:
      raise ValueError(
          f'Got ambiguous initialization from both {data_keys}, expected '
          'at most one.'
      )
    if data_keys:
      [key] = data_keys
      return key, key.startswith('nondim_')
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
      assert self.nondim_transform is not None
      new_state = self.nondim_transform(new_state)
    self.prognostics.set_value(new_state)
    self._steps_done.set_value(cx.field(jnp.zeros((), dtype=jnp.int32)))
    self._staged_updates.set_value(self._default_staged_updates(new_state))
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

  @module_utils.ensure_unchanged_state_structure
  def observe(self, queries: typing.Queries) -> typing.Observation:
    assert self.redim_transform is not None
    state_pool = self.prognostic_values() | self.diagnostic_values()
    operators = {
        self.data_key: observation_operators.DataObservationOperator(
            state_pool
        ),
        'diagnostics': observation_operators.DataObservationOperator(
            self.diagnostic_values()
        ),
    } | self.operators
    result = {}
    for k, q in queries.items():
      op_key = (
          k.removeprefix('nondim_')
          if k not in operators and k.startswith('nondim_')
          else k
      )
      if op_key not in operators:
        raise ValueError(
            f'Query with key {k!r} does not match data_key {self.data_key!r} '
            f'or available operators {sorted(operators.keys())}.'
        )
      obs = operators[op_key].observe(state_pool, q)
      result[k] = obs if k.startswith('nondim_') else self.redim_transform(obs)
    return result

  @property
  def timestep(self) -> np.timedelta64:
    return self.model_timestep

  @property
  def inputs_spec(
      self,
  ) -> dict[str, dict[str, cx.Coordinate | data_specs.CoordLikeSpec]]:
    specs: dict[str, cx.Coordinate | data_specs.CoordLikeSpec] = {
        k: data_specs.CoordSpec.with_any_timedelta(v, optional_timedelta=True)
        for k, v in self.prognostic_coords.items()
    }
    result: dict[str, dict[str, cx.Coordinate | data_specs.CoordLikeSpec]] = {
        self.data_key: specs
    }
    if self.prognostics_data_key:
      opt_specs = jax.tree.map(
          data_specs.OptionalSpec,
          specs,
          is_leaf=lambda x: isinstance(x, data_specs.CoordSpec),
      )
      result[self.prognostics_data_key] = opt_specs
      result[f'nondim_{self.prognostics_data_key}'] = opt_specs
    return result


@nnx.dataclass
class WithObservedState(api.Model):
  """Couples a base model with observation state models.

  Wraps ``base_model`` with independent ``observation_models`` that track
  observation-anchored prognostics. On ``advance``, coupling queries extract
  updates from the base model and stage them on each observation model. On
  ``observe``, forwarded query entries are resolved against observation models
  and injected as fields into the base model query. Queries keyed as
  ``{ds_key}_state`` bypass the base model and query observation model directly.

  All keys in ``observation_models``, ``observation_query_forwarding``, and
  ``coupling_query_forwarding`` must match observation operator keys in
  ``base_model``.

  Attributes:
    base_model: The primary base model to wrap.
    observation_models: Operator key → observation state model.
    coupling_query: Queries for extracting updates from ``base_model``.
    observation_query_forwarding: Forwarding rules applied during ``observe``.
    coupling_query_forwarding: Forwarding rules applied during ``advance``.
  """

  base_model: api.Model
  observation_models: dict[str, StateModelABC]
  coupling_query: typing.Queries
  observation_query_forwarding: dict[str, tuple[str, ...] | dict[str, str]] = (
      nnx.static(default_factory=dict)
  )
  coupling_query_forwarding: dict[str, tuple[str, ...] | dict[str, str]] = (
      nnx.static(default_factory=dict)
  )

  def __post_init__(self):
    base_operators = getattr(self.base_model, 'operators', {})
    if base_operators:
      for ds_key in self.observation_models:
        if ds_key not in base_operators:
          raise ValueError(
              f'Observation model key {ds_key!r} must match an observation'
              ' operator in base_model. Available keys:'
              f' {list(base_operators.keys())}'
          )
    all_forwarding = set(self.observation_query_forwarding) | set(
        self.coupling_query_forwarding
    )
    for ds_key in all_forwarding:
      if ds_key not in self.observation_models:
        raise ValueError(
            f'Forwarding key {ds_key!r} must match a key in observation_models.'
        )
      if base_operators and ds_key not in base_operators:
        raise ValueError(
            f'Forwarding key {ds_key!r} must match an observation operator '
            f'in base_model. Available keys: {list(base_operators.keys())}'
        )

  def assimilate(self, inputs: typing.Observation) -> None:
    self.base_model.assimilate(inputs)
    for obs_model in self.observation_models.values():
      obs_model.assimilate(inputs)

  def advance(self) -> None:
    self.base_model.advance()  # base model (t -> t+1).
    coupling_obs = self._observe_with_forwarding(
        self.coupling_query, self.coupling_query_forwarding
    )
    for ds_key, obs_model in self.observation_models.items():
      obs_model.stage_updates(coupling_obs.get(ds_key, {}))
      obs_model.advance()

  def observe(self, queries: typing.Queries) -> typing.Observation:
    return self._observe_with_forwarding(
        queries, self.observation_query_forwarding
    )

  def _observe_direct_state(
      self, direct_queries: typing.Queries
  ) -> typing.Observation:
    direct_outputs = {}
    for direct_key, q in direct_queries.items():
      ds_key = direct_key.removesuffix('_state')
      obs_model = self.observation_models[ds_key]
      raw_obs = obs_model.observe({obs_model.data_key: q})
      direct_outputs[direct_key] = raw_obs[obs_model.data_key]
    return direct_outputs

  def _observe_with_forwarding(
      self,
      queries: typing.Queries,
      forwarding_rules: dict[str, tuple[str, ...] | dict[str, str]],
  ) -> typing.Observation:
    """Resolves queries in three phases.

    1. **Direct state bypass**: queries keyed ``{ds_key}_state`` are routed
       directly to the matching observation model, bypassing the base model.
    2. **Query enrichment**: for each remaining query whose ``ds_key`` has a
       forwarding rule, the specified entries are resolved against the
       observation model and injected back into the query as ``Field`` values
       (preserving ``Auxiliary`` wrapping). This lets the base model's
       observation operators receive pre-resolved state from observation models.
    3. **Base model observation**: enriched queries are forwarded to
       ``base_model.observe`` and the results are merged with any direct
       state observations.

    Args:
      queries: Mapping from dataset key to per-variable query entries.
      forwarding_rules: Mapping from dataset key to forwarding specification.
        A tuple of key names forwards with identity naming; a dict maps
        query keys to observation model keys.

    Returns:
      Merged observations from base model and any direct state queries.
    """
    is_direct = (
        lambda k: k.endswith('_state')
        and k.removesuffix('_state') in self.observation_models
    )
    direct_queries = {k: q for k, q in queries.items() if is_direct(k)}
    in_queries = {k: q for k, q in queries.items() if not is_direct(k)}

    direct_observations = (
        self._observe_direct_state(direct_queries) if direct_queries else {}
    )
    if not in_queries:
      return direct_observations

    enriched_queries = {}
    as_mapping = lambda x: x if isinstance(x, dict) else {k: k for k in x}
    for ds_key, in_query in in_queries.items():
      if ds_key in forwarding_rules:
        obs_model = self.observation_models[ds_key]
        forwarding_key_mapping = as_mapping(forwarding_rules[ds_key])
        forwarded_query = {}
        for in_k, obs_k in forwarding_key_mapping.items():
          if in_k not in in_query:
            raise ValueError(
                f'Operator {ds_key!r} requires key {in_k!r} to be provided '
                f'for query forwarding. Got keys: {list(in_query.keys())}'
            )
          q, _ = typing.unwrap_auxiliary(in_query[in_k])
          forwarded_query[obs_k] = q

        if forwarded_query:
          aux_obs = obs_model.observe({obs_model.data_key: forwarded_query})
          aux_results = aux_obs[obs_model.data_key]
          injected_fields = {}
          for in_k, obs_k in forwarding_key_mapping.items():
            if obs_k in aux_results:
              f = aux_results[obs_k]
              is_aux = isinstance(in_query[in_k], typing.Auxiliary)
              injected_fields[in_k] = typing.Auxiliary(f) if is_aux else f
          enriched_queries[ds_key] = dict(in_query) | injected_fields
        else:
          enriched_queries[ds_key] = in_query
      else:
        enriched_queries[ds_key] = in_query

    base_observations = self.base_model.observe(enriched_queries)
    return base_observations | direct_observations

  @property
  def timestep(self) -> np.timedelta64:
    return self.base_model.timestep

  @property
  def inputs_spec(
      self,
  ) -> dict[str, dict[str, cx.Coordinate | data_specs.CoordLikeSpec]]:
    specs = dict(self.base_model.inputs_spec)
    for obs_model in self.observation_models.values():
      for k, v in obs_model.inputs_spec.items():
        if k in specs:
          specs[k] = specs[k] | v
        else:
          specs[k] = v
    return specs
