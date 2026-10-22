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

"""Observation operators and dispatch utilities for NeuralGCM models."""

from collections.abc import Callable, Mapping
from typing import Any, TypeAlias, TypeVar

import coordax as cx
import jax.numpy as jnp
from neuralgcm.base import model
from terrax.core import observation_operators
from terrax.core import spherical_harmonics
from terrax.core import transforms
from terrax.core import typing
from terrax.couplers import generic

DataCoupler: TypeAlias = (
    generic.InstantDataCoupler | generic.MultiTimeDataCoupler
)
T = TypeVar('T')


def merge_inputs_specs(
    *specs: Mapping[str, Mapping[str, T]],
) -> dict[str, dict[str, T]]:
  """Merges nested `inputs_spec` mappings, raising on conflicting entries."""
  merged: dict[str, dict[str, T]] = {}
  for spec in specs:
    for data_key, dataset_spec in spec.items():
      existing = merged.setdefault(data_key, {})
      for var_name, var_spec in dataset_spec.items():
        if var_name in existing and existing[var_name] != var_spec:
          raise ValueError(
              f'Conflicting input specs for {var_name!r} in dataset '
              f'{data_key!r}: {existing[var_name]} vs {var_spec}.'
          )
        existing[var_name] = var_spec
  return merged


def merge_prognostics(*components: model.Model) -> dict[str, cx.Field]:
  """Merges `prognostic_values()` across `components`, raising on conflicts."""
  merged: dict[str, cx.Field] = {}
  for component in components:
    for k, v in component.prognostic_values().items():
      if k in merged and k != 'time':
        raise ValueError(
            f'Overlapping prognostic variable {k!r} across model components.'
        )
      merged[k] = v
  return merged


def get_all_operators(
    model_instance: model.Model,
) -> dict[str, typing.ObservationOperator]:
  """Collects `operators` from `model_instance` and any nested `Model`s."""
  all_ops: dict[str, typing.ObservationOperator] = {}
  for value in vars(model_instance).values():
    if isinstance(value, model.Model):
      all_ops.update(get_all_operators(value))
    elif isinstance(value, Mapping):
      for sub_value in value.values():
        if isinstance(sub_value, model.Model):
          all_ops.update(get_all_operators(sub_value))
  all_ops.update(model_instance.operators)
  return all_ops


def build_coupler_query(*couplers: DataCoupler | None) -> typing.Queries:
  """Builds and merges observation queries for `couplers`."""
  query: dict[str, dict[str, cx.Coordinate]] = {}
  for coupler in couplers:
    if coupler is None:
      continue
    for var_name, coord in coupler.field_coords.items():
      data_key = coupler.var_to_data_keys[var_name]
      dataset_query = query.setdefault(data_key, {})
      if var_name in dataset_query and dataset_query[var_name] != coord:
        raise ValueError(
            f'Conflicting coordinates for {var_name!r} in dataset '
            f'{data_key!r}: {dataset_query[var_name]} vs {coord}.'
        )
      dataset_query[var_name] = coord
    if isinstance(coupler, generic.MultiTimeDataCoupler):
      dataset_query = query.setdefault(coupler.time_data_key, {})
      if 'time' in dataset_query and dataset_query['time'] != cx.Scalar():
        raise ValueError(
            f"Conflicting coordinates for 'time' in dataset "
            f'{coupler.time_data_key!r}: {dataset_query["time"]} vs '
            f'{cx.Scalar()}.'
        )
      dataset_query['time'] = cx.Scalar()
  return {k: v for k, v in query.items()}


def _resolve_operator_key(
    key: str, operators: Mapping[str, Any]
) -> tuple[str | None, bool]:
  """Returns `(matched_operator_key, is_nondim)` for `key` in `operators`."""
  if key in operators:
    return key, key.startswith('nondim_')
  if key.startswith('nondim_'):
    base_key = key.removeprefix('nondim_')
    if base_key in operators:
      return base_key, True
  return None, False


def dispatch_observe(
    queries: typing.Queries,
    *,
    components: Mapping[str, model.Model],
    operators: Mapping[str, typing.ObservationOperator] | None = None,
    redim_transform: typing.Transform | None = None,
    get_diagnostic_values_fn: Callable[[], dict[str, cx.Field]] | None = None,
) -> typing.Observation:
  """Dispatches `queries` across top-level `operators` and `components`.

  Each query key in `queries` is resolved in the following order:
    1. Top-level `operators` take precedence over component operators, are
       evaluated with the merged `prognostic_values()` across `components` as
       `inputs`, and are passed through `redim_transform` (when provided).
    2. If `get_diagnostic_values_fn` is provided, the `'diagnostics'` key
       observes the fields returned by `get_diagnostic_values_fn()`, which are
       passed through `redim_transform` (when provided).
    3. Remaining keys are routed to the unique component in `components` whose
       `operators` contain the key, and all queries for a given component are
       evaluated in a single `component.observe(sub_queries)` call.
    4. Any query key prefixed with `'nondim_'` (when not explicitly present in
       `operators`) matches the operator for `key.removeprefix('nondim_')` and
       returns nondimensional values directly (skipping `redim_transform` for
       top-level operators and `'diagnostics'`, and forwarding the
       `'nondim_'`-prefixed key to `component.observe` for component
       operators).
  The merged prognostics and diagnostic values are only computed if a query key
  requires them.

  Args:
    queries: Observation queries keyed by dataset/operator key.
    components: Component models keyed by component name.
    operators: Optional top-level observation operators on the composite model.
    redim_transform: Optional transform applied to dimensional outputs of
      top-level `operators` and `'diagnostics'`.
    get_diagnostic_values_fn: Optional function returning nondimensional
      diagnostic values of the composite model, observed with the
      `'diagnostics'` key.

  Returns:
    Observations keyed by dataset/operator key in the same order as `queries`.
  """
  top_operators = dict(operators) if operators is not None else {}
  component_operators = {
      name: component.operators for name, component in components.items()
  }

  prognostics: dict[str, cx.Field] | None = None
  diagnostics: dict[str, cx.Field] | None = None
  top_outputs: typing.Observation = {}
  component_queries: dict[str, typing.Queries] = {}
  for k, query in queries.items():
    top_key, is_nondim = _resolve_operator_key(k, top_operators)
    if top_key is not None:
      if prognostics is None:
        prognostics = merge_prognostics(*components.values())
      obs = top_operators[top_key].observe(prognostics, query)
      if redim_transform is not None and not is_nondim:
        obs = redim_transform(obs)
      top_outputs[k] = obs
      continue
    is_diagnostics = k.removeprefix('nondim_') == 'diagnostics'
    if get_diagnostic_values_fn is not None and is_diagnostics:
      if diagnostics is None:
        diagnostics = get_diagnostic_values_fn()
      diagnostics_operator = observation_operators.DataObservationOperator(
          diagnostics
      )
      obs = diagnostics_operator.observe({}, query)
      if redim_transform is not None and not k.startswith('nondim_'):
        obs = redim_transform(obs)
      top_outputs[k] = obs
      continue
    matching = [
        name
        for name, ops in component_operators.items()
        if _resolve_operator_key(k, ops)[0] is not None
    ]
    if len(matching) > 1:
      raise ValueError(
          f'Observation operator key {k!r} is present on multiple model '
          f'components {matching} and cannot be uniquely dispatched.'
      )
    if not matching:
      available = set(top_operators).union(*component_operators.values())
      if get_diagnostic_values_fn is not None:
        available.add('diagnostics')
      raise ValueError(
          f'No observation operator for key {k!r}, available: '
          f'{sorted(available)}.'
      )
    [matched_component] = matching
    component_queries.setdefault(matched_component, {})[k] = query

  component_outputs: typing.Observation = {}
  for name, sub_queries in component_queries.items():
    component_outputs.update(components[name].observe(sub_queries))

  return {
      k: top_outputs[k] if k in top_outputs else component_outputs[k]
      for k in queries
  }


def primitive_equations_nodal_operator(
    ylm_map: spherical_harmonics.FixedYlmMapping,
) -> observation_operators.TransformObservationOperator:
  """Returns an operator observing nodal fields from primitive equation prognostics.

  Converts modal prognostic fields on `ylm_map.ylm_grid` to nodal fields on
  `ylm_map.lon_lat_grid` (passing through any already-nodal fields and `time`),
  computes `u_component_of_wind` and `v_component_of_wind` from `divergence`
  and `vorticity`, and computes `surface_pressure` by exponentiating
  `log_surface_pressure`.

  Args:
    ylm_map: Spherical harmonic mapping between the model's modal and nodal
      grids.
  """
  to_nodal = transforms.ModalToNodal(ylm_map, include_remaining=True)
  surface_pressure = transforms.Sequential([
      transforms.ApplyFnToKeys(cx.cpmap(jnp.exp), ['log_surface_pressure']),
      transforms.RenameKeys({'log_surface_pressure': 'surface_pressure'}),
  ])
  transform = transforms.Sequential([  # pyrefly: ignore[bad-argument-type]
      transforms.Merge({
          'uv': transforms.VelocityFromDivCurl(ylm_map),
          'nodal': to_nodal,
      }),
      transforms.Merge({
          'nodal': transforms.Identity(),  # pyrefly: ignore[bad-assignment]
          'sp': surface_pressure,
      }),
  ])
  return observation_operators.TransformObservationOperator(transform)
