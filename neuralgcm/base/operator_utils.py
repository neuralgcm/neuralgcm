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

from collections.abc import Mapping
from typing import TypeAlias, TypeVar

import coordax as cx
import jax.numpy as jnp
from neuralgcm.base import model
from neuralgcm.base import units_utils
from terrax.core import module_utils
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
  """Collects `operators` from `model_instance` and any nested `Model`s.

  Operators of outer models take precedence over those of nested models.
  Nested models at equal depth are merged in traversal order, so a later
  sibling overrides an earlier one on duplicate keys.

  Args:
    model_instance: model whose own and nested `operators` are collected.

  Returns:
    Mapping from query key to observation operator, with outer models' entries
    overriding those of nested models.
  """
  models, paths = module_utils.retrieve_subclass_modules(
      model_instance, model.Model, return_paths=True
  )
  all_operators: dict[str, typing.ObservationOperator] = {}
  # Deepest models first, so that outer models override their operators.
  for _, nested_model in sorted(
      zip(paths, models), key=lambda item: len(item[0]), reverse=True
  ):
    all_operators.update(nested_model.operators)
  all_operators.update(model_instance.operators)
  return all_operators


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


def dispatch_observe_nondim(
    queries: typing.Queries,
    *,
    composite: model.Model,
    components: Mapping[str, model.Model],
) -> typing.Observation:
  """Dispatches nondimensional `queries` across `composite` and `components`.

  Implements `observe_nondim` for composite models: each query key (without
  `'nondim_'` prefix) is resolved in the following order:
    1. Top-level `composite.operators` take precedence over component operators
       and are evaluated with `composite.prognostic_values()` as `inputs`.
    2. The `'diagnostics'` key observes `composite.diagnostic_values()`.
    3. Remaining keys are routed to the unique component in `components` whose
       `operators` contain the key. All queries for a given component are
       evaluated in a single `component.observe_nondim(sub_queries)` call, and
       the results are re-expressed in `composite.sim_units` via
       `units_utils.units_passage` (a no-op when the component shares the
       composite's scale).
  Field entries of `queries` are expected in `composite.sim_units` and are
  re-expressed in the receiving component's `sim_units` accordingly.

  Args:
    queries: Nondimensional observation queries keyed by dataset/operator key.
    composite: The composite model whose `observe_nondim` is being implemented.
    components: Component models keyed by component name.

  Returns:
    Nondimensional observations in `composite.sim_units`, keyed like `queries`.
  """
  assert composite.sim_units is not None and composite.units_mapping is not None
  component_operators = {
      name: component.operators for name, component in components.items()
  }
  prognostics: dict[str, cx.Field] | None = None
  top_outputs: typing.Observation = {}
  component_queries: dict[str, typing.Queries] = {}
  for k, query in queries.items():
    if k in composite.operators:
      if prognostics is None:
        prognostics = composite.prognostic_values()
      top_outputs[k] = composite.operators[k].observe(prognostics, query)
      continue
    if k == 'diagnostics':
      diagnostics_operator = observation_operators.DataObservationOperator(
          composite.diagnostic_values()
      )
      top_outputs[k] = diagnostics_operator.observe({}, query)
      continue
    matching = [name for name, ops in component_operators.items() if k in ops]
    if len(matching) > 1:
      raise ValueError(
          f'Observation operator key {k!r} is present on multiple model '
          f'components {matching} and cannot be uniquely dispatched.'
      )
    if not matching:
      available = {*composite.operators, 'diagnostics'}.union(
          *component_operators.values()
      )
      raise ValueError(
          f'No observation operator for key {k!r}, available: '
          f'{sorted(available)}.'
      )
    [matched_component] = matching
    component_queries.setdefault(matched_component, {})[k] = query

  component_outputs: typing.Observation = {}
  for name, sub_queries in component_queries.items():
    component = components[name]
    assert component.sim_units is not None
    assert component.units_mapping is not None
    sub_queries = {
        k: _passage_query(q, composite, component)
        for k, q in sub_queries.items()
    }
    for k, obs in component.observe_nondim(sub_queries).items():
      component_outputs[k] = units_utils.units_passage(
          obs, component.sim_units, composite.sim_units, component.units_mapping
      )

  return {
      k: top_outputs[k] if k in top_outputs else component_outputs[k]
      for k in queries
  }


def _passage_query(
    query: typing.Query, source_model: model.Model, target_model: model.Model
) -> typing.Query:
  """Converts nondimensional `Field` entries of `query` for `target_model`."""
  source_units, target_units = source_model.sim_units, target_model.sim_units
  units_mapping = source_model.units_mapping
  assert source_units is not None and target_units is not None
  assert units_mapping is not None
  if source_units.scale == target_units.scale:
    return query
  outputs = {}
  for k, raw_entry in query.items():
    entry, is_aux = typing.unwrap_auxiliary(raw_entry)
    if cx.is_field(entry):
      [entry] = units_utils.units_passage(
          {k: entry}, source_units, target_units, units_mapping
      ).values()
      outputs[k] = typing.Auxiliary(entry) if is_aux else entry
    else:
      outputs[k] = raw_entry
  return outputs


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
