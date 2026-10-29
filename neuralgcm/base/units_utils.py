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

"""Unit conversion utilities for NeuralGCM models.

NeuralGCM models evolve nondimensional state in their `sim_units` and expose a
dimensional interface in physical units defined by their `units_mapping`. The
helpers here convert fields between physical units and (possibly distinct)
simulation unit systems, and are exact no-ops when no conversion is required.
"""

from collections.abc import Mapping

import coordax as cx
import jax.numpy as jnp
import jax_datetime as jdt
from terrax.core import typing
from terrax.core import units


def unit_factor(sim_units: units.SimUnits | None, unit: str) -> float:
  """Returns the nondimensional value of one `unit` in `sim_units`.

  Args:
    sim_units: simulation units to express `unit` in. `None` denotes physical
      units, for which the factor is `1.0`.
    unit: physical unit string, e.g. `'meter / second'`.

  Returns:
    The nondimensional value of `1 unit` in `sim_units`.
  """
  if sim_units is None:
    return 1.0
  return float(sim_units.nondimensionalize(typing.Quantity(unit)))


def units_passage(
    fields: Mapping[str, cx.Field],
    source_units: units.SimUnits | None,
    target_units: units.SimUnits | None,
    units_mapping: Mapping[str, str],
) -> dict[str, cx.Field]:
  """Converts `fields` from `source_units` to `target_units`.

  Each field is rescaled according to its physical unit in `units_mapping`.
  `None` stands for physical units, so the same function nondimensionalizes
  (`source_units=None`), redimensionalizes (`target_units=None`) or converts
  between two simulation unit systems.

  Conversions between simulation units with the same `scale` are exact no-ops
  and return `fields` unchanged. Fields with a unit factor of one (e.g.
  `dimensionless`) and datetime fields are also returned as-is, which preserves
  their dtype; other fields are multiplied by a static factor in their own
  floating point dtype.

  Args:
    fields: Fields keyed by variable name.
    source_units: Units in which `fields` are expressed; `None` for physical
      units.
    target_units: Units to express `fields` in; `None` for physical units.
    units_mapping: Physical unit of each variable, defining the quantity that
      its nondimensional values represent (the units mapping of the model on
      the nondimensional side; with both sides nondimensional, the source's).

  Returns:
    `fields` in `target_units`, in the same key order.

  Raises:
    ValueError: If a rescaled key is missing from `units_mapping` or has an
      integer or boolean dtype.
  """
  if (
      source_units is not None
      and target_units is not None
      and source_units.scale == target_units.scale
  ):
    return dict(fields)
  outputs = {}
  for k, field in fields.items():
    if isinstance(field.data, jdt.Datetime):
      outputs[k] = field
      continue
    if k not in units_mapping:
      raise ValueError(f'Key {k!r} not found in {units_mapping=}')
    unit = units_mapping[k]
    factor = unit_factor(target_units, unit) / unit_factor(source_units, unit)
    if factor == 1.0:
      outputs[k] = field
      continue
    dtype = field.dtype
    if dtype is None or not jnp.issubdtype(dtype, jnp.inexact):
      raise ValueError(
          f'Cannot rescale {k!r} of dtype {dtype} by {factor}; only '
          'floating point fields can change units.'
      )
    outputs[k] = field * factor
  return outputs


def nondim_query(
    query: typing.Query,
    sim_units: units.SimUnits,
    units_mapping: Mapping[str, str],
) -> typing.Query:
  """Nondimensionalizes physical `Field` entries of `query` into `sim_units`.

  Coordinate entries are returned as-is and `typing.Auxiliary` wrapping is
  preserved. Field entries are converted with `units_passage`, so entries with
  a unit factor of one keep their identity and dtype.

  Args:
    query: query whose `Field` entries are expressed in physical units.
    sim_units: simulation units to nondimensionalize `Field` entries into.
    units_mapping: mapping from `Field` entry names to their physical units.

  Returns:
    `query` with `Field` entries nondimensionalized into `sim_units`.
  """
  outputs = {}
  for k, raw_entry in query.items():
    entry, is_aux = typing.unwrap_auxiliary(raw_entry)
    if cx.is_field(entry):
      [entry] = units_passage(
          {k: entry}, None, sim_units, units_mapping
      ).values()
      outputs[k] = typing.Auxiliary(entry) if is_aux else entry
    else:
      outputs[k] = raw_entry
  return outputs


def merge_units_mappings(*mappings: Mapping[str, str]) -> dict[str, str]:
  """Merges units mappings, raising on conflicting units for a variable."""
  merged: dict[str, str] = {}
  for mapping in mappings:
    for k, unit in mapping.items():
      if k in merged and merged[k] != unit:
        raise ValueError(
            f'Conflicting units for {k!r} across units mappings: '
            f'{merged[k]!r} vs {unit!r}.'
        )
      merged[k] = unit
  return merged
