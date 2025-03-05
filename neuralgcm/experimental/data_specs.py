# Copyright 2025 Google LLC
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

"""Defines DataSpec objects that parameterize inputs/queries in experiments."""

import abc
import dataclasses

from typing import Any, Callable

import jax
from neuralgcm.experimental import coordax as cx


@dataclasses.dataclass
class DataSpec(abc.ABC):
  """Base class for data specs that parameterize inputs/queries in experiments.

  Experiments need to be able to read observation data and format queries that
  will be issued to the model when making predictions. DataSpec objects
  accomplish exactly that by specifying the coordinate system for each
  input/query entry. We use subclasses of DataSpec to differentiate how entries
  behave in data reading and query formatting contexts. Unless specified
  otherwise, all DataSpec subclasses configure both inputs and queries.
  """

  coord: cx.Coordinate


class CoordQuerySpec(DataSpec):
  """DataSpec that will be converted to a coordinate when formatting a query."""
  ...


class FieldQuerySpec(DataSpec):
  """DataSpec that will be converted to a field when formatting a query."""
  ...


class QueryOnlySpec(DataSpec):
  """Similar to CoordQuerySpec, but is ignored when reading data."""
  ...


class InputOnlySpec(DataSpec):
  """DataSpec that is ignored when formatting a query."""
  ...


def _remove_empty(inputs: dict[str, Any]) -> dict[str, Any]:
  """Recursively removes None or empty values from a nested dict."""
  is_dict = lambda x: isinstance(x, dict)
  result = {k: _remove_empty(v) if is_dict(v) else v for k, v in inputs.items()}
  is_empty = lambda x: x is None or (is_dict(x) and not x)
  result = {k: v for k, v in result.items() if not is_empty(v)}
  return result


def as_query(
    specs: dict[str, dict[str, DataSpec]],
    inputs: dict[str, dict[str, cx.Field]] | None = None,
) -> dict[str, dict[str, cx.Field | cx.Coordinate]]:
  """Transforms structure of DataSpec objects into a query structure."""
  is_coord_query = lambda x: isinstance(x, (CoordQuerySpec, QueryOnlySpec))
  is_field_query = lambda x: isinstance(x, FieldQuerySpec)
  def get_query(s, x):
    if is_coord_query(s):
      return s.coord
    elif is_field_query(s):
      return x
    else:
      return None

  return _remove_empty(jax.tree.map(get_query, specs, inputs))


def as_input_specs(
    specs: dict[str, dict[str, DataSpec]],
) -> dict[str, dict[str, cx.Coordinate]]:
  """Transforms structure of DataSpec objects into an input specs structure."""
  is_query_only = lambda x: isinstance(x, QueryOnlySpec)
  get_coord = lambda s: None if is_query_only(s) else s.coord
  return _remove_empty(jax.tree.map(get_coord, specs))


def filter_by_specs(
    filter_fn: Callable[[DataSpec], bool],
    specs: dict[str, dict[str, DataSpec]],
    inputs: dict[str, dict[str, cx.Field]],
):
  """Returns `inputs` with entries filtered by `filter_fn` on `specs`."""
  replace_skip_with_none = lambda s, x: x if filter_fn(s) else None
  return _remove_empty(jax.tree.map(replace_skip_with_none, specs, inputs))
