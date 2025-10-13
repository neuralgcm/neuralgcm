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
from typing import Any, Callable, Literal

import coordax as cx
import jax
from neuralgcm.experimental.core import coordinates
import numpy as np


@dataclasses.dataclass
class DataSpec(abc.ABC):
  """Base class for data specs that parameterize inputs/queries in experiments.

  Experiments need to be able to read input data and format queries that
  will be issued to the model when making predictions. To specify input data
  schema it is necessary to provide a coordinate system for each variable that
  is read from the dataset. Queries additionally require differentiation of
  whether the query should contain a coordinate description or provide
  additional inputs in the form of coordax.Field. DataSpec objects
  facilitate both of these use cases by specifying the coordinate system for
  each input/query entry and differentiating how queries should be formatted via
  subclasses. See (`as_query`, `CoordQuerySpec`, `FieldQuerySpec`) and
  `as_input_specs` for details on how queries and data specs are formatted.
  """

  coord: cx.Coordinate


class CoordQuerySpec(DataSpec):
  """DataSpec that will be converted to a coordinate when formatting a query.

  When formatting a query, entries of this type in data specification are
  converted to the underlying coordinate object. e.g.
  ```
  specs = {'burgers': {'y': CoordQuerySpec(x_coordinate)}}
  actual_query = as_query(specs, inputs)
  expected_query = {'burgers': {'y': x_coordinate}}
  ```
  """

  ...


class FieldQuerySpec(DataSpec):
  """DataSpec that will be converted to a field when formatting a query.

  When formatting a query, entries of this type in data specification are
  replaced by the values in inputs (with matching coordinates).
  ```
  specs = {'burgers': {'time_to_eval_at': FieldQuerySpec(cx.Scalar())}}
  input_time = cx.wrap(jdt.to_datetime('2025-03-05-10:12:23'), cx.Scalar())
  inputs = {'burgers': {'time_to_eval_at': input_time}}
  actual_query = as_query(specs, inputs)
  expected_query = {'burgers': {'time_to_eval_at': input_time}}
  ```
  """

  ...


class QueryOnlySpec(DataSpec):
  """Similar to CoordQuerySpec, but is ignored when reading data.

  This class is helpful when specification of input data and queries are
  provided in a single structure. When formatting input_specs, entries of this
  class are dropped. A common use-case is specifying queries for which no data
  exists. e.g. production runs.
  """

  ...


class InputOnlySpec(DataSpec):
  """DataSpec that is ignored when formatting a query.

  This class is helpful when specification of input data and queries are
  provided in a single structure. When formatting the query, entries of this
  class are dropped. A common use-case is labeling dynamic input data
  (e.g. sea ice cover) that is required to run the model.
  """

  ...


DimMatchRules = Literal['exact', 'present', 'subset']


@dataclasses.dataclass
class InputDataSpec:
  """Specifies input data requirements with coordinate and axis matching rule.

  InputDataSpecs are used to specify data configuration required for a given
  task. To express variety of supported inputs variables can be marked as
  optional via `optional` attribute. Additionally, a different compatibility
  schema can be used for coordinate axes stored in `dim_match_rules`.
  Supported values are:
    - 'exact': coordinates must match exactly.
    - 'present': dimension must be present, but no other checks are performed.
    - 'subset': values in InputDataSpec must be a subset of values in data.

  Attributes:
    coord: Coordinate that describes the required data.
    dim_match_rules: Dictionary mapping dimension name to matching rule. If not
      specified, the default value is set to `exact` in `__post_init__`.
    optional: If True, the data is optional and will not cause
      `are_valid_input_specs` to fail if it is not present in the data.
  """

  coord: cx.Coordinate
  dim_match_rules: dict[str, DimMatchRules] = dataclasses.field(
      default_factory=dict
  )
  optional: bool = False

  def __post_init__(self):
    for dim in self.coord.dims:
      if dim not in self.dim_match_rules:
        self.dim_match_rules[dim] = 'exact'
    if set(self.coord.dims) != set(self.dim_match_rules.keys()):
      raise ValueError(
          f'{self.dim_match_rules=} and {self.coord=} have incompatible sets of'
          ' dimensions.'
      )

  @property
  def exact_match_coord(self) -> cx.Coordinate:
    """Returns self.coord without axes for which dim_match_rules != 'exact'."""
    return cx.compose_coordinates(*[
        ax
        for dim, ax in zip(self.coord.dims, self.coord.axes)
        if self.dim_match_rules[dim] == 'exact'
    ])

  @classmethod
  def with_any_timedelta(cls, coord: cx.Coordinate):
    """Constructs InputDataSpec with added timedelta and presence match rule."""
    dummy_timedelta = coordinates.TimeDelta(np.timedelta64(0, 's')[None])
    return cls(  # pytype: disable=wrong-arg-types
        coord=cx.compose_coordinates(dummy_timedelta, coord),
        dim_match_rules={dummy_timedelta.dims[0]: 'present'},
    )

  @classmethod
  def with_given_timedelta(
      cls,
      coord: cx.Coordinate,
      timedelta: np.ndarray = np.timedelta64(0, 's')[None],
  ):
    """Constructs InputDataSpec with added timedelta and subset match rule."""
    dummy_timedelta = coordinates.TimeDelta(timedelta)
    return cls(  # pytype: disable=wrong-arg-types
        coord=cx.compose_coordinates(dummy_timedelta, coord),
        dim_match_rules={dummy_timedelta.dims[0]: 'subset'},
    )


@dataclasses.dataclass
class OutputDataSpec:
  """Specifies outputs by coordinate, or indicates data should be provided."""

  coord: cx.Coordinate
  is_field: bool = False


def _are_compatible(
    coord: cx.Coordinate,
    required_spec: InputDataSpec,
) -> bool:
  """Returns True if provided_coord is compatible with required_spec."""
  if coord.ndim != required_spec.coord.ndim:
    return False

  for ax, required_ax in zip(coord.axes, required_spec.coord.axes):
    dim = required_ax.dims[0]
    match_schema = required_spec.dim_match_rules[dim]
    if match_schema == 'exact':
      if required_ax != ax:
        return False
    elif match_schema == 'present':
      if not isinstance(ax, type(required_ax)) or dim not in ax.dims:
        return False
    elif match_schema == 'subset':
      if not isinstance(ax, type(required_ax)):
        return False  # must be coordinates of the same type.
      required_ticks = required_ax.fields.get(dim)
      present_ticks = ax.fields.get(dim)
      if required_ticks is not None:
        return present_ticks is not None and np.all(
            np.isin(required_ticks.data, present_ticks.data)
        )
      else:  # no ticks, fall back to checking the size.
        return ax.shape >= required_ax.shape
    else:
      raise NotImplementedError(f'Unknown match schema: {match_schema}')
  return True


def are_valid_input_specs(
    data_specs: dict[str, dict[str, cx.Coordinate]],
    input_data_specs: dict[str, dict[str, InputDataSpec]],
) -> bool:
  """Validates that data_specs satisfy the requirements of input_data_specs."""
  for source, required_vars in input_data_specs.items():
    for var_name, required_spec in required_vars.items():
      is_present = source in data_specs and var_name in data_specs[source]
      if not is_present and not required_spec.optional:
        return False  # missing required var
      if is_present and not _are_compatible(
          data_specs[source][var_name], required_spec
      ):
        return False  # present but incompatible
  return True


def are_valid_inputs(
    inputs: dict[str, dict[str, cx.Field]],
    input_data_specs: dict[str, dict[str, InputDataSpec]],
) -> bool:
  """Validates that `inputs` satisfy the requirements of `input_data_specs."""
  data_specs = jax.tree.map(lambda x: x.coordinate, inputs, is_leaf=cx.is_field)
  return are_valid_input_specs(data_specs, input_data_specs)


def construct_query(
    data: dict[str, dict[str, cx.Field]],
    specs: dict[str, dict[str, OutputDataSpec]],
) -> dict[str, dict[str, cx.Coordinate | cx.Field]]:
  """Constructs query from data and OutputDataSpecs."""
  query = {}
  for source, spec_vars in specs.items():
    query[source] = {}
    for var_name, spec in spec_vars.items():
      if spec.is_field:
        query[source][var_name] = data[source][var_name]
      else:
        query[source][var_name] = spec.coord
  return query


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
  """Transforms structure of DataSpec objects into a query structure.

  Args:
    specs: nested disctionary of DataSpec that defines the query structure.
    inputs: input data of the same structure as specs that is used for query
      entries of type FieldQuerySpec that provide additional inputs.

  Returns:
    Formatted query.
  """
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
