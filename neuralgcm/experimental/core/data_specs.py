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
import enum
from typing import Any, Callable, Generic, TypeAlias, TypeVar

import coordax as cx
import jax
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import typing
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


@enum.unique
class DimMatchRules(enum.Enum):
  """Rules for matching dimensions in CoordSpec.

  Attributes:
    EXACT: coordinates must match exactly.
    PRESENT: dimension must be present, but no other checks are performed.
    SUBSET: values in CoordSpec must be a subset of values in the data.
  """

  EXACT = 'exact'
  PRESENT = 'present'
  SUBSET = 'subset'


@dataclasses.dataclass
class CoordSpec:
  """A partial specification of the coordinate to be completed from data.

  Can be used to express flexible data expectations by specifying validation
  rules for coordinate axes individually. By default sets EXACT matching rule
  for all axes not included in `dim_match_rules`. If all `dim_match_rules` are
  set to `DimMatchRules.EXACT`, the specification is equivalent to the
  coordinate that it wraps.

  Examples:
    dummy_delta = coordinates.TimeDelta(np.timedelta64(0, 's')[None])
    levels = coordinates.PressureLevels.with_13_era5_levels()
    x = cx.LabeledAxis('x', np.linspace(0, np.pi, num=50))
    c_spec = CoordSpec(
        cx.compose_coordinates(dummy_delta, levels, x),
        {'timedelta': DimMatchRules.PRESENT, 'pressure': DimMatchRules.SUBSET},
    )
    delta_a = coordinates.TimeDelta(np.timedelta64(1, 's')[None])
    delta_b = coordinates.TimeDelta(np.arange(10) * np.timedelta64(1, 's'))
    good_levels = coordinates.PressureLevels.with_era5_levels()
    bad_levels = coordinates.PressureLevels([400, 1000])
    xp = cx.LabeledAxis('x', np.linspace(0, np.pi, num=100))
    # Pass:
    c_spec.validate_compatible(cx.compose_coordinates(delta_a, good_levels, x))
    c_spec.validate_compatible(cx.compose_coordinates(delta_b, good_levels, x))
    # Raise:
    c_spec.validate_compatible(cx.compose_coordinates(delta_a, bad_levels, x))
    c_spec.validate_compatible(cx.compose_coordinates(delta_a, good_levels, xp))
    c_spec.validate_compatible(cx.compose_coordinates(good_levels, x))

  Attributes:
    coord: Coordinate that describes the supported data.
    dim_match_rules: Dictionary mapping dimension name to matching rule.
      Dimensions without set values default to `DimMatchRules.EXACT`.
  """

  coord: cx.Coordinate
  dim_match_rules: dict[str, DimMatchRules] = dataclasses.field(
      default_factory=dict
  )

  def __post_init__(self):
    for dim in self.coord.dims:
      if dim not in self.dim_match_rules:
        self.dim_match_rules[dim] = DimMatchRules.EXACT
    if set(self.coord.dims) != set(self.dim_match_rules.keys()):
      raise ValueError(
          f'{self.dim_match_rules=} and {self.coord=} have incompatible sets of'
          ' dimensions.'
      )

  def validate_compatible(self, coord: cx.Coordinate):
    """Raises an informative error if not compatible with `coord`."""
    if coord.dims != self.coord.dims:
      raise ValueError(
          f'Coordinates {self.coord} and {coord} have different dimensions'
      )

    for ax, expected_ax in zip(coord.axes, self.coord.axes):
      dim = expected_ax.dims[0]
      match_schema = self.dim_match_rules[dim]
      if match_schema == DimMatchRules.EXACT:
        if expected_ax != ax:
          raise ValueError(
              f'Coordinate axis {ax} for dimension {dim} does not'
              f' match expected axis {expected_ax}.'
          )
      elif match_schema == DimMatchRules.PRESENT:
        if not isinstance(ax, type(expected_ax)) or dim not in ax.dims:
          raise ValueError(
              f'Coordinate axis {ax} for dimension {dim} is not'
              ' present or not of the expected type'
              f' {type(expected_ax)}.'
          )
      elif match_schema == DimMatchRules.SUBSET:
        if not isinstance(ax, type(expected_ax)):
          raise ValueError(
              f'Coordinate axis {ax} for dimension {dim} is not'
              f' of the expected type {type(expected_ax)}.'
          )
        required_ticks = expected_ax.fields.get(dim)
        present_ticks = ax.fields.get(dim)
        if required_ticks is not None:
          if present_ticks is None or not np.all(
              np.isin(required_ticks.data, present_ticks.data)
          ):
            raise ValueError(
                f'Coordinate axis {ax} for dimension {dim} does'
                ' not contain all required ticks from'
                f' {expected_ax}.'
            )
        else:  # no ticks, fall back to checking the size.
          if ax.shape < expected_ax.shape:
            raise ValueError(
                f'Coordinate axis {ax} for dimension {dim} has'
                f' shape {ax.shape} which is smaller than the'
                f' expected shape {expected_ax.shape}.'
            )
      else:
        raise NotImplementedError(f'Unknown match schema: {match_schema}')

  def to_coordinate(self, candidate: cx.Coordinate | None) -> cx.Coordinate:
    """Returns a verified, concrete coordinate compatible with this spec.

    If `candidate` is provided, verifies that it is is compatible with this spec
    and returns it. If no candidate is provided, returns self.coord if all
    dimensions are labeled as `DimMatchRules.EXACT`, otherwise raises an error.

    Args:
      candidate: A coordinate to use as the concrete coordinate.

    Returns:
      A concrete coordinate compatible with this spec.

    Raises:
      ValueError: If `candidate` is not None and not compatible with this spec,
        or if no candidate is provided and not all dimensions are labeled as
        `DimMatchRules.EXACT`.
    """
    if candidate is None:
      if set(self.dim_match_rules.values()) != {DimMatchRules.EXACT}:
        raise ValueError(
            'Cannot make concrete coordinate without reference data'
        )
      return self.coord
    self.validate_compatible(candidate)
    return candidate

  @classmethod
  def with_any_timedelta(cls, coord: cx.Coordinate):
    """Constructs CoordSpec with added timedelta and presence match rule."""
    dummy_timedelta = coordinates.TimeDelta(np.timedelta64(0, 's')[None])
    return cls(  # pytype: disable=wrong-arg-types
        coord=cx.compose_coordinates(dummy_timedelta, coord),
        dim_match_rules={dummy_timedelta.dims[0]: DimMatchRules.PRESENT},
    )

  @classmethod
  def with_given_timedelta(
      cls,
      coord: cx.Coordinate,
      timedelta: np.ndarray = np.timedelta64(0, 's')[None],
  ):
    """Constructs CoordSpec with added timedelta and subset match rule."""
    dummy_timedelta = coordinates.TimeDelta(timedelta)
    return cls(  # pytype: disable=wrong-arg-types
        coord=cx.compose_coordinates(dummy_timedelta, coord),
        dim_match_rules={dummy_timedelta.dims[0]: DimMatchRules.SUBSET},
    )


T = TypeVar('T')


@dataclasses.dataclass
class OptionalSpec(Generic[T]):
  """Wrapper that indicates that a spec is optional."""
  spec: T


@dataclasses.dataclass
class FieldInQuerySpec(Generic[T]):
  """Wrapper that indicates that the entry should be of type `cx.Field`."""
  spec: T


# Type alias for extended Spec objects that are used as InputsSpec.
CoordLikeSpec: TypeAlias = CoordSpec | OptionalSpec[cx.Coordinate | CoordSpec]
QuerySpec: TypeAlias = CoordSpec | FieldInQuerySpec[cx.Coordinate | CoordSpec]


def get_coord_types(
    coordinate: cx.Coordinate | CoordSpec,
) -> tuple[type[cx.Coordinate], ...]:
  """Returns tuple of coordinate types present in `coordinate`."""
  if isinstance(coordinate, CoordSpec):
    coordinate = coordinate.coord

  is_cartesian_prod = lambda x: isinstance(x, cx.CartesianProduct)
  if is_cartesian_prod(coordinate):
    # using dict.fromkeys to preserve order of appearance.
    types = list(dict.fromkeys(type(x) for x in coordinate.coordinates))
  else:
    types = [type(coordinate)]
  # if LabeledAxis is present, move it to the end of the list to ensure that we
  # try inferring other coordinates first.
  if cx.LabeledAxis in types:
    types.remove(cx.LabeledAxis)
    types.append(cx.LabeledAxis)
  return tuple(types)


def unwrap_optional(spec: T | OptionalSpec[T]) -> tuple[T, bool]:
  """Returns underlying spec and a bool indicating if spec is Optional."""
  is_optional = isinstance(spec, OptionalSpec)
  inner_spec = spec.spec if is_optional else spec  # pytype: disable=attribute-error
  return inner_spec, is_optional


def _maybe_unwrap_field_spec(spec: T | FieldInQuerySpec[T]) -> tuple[T, bool]:
  """Returns underlying spec and a bool indicating field in query request."""
  is_field_spec = isinstance(spec, FieldInQuerySpec)
  inner_spec = spec.spec if is_field_spec else spec  # pytype: disable=attribute-error
  return inner_spec, is_field_spec


def validate_inputs(
    inputs: dict[str, dict[str, cx.Coordinate]] | typing.InputFields,
    in_spec: dict[str, dict[str, cx.Coordinate | CoordLikeSpec]],
):
  """Validates that `inputs` satisfy expectations of `in_spec`."""
  for dataset_key, dataset_spec in in_spec.items():
    if dataset_key not in inputs:
      raise ValueError(f'Data key {dataset_key} is missing in {inputs.keys()=}')

    in_data = inputs[dataset_key]
    for var_name, var_spec in dataset_spec.items():
      inner_spec, is_optional = unwrap_optional(var_spec)
      if var_name not in in_data:
        if is_optional:
          continue
        else:
          raise ValueError(f'Missing non-optional variables "{var_name}"')

      x = in_data[var_name]
      data_coord = x.coordinate if cx.is_field(x) else x
      if isinstance(inner_spec, cx.Coordinate):
        inner_spec = CoordSpec(inner_spec)
      if isinstance(inner_spec, CoordSpec):
        inner_spec.validate_compatible(data_coord)
      else:
        raise ValueError(f'Got in_spec entry {in_spec} of unsupported type')


def construct_query(
    inputs: typing.InputFields,
    queries_spec: dict[str, dict[str, cx.Coordinate | QuerySpec]],
) -> typing.Queries:
  """Constructs query from data and OutputDataSpecs."""
  queries = {}
  for data_key, query_spec in queries_spec.items():
    queries[data_key] = {}
    for var_name, spec in query_spec.items():
      spec, is_field_in_query = _maybe_unwrap_field_spec(spec)

      if is_field_in_query:
        queries[data_key][var_name] = inputs[data_key][var_name]
      elif isinstance(spec, CoordSpec):
        in_data = inputs.get(data_key, {})
        x = in_data.get(var_name, None)
        coord = x.coordinate if (x is not None and cx.is_field(x)) else None
        queries[data_key][var_name] = spec.to_coordinate(coord)
      elif isinstance(spec, cx.Coordinate):
        queries[data_key][var_name] = spec
  return queries


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
