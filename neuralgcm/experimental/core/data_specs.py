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

import dataclasses
import enum
from typing import Generic, TypeAlias, TypeVar

import coordax as cx
import jax
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import typing
import numpy as np


class AxisMatchRules(enum.Enum):
  """Rules for matching candidate coordinate axes to those in CoordSpec.

  Attributes:
    EXACT: candidate axis must match exactly.
    REPLACED: candidate axis must be replaced when calling `finalize_spec`.
    SUPERSET: candidate axis must be of the same type and contain all values.
    SHAPE: candidate axis must have the same shape.
    TYPE: candidate axis must be of the same type.
    ANY: no checks are performed.
  """

  EXACT = enum.auto()
  REPLACED = enum.auto()
  SUPERSET = enum.auto()
  SHAPE = enum.auto()
  TYPE = enum.auto()
  ANY = enum.auto()


@dataclasses.dataclass
class CoordSpec:
  """Specifies coordinate expectations.

  CoordSpec is used to express flexible data expectations by specifying rules
  for how to validate compatibility with a concrete coordinate on a per-axis
  basis. See examples below for common use-cases and `AxisMatchRules` for all
  available rules. By default sets `AxisMatchRules.EXACT` matching rule for all
  axes not explicitly included in `dim_match_rules`. If all `dim_match_rules`
  are set to `AxisMatchRules.EXACT`, the validation is equivalent to equality
  check with the coordinate that it wraps.

  Examples::

    dummy_delta = coordinates.TimeDelta(np.timedelta64(0, 's')[None])
    levels = coordinates.PressureLevels([100, 500])
    x = cx.LabeledAxis('x', np.linspace(0, np.pi, num=50))
    # expects exact `x`, any TimeDelta type and levels that include [100, 500].
    c_spec = CoordSpec(
        cx.compose_coordinates(dummy_delta, levels, x),
        {'timedelta': AxisMatchRules.TYPE, 'pressure': AxisMatchRules.SUPERSET},
    )
    delta_a = coordinates.TimeDelta(np.timedelta64(1, 's')[None])
    delta_b = coordinates.TimeDelta(np.arange(10) * np.timedelta64(1, 's'))
    good_levels = coordinates.PressureLevels.with_era5_levels()
    bad_levels = coordinates.PressureLevels([400, 500])  # missing 100
    xp = cx.LabeledAxis('x', np.linspace(0, np.pi, num=100))  # not == x.

    # Pass:
    c_spec.validate_compatible(cx.compose_coordinates(delta_a, good_levels, x))
    c_spec.validate_compatible(cx.compose_coordinates(delta_b, good_levels, x))

    # Raise:
    c_spec.validate_compatible(cx.compose_coordinates(delta_a, bad_levels, x))
    c_spec.validate_compatible(cx.compose_coordinates(delta_a, good_levels, xp))
    c_spec.validate_compatible(cx.compose_coordinates(good_levels, x))

  Attributes:
    coord: Coordinate that describes the supported data.
    dim_match_rules: Dictionary mapping dimension name to an axis matching rule.
      Dimensions without set values default to `AxisMatchRules.EXACT`.
  """

  coord: cx.Coordinate
  dim_match_rules: dict[str, AxisMatchRules] = dataclasses.field(
      default_factory=dict
  )

  def __post_init__(self):
    if not set(self.dim_match_rules.keys()).issubset(set(self.coord.dims)):
      raise ValueError(
          f'{self.dim_match_rules=} contains dimensions not present in'
          f' {self.coord=}.'
      )

  def validate_compatible(self, coord: cx.Coordinate):
    """Raises an informative error if not compatible with inferred candidate."""
    candidate = finalize_spec(self, coord)
    if candidate.dims != self.coord.dims:
      raise ValueError(
          f'Coordinate {self.coord} and {candidate=} have different dimensions'
      )

    for ax, expected_ax in zip(candidate.axes, self.coord.axes, strict=True):
      [dim] = expected_ax.dims
      match_schema = self.dim_match_rules.get(dim, AxisMatchRules.EXACT)
      if (
          match_schema == AxisMatchRules.EXACT
          or match_schema == AxisMatchRules.REPLACED
      ):
        if expected_ax != ax:
          raise ValueError(
              f'Coordinate axis {ax} for dimension {dim} does not'
              f' match expected axis {expected_ax}.'
          )
      elif match_schema == AxisMatchRules.TYPE:
        if not isinstance(ax, type(expected_ax)) or dim not in ax.dims:
          raise ValueError(
              f'Coordinate axis {ax} for dimension {dim} is not'
              ' present or not of the expected type'
              f' {expected_ax=}.'
          )
      elif match_schema == AxisMatchRules.SUPERSET:
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
      elif match_schema == AxisMatchRules.SHAPE:
        if ax.shape != expected_ax.shape:
          raise ValueError(
              f'Coordinate axis {ax} for dimension {dim} has shape'
              f' {ax.shape} which does not match the expected shape'
              f' {expected_ax.shape}.'
          )
      elif match_schema == AxisMatchRules.ANY:
        pass
      else:
        raise NotImplementedError(f'Unknown match schema: {match_schema}')

  @classmethod
  def with_any_timedelta(
      cls,
      coord: cx.Coordinate,
      dim_match_rules: dict[str, AxisMatchRules] | None = None,
  ):
    """Constructs CoordSpec with added timedelta and type match rule."""
    if dim_match_rules is None:
      dim_match_rules = {}
    dummy_timedelta = coordinates.TimeDelta(np.timedelta64(0, 's')[None])
    delta_dim = dummy_timedelta.dims[0]
    if (
        delta_dim in dim_match_rules
        and dim_match_rules[delta_dim] != AxisMatchRules.TYPE
    ):
      raise ValueError(
          f'with_any_timedelta got {dim_match_rules=} that conflicts with rule '
          f'"{AxisMatchRules.TYPE}" for "{delta_dim}" dimension.'
      )
    dim_match_rules |= {delta_dim: AxisMatchRules.TYPE}
    return cls(  # pytype: disable=wrong-arg-types
        coord=cx.coords.compose(dummy_timedelta, coord),
        dim_match_rules=dim_match_rules,
    )

  @classmethod
  def with_given_timedelta(
      cls,
      coord: cx.Coordinate,
      timedelta: np.ndarray = np.timedelta64(0, 's')[None],
      dim_match_rules: dict[str, AxisMatchRules] | None = None,
  ):
    """Constructs CoordSpec with added timedelta and superset match rule."""
    if dim_match_rules is None:
      dim_match_rules = {}
    dummy_timedelta = coordinates.TimeDelta(timedelta)
    delta_dim = dummy_timedelta.dims[0]
    if (
        delta_dim in dim_match_rules
        and dim_match_rules[delta_dim] != AxisMatchRules.SUPERSET
    ):
      raise ValueError(
          f'with_given_timedelta got {dim_match_rules=} that conflicts with'
          f' rule "{AxisMatchRules.SUPERSET}" for "{delta_dim}" dimension.'
      )
    dim_match_rules |= {delta_dim: AxisMatchRules.SUPERSET}
    return cls(  # pytype: disable=wrong-arg-types
        coord=cx.coords.compose(dummy_timedelta, coord),
        dim_match_rules=dim_match_rules,
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


def finalize_spec(
    coord_spec: CoordSpec | cx.Coordinate,
    source_coord: cx.Coordinate | None,
) -> cx.Coordinate:
  """Returns a concrete coordinate candidate for `coord_spec`.

  If `source_coord` is provided, the result is constructed by replacing axes of
  `source_coord` with axes from `coord_spec` for dimensions with `REPLACED`
  matching rule and equal shapes. In other cases a `source_coord` axis is
  retained. If axis in `coord_spec` is of type `CoordinateShard` and
  corresponding axis in `source_coord` is not, the sharding information is
  included in the output.

  If `source_coord` is `None` and coord_spec is a `CoordSpec`, returns
  `coord_spec.coord` as long as all dimensions are labeled as
  `AxisMatchRules.EXACT`, which conventionally is equivalent to using a
  `coord_spec.coord` directly. Otherwise an error is raised.

  If `coord_spec` is a `cx.Coordinate`, it is treated as `CoordSpec` with
  `AxisMatchRules.EXACT` for all dimensions.

  Args:
    coord_spec: A CoordSpec for which to construct a coordinate candidate.
    source_coord: A coordinate used to build a candidate coordinate from
      `coord_spec`.

  Returns:
    A concrete coordinate candidate for `coord_spec`.
  """
  if isinstance(coord_spec, cx.Coordinate):
    coord_spec = CoordSpec(coord_spec)  # default to exact match for all dims.

  if source_coord is None:
    if any(
        rule != AxisMatchRules.EXACT
        for rule in coord_spec.dim_match_rules.values()
    ):
      raise ValueError('Cannot derive coordinate without reference coord')
    return coord_spec.coord

  coord_in_spec = coord_spec.coord
  if coord_in_spec.dims != source_coord.dims:
    raise ValueError(
        f'CoordSpec {coord_spec} and ref coordinate {source_coord} have'
        ' different dims'
    )
  coord_axes = []
  is_shard = lambda x: isinstance(x, parallelism.CoordinateShard)
  dims_and_axes = zip(
      source_coord.dims, coord_in_spec.axes, source_coord.axes, strict=True
  )
  for dim, spec_ax, coord_ax in dims_and_axes:
    if (
        spec_ax.shape == coord_ax.shape
        and coord_spec.dim_match_rules.get(dim, AxisMatchRules.EXACT)
        == AxisMatchRules.REPLACED
    ):
      coord_axes.append(spec_ax)
    else:
      if is_shard(spec_ax) and not is_shard(coord_ax):
        spmd_shape = spec_ax.spmd_mesh_shape
        partitions = spec_ax.dimension_partitions
        coord_axes.append(
            parallelism.CoordinateShard(coord_ax, spmd_shape, partitions)
        )
      else:
        coord_axes.append(coord_ax)
  return cx.coords.compose(*coord_axes)


# Type alias for extended Spec objects that are used as InputsSpec.
CoordLikeSpec: TypeAlias = CoordSpec | OptionalSpec[cx.Coordinate | CoordSpec]
QuerySpec: TypeAlias = CoordSpec | FieldInQuerySpec[cx.Coordinate | CoordSpec]
QueriesSpec: TypeAlias = dict[str, dict[str, cx.Coordinate | QuerySpec]]
FinalizedQuerySpec: TypeAlias = cx.Coordinate | FieldInQuerySpec[cx.Coordinate]
FinalizedQueriesSpec: TypeAlias = dict[str, dict[str, FinalizedQuerySpec]]


def finalize_query_spec(
    query_spec: cx.Coordinate | QuerySpec,
    source_coord: cx.Coordinate | None,
) -> cx.Coordinate | FieldInQuerySpec[cx.Coordinate]:
  """Returns a `query_spec` with CoordSpec components being finalized."""
  if isinstance(query_spec, FieldInQuerySpec):
    if isinstance(query_spec.spec, CoordSpec):
      return FieldInQuerySpec(finalize_spec(query_spec.spec, source_coord))
    return FieldInQuerySpec(finalize_spec(query_spec.spec, source_coord))
  return finalize_spec(query_spec, source_coord)


def finalize_query_spec_pytree(
    query_spec_tree: typing.Pytree,
    source_tree: typing.Pytree | None = None,
) -> typing.Pytree:
  """Returns finalized leaves of `query_spec_tree`."""
  spec_types = (cx.Coordinate, CoordSpec, FieldInQuerySpec)
  is_coord_or_spec = lambda c: isinstance(c, spec_types)
  if source_tree is None:
    source_tree = jax.tree.map(
        lambda x: None, query_spec_tree, is_leaf=is_coord_or_spec
    )
  field_to_coord = lambda x: x.coordinate if cx.is_field(x) else x
  source_tree = jax.tree.map(
      field_to_coord, source_tree, is_leaf=cx.is_field
  )
  return jax.tree.map(
      finalize_query_spec,
      query_spec_tree,
      source_tree,
      is_leaf=is_coord_or_spec,
  )


def finalize_spec_pytree(
    coord_spec_tree: typing.Pytree,
    source_tree: typing.Pytree | None = None,
) -> typing.Pytree:
  """Returns finalized leaves of `coord_spec_tree`."""
  is_coord_or_spec = lambda c: isinstance(c, (cx.Coordinate, CoordSpec))
  if source_tree is None:
    source_tree = jax.tree.map(
        lambda x: None, coord_spec_tree, is_leaf=is_coord_or_spec
    )
  field_to_coord = lambda x: x.coordinate if cx.is_field(x) else x
  source_tree = jax.tree.map(
      field_to_coord, source_tree, is_leaf=cx.is_field
  )
  return jax.tree.map(
      finalize_spec,
      coord_spec_tree,
      source_tree,
      is_leaf=is_coord_or_spec,
  )


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
        raise ValueError(
            f'Got in_spec entry {var_spec} of unsupported type'
            f' "{type(var_spec)}"'
        )


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
        queries[data_key][var_name] = finalize_spec(spec, coord)
      elif isinstance(spec, cx.Coordinate):
        queries[data_key][var_name] = spec
  return queries
