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
from typing import Literal

import coordax as cx
import jax
from neuralgcm.experimental.core import coordinates
import numpy as np


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
