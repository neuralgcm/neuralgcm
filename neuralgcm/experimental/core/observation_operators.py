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

"""Defines observation operator API and sample operators for NeuralGCM."""

import abc
import dataclasses

import coordax as cx
from flax import nnx
from neuralgcm.experimental.core import typing
import numpy as np


# pylint: disable=g-classes-have-attributes


class ObservationOperatorABC(nnx.Module, abc.ABC):
  """Base class for observation operators."""

  @abc.abstractmethod
  def observe(
      self,
      inputs: dict[str, cx.Field],
      query: dict[str, cx.Field | cx.Coordinate],
  ) -> dict[str, cx.Field]:
    """Returns observations for `query`."""
    ...

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(pytree=False, **kwargs)


def _collect_coord_components(
    coord: cx.Coordinate, dim: str
) -> list[cx.Coordinate]:
  """Returns the coordinate component with the given name, if it exists."""
  return [c for c in cx.coords.canonicalize(coord) if c.dims == (dim,)]


def _subset_query_from_field(
    field: cx.Field, query_coords: cx.Coordinate
) -> cx.Field | None:
  """Returns a subset of `field` that matches `query_coords`, or `None`."""
  if query_coords == field.coordinate:
    return field

  if query_coords.dims != field.dims:
    return None

  # TODO(shoyer): Consider making this logic more generic and potentially moving
  # it into coordax. Currently we use a hard-coded list of coordinates for which
  # we know no other metadata is necessary to match indices.
  for dim in ['pressure', 'sigma', 'layer_index', 'soil_levels']:
    if dim not in query_coords.dims:
      continue
    (query_component,) = _collect_coord_components(query_coords, dim)
    (field_component,) = _collect_coord_components(field.coordinate, dim)
    query_levels = query_component.fields[dim].data.tolist()
    field_levels = field_component.fields[dim].data.tolist()
    try:
      indices = [
          field_levels.index(query_levels) for query_levels in query_levels
      ]
    except ValueError as e:
      raise ValueError(
          f'query vertical coordinate {query_component} is not a subset of '
          f'field vertical coordinate {field_component}'
      ) from e

    untagged_field = field.untag(field_component)
    out_axes = untagged_field.named_axes
    field = cx.cmap(lambda x: x[np.array(indices)], out_axes)(  # pylint: disable=cell-var-from-loop
        untagged_field
    ).tag(query_component)

  if query_coords == field.coordinate:
    return field  # successful indexing

  return None


@dataclasses.dataclass
class DataObservationOperator(ObservationOperatorABC):
  """Operator that returns pre-computed fields for matching coordinate queries.

  This observation operator matches keys and coordinates in the pre-computed
  dictionary of `coordax.Field`s and the query to the observation operator. This
  operator requires that all `query` entries are of `coordax.Coordinate` type.

  Attributes:
    fields: A dictionary of `coordax.Field`s to return in the observation.
  """

  fields: dict[str, cx.Field]

  def observe(
      self,
      inputs: dict[str, cx.Field],
      query: dict[str, cx.Field | cx.Coordinate],
  ) -> dict[str, cx.Field]:
    """Returns observations for `query` matched against `self.fields`."""
    del inputs  # unused.
    observations = {}
    is_coordinate = lambda x: isinstance(x, cx.Coordinate)
    valid_keys = list(self.fields.keys())
    for k, query_coord in query.items():
      if k not in valid_keys:
        raise ValueError(f'query contains {k=} not in {valid_keys}')
      if not is_coordinate(query_coord):
        raise ValueError(
            'DataObservationOperator only supports coordinate queries, got'
            f' {query_coord}'
        )
      field = self.fields[k]
      result = _subset_query_from_field(field, query_coord)
      if result is None:
        raise ValueError(
            f'query coordinate for {k!r} does not match field:\n'
            f'{query_coord}\nvs\n{field.coordinate}'
        )
      observations[k] = result

    return observations


@dataclasses.dataclass
class TransformObservationOperator(ObservationOperatorABC):
  """Operator that returns transformed inputs as observations."""

  transform: typing.Transform
  requested_fields_from_query: tuple[str, ...] = ()

  def observe(
      self,
      inputs: dict[str, cx.Field],
      query: dict[str, cx.Field | cx.Coordinate],
  ) -> typing.Observation:
    q_fields = {k: query.get(k, None) for k in self.requested_fields_from_query}
    bad_fields = {k: v for k, v in q_fields.items() if not cx.is_field(v)}
    if bad_fields:
      raise ValueError(f'Got {query=} that contains {bad_fields=}.')
    data_obserator = DataObservationOperator(self.transform(inputs | q_fields))
    coord_queries = {k: v for k, v in query.items() if cx.is_coord(v)}
    return data_obserator.observe({}, coord_queries)


@dataclasses.dataclass
class ObservationOperatorWithRenaming(ObservationOperatorABC):
  """Operator wrapper that converts between different naming conventions.

  Attributes:
    operator: Observation operator that performs computation.
    renaming_dict: A dictionary mapping new names to those used by `operator`.
  """

  operator: typing.ObservationOperator
  renaming_dict: dict[str, str]

  def observe(
      self,
      inputs: dict[str, cx.Field],
      query: dict[str, cx.Field | cx.Coordinate],
  ) -> dict[str, cx.Field]:
    """Returns observations for `query` matched against `self.fields`."""
    renamed_query = {self.renaming_dict.get(k, k): v for k, v in query.items()}
    observation = self.operator.observe(inputs, renamed_query)
    inverse_renaming_dict = {v: k for k, v in self.renaming_dict.items()}
    return {inverse_renaming_dict.get(k, k): v for k, v in observation.items()}


@dataclasses.dataclass
class MultiObservationOperator(ObservationOperatorABC):
  """Operator that dispatches queries to multiple operators.

  Attributes:
    keys_to_operator: A dictionary mapping query keys to observation operators.
  """

  keys_to_operator: dict[tuple[str, ...], typing.ObservationOperator]

  def observe(
      self,
      inputs: dict[str, cx.Field],
      query: dict[str, cx.Field | cx.Coordinate],
  ) -> dict[str, cx.Field]:
    outputs = {}
    supported_keys = set(sum(self.keys_to_operator.keys(), start=()))
    query_keys = set(query.keys())
    if not query_keys.issubset(supported_keys):
      raise ValueError(
          f'query keys {query_keys} are not a subset of supported keys'
          f' {supported_keys}'
      )
    for key_tuple, obs_op in self.keys_to_operator.items():
      sub_query = {}
      for key in key_tuple:
        if key in query:
          sub_query[key] = query[key]
      outputs |= obs_op.observe(inputs, sub_query)
    return outputs
