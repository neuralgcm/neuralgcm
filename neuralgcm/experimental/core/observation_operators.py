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
    valid_keys = list(self.fields.keys())
    for k, query_coord in query.items():
      if k not in valid_keys:
        raise ValueError(f'query contains {k=} not in {valid_keys}')
      if not cx.is_coord(query_coord):
        raise ValueError(
            'DataObservationOperator only supports coordinate queries, got'
            f' {query_coord}'
        )
      field = self.fields[k]
      if field.coordinate == query_coord:
        result = field
      else:
        try:
          result = field.sel({field.coordinate: query_coord})
        except KeyError as e:
          raise ValueError(
              f'query coordinate for {k!r} is not a valid slice of field:\n'
              f'{query_coord}\nvs\n{field.coordinate}'
          ) from e
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
