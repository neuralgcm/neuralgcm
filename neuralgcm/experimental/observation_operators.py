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
from flax import nnx
from neuralgcm.experimental import coordax as cx


class ObservationOperator(nnx.Module, abc.ABC):
  """Base class for observation operators."""

  @abc.abstractmethod
  def observe(
      self,
      inputs: dict[str, cx.Field],
      query: dict[str, cx.Field | cx.Coordinate],
  ) -> dict[str, cx.Field]:
    """Returns observations for `query`."""
    ...


@dataclasses.dataclass
class DataObservationOperator(ObservationOperator):
  """Operator that returns pre-computed fields for matching coordinate queries.

  This observation operator matches keys and coordinates in the pre-computed
  dictionary of cx.Fields and the query to the observation operator. Since
  it is not possible to validate that the alignment data in the query matches
  the values in `self.fields`, this operator only supports queries expressed
  as cx.Coordinate.

  Attributes:
    fields: A dictionary of cx.Fields to return in the observation.
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
    is_coordinate = lambda x: isinstance(x, (cx.Coordinate, type(None)))
    valid_keys = list(self.fields.keys())
    for k, v in query.items():
      if k not in valid_keys:
        raise ValueError(f'Query contains {k=} not in {valid_keys}')
      if not is_coordinate(v):
        raise ValueError(
            f'DataObservationOperator only supports coordinate queries, got {v}'
        )
      coord = cx.extract_coordinate(self.fields[k])
      if v != coord:
        raise ValueError(f'Query ({k}, {v}) does not match field.{coord=}')
      observations[k] = self.fields[k]
    return observations
