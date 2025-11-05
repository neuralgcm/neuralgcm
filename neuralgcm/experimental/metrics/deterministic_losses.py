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

"""Defines deterministic losses."""

from __future__ import annotations
import dataclasses
import coordax as cx
from neuralgcm.experimental.metrics import base
from neuralgcm.experimental.metrics import deterministic_metrics


@dataclasses.dataclass
class MSE(base.PerVariableLoss):
  """Mean Squared Error loss."""

  @property
  def statistics(self) -> dict[str, base.Statistic]:
    return {'SquaredError': deterministic_metrics.SquaredError()}

  def _values_from_mean_statistics_per_variable(
      self,
      statistic_values: dict[str, cx.Field],
  ) -> cx.Field:
    return statistic_values['SquaredError']


@dataclasses.dataclass
class MAE(base.PerVariableLoss):
  """Mean Absolute Error loss."""

  @property
  def statistics(self) -> dict[str, base.Statistic]:
    return {'AbsoluteError': deterministic_metrics.AbsoluteError()}

  def _values_from_mean_statistics_per_variable(
      self,
      statistic_values: dict[str, cx.Field],
  ) -> cx.Field:
    return statistic_values['AbsoluteError']
