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

"""Defines deterministic metrics."""

from __future__ import annotations
import dataclasses
import coordax as cx
import jax.numpy as jnp
from neuralgcm.experimental.metrics import base


@dataclasses.dataclass
class SquaredError(base.PerVariableStatistic):
  """Squared error statistics."""

  @property
  def unique_name(self):
    return 'SquaredError'

  def _compute_per_variable(
      self, predictions: cx.Field, targets: cx.Field
  ) -> cx.Field:
    return (predictions - targets) ** 2


@dataclasses.dataclass
class AbsoluteError(base.PerVariableStatistic):
  """Absolute error statistics."""

  @property
  def unique_name(self):
    return 'AbsoluteError'

  def _compute_per_variable(
      self, predictions: cx.Field, targets: cx.Field
  ) -> cx.Field:
    return cx.cmap(jnp.abs)(predictions - targets)


@dataclasses.dataclass
class Error(base.PerVariableStatistic):
  """Error statistics."""

  @property
  def unique_name(self):
    return 'Error'

  def _compute_per_variable(
      self, predictions: cx.Field, targets: cx.Field
  ) -> cx.Field:
    return predictions - targets


@dataclasses.dataclass
class WindVectorSquaredError(base.Statistic):
  """Computes squared error between two wind components."""
  u_name: str = 'u_component_of_wind'
  v_name: str = 'v_component_of_wind'
  vector_name: str = 'wind_vector'

  @property
  def unique_name(self) -> str:
    return 'WindVectorSquaredError_' + self.vector_name

  def compute(
      self,
      predictions: dict[str, cx.Field],
      targets: dict[str, cx.Field],
  ) -> dict[str, cx.Field]:
    u, v = predictions[self.u_name], predictions[self.v_name]
    u_target, v_target = targets[self.u_name], targets[self.v_name]
    return {self.vector_name: (u - u_target) ** 2 + (v - v_target) ** 2}


@dataclasses.dataclass
class RMSE(base.PerVariableMetric):
  """Root mean squared error metric."""

  @property
  def statistics(self) -> dict[str, base.Statistic]:
    return {'SquaredError': SquaredError()}

  def _values_from_mean_statistics_per_variable(
      self,
      statistic_values: dict[str, cx.Field],
  ) -> cx.Field:
    return cx.cmap(jnp.sqrt)(statistic_values['SquaredError'])


@dataclasses.dataclass
class WindVectorRMSE(base.Metric):
  """Computes vector RMSE between two wind components."""
  u_name: str = 'u_component_of_wind'
  v_name: str = 'v_component_of_wind'
  vector_name: str = 'wind_vector'

  @property
  def statistics(self) -> dict[str, base.Statistic]:
    return {
        'WindVectorSquaredError': WindVectorSquaredError(
            self.u_name, self.v_name, self.vector_name
        )
    }

  def _values_from_mean_statistics_with_internal_names(
      self,
      statistic_values: dict[str, dict[str, cx.Field]],
  ) -> dict[str, cx.Field]:
    wind_vector_se = statistic_values['WindVectorSquaredError']
    return {k: cx.cmap(jnp.sqrt)(v) for k, v in wind_vector_se.items()}


MSE = SquaredError
MAE = AbsoluteError
Bias = Error
