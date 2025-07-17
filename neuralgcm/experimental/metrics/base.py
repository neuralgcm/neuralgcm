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

"""Defines base Statistic, Metric and Loss classes operating on cx.Field."""

from __future__ import annotations
import abc
import collections
import dataclasses
import coordax as cx


@dataclasses.dataclass
class Statistic(abc.ABC):
  """Abstract base class for a statistic."""

  @property
  @abc.abstractmethod
  def unique_name(self) -> str:
    """A unique name for this statistic."""
    ...

  @abc.abstractmethod
  def compute(
      self,
      predictions: dict[str, cx.Field],
      targets: dict[str, cx.Field],
  ) -> dict[str, cx.Field]:
    """Computes the statistic values."""
    ...


@dataclasses.dataclass
class PerVariableStatistic(Statistic):
  """Abstract base class for a statistic computed independently per variable."""

  @abc.abstractmethod
  def _compute_per_variable(
      self, predictions: cx.Field, targets: cx.Field
  ) -> cx.Field:
    """Computes the statistic for a single variable."""
    ...

  def compute(
      self,
      predictions: dict[str, cx.Field],
      targets: dict[str, cx.Field],
  ) -> dict[str, cx.Field]:
    return {
        k: self._compute_per_variable(predictions[k], targets[k])
        for k in targets
    }


@dataclasses.dataclass
class Metric(abc.ABC):
  """Base class for a metric."""

  @property
  @abc.abstractmethod
  def statistics(self) -> dict[str, Statistic]:
    """A dictionary of Statistic objects required for this metric."""
    ...

  def values_from_mean_statistics(
      self,
      statistic_values: dict[str, dict[str, cx.Field]],
  ) -> dict[str, cx.Field]:
    """Computes final metric values from averaged statistics.

    Args:
      statistic_values: A dictionary where keys are unique statistic names (from
        Statistic.unique_name) and values are dictionaries mapping statistics
        variable names to their averaged statistic data.

    Returns:
      A dictionary mapping metric names to the computed final metric values.
    """
    statistic_values = {
        k: statistic_values[v.unique_name] for k, v in self.statistics.items()
    }
    return self._values_from_mean_statistics_with_internal_names(
        statistic_values
    )

  @abc.abstractmethod
  def _values_from_mean_statistics_with_internal_names(
      self,
      statistic_values: dict[str, dict[str, cx.Field]],
  ) -> dict[str, cx.Field]:
    """Computes metric from statistics using internal names."""
    ...


@dataclasses.dataclass
class PerVariableMetric(Metric):
  """Base class for a metric computed independently per variable."""

  @abc.abstractmethod
  def _values_from_mean_statistics_per_variable(
      self, statistic_values: dict[str, cx.Field]
  ) -> cx.Field:
    """Computes the metric for a single variable from its mean statistics.

    Args:
      statistic_values: A dictionary where keys are internal statistic names
        (keys from self.statistics) and values are the averaged statistic data
        for a single variable.

    Returns:
      The computed metric value for the variable.
    """
    ...

  def _values_from_mean_statistics_with_internal_names(
      self,
      statistic_values: dict[str, dict[str, cx.Field]],
  ) -> dict[str, cx.Field]:
    """Computes metric from statistics using internal names."""
    common_variables = set.intersection(
        *[set(statistic_values[k]) for k in self.statistics]
    )
    values = {}
    for v in common_variables:  # Compute values for all common variables.
      stats_per_var = {k: statistic_values[k][v] for k in self.statistics}
      values[v] = self._values_from_mean_statistics_per_variable(stats_per_var)
    return values


@dataclasses.dataclass
class MultiMetric(Metric):
  """A metric that combines multiple metrics.

  This allows sharing statistics computation when multiple metrics are evaluated
  together using a single Evaluator.
  """

  terms: dict[str, Metric]

  @property
  def statistics(self) -> dict[str, Statistic]:
    """Union of statistics from all term metrics."""
    all_stats = {}
    for metric in self.terms.values():
      all_stats.update(metric.statistics)
    return all_stats

  def _values_from_mean_statistics_with_internal_names(
      self,
      statistic_values: dict[str, dict[str, cx.Field]],
  ) -> dict[str, cx.Field]:
    """Computes values for each term and returns a flattened dictionary."""
    all_metric_values = {}
    for term_name, metric_term in sorted(self.terms.items()):
      term_statistic_values = {
          k: statistic_values[k]
          for k in metric_term.statistics
          if k in statistic_values
      }
      # pylint: disable=protected-access
      term_values = (
          metric_term._values_from_mean_statistics_with_internal_names(
              term_statistic_values
          )
      )
      # pylint: enable=protected-access
      for metric_name, value in sorted(term_values.items()):
        all_metric_values[f'{term_name}.{metric_name}'] = value
    return all_metric_values


@dataclasses.dataclass
class Loss(Metric, abc.ABC):
  """Abstract base class for a loss function."""

  @abc.abstractmethod
  def total(
      self,
      metric_values: dict[str, cx.Field],
  ) -> cx.Field:
    """Computes the total scalar loss from metric values."""
    ...

  def debug_terms(
      self,
      mean_stats: dict[str, dict[str, cx.Field]],
      metric_values: dict[str, cx.Field],
  ) -> dict[str, cx.Field]:
    """Returns debug terms for the loss, typically including relative losses."""
    del mean_stats, metric_values  # unused.
    return {}


@dataclasses.dataclass
class PerVariableLoss(Loss, PerVariableMetric):
  """A loss computed independently per variable and then aggregated."""

  variable_weights: dict[str, float | cx.Field] | None = None

  def total(
      self,
      metric_values: dict[str, cx.Field],
  ) -> cx.Field:
    """Sums up the per-variable losses with optional weights."""
    if not metric_values:
      return cx.wrap(0.0)

    total_loss = 0.0
    ws = collections.defaultdict(lambda: 1.0) | (self.variable_weights or {})
    for var_name, value in sorted(metric_values.items()):
      total_loss += ws[var_name] * value
    return total_loss

  def debug_terms(
      self,
      mean_stats: dict[str, dict[str, cx.Field]],
      metric_values: dict[str, cx.Field],
  ) -> dict[str, cx.Field]:
    """Returns debug terms. Defaults to metric_values and relative losses."""
    del mean_stats  # unused.
    total = self.total(metric_values)
    ws = collections.defaultdict(lambda: 1.0) | (self.variable_weights or {})
    relative_metric_values = {
        'relative_' + k: (ws[k] * v) / total for k, v in metric_values.items()
    }
    return relative_metric_values


@dataclasses.dataclass
class SumLoss(Loss):
  """A loss that is a weighted sum of other losses."""

  terms: dict[str, Loss]
  term_weights: dict[str, float] | None = None

  @property
  def statistics(self) -> dict[str, Statistic]:
    """Union of statistics from all term losses."""
    all_stats = {}
    for loss in self.terms.values():
      all_stats.update(loss.statistics)
    return all_stats

  def _values_from_mean_statistics_with_internal_names(
      self,
      statistic_values: dict[str, dict[str, cx.Field]],
  ) -> dict[str, cx.Field]:
    """Computes values for each term and returns a flattened dictionary."""
    all_metric_values = {}
    for term_name, loss_term in sorted(self.terms.items()):
      # Get the stats required by the loss term, using its internal names.
      term_statistic_values = {
          k: statistic_values[k] for k in loss_term.statistics
      }
      # pylint: disable=protected-access
      term_values = loss_term._values_from_mean_statistics_with_internal_names(
          term_statistic_values
      )
      # pylint: enable=protected-access
      for var_name, value in term_values.items():
        all_metric_values[f'{term_name}.{var_name}'] = value
    return all_metric_values

  def total(
      self,
      metric_values: dict[str, cx.Field],
  ) -> cx.Field:
    """Computes total loss by summing weighted term totals."""
    term_totals = {}
    for term_name, loss_term in sorted(self.terms.items()):
      term_metric_values = {
          k.split('.', 1)[1]: v
          for k, v in metric_values.items()
          if k.startswith(term_name + '.')
      }
      term_totals[term_name] = loss_term.total(term_metric_values)

    final_loss = 0.0
    ws = collections.defaultdict(lambda: 1.0) | (self.term_weights or {})
    for term_name, term_total_loss in sorted(term_totals.items()):
      final_loss += ws[term_name] * term_total_loss
    return final_loss

  def debug_terms(
      self,
      mean_stats: dict[str, dict[str, cx.Field]],
      metric_values: dict[str, cx.Field],
  ) -> dict[str, cx.Field]:
    """Return total loss for each term and their individual debug terms."""
    all_debug_terms = {}
    total = self.total(metric_values)
    ws = collections.defaultdict(lambda: 1.0) | (self.term_weights or {})
    for term_name, loss_term in sorted(self.terms.items()):
      term_metric_values = {
          k.split('.', 1)[1]: v
          for k, v in metric_values.items()
          if k.startswith(term_name + '.')
      }
      term_total = loss_term.total(term_metric_values)
      all_debug_terms[f'{term_name}_total'] = term_total
      all_debug_terms[f'relative_{term_name}_total'] = (
          ws[term_name] * term_total / total
      )

      # Add debug terms from each of the loss terms with term_name prefix.
      term_mean_stats = {
          stat.unique_name: mean_stats[stat.unique_name]
          for stat in loss_term.statistics.values()
          if stat.unique_name in mean_stats
      }
      sub_debug_terms = loss_term.debug_terms(
          term_mean_stats, term_metric_values
      )
      for k, v in sorted(sub_debug_terms.items()):
        all_debug_terms[f'{term_name}.{k}'] = v
    return all_debug_terms
