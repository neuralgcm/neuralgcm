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

"""Defines evaluators that simplify online metric and loss evaluations."""

from __future__ import annotations
import collections
import dataclasses
from typing import Generic, TypeVar
import coordax as cx
import jax
from neuralgcm.experimental.core import transforms
from neuralgcm.experimental.metrics import aggregation
from neuralgcm.experimental.metrics import base


M = TypeVar('M', bound=base.Metric)


@dataclasses.dataclass
class Evaluator(Generic[M]):
  """A class for parameterizing online evaluations."""

  metrics: dict[str, M]
  aggregators: dict[str, aggregation.Aggregator] | aggregation.Aggregator
  getters: dict[str, transforms.Transform] | transforms.Transform | None = None
  term_weights: dict[str, float] | None = None
  is_loss_evaluator: bool = dataclasses.field(init=False)

  def __post_init__(self):
    if all(isinstance(m, base.Loss) for m in self.metrics.values()):
      self.is_loss_evaluator = True
    else:
      self.is_loss_evaluator = False
      if self.term_weights is not None:
        raise TypeError(
            f'{self.term_weights=} can only be set when all metrics are Losses.'
        )

  def _get_getter(self, key: str) -> transforms.Transform | None:
    """Returns getter for a given metric key."""
    if isinstance(self.getters, dict):
      return self.getters[key]
    return self.getters

  def _get_aggregator(self, key: str) -> aggregation.Aggregator:
    """Returns aggregator for a given metric key."""
    if isinstance(self.aggregators, dict):
      return self.aggregators[key]
    return self.aggregators

  def _evaluate_group(
      self,
      metric_keys: list[str],
      getter: transforms.Transform | None,
      predictions: dict[str, cx.Field],
      targets: dict[str, cx.Field],
  ) -> dict[str, aggregation.AggregationState]:
    """Returns aggregation states for metrics that use the same getter."""
    if getter:
      predictions = getter(predictions)
      targets = getter(targets)
    metrics_group = {key: self.metrics[key] for key in metric_keys}
    unique_stats = base.compute_unique_statistics_for_all_metrics(
        metrics_group, predictions, targets
    )
    # TODO(dkochkov): Consider supporting direct aggregator comparison.
    aggregator_groups = collections.defaultdict(list)
    for key in metric_keys:
      aggregator_groups[id(self._get_aggregator(key))].append(key)
    id_to_aggregator = {
        id(self._get_aggregator(k)): self._get_aggregator(k)
        for k in metric_keys
    }
    agg_states = {}
    for aggregator_id, keys in aggregator_groups.items():
      aggregator = id_to_aggregator[aggregator_id]
      stats_for_group = {}
      for key in keys:
        for stat in self.metrics[key].statistics.values():
          stats_for_group[stat.unique_name] = unique_stats[stat.unique_name]
      group_agg_state = aggregator.aggregate_statistics(stats_for_group)
      group_metrics = {k: self.metrics[k] for k in keys}
      agg_states |= aggregation.split_aggregation_state_for_metrics(
          group_metrics, group_agg_state
      )
    return agg_states

  def evaluate(
      self,
      predictions: dict[str, cx.Field],
      targets: dict[str, cx.Field],
  ) -> dict[str, aggregation.AggregationState]:
    """Evaluates statistics and aggregates them, returning a dict of states."""
    getter_groups = collections.defaultdict(list)
    for key in self.metrics:
      getter = self._get_getter(key)
      getter_groups[id(getter)].append(key)

    id_to_getter = {
        id(self._get_getter(k)): self._get_getter(k) for k in self.metrics
    }
    agg_states = {}
    for getter_id, keys in getter_groups.items():
      getter = id_to_getter[getter_id]
      agg_states |= self._evaluate_group(keys, getter, predictions, targets)
    return agg_states

  def evaluate_total(
      self,
      predictions: dict[str, cx.Field],
      targets: dict[str, cx.Field],
      agg_states: dict[str, aggregation.AggregationState] | None = None,
  ) -> cx.Field:
    """Evaluates total loss, enabled only if metrics are Losses."""
    if not self.is_loss_evaluator:
      raise TypeError(
          'evaluate_total() can only be called when'
          f' {self.metrics.values()=} are all Losses.'
      )
    if agg_states is None:
      agg_states = self.evaluate(predictions, targets)
    total_loss = cx.wrap(0.0)
    for loss_key, loss in sorted(self.metrics.items()):
      assert isinstance(loss, base.Loss)  # make pytype happy.
      metric_values = agg_states[loss_key].metric_values(loss)
      term_total = loss.total(metric_values)
      weight = self.term_weights.get(loss_key, 1) if self.term_weights else 1
      total_loss += weight * term_total
    return total_loss

  def with_context(self, context: dict[str, cx.Field]) -> Evaluator:
    """Returns a copy of the evaluator with context set in aggregators."""
    if isinstance(self.aggregators, dict):
      new_aggregators = {
          k: agg.with_context(context) for k, agg in self.aggregators.items()
      }
    else:
      new_aggregators = self.aggregators.with_context(context)
    return dataclasses.replace(self, aggregators=new_aggregators)


jax.tree_util.register_dataclass(
    Evaluator,
    data_fields=['aggregators'],
    meta_fields=['metrics', 'getters', 'term_weights'],
    drop_fields=['is_loss_evaluator'],
)
