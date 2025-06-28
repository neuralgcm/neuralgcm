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

  def _evaluate_single(
      self,
      metric: base.Metric,
      aggregator: aggregation.Aggregator,
      getter: transforms.Transform | None,
      predictions: dict[str, cx.Field],
      targets: dict[str, cx.Field],
  ) -> aggregation.AggregationState:
    """Evaluates statistics and aggregates them, returning a dict of states."""
    if getter:
      predictions = getter(predictions)
      targets = getter(targets)
    raw_stats = {
        stat.unique_name: stat.compute(predictions, targets)
        for stat in metric.statistics.values()
    }
    return aggregator.aggregate_statistics(raw_stats)

  def evaluate(
      self,
      predictions: dict[str, cx.Field],
      targets: dict[str, cx.Field],
  ) -> dict[str, aggregation.AggregationState]:
    """Evaluates statistics and aggregates them, returning a dict of states."""
    getters = (
        self.getters
        if isinstance(self.getters, dict)
        else collections.defaultdict(lambda: self.getters)
    )
    aggregators = (
        self.aggregators
        if isinstance(self.aggregators, dict)
        else collections.defaultdict(lambda: self.aggregators)
    )  # defaultdict effectively implements sharing of single getter/aggregator.
    agg_states = {}
    for k, metric in sorted(self.metrics.items()):
      agg_states[k] = self._evaluate_single(
          metric, aggregators[k], getters[k], predictions, targets
      )
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
