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
from typing import Any, Generic, TypeVar
import coordax as cx
import jax
from neuralgcm.experimental.core import typing
from neuralgcm.experimental.metrics import aggregation
from neuralgcm.experimental.metrics import base


M = TypeVar('M', bound=base.Metric)


@dataclasses.dataclass
class Evaluator(Generic[M]):
  """A class for parameterizing online evaluations."""

  metrics: dict[str, M]
  aggregators: dict[str, aggregation.Aggregator] | aggregation.Aggregator
  getters: dict[str, typing.Transform] | typing.Transform | None = None
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

  def _get_getter(self, key: str) -> typing.Transform | None:
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
      getter: typing.Transform | None,
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
      # process aggregation for a group of metrics.
      group_agg_state = aggregator.aggregate_statistics(stats_for_group)
      group_metrics = {k: self.metrics[k] for k in keys}
      agg_states |= aggregation.split_aggregation_state_for_metrics(
          group_metrics, group_agg_state
      )  # results in agg_states indexed by metric keys.
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

  def evaluate_metrics(
      self,
      predictions: dict[str, cx.Field],
      targets: dict[str, cx.Field],
      agg_states: dict[str, aggregation.AggregationState] | None = None,
  ) -> dict[str, dict[str, cx.Field]]:
    """Evaluates metrics and returns their values."""
    if agg_states is None:
      agg_states = self.evaluate(predictions, targets)
    metric_values = {}
    for key, metric in self.metrics.items():
      metric_values[key] = agg_states[key].metric_values(metric)
    return metric_values

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
    total_loss = cx.field(0.0)
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

  def zeros_aggregation_states(
      self,
      predictions: dict[str, cx.Field],
      targets: dict[str, cx.Field],
  ) -> dict[str, aggregation.AggregationState]:
    """Returns zero AggregationStates for all metrics without computation."""

    def apply_getter(getter, x):
      if getter is None:
        return x
      return getter.output_shapes(x)

    return {
        k: (
            self._get_aggregator(k).zeros_aggregation_state(
                metric,
                apply_getter(self._get_getter(k), predictions),
                apply_getter(self._get_getter(k), targets),
            )
        )
        for k, metric in self.metrics.items()
    }


jax.tree_util.register_dataclass(
    Evaluator,
    data_fields=['aggregators'],
    meta_fields=['metrics', 'getters', 'term_weights'],
    drop_fields=['is_loss_evaluator'],
)


def _flatten_dict(data: dict[str, dict[str, cx.Field]]) -> dict[str, cx.Field]:
  """Flattens dict['dataset']['var'] to dict['dataset.var']."""
  flat_dict = {}
  for dataset_key, variables in data.items():
    for var_name, value in variables.items():
      flat_dict[f'{dataset_key}.{var_name}'] = value
  return flat_dict


@dataclasses.dataclass
class FlattenedEvaluator:
  """Evaluator wrapper that flattens nested inputs before evaluation.

  This evaluator takes nested prediction and target dictionaries of the form
  `{dataset_key: {variable_name: cx.Field}}` and flattens them into a single
  dictionary of the form `{dataset_key.variable_name: cx.Field}` before
  passing them to the wrapped `Evaluator`. This is useful when a single metric
  or loss function needs to operate on variables from different datasets.

  Attributes:
    evaluator: The `Evaluator` instance to apply to flattened inputs.
    is_loss_evaluator: True if the wrapped evaluator computes losses.
  """

  evaluator: Evaluator
  is_loss_evaluator: bool = dataclasses.field(init=False)

  def __post_init__(self):
    self.is_loss_evaluator = self.evaluator.is_loss_evaluator

  def evaluate(
      self,
      predictions: dict[str, dict[str, cx.Field]],
      targets: dict[str, dict[str, cx.Field]],
  ) -> dict[str, aggregation.AggregationState]:
    """Flattens inputs and evaluates metrics."""
    return self.evaluator.evaluate(
        _flatten_dict(predictions), _flatten_dict(targets)
    )

  def evaluate_metrics(
      self,
      predictions: dict[str, dict[str, cx.Field]],
      targets: dict[str, dict[str, cx.Field]],
      agg_states: dict[str, aggregation.AggregationState] | None = None,
  ) -> dict[str, dict[str, cx.Field]]:
    """Flattens inputs and evaluates metrics."""
    return self.evaluator.evaluate_metrics(
        _flatten_dict(predictions), _flatten_dict(targets), agg_states
    )

  def evaluate_total(
      self,
      predictions: dict[str, dict[str, cx.Field]],
      targets: dict[str, dict[str, cx.Field]],
      agg_states: dict[str, aggregation.AggregationState] | None = None,
  ) -> cx.Field:
    """Flattens inputs and evaluates total loss."""
    if not self.is_loss_evaluator:
      raise TypeError('evaluate_total() requires the evaluator to be a loss.')
    return self.evaluator.evaluate_total(
        _flatten_dict(predictions), _flatten_dict(targets), agg_states
    )

  def with_context(self, context: dict[str, cx.Field]) -> FlattenedEvaluator:
    """Returns a copy with context set in the wrapped evaluator."""
    return dataclasses.replace(
        self, evaluator=self.evaluator.with_context(context)
    )

  def zeros_aggregation_states(
      self,
      predictions: dict[str, dict[str, cx.Field]],
      targets: dict[str, dict[str, cx.Field]],
  ) -> dict[str, aggregation.AggregationState]:
    """Flattens inputs and returns zero aggregation states."""
    return self.evaluator.zeros_aggregation_states(
        _flatten_dict(predictions), _flatten_dict(targets)
    )


@dataclasses.dataclass
class NestedEvaluators:
  """An evaluator that applies different evaluators to nested inputs.

  This class holds a dictionary of `Evaluator` instances, where each evaluator
  corresponds to a `dataset_key` in the nested predictions and targets
  dictionaries (`{dataset_key: {variable_name: cx.Field}}`). Evaluation is
  performed independently for each dataset. If all contained evaluators are
  loss evaluators, `evaluate_total` can be used to compute a weighted sum of
  total losses from each dataset.

  Attributes:
    evaluators: A dictionary mapping dataset_keys to `Evaluator` instances. Can
      include `...` as a key to specify a default evaluator for dataset_keys not
      explicitly listed.
    evaluator_weights: Optional weights used to scale contributions of
      individual evaluators to the total loss. Keys must match those of
      `evaluators`. Missing keys defaults to the weight of 1.0.
    is_loss_evaluator: True if all wrapped evaluators compute losses.
    default_evaluator: The default evaluator specified with `...`, or None.
  """

  evaluators: dict[Any, Evaluator]
  evaluator_weights: dict[str, float] | None = None
  is_loss_evaluator: bool = dataclasses.field(init=False)
  default_evaluator: Evaluator | None = dataclasses.field(init=False)

  def __post_init__(self):
    self.evaluators = dict(self.evaluators)  # make mutable copy
    self.default_evaluator = self.evaluators.pop(..., None)
    all_evals = list(self.evaluators.values())
    if self.default_evaluator:
      all_evals.append(self.default_evaluator)

    self.is_loss_evaluator = all(ev.is_loss_evaluator for ev in all_evals)
    if not self.is_loss_evaluator and self.evaluator_weights is not None:
      raise TypeError(
          f'{self.evaluator_weights=} can only be set when all evaluators are'
          ' loss evaluators.'
      )
    if self.evaluator_weights is not None:
      if not set(self.evaluator_weights).issubset(set(self.evaluators)):
        raise ValueError(
            f'Keys in {self.evaluator_weights=} must be a subset of keys in'
            f' {self.evaluators=}.'
        )

  def evaluate(
      self,
      predictions: dict[str, dict[str, cx.Field]],
      targets: dict[str, dict[str, cx.Field]],
  ) -> dict[str, dict[str, aggregation.AggregationState]]:
    """Evaluates metrics for each dataset, returning nested aggregation states."""
    result = {}
    all_keys = set(predictions.keys()) & set(targets.keys())
    for key in all_keys:
      evaluator = self.evaluators.get(key, self.default_evaluator)
      if evaluator is None:
        raise ValueError(
            f"No evaluator found for key '{key}' and no default (...) was "
            'provided in NestedEvaluators.'
        )
      result[key] = evaluator.evaluate(predictions[key], targets[key])
    return result

  def evaluate_metrics(
      self,
      predictions: dict[str, dict[str, cx.Field]],
      targets: dict[str, dict[str, cx.Field]],
      agg_states: (
          dict[str, dict[str, aggregation.AggregationState]] | None
      ) = None,
  ) -> dict[str, dict[str, dict[str, cx.Field]]]:
    """Evaluates metrics for each dataset and returns nested metric values."""
    if agg_states is None:
      agg_states = self.evaluate(predictions, targets)
    metrics_results = {}
    for key, states in agg_states.items():
      evaluator = self.evaluators.get(key, self.default_evaluator)
      metrics_results[key] = evaluator.evaluate_metrics(  # pytype: disable=attribute-error
          predictions.get(key, {}), targets.get(key, {}), states
      )
    return metrics_results

  def evaluate_total(
      self,
      predictions: dict[str, dict[str, cx.Field]],
      targets: dict[str, dict[str, cx.Field]],
      agg_states: (
          dict[str, dict[str, aggregation.AggregationState]] | None
      ) = None,
  ) -> cx.Field:
    """Evaluates and sums total losses from all datasets."""
    if not self.is_loss_evaluator:
      raise TypeError('evaluate_total() requires all evaluators to be losses.')
    if agg_states is None:
      agg_states = self.evaluate(predictions, targets)
    total_loss = cx.field(0.0)
    weights = self.evaluator_weights or {}
    for key, states in sorted(agg_states.items()):
      evaluator = self.evaluators.get(key, self.default_evaluator)
      term_total = evaluator.evaluate_total(  # pytype: disable=attribute-error
          predictions.get(key, {}), targets.get(key, {}), states
      )
      total_loss += weights.get(key, 1.0) * term_total
    return total_loss

  def with_context(self, context: dict[str, cx.Field]) -> NestedEvaluators:
    """Returns a copy with context set in all nested evaluators."""
    new_evaluators = {
        k: ev.with_context(context) for k, ev in self.evaluators.items()
    }
    if self.default_evaluator:
      new_evaluators[...] = self.default_evaluator.with_context(context)
    return dataclasses.replace(
        self,
        evaluators=new_evaluators,
        evaluator_weights=self.evaluator_weights,
    )

  def zeros_aggregation_states(
      self,
      predictions: dict[str, dict[str, cx.Field]],
      targets: dict[str, dict[str, cx.Field]],
  ) -> dict[str, dict[str, aggregation.AggregationState]]:
    """Returns zero aggregation states for each dataset."""
    result = {}
    all_keys = set(predictions.keys()) & set(targets.keys())
    for key in all_keys:
      evaluator = self.evaluators.get(key, self.default_evaluator)
      if evaluator is None:
        raise ValueError(
            f"No evaluator found for key '{key}' and no default (...) was "
            'provided in NestedEvaluators.'
        )
      result[key] = evaluator.zeros_aggregation_states(
          predictions[key], targets[key]
      )
    return result


jax.tree_util.register_dataclass(
    FlattenedEvaluator,
    data_fields=['evaluator'],
    meta_fields=[],
    drop_fields=['is_loss_evaluator'],
)
jax.tree_util.register_dataclass(
    NestedEvaluators,
    data_fields=['evaluators'],
    meta_fields=['evaluator_weights'],
    drop_fields=['is_loss_evaluator'],
)
