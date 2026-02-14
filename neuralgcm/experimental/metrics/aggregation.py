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

"""Defines Aggregator and AggregationState  and Metric classes operating on cx.Field."""

from __future__ import annotations

import collections
import dataclasses
import functools
import operator
from typing import Sequence, overload

import coordax as cx
import jax
import jax.numpy as jnp
from neuralgcm.experimental.metrics import base
from neuralgcm.experimental.metrics import binning
from neuralgcm.experimental.metrics import scaling
from neuralgcm.experimental.metrics import weighting
import numpy as np


# pytype: disable=invalid-annotation


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=['sum_weighted_statistics', 'sum_weights'],
    meta_fields=[],
)
@dataclasses.dataclass
class AggregationState:
  """An object that contains sum of weighted statistics and sum of weights.

  Allows for aggregation over multiple batches/chunks, e.g. during online evals
  or in a Beam pipeline.

  Attributes:
    sum_weighted_statistics: Structure containing summed/aggregated statistics,
      as a nested dictionary: {statistic_name: {term_name: cx.Field}}.
    sum_weights: Structure containing the corresponding summed weights, with the
      same nested structure as `sum_weighted_statistics`.
  """

  # statistics_name -> dict[statistic_term_name -> F]
  sum_weighted_statistics: dict[str, dict[str, cx.Field]]
  sum_weights: dict[str, dict[str, cx.Field]]

  @classmethod
  def empty(cls) -> AggregationState:
    """An initial/'zero' aggregation state (empty dicts)."""
    return cls(sum_weighted_statistics={}, sum_weights={})

  @jax.jit
  def __add__(self, other: AggregationState) -> AggregationState:
    # Weight and weighted stats are aggregated using `Field.sum` method, which
    # by default contains all named dimensions from all input fields. To avoid
    # accidental broadcasting during aggregation, we explicitly check that all
    # aggregation states have the same coordinates/dimensions. This implies that
    # aggregation states can only be summed when they represent uniform chunks
    # of statistics. This is different from what is done in WeatherBenchX, where
    # stats form different AggregationStates can be aligned and concatenated
    # along shared dimensions. This functionality could be added here in the
    # future by pre-padding stats with zeros or separate disjoint chunks in
    # different AggregationStates.
    self_is_empty = not self.sum_weighted_statistics
    other_is_empty = not other.sum_weighted_statistics
    if self_is_empty:
      return other
    if other_is_empty:
      return self
    self_coords = jax.tree.map(cx.get_coordinate, self, is_leaf=cx.is_field)
    other_coords = jax.tree.map(cx.get_coordinate, other, is_leaf=cx.is_field)
    if self_coords != other_coords:
      raise ValueError(
          'Aggregation states must represent uniform chunks of statistics, '
          f'but have different coordinates: {self_coords} vs {other_coords}'
      )
    tree_add = functools.partial(jax.tree.map, operator.add)
    return AggregationState(
        tree_add(self.sum_weighted_statistics, other.sum_weighted_statistics),
        tree_add(self.sum_weights, other.sum_weights),
    )

  @classmethod
  def sum(
      cls, aggregation_states: Sequence[AggregationState]
  ) -> AggregationState:
    """Sums sequence of aggregation states."""
    return sum(aggregation_states, start=cls.empty())

  def mean_statistics(self) -> dict[str, dict[str, cx.Field]]:
    """Returns the statistics normalized by their corresponding weights."""
    if not self.sum_weighted_statistics or not self.sum_weights:
      assert (not self.sum_weighted_statistics) and (
          not self.sum_weighted_statistics
      )
      return {}

    return jax.tree.map(
        lambda x, w: x / w,
        self.sum_weighted_statistics,
        self.sum_weights,
        is_leaf=cx.is_field,
    )

  @overload
  def metric_values(self, metrics: base.Metric) -> dict[str, cx.Field]:
    ...

  @overload
  def metric_values(
      self, metrics: dict[str, base.Metric]
  ) -> dict[str, dict[str, cx.Field]]:
    ...

  def metric_values(
      self, metrics: dict[str, base.Metric] | base.Metric
  ) -> dict[str, dict[str, cx.Field]] | dict[str, cx.Field]:
    """Returns metrics computed from the normalized statistics."""
    mean_stats = self.mean_statistics()
    if isinstance(metrics, base.Metric):
      return metrics.values_from_mean_statistics(mean_stats)
    return {
        k: metric.values_from_mean_statistics(mean_stats)
        for k, metric in sorted(metrics.items())
    }


def _is_present(dim: cx.Coordinate | str, field: cx.Field) -> bool:
  """Returns True if dim is present in field's dims/axes, False otherwise."""
  if isinstance(dim, cx.Coordinate):
    return all(ax == field.axes[ax.dims[0]] for ax in dim.axes)
  else:
    return dim in field.dims


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=['context'],
    meta_fields=[
        'dims_to_reduce',
        'weight_by',
        'scale_by',
        'bin_by',
        'skip_missing',
        'skipna',
        'keep_weights_for_nans',
    ],
)
@dataclasses.dataclass
class Aggregator:
  """Defines aggregation process over dimensions of statistics.

  This class configures the process of computing an `AggregationState` from raw
  statistics. It specifies which dimensions to reduce over, and what weighting,
  scaling or binning to apply before the reduction. Weighting, scaling and
  binning can be inferred either from the coordinates of the processed
  statistics or from the `context` values on the Aggregator, that holds
  coordinate values used to aggregate statistics over dimensions one slice at a
  time.

  A common use case for the `context` is performing aggregation over data slices
  (e.g. 'timedelta') one at a time. `context` plays a role of passing the
  coordinate values to the weighting, scaling and binning functions, since those
  are not available for individual slices.

  Attributes:
    dims_to_reduce: Sequence of coordinates or dimension names to reduce over.
    weight_by: Sequence of `weighting.Weighting` instances to apply.
    scale_by: Optional sequence of `scaling.Scaler` instances to apply.
    bin_by: Optional sequence of `binning.Binning` instances to apply.
    skip_missing: If True, `dims_to_reduce` that are not present in a given
      field will be skipped. If False, a `ValueError` is raised.
    skipna: If True, NaNs will be omitted in the aggregation.
    keep_weights_for_nans: If True, weights corresponding to NaN values are
      retained. Such behavior might be desired when computing the loss, where a
      NaN value is slipped without increasing the relative weight of the other
      entries in the statistics. Has effective only when `skipna` is True.
    context: Optional dictionary of context fields, used for coordinate
      dependent weighting, binning and scaling. Keys must include all dimensions
      expected by the weighting, binning and scaling instances that are not
      included on the statistis `Field`s themselves.
  """

  # TODO(dkochkov): Consider introducing a Protocol for added flexibility.
  # TODO(dkochkov): Add support for masking and nan handling.

  dims_to_reduce: Sequence[cx.Coordinate | str]
  weight_by: Sequence[weighting.Weighting] = tuple()
  scale_by: Sequence[scaling.ScaleFactor] | None = None
  bin_by: Sequence[binning.Binning] | None = None
  skip_missing: bool = True
  skipna: bool = False
  keep_weights_for_nans: bool = False
  context: dict[str, cx.Field] | None = None

  def __post_init__(self):
    self.dims_to_reduce = tuple(self.dims_to_reduce)
    self.weight_by = tuple(self.weight_by)
    if self.scale_by is not None:
      self.scale_by = tuple(self.scale_by)
    if self.bin_by is not None:
      self.bin_by = tuple(self.bin_by)

  def with_context(self, context: dict[str, cx.Field]) -> Aggregator:
    """Returns a copy of the aggregator with context set."""
    return dataclasses.replace(self, context=context)

  def aggregation_fn(
      self,
      stat_field: cx.Field,
      field_name: str,
      apply_scales: bool,
  ) -> cx.Field:
    """Applies configured reductions, (optional) weightings, and binnings."""
    weights = cx.field(1)
    for weighting_instance in self.weight_by:
      weights *= weighting_instance.weights(
          stat_field, field_name=field_name, context=self.context
      )

    if self.bin_by:
      bin_mask = cx.field(1)
      for binner in self.bin_by:
        bin_mask *= binner.create_bin_mask(
            stat_field, field_name=field_name, context=self.context
        )
      weights *= bin_mask

    untags = [d for d in self.dims_to_reduce if _is_present(d, stat_field)]
    if not self.skip_missing and set(untags) != set(self.dims_to_reduce):
      missing_dims = set(self.dims_to_reduce) - set(untags)
      raise ValueError(f'skip_missing is False but have a {missing_dims=}')

    if apply_scales and self.scale_by is not None:
      for scaler in self.scale_by:
        stat_field *= scaler.scales(
            stat_field, field_name=field_name, context=self.context
        )

    # TODO(dkochkov): Consider using `jnp.dot` + `jnp.sum` here for efficiency.
    sum_positional = cx.cmap(jnp.sum)
    return sum_positional((stat_field * weights).untag(*untags))

  def aggregate_statistics(
      self,
      statistics: dict[str, dict[str, cx.Field]],
  ) -> AggregationState:
    """Aggregate `statistics` with configured weightings and binnings."""
    sum_weighted_stats_result = collections.defaultdict(dict)
    sum_weights_result = collections.defaultdict(dict)
    for stat_name, statistic_values in sorted(statistics.items()):
      for term_name, stat_field in sorted(statistic_values.items()):
        # TODO(dkochkov): Could weights averaging be done more efficiently by
        # exposing the outer product structure?
        weight_field = cx.field(
            np.ones(stat_field.shape), stat_field.coordinate
        )
        if self.skipna:

          def _apply_nan_mask(x, nan_mask):
            return jnp.where(nan_mask, 0.0, x)

          # TODO(dkochkov): Consider requiring explicit nan mask.
          nan_mask = cx.cmap(jnp.isnan)(stat_field)
          stat_field = cx.cmap(_apply_nan_mask)(stat_field, nan_mask)
          if not self.keep_weights_for_nans:
            weight_field = cx.cmap(_apply_nan_mask)(weight_field, nan_mask)

        sum_weighted_stats_result[stat_name][term_name] = self.aggregation_fn(
            stat_field, field_name=term_name, apply_scales=True
        )
        sum_weights_result[stat_name][term_name] = self.aggregation_fn(
            weight_field, field_name=term_name, apply_scales=False
        )
    return AggregationState(
        dict(sum_weighted_stats_result), dict(sum_weights_result)
    )

  def zeros_aggregation_state(
      self,
      metric: base.Metric,
      predictions: dict[str, cx.Field],
      targets: dict[str, cx.Field],
  ) -> AggregationState:
    """Constructs zero aggregation state for metric without computation.

    This is useful for initializing an `AggregationState` with the correct
    structure (including shapes and coordinates) before accumulating actual
    statistics. It uses `jax.eval_shape` under the hood to avoid actually
    computing the statistics and their aggregation.

    Args:
      metric: The metric for which to create the zero state.
      predictions: Predictions struct used to infer the shapes and
        coordinates of the aggregated statistics. Can be obtained from actual
        values using `pytree_utils.shape_structure(predictions)` or constructed
        from coordinates using `cx.shape_struct_field`.
      targets: Targets used to infer the shapes and
        coordinates of the aggregated statistics. Can be obtained from actual
        values using `pytree_utils.shape_structure(targets)` or constructed from
        coordinates using `cx.shape_struct_field`.

    Returns:
      An `AggregationState` with the same structure as would be produced by
      aggregating the given metric's statistics, but with all numerical values
      initialized to zeros.
    """

    def _get_dummy_aggregation_state():
      statistics = {
          s.unique_name: s.compute(predictions, targets)
          for s in metric.statistics.values()
      }
      return self.aggregate_statistics(statistics)

    dummy_aggregation_state = jax.eval_shape(_get_dummy_aggregation_state)
    return jax.tree.map(jnp.zeros_like, dummy_aggregation_state)


def split_aggregation_state_for_metrics(
    metrics: dict[str, base.Metric],
    aggregation_state: AggregationState,
) -> dict[str, AggregationState]:
  """Splits aggregation state for unique statistics to per-metric aggregations.

  Args:
    metrics: A dictionary mapping metric names to `Metric` instances.
    aggregation_state: An `AggregationState` containing aggregated statistics
      indexed by their unique names. Can be computed by aggregating outputs of
      `base.compute_unique_statistics_for_all_metrics(metrics, ...)`.

  Returns:
    A dictionary mapping metric names to `AggregationState` instances, where
    each `AggregationState` contains only the statistics required by the
    corresponding metric.
  """
  aggregation_state_per_metric = {}
  for metric_name, metric in metrics.items():
    metric_stats = {s.unique_name for s in metric.statistics.values()}
    # Missing statistics is allowed at aggregation level, since it might be
    # filled in at a later stage. If not, the error would be raised when metric
    # values are being computed.
    metric_sum_weighted_statistics = {
        k: aggregation_state.sum_weighted_statistics[k] for k in metric_stats
        if k in aggregation_state.sum_weighted_statistics
    }
    metric_sum_weights = {
        k: aggregation_state.sum_weights[k] for k in metric_stats
        if k in aggregation_state.sum_weights
    }
    aggregation_state_per_metric[metric_name] = AggregationState(
        sum_weighted_statistics=metric_sum_weighted_statistics,
        sum_weights=metric_sum_weights,
    )
  return aggregation_state_per_metric
