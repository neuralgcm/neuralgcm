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
from typing import Sequence

import jax
import jax.numpy as jnp
from neuralgcm.experimental import coordax as cx
from neuralgcm.experimental.metrics import base
from neuralgcm.experimental.metrics import binning
from neuralgcm.experimental.metrics import weighting
import numpy as np


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
  def zero(cls) -> AggregationState:
    """An initial/'zero' aggregation state (empty dicts)."""
    return cls(sum_weighted_statistics={}, sum_weights={})

  def __add__(self, other: AggregationState) -> AggregationState:
    return self.sum([self, other])

  @classmethod
  def sum(
      cls, aggregation_states: Sequence[AggregationState]
  ) -> AggregationState:
    """Sums sequence of aggregation states."""
    if not aggregation_states:
      return cls.zero()

    all_weighted_stats = [x.sum_weighted_statistics for x in aggregation_states]
    all_weights = [x.sum_weights for x in aggregation_states]
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

    def _check_aligned_coords(inputs: Sequence[dict[str, dict[str, cx.Field]]]):
      input_coords = [
          jax.tree.map(cx.get_coordinate, nested, is_leaf=cx.is_field)
          for nested in inputs
      ]
      # Transpose list of nested dicts to a nested dict of lists of coordinates.
      coords_by_term = jax.tree.map(lambda *x: list(x), *input_coords)
      unique_coord_check = lambda *coords: len(set(coords)) == 1
      is_list = lambda x: isinstance(x, list)
      unique_coords = jax.tree.map(
          unique_coord_check, coords_by_term, is_leaf=is_list
      )
      if not all(jax.tree.leaves(unique_coords)):
        raise ValueError(
            'Aggregation states must represent uniform chunks of statistics, '
            f'but have different coordinates: {coords_by_term}'
        )

    _check_aligned_coords(all_weighted_stats)
    _check_aligned_coords(all_weights)
    sum_weighted_stats = jax.tree.map(
        lambda *field_leaves: sum(field_leaves),  # assume that all entries are
        *all_weighted_stats,
        is_leaf=cx.is_field,
    )
    sum_weights = jax.tree.map(
        lambda *field_leaves: sum(field_leaves),
        *all_weights,
        is_leaf=cx.is_field,
    )
    return cls(sum_weighted_stats, sum_weights)

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

  def metric_values(
      self, metrics: dict[str, base.Metric] | base.Metric
  ) -> dict[str, cx.Field]:
    """Returns metrics computed from the normalized statistics."""
    mean_stats = self.mean_statistics()
    if isinstance(metrics, base.Metric):
      return metrics.values_from_mean_statistics(mean_stats)
    # If multiple metrics is provided, we prefix value keys with metric_name.
    metric_values_nested = {
        k: metric.values_from_mean_statistics(mean_stats)
        for k, metric in metrics.items()
    }
    flat_metric_values: dict[str, cx.Field] = {}
    for metric_name, var_metrics in metric_values_nested.items():
      for var_name, field_value in var_metrics.items():
        flat_metric_values[f'{metric_name}.{var_name}'] = field_value
    return flat_metric_values


def _is_present(dim: cx.Coordinate | str, field: cx.Field) -> bool:
  """Returns True if dim is present in field's dims/axes, False otherwise."""
  if isinstance(dim, cx.Coordinate):
    return all(ax == field.axes[ax.dims[0]] for ax in dim.axes)
  else:
    return dim in field.dims


@dataclasses.dataclass
class Aggregator:
  """Defines aggregation process over dimensions of statistics.

  This class configures the process of computing an `AggregationState` from raw
  statistics. It specifies which dimensions to reduce over, and what weighting
  or binning to apply before the reduction.

  Attributes:
    dims_to_reduce: Sequence of coordinates or dimension names to reduce over.
    weight_by: Sequence of `weighting.Weighting` instances to apply.
    bin_by: Optional sequence of `binning.Binning` instances to apply.
    skip_missing: If True, `dims_to_reduce` that are not present in a given
      field will be skipped. If False, a `ValueError` is raised.
    skipna: If True, NaNs will be omitted in the aggregation.
  """

  # TODO(dkochkov): Consider introducing a Protocol for added flexibility.
  # TODO(dkochkov): Add support for masking and nan handling.

  dims_to_reduce: Sequence[cx.Coordinate | str]
  weight_by: Sequence[weighting.Weighting]
  bin_by: Sequence[binning.Binning] | None = None
  skip_missing: bool = True
  skipna: bool = False

  def aggregation_fn(
      self,
      stat_field: cx.Field,
      field_name: str,
  ) -> cx.Field:
    """Applies configured reductions, (optional) weightings, and binnings."""
    weights = cx.wrap(1)
    for weighting_instance in self.weight_by:
      weights *= weighting_instance.weights(stat_field, field_name=field_name)

    if self.bin_by:
      bin_mask = cx.wrap(1)
      for binner in self.bin_by:
        bin_mask *= binner.create_bin_mask(stat_field, field_name=field_name)
      weights *= bin_mask

    untags = [d for d in self.dims_to_reduce if _is_present(d, stat_field)]
    if not self.skip_missing and set(untags) != set(self.dims_to_reduce):
      missing_dims = set(self.dims_to_reduce) - set(untags)
      raise ValueError(f'skip_missing is False but have a {missing_dims=}')

    sum_positional = cx.cmap(jnp.sum)
    return sum_positional((stat_field * weights).untag(*untags))

  def aggregate_statistics(
      self,
      statistics: dict[str, dict[str, cx.Field]],
  ) -> AggregationState:
    """Aggregate `statistics` with configured weightings and binnings."""
    sum_weighted_stats_result = collections.defaultdict(dict)
    sum_weights_result = collections.defaultdict(dict)
    for stat_name, statistic_values in statistics.items():
      for term_name, stat_field in statistic_values.items():
        # TODO(dkochkov): Could weights averaging be done more efficiently by
        # exposing the outer product structure?
        weight_field = cx.wrap_like(np.ones(stat_field.shape), stat_field)
        if self.skipna:
          def _apply_nan_mask(x, nan_mask):
            return jnp.where(nan_mask, 0.0, x)

          nan_mask = cx.cmap(jnp.isnan)(stat_field)
          stat_field = cx.cmap(_apply_nan_mask)(stat_field, nan_mask)
          weight_field = cx.cmap(_apply_nan_mask)(weight_field, nan_mask)

        sum_weighted_stats_result[stat_name][term_name] = self.aggregation_fn(
            stat_field, field_name=term_name
        )
        sum_weights_result[stat_name][term_name] = self.aggregation_fn(
            weight_field, field_name=term_name
        )
    return AggregationState(
        dict(sum_weighted_stats_result), dict(sum_weights_result)
    )
