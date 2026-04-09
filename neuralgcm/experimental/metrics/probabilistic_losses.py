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

"""Defines probabilistic losses."""

from __future__ import annotations
import collections
import dataclasses
import coordax as cx
from neuralgcm.experimental.metrics import base
from neuralgcm.experimental.metrics import probabilistic_metrics


@dataclasses.dataclass
class CRPS(probabilistic_metrics.CRPS, base.PerVariableLoss):
  """CRPS loss."""

  def debug_terms(
      self,
      mean_stats: dict[str, dict[str, cx.Field]],
      metric_values: dict[str, cx.Field],
  ) -> dict[str, cx.Field]:
    """Returns debug terms. Defaults to metric_values and relative losses."""
    total = self.total(metric_values)
    mean_stats = {
        k: mean_stats[v.unique_name] for k, v in self.statistics.items()
    }
    skills = mean_stats['skill']
    spreads = mean_stats['spread']
    weighted_spreads = {
        k: self.spread_term_weight * v for k, v in spreads.items()
    }
    ws = collections.defaultdict(lambda: 1.0) | (self.variable_weights or {})
    relative_crps = {
        f'relative_crps_{k}': ws[k] * (skills[k] - weighted_spreads[k]) / total
        for k in skills
    }
    skill_spread_ratio = {
        f'skill_spread_ratio_{k}': skills[k] / spreads[k]
        for k in skills
    }
    return relative_crps | skill_spread_ratio
