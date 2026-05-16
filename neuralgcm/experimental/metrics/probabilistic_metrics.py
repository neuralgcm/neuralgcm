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

"""Defines probabilistic metrics."""

from __future__ import annotations
import dataclasses
import functools
import coordax as cx
import jax
import jax.numpy as jnp
from neuralgcm.experimental.metrics import base


def _validate_ens_dim(predictions: cx.Field, ens_dim: str) -> None:
  if ens_dim not in predictions.dims:
    raise ValueError(f'Prediction field must have an "{ens_dim}" axis.')


def _abs_beta(x: cx.Field, beta: float) -> cx.Field:
  if beta >= 1:
    abs_fn = cx.cmap(jnp.abs)
  else:
    abs_fn = cx.cmap(lambda x: base.safe_sqrt(x**2))
  return abs_fn(x) ** beta


def _flip_2_ens_spread(x: cx.Field, ens_dim: str, beta: float) -> cx.Field:
  """Computes spread for exactly 2 ensemble members using jnp.flip."""
  if x.named_shape[ens_dim] != 2:
    raise ValueError(
        '_flip_2_ens_spread requires exactly 2 ensemble members, got'
        f' {x.named_shape[ens_dim]}'
    )
  x_prime = cx.cmap(jnp.flip)(x.untag(ens_dim)).tag(ens_dim)
  err = x - x_prime
  return cx.cmap(jnp.mean)(_abs_beta(err, beta).untag(ens_dim))


def _sort_ens_spread(x: cx.Field, ens_dim: str, beta: float) -> cx.Field:
  """Computes unbiased expected pairwise difference using sorted members."""
  if beta != 1.0:
    raise ValueError(f'_sort_ens_spread only supports beta=1.0, got {beta}')
  n = x.named_shape[ens_dim]
  if n < 2:
    raise ValueError('_sort_ens_spread requires at least 2 ensemble members.')

  def fair_spread_fn(arr):
    arr_sorted = jnp.sort(arr)
    i = jnp.arange(1, n + 1, dtype=arr.dtype)
    weights = 2 * i - n - 1
    return jnp.sum(weights * arr_sorted) * 2.0 / (n * (n - 1))

  return cx.cmap(fair_spread_fn)(x.untag(ens_dim))


def _all_pairs_ens_spread(
    x: cx.Field, ens_dim: str, beta: float
) -> cx.Field:
  """Computes unbiased expected pairwise difference over rolling shifts."""
  n = x.named_shape[ens_dim]
  if n < 2:
    raise ValueError(
        '_all_pairs_ens_spread requires at least 2 ensemble members.'
    )
  roll_m = cx.cmap(lambda arr, m: jnp.roll(arr, m, axis=0))

  def shift_spread(shift):
    x_shifted = roll_m(x.untag(ens_dim), shift).tag(ens_dim)
    err = x - x_shifted
    return cx.cmap(jnp.mean)(_abs_beta(err, beta).untag(ens_dim))

  total_spread = sum(shift_spread(s) for s in range(1, n))
  return total_spread * (1.0 / (n - 1))


def _ring_chain_ens_spread(
    x: cx.Field, ens_dim: str, beta: float
) -> cx.Field:
  """Computes unbiased expected pairwise difference using ring chaining."""
  n = x.named_shape[ens_dim]
  if n < 2:
    raise ValueError(
        '_ring_chain_ens_spread requires at least 2 ensemble members.'
    )
  roll_once = cx.cmap(lambda arr: jnp.roll(arr, 1, axis=0))

  total_spread = None
  curr_x = x
  for _ in range(1, n):
    curr_x = roll_once(curr_x.untag(ens_dim)).tag(ens_dim)
    err = x - curr_x
    spread_s = cx.cmap(jnp.mean)(_abs_beta(err, beta).untag(ens_dim))
    if total_spread is None:
      total_spread = spread_s
    else:
      total_spread = total_spread + spread_s

  return total_spread * (1.0 / (n - 1))


@dataclasses.dataclass
class EnergySkill(base.PerVariableStatistic):
  """Statistic for the skill term of an energy-like score: E[|X - Y|^β].

  With X, X' two i.i.d. predictions and Y the target, the skill is defined as
  the expected error against the target. In a 2-member ensemble setting, this
  is computed as:
    Skill = 0.5 * (|X - Y|^β + |X' - Y|^β)

  When β=1, this is the skill term for the CRPS. For energy score based on this
  statistics to be strictly proper beta must be belong to `(0, 2)`.
  """

  ensemble_dim: str = 'ensemble'
  beta: float = 1.0

  @property
  def unique_name(self) -> str:
    return f'EnergySkill_{self.ensemble_dim}_beta_{self.beta}'

  def _compute_per_variable(
      self, predictions: cx.Field, targets: cx.Field
  ) -> cx.Field:
    """Computes E|prediction - target|^β over the ensemble axis."""
    _validate_ens_dim(predictions, self.ensemble_dim)
    targets = targets.broadcast_like(predictions)  # Needed?
    err = predictions - targets
    return cx.cmap(jnp.mean)(_abs_beta(err, self.beta).untag(self.ensemble_dim))


@dataclasses.dataclass
class EnergySpread(base.PerVariableStatistic):
  """Statistic for the spread term of an energy-like score: E[|X - X'|^β].

  With X, X' two i.i.d. predictions, the spread is defined as the expected
  difference between predictions.

  Attributes:
    ensemble_dim: Name of the ensemble dimension in predictions.
    beta: Power parameter for the energy spread calculation.
    broadcast_target_nans: If True, ensures computed spread becomes NaN at
      locations where target fields are NaN (masking unobserved targets).
    algorithm: Method used to compute the expected pairwise difference. Options:
      - 'auto': Dynamically selects optimal algorithm ('flip' for n=2; 'sort' on
        CPU and 'all_pairs' on TPU/GPU for n>2).
      - 'all_pairs': Evaluates pairwise differences via sequential shifts.
      - 'ring_chain': Similar to 'all_pairs', but uses sequential ring shifts.
      - 'flip': Evaluates difference with reversed array (for n=2 only).
      - 'sort': Evaluates difference using sorted predictions (beta=1.0 only).
  """

  ensemble_dim: str = 'ensemble'
  beta: float = 1.0
  broadcast_target_nans: bool = False
  algorithm: str = 'auto'

  def __post_init__(self):
    if self.algorithm == 'sort' and self.beta != 1.0:
      raise ValueError(
          f'sort algorithm only supports beta=1.0, got {self.beta}'
      )

  @property
  def unique_name(self) -> str:
    return f'EnergySpread_{self.ensemble_dim}_beta_{self.beta}'

  def _compute_per_variable(
      self, predictions: cx.Field, targets: cx.Field
  ) -> cx.Field:
    """Computes E|prediction - prediction'|^β over the ensemble axis."""
    ens_dim = self.ensemble_dim
    beta = self.beta
    _validate_ens_dim(predictions, ens_dim)

    if self.algorithm == 'flip':
      spread = _flip_2_ens_spread(predictions, ens_dim, beta)
    elif self.algorithm == 'sort':
      spread = _sort_ens_spread(predictions, ens_dim, beta)
    elif self.algorithm == 'all_pairs':
      spread = _all_pairs_ens_spread(predictions, ens_dim, beta)
    elif self.algorithm == 'ring_chain':
      spread = _ring_chain_ens_spread(predictions, ens_dim, beta)
    elif self.algorithm == 'auto':
      if predictions.named_shape[ens_dim] == 2:
        spread = _flip_2_ens_spread(predictions, ens_dim, beta)
      else:
        cpu_fn = functools.partial(
            _sort_ens_spread if beta == 1.0 else _all_pairs_ens_spread,
            ens_dim=ens_dim,
            beta=beta,
        )
        default_fn = functools.partial(
            _all_pairs_ens_spread, ens_dim=ens_dim, beta=beta
        )
        spread = jax.lax.platform_dependent(
            predictions, cpu=cpu_fn, default=default_fn
        )
    else:
      raise ValueError(f'Unknown algorithm: {self.algorithm}')

    if self.broadcast_target_nans:
      # Ensure spread is nan where targets are nan.
      spread = spread + 0.0 * targets.broadcast_like(spread)
    return spread


@dataclasses.dataclass
class CRPS(base.PerVariableMetric):
  """Continuously Ranked Probability Score.

  CRPS takes the form (with E being expectation over the ensemble):
    CRPS = E[|X - Y|] - spread_term_weight * E[|X - X'|]
  where X, X' are i.i.d. predictions and Y is the target. This corresponds to
  the energy-like score with β=1. It can be thought of as the sum of
  component-wise energy score losses.

  A naive implementation computes the total skill and spread as scalars and then
  subtracts them. However, this is unstable if Spread ≈ 2 * Skill. A more
  stable estimate is to compute the difference at each point first, and then
  aggregate:
    CRPS = Mean( E[|X-Y|] - spread_term_weight * E[|X-X'|] )
  The triangle inequality ensures the terms being averaged are non-negative
  when spread_term_weight=0.5.

  This implementation uses separate Statistics for skill and spread, but
  combines them before any spatial aggregation, thus using the stable method.

  Based on formula 21 in [1]; http://shortn/_Lyu0etEy1F

  References:
    [1]: Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules,
         prediction, and estimation. Journal of the American statistical
         Association, 102(477), 359-378.
  """

  ensemble_dim: str = 'ensemble'
  spread_term_weight: float = 0.5

  @property
  def statistics(self) -> dict[str, base.Statistic]:
    return {
        'skill': EnergySkill(ensemble_dim=self.ensemble_dim, beta=1.0),
        'spread': EnergySpread(self.ensemble_dim, 1.0, True),
    }

  def _values_from_mean_statistics_per_variable(
      self, statistic_values: dict[str, cx.Field]
  ) -> cx.Field:
    """Computes CRPS from skill and spread statistics."""
    # With X, X' two i.i.d. predictions,
    #   Skill  = E[|X-Y|^β]
    #   Spread = E[|X-X'|^β]
    # Then CRPS = Skill - spread_term_weight * Spread.
    # This subtraction is performed per-location, before spatial aggregation,
    # which is more numerically stable.
    return (
        statistic_values['skill']
        - self.spread_term_weight * statistic_values['spread']
    )


@dataclasses.dataclass
class EnsembleVariance(base.PerVariableStatistic):
  """Computes unbiased sample variance along the ensemble dimension."""

  ensemble_dim: str = 'ensemble'

  @property
  def unique_name(self) -> str:
    return f'EnsembleVariance_{self.ensemble_dim}'

  def _compute_per_variable(
      self, predictions: cx.Field, targets: cx.Field
  ) -> cx.Field:
    """Computes unbiased variance over the ensemble axis."""
    _validate_ens_dim(predictions, self.ensemble_dim)
    # ddof=1 for unbiased variance
    return cx.cmap(lambda x: jnp.var(x, ddof=1))(
        predictions.untag(self.ensemble_dim)
    )


@dataclasses.dataclass
class UnbiasedEnsembleMeanSquaredError(base.PerVariableStatistic):
  """Computes unbiased ensemble mean squared error."""

  ensemble_dim: str = 'ensemble'

  @property
  def unique_name(self) -> str:
    return f'UnbiasedEnsembleMeanSquaredError_{self.ensemble_dim}'

  def _compute_per_variable(
      self, predictions: cx.Field, targets: cx.Field
  ) -> cx.Field:
    """Computes unbiased ensemble mean squared error."""
    _validate_ens_dim(predictions, self.ensemble_dim)
    n = predictions.named_shape[self.ensemble_dim]
    predictions_mean = cx.cmap(jnp.mean)(predictions.untag(self.ensemble_dim))
    predictions_var = cx.cmap(lambda x: jnp.var(x, ddof=1))(
        predictions.untag(self.ensemble_dim)
    )
    biased_mse = (predictions_mean - targets) ** 2
    return biased_mse - predictions_var / n


@dataclasses.dataclass
class UnbiasedEnsembleMeanRMSE(base.PerVariableMetric):
  """Unbiased ensemble mean root mean squared error."""

  ensemble_dim: str = 'ensemble'

  @property
  def statistics(self) -> dict[str, base.Statistic]:
    return {'unbiased_mse': UnbiasedEnsembleMeanSquaredError(self.ensemble_dim)}

  def _values_from_mean_statistics_per_variable(
      self, statistic_values: dict[str, cx.Field]
  ) -> cx.Field:
    return cx.cmap(jnp.sqrt)(statistic_values['unbiased_mse'])


@dataclasses.dataclass
class UnbiasedSpreadSkillRatio(base.PerVariableMetric):
  """Unbiased spread-skill ratio."""

  ensemble_dim: str = 'ensemble'

  @property
  def statistics(self) -> dict[str, base.Statistic]:
    return {
        'variance': EnsembleVariance(self.ensemble_dim),
        'unbiased_mse': UnbiasedEnsembleMeanSquaredError(self.ensemble_dim),
    }

  def _values_from_mean_statistics_per_variable(
      self, statistic_values: dict[str, cx.Field]
  ) -> cx.Field:
    return cx.cmap(jnp.sqrt)(
        statistic_values['variance'] / statistic_values['unbiased_mse']
    )
