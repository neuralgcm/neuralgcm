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
import coordax as cx
import jax
import jax.numpy as jnp
from neuralgcm.experimental.core import typing
from neuralgcm.experimental.metrics import base


@jax.custom_jvp
def safe_sqrt(x: typing.Array) -> jax.Array:
  """Sqrt(x) with gradient = 0 for x near 0."""
  return jnp.sqrt(x)


@safe_sqrt.defjvp
def safe_sqrt_jvp(
    primals: typing.Array,
    tangents: typing.Array,
) -> tuple[jax.Array, jax.Array]:
  (x,) = primals
  (x_dot,) = tangents
  primal_out = safe_sqrt(x)
  eps = jnp.finfo(x.dtype).eps
  safe_x = jnp.where(x > eps, x, 1.0)
  tangent_out = jnp.where(x > eps, x_dot / (2 * safe_sqrt(safe_x)), 0)
  return primal_out, tangent_out


def abs_beta(x: cx.Field, beta: float) -> cx.Field:
  if beta >= 1:
    abs_fn = cx.cmap(jnp.abs)
  else:
    abs_fn = cx.cmap(lambda x: safe_sqrt(x**2))
  return abs_fn(x) ** beta


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
    if self.ensemble_dim not in predictions.dims:
      raise ValueError(
          f'Prediction field must have an "{self.ensemble_dim}" axis.'
      )
    targets = targets.broadcast_like(predictions)  # Needed?
    err = predictions - targets
    return cx.cmap(jnp.mean)(abs_beta(err, self.beta).untag(self.ensemble_dim))


@dataclasses.dataclass
class EnergySpread(base.PerVariableStatistic):
  """Statistic for the spread term of an energy-like score: E[|X - X'|^β].

  With X, X' two i.i.d. predictions, the spread is defined as the expected
  difference between predictions. In a 2-member ensemble setting, this is
  computed as:
    Spread = |X - X'|^β
  """

  ensemble_dim: str = 'ensemble'
  beta: float = 1.0
  broadcast_target_nans: bool = False

  @property
  def unique_name(self) -> str:
    return f'EnergySpread_{self.ensemble_dim}_beta_{self.beta}'

  def _compute_per_variable(
      self, predictions: cx.Field, targets: cx.Field
  ) -> cx.Field:
    """Computes E|prediction - prediction'|^β over the ensemble axis."""
    if self.ensemble_dim not in predictions.dims:
      raise ValueError(
          f'Prediction field must have an "{self.ensemble_dim}" axis.'
      )
    if predictions.named_shape[self.ensemble_dim] != 2:
      raise ValueError(
          'EnergySpread currently only supports 2-member ensembles.'
      )
    # For a 2-member ensemble, X' is just the other member.
    # We can get this by reversing the array along the ensemble axis.
    x = predictions
    ens_dim = self.ensemble_dim
    x_prime = cx.cmap(jnp.flip)(x.untag(ens_dim)).tag(ens_dim)
    err = x - x_prime
    # The ensemble mean of |X - X'|^β and |X' - X|^β is just |X - X'|^β.
    spread = cx.cmap(jnp.mean)(abs_beta(err, self.beta).untag(ens_dim))
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
