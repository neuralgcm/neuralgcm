# Copyright 2024 Google LLC
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
"""Stochastic losses for NeuralGCM."""
import abc
from typing import Callable, Optional, Sequence
from dinosaur import typing
import gin
import jax
import jax.numpy as jnp
import linear_transforms
import metrics_base
import metrics_util
from neuralgcm.legacy import model_utils
import numpy as np


Pytree = typing.Pytree
TrajectoryRepresentations = typing.TrajectoryRepresentations

AggregationTransformConstructor = metrics_util.AggregationTransformConstructor

tree_leaves = jax.tree_util.tree_leaves
tree_map = jax.tree_util.tree_map


def replicate(
    x: Pytree,
    axis_name: str = 'batch',
    times: Optional[int] = None,
) -> Pytree:
  """Replicated a pytree across devices."""
  if times is None:
    times = jax.local_device_count()

  def _replicate(_):
    return x

  return jax.pmap(_replicate, axis_name)(np.ones(times))


class EnergyLikeLoss(metrics_base.Loss, abc.ABC):
  """Energy-score like loss function.

  Both CRPS and EnergyScore take the form (with E expectation)
    E‖X - Y‖^β - ensemble_term_weight * E‖X - X'‖^β
  where for CRPS ‖⋅‖ is the L1 norm, and for EnergyScore it is the L2 norm.

  To create a general implementation, we decompose the norm as
    ‖Z‖ := _norm_reduction_fn(_norm_inner_fn(Z))

  For more see (21) and (22) in [1]; http://shortn/_Lyu0etEy1F

  References:
    [1]: Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules,
         prediction, and estimation. Journal of the American statistical
         Association, 102(477), 359-378.
  """

  def __init__(
      self,
      trajectory_spec: metrics_util.TrajectorySpec,
      components: Sequence[linear_transforms.LinearTransformConstructor],
      time_step: Optional[int | slice] = None,
      level: Optional[int] = None,
      getter: Callable[[Pytree], Pytree] = (
          metrics_util.filter_sim_time_and_diagnostics
      ),
      beta: float = 1.0,
      ensemble_term_weight: float = 0.5,
      is_nodal: bool = True,
      is_encoded: bool = False,
      coarsen_aggregation: AggregationTransformConstructor = (
          metrics_util.AggregateIdentity
      ),
      vector_norm_squared_aggregation: AggregationTransformConstructor = (
          metrics_util.AggregateIdentity
      ),
  ):
    """Constructs an instance of EnergyLikeLoss.

    Args:
      trajectory_spec: Specification of spatial and temporal trajectory sizes.
      components: Sequence of linear transformations to be applied to errors.
      time_step: Step or slice at which to compute loss, or None for all steps.
      level: Level to compute loss at, or None to use mean over all levels.
      getter: Function for extracting a sub-pytree on which errors are computed.
      beta: Power parameter of the loss. For energy score to be strictly proper
        beta must be belong to `(0, 2)`.
      ensemble_term_weight: Coefficient that specifcies how much weight is put
        on the terms that captures the spread of the 2-ensemble. For standard
        energy score this value should be set to `0.5`. It can be used to
        interpolate to other scoring rules that are not strictly proper. For
        example setting this value to `0.0` and setting `beta = 2.0` will result
        in a squared error loss.
      is_nodal: Indicator whether loss is computed in nodal space.
      is_encoded: Indicator whether loss is computed in encoded(model) space.
      coarsen_aggregation: Transform class that is used to aggregate errors
        before computing the loss elements. This enables defining losses on
        coarser representations that accentuate larger scale structure.
        Currently this argument should be used only by PatchEnergyLoss. Example
        coarsening operators include `RegriddingAggregation`, `TimeWindowSum`.
      vector_norm_squared_aggregation: Transform class that is used to aggregate
        components of the squared errors to form the distance for computing the
        energy score. Currently this argument should be used only by
        PatchEnergyLoss. Suitable aggregation methods include
        `RegriddingAggregation`, `TimeWindowSum`, `SumVariables`, which would
        correspond to vectors of (1) single level, time, variable, horizontal
        neighbors; (2) single level, variable, lon-lat, sequence of time values;
        (3) all variables at a single level, time, lon-lat.
    """
    self.coarsen_fn = coarsen_aggregation(
        trajectory_spec, is_nodal=is_nodal, is_encoded=is_encoded
    )
    self.vector_norm_squared_fn = vector_norm_squared_aggregation(
        self.coarsen_fn.out_trajectory_spec,
        is_nodal=is_nodal,
        is_encoded=is_encoded,
    )
    # parent class reductions are done on the final out_trajectory_spec.
    super().__init__(
        self.vector_norm_squared_fn.out_trajectory_spec,
        is_nodal=is_nodal,
        is_encoded=is_encoded,
    )
    self.components = components
    self.time_step = time_step
    self.level = level
    self.getter = getter
    # transform is applied to raw inputs which are aligned with trajectory_spec.
    self.transform = linear_transforms.ComposedTransformForLoss(
        trajectory_spec, self.components
    )
    self._beta = beta
    self._ensemble_term_weight = ensemble_term_weight

  def a_minus_cb(self, a: Pytree, c: float, b: Pytree) -> Pytree:
    """A - c * B."""
    return tree_map(lambda a_i, b_i: a_i - c * b_i, a, b)

  def ca_minus_b(self, c: float, a: Pytree, b: Pytree) -> Pytree:
    """c * A - B."""
    return tree_map(lambda a_i, b_i: c * a_i - b_i, a, b)

  def component_mean(self, tree: Pytree) -> jax.Array:
    """Mean over variable, time, pressure, lat, lon."""
    leaf_means = tree_leaves(self.mean_per_variable(tree))
    return sum(leaf_means) / len(leaf_means)

  def ensemble_mean(self, tree: Pytree) -> Pytree:
    return jax.lax.pmean(tree, 'ensemble')

  def _prepare(self, trajectory: TrajectoryRepresentations) -> Pytree:
    """Prepares target or predictions."""
    # Cannot consolidate with RMSE.prepare since this one
    # * does not take ensemble mean of trajectory.
    trajectory = metrics_util.extract_variable(
        trajectory,
        self.trajectory_spec,
        self.time_step,
        self.level,
        self.getter,
        self.is_nodal,
        self.is_encoded,
    )
    return trajectory

  def evaluate(
      self,
      prediction: TrajectoryRepresentations,
      target: TrajectoryRepresentations,
  ) -> Pytree:
    """Evaluates giving values of interest."""
    pv2ss = self._per_variable_spread_skill_errors(prediction, target)
    return self._spread_skill_and_loss(
        x_minus_y=pv2ss['x_minus_y'],
        x_minus_xprime=pv2ss['x_minus_xprime'],
    )['loss']

  def debug_loss_terms_instance(self) -> metrics_base.EvaluateFunctionWrapper:
    """Returns class that evaluates rel loss per variable and spread/skill."""

    def evaluate_fn(
        prediction: TrajectoryRepresentations,
        target: TrajectoryRepresentations,
    ) -> Pytree:
      # self.loss.evaluate takes ensemble mean (to evaluate on ensemble mean) if
      # needed.
      pv2ss = self._per_variable_spread_skill_errors(prediction, target)
      overall_spread_skill_loss = self._spread_skill_and_loss(
          x_minus_y=pv2ss['x_minus_y'],
          x_minus_xprime=pv2ss['x_minus_xprime'],
      )

      all_vars = pv2ss['x_minus_y'].keys()

      per_variable_terms = {
          var: self._spread_skill_and_loss(
              x_minus_y=pv2ss['x_minus_y'][var],
              x_minus_xprime=pv2ss['x_minus_xprime'][var],
          )
          for var in all_vars
      }
      # here we reduce terms by summation to expose relative contributions,
      # even though the actual total_loss might be different.
      per_variable_losses = {
          var: per_variable_terms[var]['loss'] for var in all_vars
      }
      sum_of_losses = sum(per_variable_losses.values())
      per_variable_relative_losses = tree_map(
          lambda x: x / sum_of_losses, per_variable_losses
      )
      return {
          'relative_loss': per_variable_relative_losses,
          'overall': overall_spread_skill_loss,
          'per_variable_spread': {
              var: per_variable_terms[var]['spread'] for var in all_vars
          },
          'per_variable_skill': {
              var: per_variable_terms[var]['skill'] for var in all_vars
          },
      }

    return metrics_base.EvaluateFunctionWrapper(evaluate_fn)

  def _per_variable_spread_skill_errors(
      self,
      prediction: TrajectoryRepresentations,
      target: TrajectoryRepresentations,
  ) -> Pytree:
    """Computes non-reduced loss terms (skill and spread) for each variable.

    Args:
      prediction: predicted 2-ensemble of trajectories with each component
        having shape [2, time_steps, vertical, lat_axis, lon_axis], with leading
        axis corresponding to different ensemble members and last two axes being
        either spherical harmonics numbers or lat, lon values.
      target: target trajectory replicated along the ensemble axis. The shape is
        expected to be exactly the same as `trajectory`.

    Returns:
      A dictionary with keys containing transformed variables.
        `x_minus_y` = prediction - target
        `x_minus_xprime` = difference of ensemble predictions
        `prediction` = prediction
    """
    ensemble_size = jax.lax.psum(1, 'ensemble')
    if ensemble_size != 2:
      raise ValueError(f'{ensemble_size=} is not 2')

    prediction = self.transform(self._prepare(prediction), target)
    target = self.transform(self._prepare(target), target)

    x_minus_y = tree_map(jnp.subtract, prediction, target)  # X_i - Y

    xprime = jax.lax.pshuffle(prediction, 'ensemble', (1, 0))
    x_minus_xprime = tree_map(jnp.subtract, prediction, xprime)  # X_i - X_j≠i

    return {
        'x_minus_y': x_minus_y,
        'x_minus_xprime': x_minus_xprime,
        'prediction': prediction,
    }

  @abc.abstractmethod
  def _spread_skill_and_loss(
      self,
      x_minus_y: Pytree,
      x_minus_xprime: Pytree,
  ) -> dict[str, jax.Array]:
    """Gets dictionary with 'spread', 'skill', and 'loss' entries."""


@gin.register(
    denylist=['coarsen_aggregation', 'vector_norm_squared_aggregation']
)
class CRPSLoss(EnergyLikeLoss):
  """CRPS loss on linearly transformed errors.

  CRPS takes the form (with E expectation)
    E‖X - Y‖^β - ensemble_term_weight * E‖X - X'‖^β
  where ‖⋅‖ is the L1 norm. It can be thought of as the sum of component-wise
  energy score losses.

  Based on formula 21 in [1]; http://shortn/_Lyu0etEy1F

  References:
    [1]: Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules,
         prediction, and estimation. Journal of the American statistical
         Association, 102(477), 359-378.
  """

  def _spread_skill_and_loss(
      self,
      x_minus_y: Pytree,
      x_minus_xprime: Pytree,
  ) -> dict[str, jax.Array]:
    """Gets dictionary with 'spread', 'skill', and 'loss' entries."""
    a_minus_cb = self.a_minus_cb
    ensemble_mean = self.ensemble_mean
    component_mean = self.component_mean

    def abs_beta(tree: Pytree) -> Pytree:
      return tree_map(lambda x: jnp.abs(x) ** self._beta, tree)

    # With X, X' two i.i.d. predictions,
    #   Skill  = (1/2)[ (1/N)Σₙ|Xₙ-Yₙ| + (1/N)Σₙ|Xₙ'-Yₙ| ]
    #   Spread = (1/N) Σₙ|Xₙ-Xₙ'|

    # Recall x_minus_y = X-Y on one device and X'-Y on another. So the ensemble
    # mean of this (which is all-reduced) is exactly Skill above.
    skill = component_mean(ensemble_mean(abs_beta(x_minus_y)))

    # One device has X-X' and the other has X'-X, so the ensemble mean is the
    # same on both devices.
    spread = component_mean(ensemble_mean(abs_beta(x_minus_xprime)))

    # Then CRPS = Skill - (1/2) Spread
    # However, this is unstable if Spread = 2Skill + ε, where |ε| << |Spread|.
    # In particular, up to numerical precision, CRPS will equal 0!
    # This can happen if Prob[Xₙ = 1] = p << 1, and Prob[Xₙ = 0] = 1 - p.
    # a stable estimate of CRPS is
    #   CRPS = C + C'  (an ensemble mean)
    # where
    #      C = (1/N) Σₙ[ |Xₙ-Yₙ| - (1/2) |Xₙ-Xₙ'| ]
    #      C'= (1/N) Σₙ[ |Xₙ'-Yₙ| - (1/2) |Xₙ'-Xₙ| ]
    # which should be re-written as
    #   CRPS = (1/(2N)) Σₙ[ |Xₙ-Yₙ| + |Xₙ'-Yₙ| - |Xₙ-Xₙ'| ]
    # The triangle inequality ensures the summands are non-negative.
    crps = component_mean(
        ensemble_mean(
            a_minus_cb(  # |Xₙ-Yₙ| - (1/2) |Xₙ-Xₙ'|
                abs_beta(x_minus_y),
                self._ensemble_term_weight,
                abs_beta(x_minus_xprime),
            )
        )
    )
    return {'spread': spread, 'skill': skill, 'loss': crps}


@gin.register(
    denylist=['coarsen_aggregation', 'vector_norm_squared_aggregation']
)
class EnergyScoreLoss(EnergyLikeLoss):
  """Energy score loss on linearly transformed errors.

  EnergyScoreLoss takes the form (with E expectation)
    E‖X - Y‖^β - ensemble_term_weight * E‖X - X'‖^β
  where ‖⋅‖ is the L2 norm. It is a generalization of CRPS to
  multiple-dimensions.

  Based on formula 22 in [1]; http://shortn/_Lyu0etEy1F

  References:
    [1]: Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules,
         prediction, and estimation. Journal of the American Statistical
         Association, 102(477), 359-378.
  """

  def _spread_skill_and_loss(
      self,
      x_minus_y: Pytree,
      x_minus_xprime: Pytree,
  ) -> dict[str, jax.Array]:
    """Gets dictionary with 'spread', 'skill', and 'loss' entries."""
    a_minus_cb = self.a_minus_cb
    ensemble_mean = self.ensemble_mean
    component_mean = self.component_mean

    def sqrt_beta(x: jax.Array) -> jax.Array:
      return model_utils.safe_sqrt(x) ** self._beta

    def square(tree: Pytree) -> Pytree:
      return tree_map(jnp.square, tree)

    # With X, X' two i.i.d. predictions,
    #   Skill  = (1/2)[ ‖X-Y‖ + ‖X'-Y‖ ]
    #   Spread = ‖Xₙ-Xₙ'‖

    # Recall x_minus_y = X-Y on one device and X'-Y on another. So the ensemble
    # mean of this (which is all-reduced) is exactly Skill above.
    skill = ensemble_mean(sqrt_beta(component_mean(square(x_minus_y))))

    # One device has X-X' and the other has X'-X, so the ensemble mean is the
    # same on both devices. The call to ensemble_mean simply removes the
    # ensemble dim.
    spread = ensemble_mean(sqrt_beta(component_mean(square(x_minus_xprime))))

    # The straightforward implementation will lose resolution when the relative
    # difference between
    #   ‖X - X'‖  AND ‖X - Y‖ + ‖X' - Y‖,
    # is less than 1e-6. This is so unlikely that we do will not handle it.
    es_straightforward = a_minus_cb(skill, self._ensemble_term_weight, spread)
    es = es_straightforward

    # Unused demonstration of how to handle this co-linear case with lots of
    # extra complex operations.
    # if float(self._beta) != 1:
    #   es = es_straightforward
    # else:
    #   # If beta == 1, there is a high resolution fix.
    #   # See http://screen/BkvX57d9B9eqMrB
    #   #
    #   # alpha = ‖X - Y‖²
    #   alpha = component_mean(square(x_minus_y))
    #   # And if ensemble_term_weight == 1/2,
    #   # gamma_minus_alpha = ‖X - X'‖²/4 - ‖X - Y‖²
    #   #                  = (1/N) Σₙ[ (Xₙ-Xₙ')²/4 - (Xₙ-Yₙ)² ]
    #   gamma_minus_alpha = component_mean(
    #       self.ca_minus_b(
    #           self._ensemble_term_weight**2,
    #           square(x_minus_xprime),
    #           square(x_minus_y),
    #       )
    #   )
    #   # If gamma = ‖X - X'‖²/4 = 0 (e.g. at step=0), then (γ-α)/α = -1,
    #   # and then grad(sqrt1pm1) is NaN. However, in this case we can use the
    #   # straightforward version with no issues.
    #   gamma_minus_alpha_div_alpha = gamma_minus_alpha / alpha

    #   # Construct a "safe" input to use in the go/tf-where-nan trick.
    #   cutoff = -0.1
    #   safe_gamma_minus_alpha_div_alpha = jnp.maximum(
    #       gamma_minus_alpha / alpha, cutoff
    #   )

    #   # For γ ≈ α, safe_gamma_minus_alpha_div_alpha =
    #   # gamma_minus_alpha_div_alpha, and this code block will be used.
    #   es_for_small_diffs = ensemble_mean(
    #       # sqrt1pm1(z) = sqrt(z + 1) - 1, so
    #       # sqrt(α) * -1 * sqrt1pm1((γ-α)/α)
    #       # = sqrt(α) * (1 - sqrt((γ-α)/α) + 1)
    #       # = sqrt(α) * (1 - sqrt(γ/α))
    #       # = sqrt(α) - sqrt(γ)
    #       # = sqrt(‖X - Y‖²) - sqrt(‖X - X'‖²/4)
    #       jnp.sqrt(alpha)
    #       * -1
    #       * tfp.math.sqrt1pm1(safe_gamma_minus_alpha_div_alpha)
    #   )
    #   es = jnp.where(
    #       # Reminder that the triangle-inequality shows γ <= α always. So
    #       # (γ - α) / α < 0.1 is a "large diff" (despite being negative).
    #       gamma_minus_alpha_div_alpha < cutoff,
    #       es_straightforward,
    #       es_for_small_diffs,
    #   )
    return {'spread': spread, 'skill': skill, 'loss': es}
