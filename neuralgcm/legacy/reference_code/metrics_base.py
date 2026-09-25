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
"""Base classes for Metrics."""
import dataclasses
from typing import Callable
from dinosaur import typing
import jax
import jax.numpy as jnp
import metrics_util


Pytree = typing.Pytree
TrajectoryRepresentations = typing.TrajectoryRepresentations

tree_leaves = jax.tree_util.tree_leaves
tree_map = jax.tree_util.tree_map


@dataclasses.dataclass
class Evaluator:
  """Class that evaluates on (prediction, trajectory) returning Pytree."""

  def evaluate(
      self,
      prediction: TrajectoryRepresentations,
      target: TrajectoryRepresentations,
  ) -> Pytree:
    """Evaluates giving values of interest."""
    raise NotImplementedError()


@dataclasses.dataclass
class EvaluateFunctionWrapper(Evaluator):
  """Wraps `evaluate_fn` function to be used as an Evaluator."""

  def __init__(
      self,
      evaluate_fn: Callable[
          [TrajectoryRepresentations, TrajectoryRepresentations], Pytree
      ],
  ):
    self._evaluate_fn = evaluate_fn

  def evaluate(
      self,
      prediction: TrajectoryRepresentations,
      target: TrajectoryRepresentations,
  ) -> Pytree:
    return self._evaluate_fn(prediction, target)


class MetricRuntimeError(Exception):
  """Generic error for Metrics to raise in place of generic RuntimeError."""


@dataclasses.dataclass
class Metric(Evaluator):
  """An Evaluator that derives information from a TrajectorySpec."""

  trajectory_spec: metrics_util.TrajectorySpec
  is_nodal: bool = dataclasses.field(default=True, kw_only=True)
  is_encoded: bool = dataclasses.field(default=False, kw_only=True)

  def get_representation(self, x: TrajectoryRepresentations) -> Pytree:
    x_rep = x.get_representation(
        is_nodal=self.is_nodal, is_encoded=self.is_encoded
    )
    if x_rep is None:
      raise MetricRuntimeError(
          'Desired representation of `x` was None. '
          f'{self.is_nodal=}, {self.is_encoded=}'
      )
    return x_rep

  def surface_mean(self, trajectory: Pytree) -> Pytree:
    if self.is_encoded:
      coords = self.trajectory_spec.coords
    else:
      coords = self.trajectory_spec.data_coords
    if self.is_nodal:
      # Mean over lat/lon. Converts shapes
      #   (n_time, n_level, n_lon, n_lat) --> (n_time, n_level)
      fn = lambda x: metrics_util.nodal_surface_mean(x, coords)
    else:
      fn = lambda x: metrics_util.modal_surface_mean(x, coords)
    return tree_map(fn, trajectory)

  def mean_per_variable(self, trajectory: Pytree) -> Pytree:
    # In practice this is used to reduce shape (n_time, n_level) --> ()
    return tree_map(jnp.mean, self.surface_mean(trajectory))


class ScalarMetric(Metric):
  """Metric that compute scalar quantities."""


@dataclasses.dataclass
class Loss(ScalarMetric):
  """Metric that can be used as a loss."""

  trajectory_spec: metrics_util.TrajectorySpec
  is_nodal: bool = dataclasses.field(default=True, kw_only=True)
  is_encoded: bool = dataclasses.field(default=False, kw_only=True)
  time_step: int | slice | None = dataclasses.field(default=None, kw_only=True)

  def evaluate_per_variable(
      self,
      prediction: TrajectoryRepresentations,
      target: TrajectoryRepresentations,
  ) -> Pytree:
    raise NotImplementedError()

  def evaluate(
      self,
      prediction: TrajectoryRepresentations,
      target: TrajectoryRepresentations,
  ) -> jnp.ndarray:
    error_per_variable = self.evaluate_per_variable(prediction, target)
    return sum(tree_leaves(error_per_variable))

  def debug_loss_terms_instance(self) -> EvaluateFunctionWrapper:
    """Returns class that evaluates relative loss per variable."""

    def evaluate_fn(
        prediction: TrajectoryRepresentations,
        target: TrajectoryRepresentations,
    ) -> Pytree:
      # self.loss.evaluate takes ensemble mean (to evaluate on ensemble mean) if
      # needed.
      loss_per_variable = self.evaluate_per_variable(prediction, target)
      # here we reduce terms by summation to expose relative contributions,
      # even though the actual total_loss might be different.
      sum_of_all_terms = sum(tree_leaves(loss_per_variable))
      relative_loss = tree_map(
          lambda x: x / sum_of_all_terms, loss_per_variable
      )
      return {'relative_loss': relative_loss}

    return EvaluateFunctionWrapper(evaluate_fn)
