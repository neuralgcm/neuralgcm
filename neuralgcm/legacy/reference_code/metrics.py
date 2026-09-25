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
"""Metrics and loss functions for NeuralGCM."""

from __future__ import annotations

import dataclasses
import functools
from typing import Callable, Optional, Sequence

from dinosaur import coordinate_systems
from dinosaur import horizontal_interpolation
from dinosaur import spherical_harmonic
from dinosaur import typing
from dinosaur import vertical_interpolation
import gin
import jax
import jax.numpy as jnp
import linear_transforms
import metrics_base
import metrics_util
from neuralgcm.legacy import model_utils
import numpy as np
import train_utils


Pytree = typing.Pytree
TrajectoryRepresentations = typing.TrajectoryRepresentations

tree_leaves = jax.tree_util.tree_leaves
tree_map = jax.tree_util.tree_map


def _compute_spectral_norm(
    x: typing.Array, coords: coordinate_systems.CoordinateSystem
) -> typing.Array:
  """Computes spectral norm of nodal inputs `x`."""
  x = coordinate_systems.maybe_to_modal(x, coords)
  # axis = -2 corresponds to the longitudinal wavenumber.
  return model_utils.safe_sqrt(
      jnp.sum((x * x.conj()).real, axis=-2, keepdims=True)
  )


@gin.register
def _spectral_amplitude(
    x: typing.Array, coords: coordinate_systems.CoordinateSystem
) -> typing.Array:
  """Computes spectral amplitude ."""
  x = coordinate_systems.maybe_to_modal(x, coords)
  return jnp.abs(x)


@gin.register
@dataclasses.dataclass
class TransformedL2Loss(metrics_base.Loss):
  """L2 loss on linearly transformed errors."""

  def __init__(
      self,
      trajectory_spec: metrics_util.TrajectorySpec,
      components: Sequence[linear_transforms.LinearTransformConstructor],
      is_nodal: bool = True,
      is_encoded: bool = False,
      getter: Callable[[Pytree], Pytree] = metrics_util.filter_sim_time,
      time_step: Optional[int | slice] = None,
  ):
    super().__init__(
        trajectory_spec,
        is_nodal=is_nodal,
        is_encoded=is_encoded,
        time_step=time_step,
    )
    self.components = components
    self.getter = getter
    self.transform = linear_transforms.ComposedTransformForLoss(
        trajectory_spec, components
    )

  def evaluate_per_variable(
      self,
      prediction: TrajectoryRepresentations,
      target: TrajectoryRepresentations,
  ) -> Pytree:
    prediction = self.get_representation(prediction)
    target = self.get_representation(target)
    trajectory = self.getter(prediction)
    target = self.getter(target)
    errors = tree_map(jnp.subtract, trajectory, target)
    transformed_errors = self.transform(errors, target)
    squared_transformed_errors = tree_map(jnp.square, transformed_errors)
    return self.mean_per_variable(squared_transformed_errors)


@gin.register
@dataclasses.dataclass
class TransformedL2SpectrumLoss(metrics_base.Loss):
  """L2 loss on linearly transformed errors of spectal norms.

  Here we define spectrum norm at a given total wavenumber as the length of the
  vector formed by longitude wavenumbers. i.e. for a field `x` with indices
  `{z, m, l}` corresponding to level, longitude wavenumber, total wavenumber
  we have:

  spectrum_norm(x)_{z, l} = ||x_{z, :, l}||₂

  The loss is then computed as MSE(spectrum_norm(x), spectrum_norm(y)) where
  `x` and `y` are predicted and target signals in modal representation.
  """

  def __init__(
      self,
      trajectory_spec: metrics_util.TrajectorySpec,
      components: Sequence[linear_transforms.LinearTransformConstructor],
      is_nodal: bool = True,
      is_encoded: bool = False,
      getter: Callable[[Pytree], Pytree] = metrics_util.filter_sim_time,
      time_step: Optional[int | slice] = None,
  ):
    super().__init__(
        trajectory_spec,
        is_nodal=is_nodal,
        is_encoded=is_encoded,
        time_step=time_step,
    )
    if self.is_encoded:
      coords = trajectory_spec.coords
    else:
      coords = trajectory_spec.data_coords
    spectrum_fn = lambda x: _compute_spectral_norm(x, coords)
    self.components = components
    self.getter = getter
    self.spectrum_fn = lambda tree: tree_map(spectrum_fn, tree)
    self.transform = linear_transforms.ComposedTransformForLoss(
        trajectory_spec, components
    )

  def mean_per_variable(self, trajectory: Pytree) -> Pytree:
    return tree_map(jnp.mean, trajectory)

  def evaluate_per_variable(
      self,
      prediction: TrajectoryRepresentations,
      target: TrajectoryRepresentations,
  ) -> Pytree:
    prediction = self.get_representation(prediction)
    target = self.get_representation(target)
    trajectory_spectrum = self.spectrum_fn(self.getter(prediction))
    target_spectrum = self.spectrum_fn(self.getter(target))
    errors = tree_map(jnp.subtract, trajectory_spectrum, target_spectrum)
    transformed_errors = self.transform(errors, target)
    squared_transformed_errors = tree_map(jnp.square, transformed_errors)
    return self.mean_per_variable(squared_transformed_errors)


@gin.register
@dataclasses.dataclass
class SumLoss(metrics_base.Loss):
  """Loss that consists of a sum of separate losses."""

  def __init__(
      self,
      trajectory_spec: metrics_util.TrajectorySpec,
      terms: Sequence[Callable[..., metrics_base.Loss]],
      labels: Optional[Sequence[str]] = None,
      time_step: Optional[int | slice] = None,
  ):
    super().__init__(trajectory_spec)
    self.losses = [term(trajectory_spec, time_step=time_step) for term in terms]
    if labels is not None:
      if len(labels) != len(self.losses):
        raise ValueError(f'Not all losses are labeled: {labels}, {len(terms)=}')
      self.labels = labels
    else:
      self.labels = [''] * len(self.losses)

  def evaluate_per_variable(
      self,
      prediction: TrajectoryRepresentations,
      target: TrajectoryRepresentations,
  ) -> Pytree:
    all_per_variable_losses = [
        loss.evaluate_per_variable(prediction, target) for loss in self.losses
    ]
    output = {}
    for per_variable_loss, prefix in zip(all_per_variable_losses, self.labels):
      for k, v in per_variable_loss.items():
        if isinstance(v, dict):
          current_values = output.get(prefix + k, {})
          for ik, iv in v.items():
            current_values[ik] = current_values.get(ik, 0) + iv
          output[prefix + k] = current_values
        else:
          output[prefix + k] = output.get(prefix + k, 0) + v
    return output

  def evaluate(
      self,
      prediction: TrajectoryRepresentations,
      target: TrajectoryRepresentations,
  ) -> Pytree:
    return sum(loss.evaluate(prediction, target) for loss in self.losses)

  def debug_loss_terms_instance(self) -> metrics_base.EvaluateFunctionWrapper:
    """Returns class that evaluates relative loss per variable."""

    def evaluate_fn(
        prediction: TrajectoryRepresentations,
        target: TrajectoryRepresentations,
    ) -> Pytree:
      return train_utils.flatten_dict({
          label: loss.debug_loss_terms_instance().evaluate(prediction, target)
          for label, loss in zip(self.labels, self.losses)
      })

    return metrics_base.EvaluateFunctionWrapper(evaluate_fn)


@gin.register
def WeightedL2CumulativeLoss(  # pylint: disable=invalid-name
    trajectory_spec: metrics_util.TrajectorySpec,
    weights: Pytree = None,
    scale: float = 1.0,
) -> TransformedL2Loss:
  """Legacy wrapper for TransformedL2Loss with weighted cumulative error."""
  components = [
      linear_transforms.LegacyTimeRescaling,
      functools.partial(
          linear_transforms.PerVariableRescaling, weights=weights, scale=scale
      ),
  ]
  return TransformedL2Loss(trajectory_spec, components)


@gin.register
class RMSE(metrics_base.ScalarMetric):
  """Root mean squared error."""

  def __init__(
      self,
      trajectory_spec: metrics_util.TrajectorySpec,
      time_step: int,
      level: Optional[int] = None,
      getter: Callable[[Pytree], Pytree] = metrics_util.filter_sim_time,
      is_nodal: bool = True,
      is_encoded: bool = False,
      is_ensemble_data: bool = False,
  ):
    super().__init__(trajectory_spec, is_nodal=is_nodal, is_encoded=is_encoded)
    self.time_step = time_step
    self.level = level
    self.getter = getter
    self.is_ensemble_data = is_ensemble_data

  def _prepare(self, trajectory: TrajectoryRepresentations) -> Pytree:
    """Prepares target or predictions."""
    trajectory = metrics_util.extract_variable(
        trajectory,
        self.trajectory_spec,
        self.time_step,
        self.level,
        self.getter,
        self.is_nodal,
        self.is_encoded,
    )
    if self.is_ensemble_data:
      # Evaluate RMSE vs. the ensemble mean.
      trajectory = jax.lax.pmean(trajectory, axis_name='ensemble')
    return trajectory

  def evaluate(
      self,
      prediction: TrajectoryRepresentations,
      target: TrajectoryRepresentations,
  ) -> jnp.ndarray:
    """Evaluates RMSE between prediction and target."""
    prediction = self._prepare(prediction)
    target = self._prepare(target)
    squared_error = tree_map(lambda x, y: (x - y) ** 2, prediction, target)
    mse_per_variable = self.mean_per_variable(squared_error)
    return jnp.sqrt(sum(tree_leaves(mse_per_variable)))


@gin.register
class SpatialBiasRMSE(metrics_base.ScalarMetric):
  """Root mean squared error of spatial bias.

  This is given by the formula:

    RMSE(batch_average(prediction - target))

  where `batch_average()` denotes an average over distinct weather forecasts
  (initialization times or valid times) and ensemble members (if relevant).
  """

  def __init__(
      self,
      trajectory_spec: metrics_util.TrajectorySpec,
      time_step: int,
      level: Optional[int] = None,
      getter: Callable[[Pytree], Pytree] = metrics_util.filter_sim_time,
      is_nodal: bool = True,
      is_encoded: bool = False,
      is_batch_data: bool = True,
      is_ensemble_data: bool = False,
  ):
    super().__init__(trajectory_spec, is_nodal=is_nodal, is_encoded=is_encoded)
    self.time_step = time_step
    self.level = level
    self.getter = getter
    self.is_ensemble_data = is_ensemble_data
    self.is_batch_data = is_batch_data

  def _prepare(self, trajectory: TrajectoryRepresentations) -> Pytree:
    """Prepares target or predictions."""
    trajectory = metrics_util.extract_variable(
        trajectory,
        self.trajectory_spec,
        time_step=self.time_step,
        level=self.level,
        getter=self.getter,
        is_nodal=self.is_nodal,
        is_encoded=self.is_encoded,
    )
    if self.is_batch_data:
      trajectory = jax.lax.pmean(trajectory, axis_name='batch')
    if self.is_ensemble_data:
      trajectory = jax.lax.pmean(trajectory, axis_name='ensemble')
    return trajectory

  def evaluate(
      self,
      prediction: TrajectoryRepresentations,
      target: TrajectoryRepresentations,
  ) -> jnp.ndarray:
    """Evaluates RMSE between prediction and target."""
    prediction = self._prepare(prediction)
    target = self._prepare(target)
    squared_error = tree_map(lambda x, y: (x - y) ** 2, prediction, target)
    mse_per_variable = self.mean_per_variable(squared_error)
    return jnp.sqrt(sum(tree_leaves(mse_per_variable)))


@gin.register
class BatchMeanSquaredBias(metrics_base.Loss):
  """Mean squared error for a chosen metric.

  This is given by the formula:

    MSE(rollout_average(batch_average(prediction - target)))

  where `batch_average()` denotes an average over distinct weather forecasts
  (initialization times or valid times) or ensemble members (whichever is
  vmapped first) and 'rollout_average()' denotes an average over all predicted
  times. The MSE is taken over all nodal/modal points.
  """

  def __init__(
      self,
      trajectory_spec: metrics_util.TrajectorySpec,
      components: Sequence[linear_transforms.LinearTransformConstructor] = (),
      observation_fn=_spectral_amplitude,
      getter: Callable[[Pytree], Pytree] = metrics_util.filter_sim_time,
      is_nodal: bool = False,
      is_encoded: bool = False,
      time_step: Optional[int | slice] = None,
  ):
    super().__init__(
        trajectory_spec,
        is_nodal=is_nodal,
        is_encoded=is_encoded,
        time_step=time_step,
    )
    if self.is_encoded:
      coords = trajectory_spec.coords
    else:
      coords = trajectory_spec.data_coords
    metric_fn = lambda x: observation_fn(x, coords)
    self.components = components
    self.getter = getter
    self.metric_fn = lambda tree: tree_map(metric_fn, tree)
    self.transform = linear_transforms.ComposedTransformForLoss(
        trajectory_spec, components
    )

  def evaluate_per_variable(
      self,
      prediction: TrajectoryRepresentations,
      target: TrajectoryRepresentations,
  ) -> Pytree:
    """Evaluates the squere bias of a chosen metric between prediction and target.

    Note: this method is only valid when vmapped.

    Args:
      prediction: a TrajectoryRepresentations of prediction
      target: a TrajectoryRepresentations of ground truth

    Returns:
      Pytree of MSE
    """
    prediction = self.get_representation(prediction)
    target = self.get_representation(target)
    # because this function applies average over time axis, we apply
    # `TruncateToTrajectoryLength` prior to computing
    truncate_transform = self.transform.transforms[0]
    assert isinstance(
        truncate_transform, linear_transforms.TruncateToTrajectoryLength
    )
    getter_fn = lambda x: self.getter(truncate_transform(x, None))
    trajectory_calc = self.metric_fn(getter_fn(prediction))
    target_calc = self.metric_fn(getter_fn(target))
    # Batch mean over "ensemble" and "batch" dimensions
    trajectory_calc = tree_map(metrics_util.pmean_all_axes, trajectory_calc)
    target_calc = tree_map(metrics_util.pmean_all_axes, target_calc)
    # Time mean:
    trajectory_calc = tree_map(
        lambda x,: jnp.mean(x, axis=0, keepdims=True), trajectory_calc
    )
    target_calc = tree_map(
        lambda x,: jnp.mean(x, axis=0, keepdims=True), target_calc
    )
    errors = tree_map(jnp.subtract, trajectory_calc, target_calc)
    transformed_errors = self.transform(errors, target)
    squared_transformed_errors = tree_map(jnp.square, transformed_errors)
    mse_per_variable = tree_map(jnp.mean, squared_transformed_errors)
    return mse_per_variable


@gin.register
class MAE(metrics_base.ScalarMetric):
  """Mean absolute error."""

  def __init__(
      self,
      trajectory_spec: metrics_util.TrajectorySpec,
      time_step: int,
      level: Optional[int] = None,
      getter: Callable[[Pytree], Pytree] = metrics_util.filter_sim_time,
      is_nodal: bool = True,
      is_encoded: bool = False,
  ):
    super().__init__(trajectory_spec, is_nodal=is_nodal, is_encoded=is_encoded)
    self.time_step = time_step
    self.level = level
    self.getter = getter

  def _prepare(self, trajectory: TrajectoryRepresentations) -> Pytree:
    return metrics_util.extract_variable(
        trajectory,
        self.trajectory_spec,
        self.time_step,
        self.level,
        self.getter,
        self.is_nodal,
        self.is_encoded,
    )

  def evaluate(
      self,
      prediction: TrajectoryRepresentations,
      target: TrajectoryRepresentations,
  ) -> jnp.ndarray:
    prediction = self._prepare(prediction)
    target = self._prepare(target)
    abs_error = tree_map(lambda x, y: abs(x - y), prediction, target)
    mse_per_variable = self.mean_per_variable(abs_error)
    flat_mse = tree_leaves(mse_per_variable)
    return sum(flat_mse) / len(flat_mse)


@jax.jit
def weighted_quantile(
    data: jax.Array, quantile: jax.Array, weights: jax.Array
) -> jax.Array:
  """Calculate a weighted quantile."""
  if data.shape != weights.shape:
    raise ValueError(f'incompatible shapes: {data.shape=} != {weights.shape=}')
  data = data.ravel()
  weights = weights.ravel() / weights.sum()
  indices = jnp.argsort(data)
  cum_weights = weights[indices].cumsum()
  return jnp.interp(quantile, cum_weights, data[indices])


@dataclasses.dataclass
class AbsErrorQuantile(metrics_base.ScalarMetric):
  """Quantile of absolute error."""

  def __init__(
      self,
      trajectory_spec: metrics_util.TrajectorySpec,
      quantile: float,
      time_step: int,
      level: Optional[int] = None,
      getter: Callable[[Pytree], Pytree] = metrics_util.filter_sim_time,
      is_nodal: bool = True,
      is_encoded: bool = False,
      is_ensemble_data: bool = False,
  ):
    super().__init__(trajectory_spec, is_nodal=is_nodal, is_encoded=is_encoded)
    self.quantile = quantile
    self.time_step = time_step
    self.level = level
    self.getter = getter
    self.is_ensemble_data = is_ensemble_data

  def _prepare(self, trajectory: TrajectoryRepresentations) -> Pytree:
    return metrics_util.extract_variable(
        trajectory,
        self.trajectory_spec,
        self.time_step,
        self.level,
        self.getter,
        self.is_nodal,
        self.is_encoded,
    )

  def _get_weights(self) -> np.ndarray:
    if self.is_encoded:
      coords = self.trajectory_spec.coords
    else:
      coords = self.trajectory_spec.data_coords
    if self.is_nodal:
      weights = coords.horizontal.quadrature_weights
    else:
      weights = coords.horizontal.mask
    return weights

  def evaluate(
      self,
      prediction: TrajectoryRepresentations,
      target: TrajectoryRepresentations,
  ) -> jnp.ndarray:
    prediction = self._prepare(prediction)
    target = self._prepare(target)
    abs_error = tree_map(lambda x, y: abs(x - y), prediction, target)
    weights = jnp.broadcast_to(self._get_weights(), target.shape)
    result = tree_map(
        lambda e: weighted_quantile(e, self.quantile, weights), abs_error
    )
    if self.is_ensemble_data:
      # metrics must be consistent across the ensmble dimension.
      result = jax.lax.pmean(result, axis_name='ensemble')
    return result


def weatherbench2_rmse_metrics(
    trajectory_spec: metrics_util.TrajectorySpec,
    time_steps: Sequence[int],
    is_ensemble_data: bool = False,
    extra_metric_grids: Optional[dict[str, spherical_harmonic.Grid]] = None,
) -> dict[str, metrics_base.Metric]:
  """RMSE based metrics for WeatherBench2."""
  metric_grids = {} if extra_metric_grids is None else extra_metric_grids.copy()
  trajectory_grid = trajectory_spec.coords.horizontal
  if trajectory_grid not in metric_grids.values():
    metric_grids['Traj'] = trajectory_grid

  def get_and_regrid(tree, regrid_fn, getter):
    return tree_map(regrid_fn, getter(tree))

  metrics = {}
  for name, grid in metric_grids.items():
    if grid == trajectory_grid:
      regrid = lambda tree: tree
      rmse_traj_spec = trajectory_spec
    else:
      regrid = horizontal_interpolation.ConservativeRegridder(
          source_grid=trajectory_spec.coords.horizontal, target_grid=grid
      )
      rmse_traj_spec = dataclasses.replace(
          trajectory_spec,
          # Only data_coords needs to be replaced since RMSE.is_encoded=False.
          data_coords=dataclasses.replace(
              trajectory_spec.data_coords,
              horizontal=grid,
          ),
      )
    for time_step in time_steps:
      for var, level, getter in [
          ('T', 850, lambda x: x['t']),
          ('Z', 500, lambda x: x['z']),
          ('UV', 700, lambda x: (x['u'], x['v'])),
          ('Q', 700, lambda x: 1000 * x['tracers']['specific_humidity']),
      ]:
        t = time_step * trajectory_spec.steps_per_save
        key = f'RMSE[{name}]_{var}{level}_{t:03d}_hours'
        metrics[key] = RMSE(
            rmse_traj_spec,
            is_encoded=False,  # To make this (default) clear.
            time_step=time_step,
            level=level,
            getter=functools.partial(
                get_and_regrid, regrid_fn=regrid, getter=getter
            ),
            is_ensemble_data=is_ensemble_data,
        )
  return metrics


def default_metrics(
    trajectory_spec: metrics_util.TrajectorySpec,
    eval_time_steps: Sequence[int],
    train_loss: metrics_base.Loss,
    is_batch_data: bool = True,
    is_ensemble_data: bool = False,
) -> dict[str, metrics_base.Evaluator]:
  """Default evaluation metrics for Whirl models."""
  metrics_dict = {
      'training_loss': train_loss,
      'debug': train_loss.debug_loss_terms_instance(),
  }

  if isinstance(
      trajectory_spec.data_coords.vertical,
      vertical_interpolation.PressureCoordinates,
  ):
    tl31_grid = dataclasses.replace(
        spherical_harmonic.Grid.TL31(),
        spherical_harmonics_impl=trajectory_spec.data_coords.horizontal.spherical_harmonics_impl,
    )
    metrics_dict.update(
        weatherbench2_rmse_metrics(
            trajectory_spec,
            eval_time_steps,
            is_ensemble_data=is_ensemble_data,
            extra_metric_grids={'TL31': tl31_grid},
        )
    )

    for time_step in eval_time_steps:
      t = time_step * trajectory_spec.steps_per_save

      for var, getter in [
          ('T', lambda x: x['t']),
          ('Z', lambda x: x['z']),
          ('UV', lambda x: (x['u'], x['v'])),
          ('Q', lambda x: 1000 * x['tracers']['specific_humidity']),
      ]:
        key = f'rmse_{var}_all_levels_{t:03d}_hours'
        metrics_dict[key] = RMSE(
            trajectory_spec,
            time_step=time_step,
            level=None,
            getter=getter,
            is_ensemble_data=is_ensemble_data,
        )

      for var, level, getter in [
          ('T', 850, lambda x: x['t']),
          ('Z', 500, lambda x: x['z']),
          ('U', 700, lambda x: x['u']),
          ('V', 700, lambda x: x['v']),
          ('Q', 700, lambda x: 1000 * x['tracers']['specific_humidity']),
      ]:
        key = f'spatial_bias_rmse_{var}{level}_{t:03d}_hours'
        metrics_dict[key] = SpatialBiasRMSE(
            trajectory_spec,
            time_step=time_step,
            level=level,
            getter=getter,
            is_batch_data=is_batch_data,
            is_ensemble_data=is_ensemble_data,
        )

      for var, level, getter in [
          ('T', 850, lambda x: x['t']),
          ('Z', 500, lambda x: x['z']),
          ('U', 700, lambda x: x['u']),
          ('V', 700, lambda x: x['v']),
          ('Q', 700, lambda x: 1000 * x['tracers']['specific_humidity']),
      ]:
        for q in [0.99]:
          key = f'abs_error_q{q}_{var}{level}_{t:03d}_hours'
          metrics_dict[key] = AbsErrorQuantile(
              trajectory_spec,
              quantile=q,
              time_step=time_step,
              level=level,
              getter=getter,
              is_ensemble_data=is_ensemble_data,
          )

  return metrics_dict
