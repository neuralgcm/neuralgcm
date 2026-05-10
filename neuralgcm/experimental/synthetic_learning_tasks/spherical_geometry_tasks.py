# Copyright 2026 Google LLC
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
"""Implements spherical geometry tasks for benchmarking ML architectures."""

import coordax as cx
from flax import nnx
import jax
import jax.numpy as jnp
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import module_utils
from neuralgcm.experimental.core import random_processes
from neuralgcm.experimental.core import spatial_filters
from neuralgcm.experimental.core import spherical_harmonics
from neuralgcm.experimental.core import transforms
from neuralgcm.experimental.core import typing
from neuralgcm.experimental.core import units
from neuralgcm.experimental.metrics import binning
from neuralgcm.experimental.metrics import deterministic_losses
from neuralgcm.experimental.metrics import deterministic_metrics
from neuralgcm.experimental.metrics import evaluators
from neuralgcm.experimental.metrics import weighting
from neuralgcm.experimental.synthetic_learning_tasks import synthetic_tasks
from neuralgcm.experimental.synthetic_learning_tasks import utils
import numpy as np


def _setup_scalers(
    task: synthetic_tasks.SyntheticTask,
    rng: jax.Array,
    inputs_scaler: typing.Transform | None,
    targets_scaler: typing.Transform | None,
    n_stats_batches: int,
    independent_axes: tuple[cx.Coordinate, ...] = (),
    skip_nans: bool = False,
) -> tuple[typing.Transform, typing.Transform]:
  """Initializes and calibrates scalers for a synthetic task."""
  if inputs_scaler is not None and targets_scaler is not None:
    return inputs_scaler, targets_scaler  # no need to setup when specified.

  # Setup default scalers for missing transforms to collect statistics.
  inputs_scaler = inputs_scaler or transforms.Identity()
  task.inputs_scaler = inputs_scaler
  task.targets_scaler = targets_scaler or transforms.Identity()
  sample_batch_fn = lambda task, rng: task.sample_batch(rng)
  _, targets_struct = nnx.eval_shape(sample_batch_fn, task, rng)
  inputs_scaler_needs_stats, targets_scaler_needs_stats = False, False
  # Skipping input normalization as requested.
  if targets_scaler is None:
    targets_scaler_needs_stats = True
    targets_scaler = transforms.StreamNorm.for_inputs_struct(
        targets_struct,
        independent_axes=independent_axes,
        update_stats=True,
        skip_nans=skip_nans,
    )

  for i in range(n_stats_batches):
    inputs, targets = sample_batch_fn(task, jax.random.fold_in(rng, i))
    if inputs_scaler_needs_stats:
      inputs_scaler(inputs)
    if targets_scaler_needs_stats:
      targets_scaler(targets)
  return inputs_scaler, targets_scaler


class ColumnIntegratorTask(synthetic_tasks.SyntheticTask):
  """Task for learning vertical integration of a 3D field."""

  def __init__(
      self,
      ylm_map: spherical_harmonics.FixedYlmMapping,
      levels: coordinates.SigmaLevels | coordinates.HybridLevels,
      correlation_lengths: typing.Numeric | None = None,
      batch_size: int | None = 32,
      loss_evaluator: evaluators.Evaluator | None = None,
      metrics_evaluator: evaluators.Evaluator | None = None,
      enable_generalization: bool = True,
      inputs_scaler: typing.Transform | None = None,
      targets_scaler: typing.Transform | None = None,
      n_stats_batches: int = 10,
      *,
      rngs: nnx.Rngs,
  ):
    self.enable_generalization = enable_generalization
    self.ylm_map = ylm_map
    self.b = cx.SizedAxis('b', batch_size) if batch_size else cx.Scalar()
    self.levels = levels
    n_levels = levels.shape[0]
    if correlation_lengths is None:
      correlation_lengths = np.linspace(0.3, 0.5, n_levels)
    correlation_times = [
        1
    ] * n_levels  # Irrelevant because advance() is not called.
    self.grf = random_processes.VectorizedGaussianRandomField(
        ylm_map=ylm_map,
        dt=1.0,
        sim_units=units.SI_UNITS,
        axis=levels,
        correlation_times=correlation_times,
        correlation_lengths=correlation_lengths,
        variances=[1.0] * n_levels,
        rngs=rngs,
    )
    self.log_surface_pressure_grf = random_processes.GaussianRandomField(
        ylm_map=ylm_map,
        dt=1.0,
        sim_units=units.SI_UNITS,
        correlation_length=0.7,
        correlation_time=1.0,  # Irrelevant because advance() is not called.
        variance=1.0,
        rngs=rngs,
    )
    if self.b.ndim:
      module_utils.vectorize_module(self.grf, {typing.Randomness: self.b})
      module_utils.vectorize_module(
          self.log_surface_pressure_grf, {typing.Randomness: self.b}
      )

    self._loss_evaluator = loss_evaluator
    self._metrics_evaluator = metrics_evaluator

    self.inputs_scaler, self.targets_scaler = _setup_scalers(
        self,
        rngs.target(),
        inputs_scaler,
        targets_scaler,
        n_stats_batches,
        independent_axes=(levels,),
    )

  @property
  def loss_evaluator(self) -> evaluators.Evaluator:
    if self._loss_evaluator is not None:
      return self._loss_evaluator
    return evaluators.Evaluator(
        metrics={'mse': deterministic_losses.MSE()},
        aggregators=evaluators.aggregation.Aggregator(
            dims_to_reduce=('b', 'longitude', 'latitude')
        ),
    )

  @property
  def generalization_tasks(self) -> dict[str, 'synthetic_tasks.SyntheticTask']:
    if not self.enable_generalization:
      return {}
    n_levels = self.levels.shape[0]
    return {
        'short_corr_len': ColumnIntegratorTask(
            ylm_map=self.ylm_map,
            levels=self.levels,
            correlation_lengths=np.linspace(0.1, 0.2, n_levels),
            batch_size=self.b.size if self.b.ndim else None,
            enable_generalization=False,
            inputs_scaler=self.inputs_scaler,
            targets_scaler=self.targets_scaler,
            rngs=nnx.Rngs(1),
        )
    }

  @property
  def metrics_evaluator(self) -> evaluators.Evaluator:
    if self._metrics_evaluator is not None:
      return self._metrics_evaluator

    scalar_agg = evaluators.aggregation.Aggregator(
        dims_to_reduce=('b', 'longitude', 'latitude'),
        weight_by=[weighting.GridAreaWeighting()],
    )
    zonal_agg = evaluators.aggregation.Aggregator(
        dims_to_reduce=('b', 'longitude'),
        weight_by=[weighting.GridAreaWeighting()],
    )
    map_agg = evaluators.aggregation.Aggregator(dims_to_reduce=('b',))
    return evaluators.Evaluator(
        metrics={
            'mse': deterministic_losses.MSE(),
            'mse_zonal': deterministic_losses.MSE(),
            'mse_map': deterministic_losses.MSE(),
            'bias_map': deterministic_metrics.Bias(),
            'bias_zonal': deterministic_metrics.Bias(),
        },
        aggregators={
            'mse': scalar_agg,
            'mse_zonal': zonal_agg,
            'mse_map': map_agg,
            'bias_map': map_agg,
            'bias_zonal': zonal_agg,
        },
    )

  @property
  def batch_axis(self) -> cx.Coordinate:
    return self.b

  @property
  def input_shapes(self) -> dict[str, cx.Field]:
    grid = self.ylm_map.lon_lat_grid
    return {
        'features': cx.shape_struct_field(self.b, self.levels, grid),
        'surface_pressure': cx.shape_struct_field(self.b, grid),
    }

  @property
  def target_split_axes(self) -> dict[str, cx.Coordinate]:
    return {'integral': cx.Scalar()}

  @nnx.jit
  def sample_batch(
      self, rng: jax.Array
  ) -> tuple[dict[str, cx.Field], dict[str, cx.Field]]:
    rng, ps_rng = jax.random.split(rng)
    rng = cx.field(rng)
    ps_rng = cx.field(ps_rng)
    if self.b.ndim:
      rng = cx.cmap(jax.random.split)(rng, self.b.size).tag(self.b)
      ps_rng = cx.cmap(jax.random.split)(ps_rng, self.b.size).tag(self.b)
    self.grf.unconditional_sample(rng)
    self.log_surface_pressure_grf.unconditional_sample(ps_rng)

    inputs_field = self.grf.state_values()
    log_surface_pressure = self.log_surface_pressure_grf.state_values()
    surface_pressure = cx.cmap(jnp.exp)(log_surface_pressure)
    targets = self.levels.integrate_over_pressure(
        inputs_field, surface_pressure, sim_units=None
    )
    inputs = {'features': inputs_field, 'surface_pressure': surface_pressure}
    targets = {'integral': targets}
    inputs = self.inputs_scaler(inputs)
    targets = self.targets_scaler(targets)
    return inputs, targets


class HelmholtzDecompositionTask(synthetic_tasks.SyntheticTask):
  """Task for recovering vorticity and divergence from wind field."""

  def __init__(
      self,
      ylm_map: spherical_harmonics.FixedYlmMapping,
      correlation_length: float = 0.7,
      variance: float = 1.0,
      anisotropy_alpha: float | None = None,
      batch_size: int | None = 32,
      loss_evaluator: evaluators.Evaluator | None = None,
      metrics_evaluator: evaluators.Evaluator | None = None,
      enable_generalization: bool = True,
      inputs_scaler: typing.Transform | None = None,
      targets_scaler: typing.Transform | None = None,
      n_stats_batches: int = 10,
      *,
      rngs: nnx.Rngs,
  ):
    self.enable_generalization = enable_generalization
    self.ylm_map = ylm_map
    self.anisotropy_alpha = anisotropy_alpha
    self.b = cx.SizedAxis('b', batch_size) if batch_size else cx.Scalar()

    self.winds_grf = random_processes.GaussianRandomField(
        ylm_map=ylm_map,
        dt=1.0,
        sim_units=units.SI_UNITS,
        correlation_length=correlation_length,
        correlation_time=1.0,  # Irrelevant because advance() is not called.
        variance=variance,
        rngs=rngs,
    )
    if self.b.ndim:
      module_utils.vectorize_module(self.winds_grf, {typing.Randomness: self.b})

    self._loss_evaluator = loss_evaluator
    self._metrics_evaluator = metrics_evaluator

    self.inputs_scaler, self.targets_scaler = _setup_scalers(
        self,
        rngs.target(),
        inputs_scaler,
        targets_scaler,
        n_stats_batches,
    )

  @property
  def loss_evaluator(self) -> evaluators.Evaluator:
    if self._loss_evaluator is not None:
      return self._loss_evaluator
    return evaluators.Evaluator(
        metrics={'mse': deterministic_losses.MSE()},
        aggregators=evaluators.aggregation.Aggregator(
            dims_to_reduce=('b', 'longitude', 'latitude')
        ),
    )

  @property
  def generalization_tasks(self) -> dict[str, 'synthetic_tasks.SyntheticTask']:
    if not self.enable_generalization:
      return {}
    return {
        'short_corr_len': HelmholtzDecompositionTask(
            ylm_map=self.ylm_map,
            correlation_length=0.5,
            anisotropy_alpha=self.anisotropy_alpha,
            batch_size=self.b.size if self.b.ndim else None,
            enable_generalization=False,
            inputs_scaler=self.inputs_scaler,
            targets_scaler=self.targets_scaler,
            rngs=nnx.Rngs(1),
        )
    }

  @property
  def metrics_evaluator(self) -> evaluators.Evaluator:
    if self._metrics_evaluator is not None:
      return self._metrics_evaluator

    lat_bins = binning.Regions(
        bin_dim_name='latitude_bin',
        regions={
            '90_to_60': ((60, 90), (0, 360)),
            '60_to_30': ((30, 60), (0, 360)),
            '30_to_-30': ((-30, 30), (0, 360)),
            '-30_to_-60': ((-60, -30), (0, 360)),
            '-60_to_-90': ((-90, -60), (0, 360)),
        },
    )

    scalar_agg = evaluators.aggregation.Aggregator(
        dims_to_reduce=('b', 'longitude', 'latitude')
    )
    scalar_binned_agg = evaluators.aggregation.Aggregator(
        dims_to_reduce=('b', 'longitude', 'latitude'), bin_by=[lat_bins]
    )
    zonal_agg = evaluators.aggregation.Aggregator(
        dims_to_reduce=('b', 'longitude')
    )
    map_agg = evaluators.aggregation.Aggregator(dims_to_reduce=('b',))
    return evaluators.Evaluator(
        metrics={
            'mse': deterministic_losses.MSE(),
            'mse_binned': deterministic_losses.MSE(),
            'mse_zonal': deterministic_losses.MSE(),
            'mse_map': deterministic_losses.MSE(),
            'bias_binned': deterministic_metrics.Bias(),
            'bias_map': deterministic_metrics.Bias(),
            'bias_zonal': deterministic_metrics.Bias(),
        },
        aggregators={
            'mse': scalar_agg,
            'mse_binned': scalar_binned_agg,
            'mse_zonal': zonal_agg,
            'mse_map': map_agg,
            'bias_binned': scalar_binned_agg,
            'bias_map': map_agg,
            'bias_zonal': zonal_agg,
        },
    )

  @property
  def batch_axis(self) -> cx.Coordinate:
    return self.b

  @property
  def input_shapes(self) -> dict[str, cx.Field]:
    return {
        'u': cx.shape_struct_field(self.b, self.ylm_map.lon_lat_grid),
        'v': cx.shape_struct_field(self.b, self.ylm_map.lon_lat_grid),
    }

  @property
  def target_split_axes(self) -> dict[str, cx.Coordinate]:
    return {'vorticity': cx.Scalar(), 'divergence': cx.Scalar()}

  @nnx.jit
  def sample_batch(
      self, rng: jax.Array
  ) -> tuple[dict[str, cx.Field], dict[str, cx.Field]]:
    rng = cx.field(rng)
    if self.b.ndim:
      rng = cx.cmap(jax.random.split)(rng, self.b.size).tag(self.b)

    psi_rng, chi_rng = cx.cmap(lambda r: tuple(jax.random.split(r)))(rng)
    self.winds_grf.unconditional_sample(psi_rng)
    psi_ylm = self.winds_grf.state_values(self.ylm_map.modal_grid)
    self.winds_grf.unconditional_sample(chi_rng)
    chi_ylm = self.winds_grf.state_values(self.ylm_map.modal_grid)

    if self.anisotropy_alpha is not None:
      psi_ylm = utils.apply_anisotropy(psi_ylm, self.anisotropy_alpha)
      chi_ylm = utils.apply_anisotropy(chi_ylm, self.anisotropy_alpha)

    vort = self.ylm_map.laplacian(psi_ylm)
    div = self.ylm_map.laplacian(chi_ylm)
    u, v = spherical_harmonics.vor_div_to_uv_nodal(vort, div, self.ylm_map)

    vorticity = self.ylm_map.to_nodal(vort)
    divergence = self.ylm_map.to_nodal(div)
    inputs = {'u': u, 'v': v}
    targets = {'vorticity': vorticity, 'divergence': divergence}

    inputs = self.inputs_scaler(inputs)
    targets = self.targets_scaler(targets)

    return inputs, targets


class LagrangianAdvectionTask(synthetic_tasks.SyntheticTask):
  """Task for predicting the advection of a tracer field."""

  def __init__(
      self,
      ylm_map: spherical_harmonics.FixedYlmMapping,
      advection_dt: float = 1.0,
      correlation_length: float = 0.7,
      variance: float = 1.0,
      anisotropy_alpha: float | None = None,
      batch_size: int | None = 32,
      tracer_process_module: random_processes.RandomProcessModule | None = None,
      loss_evaluator: evaluators.Evaluator | None = None,
      metrics_evaluator: evaluators.Evaluator | None = None,
      enable_generalization: bool = True,
      inputs_scaler: typing.Transform | None = None,
      targets_scaler: typing.Transform | None = None,
      n_stats_batches: int = 10,
      *,
      rngs: nnx.Rngs,
  ):
    self.enable_generalization = enable_generalization
    self.ylm_map = ylm_map
    self.advection_dt = advection_dt
    self.anisotropy_alpha = anisotropy_alpha
    self.b = cx.SizedAxis('b', batch_size) if batch_size else cx.Scalar()

    self.winds_grf = random_processes.GaussianRandomField(
        ylm_map=ylm_map,
        dt=1.0,
        sim_units=units.SI_UNITS,
        correlation_length=correlation_length,
        correlation_time=1.0,  # Irrelevant because advance() is not called.
        variance=variance,
        rngs=rngs,
    )

    if tracer_process_module is None:
      self.tracer_process = random_processes.GaussianRandomField(
          ylm_map=ylm_map,
          dt=1.0,
          sim_units=units.SI_UNITS,
          correlation_time=1.0,  # Irrelevant because advance() is not called.
          correlation_length=correlation_length,
          variance=1.0,
          rngs=rngs,
      )
    else:
      self.tracer_process = tracer_process_module

    if self.b.ndim:
      module_utils.vectorize_module(self.winds_grf, {typing.Randomness: self.b})
      module_utils.vectorize_module(
          self.tracer_process, {typing.Randomness: self.b}
      )

    self._loss_evaluator = loss_evaluator
    self._metrics_evaluator = metrics_evaluator

    self.inputs_scaler, self.targets_scaler = _setup_scalers(
        self,
        rngs.target(),
        inputs_scaler,
        targets_scaler,
        n_stats_batches,
    )

  @property
  def loss_evaluator(self) -> evaluators.Evaluator:
    if self._loss_evaluator is not None:
      return self._loss_evaluator
    return evaluators.Evaluator(
        metrics={'mse': deterministic_losses.MSE()},
        aggregators=evaluators.aggregation.Aggregator(
            dims_to_reduce=('b', 'longitude', 'latitude')
        ),
    )

  @property
  def generalization_tasks(self) -> dict[str, 'synthetic_tasks.SyntheticTask']:
    if not self.enable_generalization:
      return {}
    return {
        'short_corr_len': LagrangianAdvectionTask(
            ylm_map=self.ylm_map,
            advection_dt=self.advection_dt,
            correlation_length=0.5,
            anisotropy_alpha=self.anisotropy_alpha,
            batch_size=self.b.size if self.b.ndim else None,
            enable_generalization=False,
            inputs_scaler=self.inputs_scaler,
            targets_scaler=self.targets_scaler,
            rngs=nnx.Rngs(1),
        )
    }

  @property
  def metrics_evaluator(self) -> evaluators.Evaluator:
    if self._metrics_evaluator is not None:
      return self._metrics_evaluator

    scalar_agg = evaluators.aggregation.Aggregator(
        dims_to_reduce=('b', 'longitude', 'latitude')
    )
    zonal_agg = evaluators.aggregation.Aggregator(
        dims_to_reduce=('b', 'longitude')
    )
    map_agg = evaluators.aggregation.Aggregator(dims_to_reduce=('b',))
    return evaluators.Evaluator(
        metrics={
            'mse': deterministic_losses.MSE(),
            'mse_zonal': deterministic_losses.MSE(),
            'mse_map': deterministic_losses.MSE(),
            'bias_map': deterministic_metrics.Bias(),
            'bias_zonal': deterministic_metrics.Bias(),
        },
        aggregators={
            'mse': scalar_agg,
            'mse_zonal': zonal_agg,
            'mse_map': map_agg,
            'bias_map': map_agg,
            'bias_zonal': zonal_agg,
        },
    )

  @property
  def batch_axis(self) -> cx.Coordinate:
    return self.b

  @property
  def input_shapes(self) -> dict[str, cx.Field]:
    return {
        'u': cx.shape_struct_field(self.b, self.ylm_map.lon_lat_grid),
        'v': cx.shape_struct_field(self.b, self.ylm_map.lon_lat_grid),
        'tracer': cx.shape_struct_field(self.b, self.ylm_map.lon_lat_grid),
    }

  @property
  def target_split_axes(self) -> dict[str, cx.Coordinate]:
    return {'tracer': cx.Scalar()}

  @nnx.jit
  def sample_batch(
      self, rng: jax.Array
  ) -> tuple[dict[str, cx.Field], dict[str, cx.Field]]:
    rng = cx.field(rng)
    if self.b.ndim:
      rng = cx.cmap(jax.random.split)(rng, self.b.size).tag(self.b)

    split_three = lambda r: tuple(jax.random.split(r, 3))
    psi_rng, chi_rng, tr_rng = cx.cmap(split_three)(rng)
    self.winds_grf.unconditional_sample(psi_rng)
    psi_ylm = self.winds_grf.state_values(self.ylm_map.modal_grid)
    self.winds_grf.unconditional_sample(chi_rng)
    chi_ylm = self.winds_grf.state_values(self.ylm_map.modal_grid)
    self.tracer_process.unconditional_sample(tr_rng)

    if self.anisotropy_alpha is not None:
      psi_ylm = utils.apply_anisotropy(psi_ylm, self.anisotropy_alpha)
      chi_ylm = utils.apply_anisotropy(chi_ylm, self.anisotropy_alpha)

    grid = self.ylm_map.lon_lat_grid
    vort = self.ylm_map.laplacian(psi_ylm)
    div = self.ylm_map.laplacian(chi_ylm)
    u, v = spherical_harmonics.vor_div_to_uv_nodal(vort, div, self.ylm_map)

    lon = grid.fields['longitude'].broadcast_like(grid)
    lat = grid.fields['latitude'].broadcast_like(grid)
    dep_lon, dep_lat = utils.compute_lon_lat_departure_points(
        lon=lon,
        lat=lat,
        u=u,
        v=v,
        dt=self.advection_dt,
        radius=self.ylm_map.radius,
    )
    tracer = self.tracer_process.state_values(grid)
    tracer_modal = self.tracer_process.state_values(self.ylm_map.modal_grid)
    advected_tracer = utils.evaluate_modal_field(tracer_modal, dep_lon, dep_lat)
    advected_tracer = cx.cmap(jnp.real)(advected_tracer)

    inputs = {'u': u, 'v': v, 'tracer': tracer}
    targets = {'tracer': advected_tracer}

    inputs = self.inputs_scaler(inputs)
    targets = self.targets_scaler(targets)

    return inputs, targets


class MaskedDiffusionTask(synthetic_tasks.SyntheticTask):
  """Task for predicting diffusion on a masked sphere."""

  def __init__(
      self,
      ylm_map: spherical_harmonics.FixedYlmMapping,
      nan_mask: cx.Field,
      nu_dt: float = 0.002,
      steps: int = 5,
      correlation_length: float = 0.7,
      batch_size: int | None = 32,
      loss_evaluator: evaluators.Evaluator | None = None,
      metrics_evaluator: evaluators.Evaluator | None = None,
      enable_generalization: bool = True,
      inputs_scaler: typing.Transform | None = None,
      targets_scaler: typing.Transform | None = None,
      n_stats_batches: int = 10,
      *,
      rngs: nnx.Rngs,
  ):
    self.enable_generalization = enable_generalization
    self.ylm_map = ylm_map
    self.nan_mask = nnx.data(nan_mask)
    self.nu_dt = nu_dt
    self.steps = steps
    self.b = cx.SizedAxis('b', batch_size) if batch_size else cx.Scalar()

    self.grf = random_processes.GaussianRandomField(
        ylm_map=ylm_map,
        dt=1.0,
        sim_units=units.SI_UNITS,
        correlation_length=correlation_length,
        correlation_time=1.0,  # irrelevant.
        variance=1.0,  # irrelevant.
        rngs=rngs,
    )
    self.filter = spatial_filters.ExponentialModalFilter(
        ylm_map=ylm_map,
        attenuation=3.0,
        order=1,
    )

    if self.b.ndim:
      module_utils.vectorize_module(self.grf, {typing.Randomness: self.b})

    self._loss_evaluator = loss_evaluator
    self._metrics_evaluator = metrics_evaluator

    self.inputs_scaler, self.targets_scaler = _setup_scalers(
        self,
        rngs.target(),
        inputs_scaler,
        targets_scaler,
        n_stats_batches,
        skip_nans=True,
    )

  @property
  def loss_evaluator(self) -> evaluators.Evaluator:
    if self._loss_evaluator is not None:
      return self._loss_evaluator
    return evaluators.Evaluator(
        metrics={'mse': deterministic_losses.MSE()},
        aggregators=evaluators.aggregation.Aggregator(
            dims_to_reduce=('b', 'longitude', 'latitude'), skipna=True
        ),
    )

  @property
  def generalization_tasks(self) -> dict[str, 'synthetic_tasks.SyntheticTask']:
    if not self.enable_generalization:
      return {}
    return {
        'short_corr_len': MaskedDiffusionTask(
            ylm_map=self.ylm_map,
            nan_mask=self.nan_mask,
            nu_dt=self.nu_dt,
            steps=self.steps,
            correlation_length=0.3,
            batch_size=self.b.size if self.b.ndim else None,
            enable_generalization=False,
            inputs_scaler=self.inputs_scaler,
            targets_scaler=self.targets_scaler,
            rngs=nnx.Rngs(1),
        )
    }

  @property
  def metrics_evaluator(self) -> evaluators.Evaluator:
    if self._metrics_evaluator is not None:
      return self._metrics_evaluator

    scalar_agg = evaluators.aggregation.Aggregator(
        dims_to_reduce=('b', 'longitude', 'latitude'), skipna=True
    )
    zonal_agg = evaluators.aggregation.Aggregator(
        dims_to_reduce=('b', 'longitude'), skipna=True
    )
    map_agg = evaluators.aggregation.Aggregator(
        dims_to_reduce=('b',), skipna=True
    )
    return evaluators.Evaluator(
        metrics={
            'mse': deterministic_losses.MSE(),
            'mse_zonal': deterministic_losses.MSE(),
            'mse_map': deterministic_losses.MSE(),
            'bias_map': deterministic_metrics.Bias(),
            'bias_zonal': deterministic_metrics.Bias(),
        },
        aggregators={
            'mse': scalar_agg,
            'mse_zonal': zonal_agg,
            'mse_map': map_agg,
            'bias_map': map_agg,
            'bias_zonal': zonal_agg,
        },
    )

  @property
  def batch_axis(self) -> cx.Coordinate:
    return self.b

  @property
  def input_shapes(self) -> dict[str, cx.Field]:
    return {
        'u': cx.shape_struct_field(self.b, self.ylm_map.lon_lat_grid),
        'nan_mask': cx.shape_struct_field(self.b, self.ylm_map.lon_lat_grid),
    }

  @property
  def target_split_axes(self) -> dict[str, cx.Coordinate]:
    return {'u': cx.Scalar()}

  @nnx.jit
  def sample_batch(
      self, rng: jax.Array
  ) -> tuple[dict[str, cx.Field], dict[str, cx.Field]]:
    rng = cx.field(rng)
    if self.b.ndim:
      rng = cx.cmap(jax.random.split)(rng, self.b.size).tag(self.b)

    self.grf.unconditional_sample(rng)
    u_initial = self.grf.state_values(self.ylm_map.lon_lat_grid)

    mask = self.nan_mask
    nan_mask = mask.broadcast_like(cx.coords.compose(self.b, mask.coordinate))

    u_initial = cx.cmap(jnp.where)(nan_mask, jnp.nan, u_initial)
    inputs = {'nan_mask': nan_mask, 'u': u_initial}

    u_current = u_initial
    for _ in range(self.steps):
      u_fill_zero = cx.cmap(jnp.nan_to_num)(u_current)
      u_modal = self.ylm_map.to_modal(u_fill_zero)
      lap_modal = self.ylm_map.laplacian(u_modal)
      lap_modal = self.filter.filter_modal(lap_modal)
      lap_nodal = self.ylm_map.to_nodal(lap_modal)
      u_current = u_current + self.nu_dt * lap_nodal
      u_current = cx.cmap(jnp.where)(nan_mask, jnp.nan, u_current)

    targets = {'u': u_current}

    inputs = self.inputs_scaler(inputs)
    targets = self.targets_scaler(targets)

    return inputs, targets
