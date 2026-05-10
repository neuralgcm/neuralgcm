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
"""Visualization utilities for synthetic learning tasks."""

import itertools
from typing import Any, Sequence

import coordax as cx
import jax
from matplotlib import pyplot as plt
import mpl_toolkits.axes_grid1
from neuralgcm.experimental.core import xarray_utils
from neuralgcm.experimental.synthetic_learning_tasks import synthetic_tasks
import numpy as np
import xarray


def plot_scalar_metrics(
    data: xarray.Dataset,
    task: Any,
    task_names: Sequence[str] | None = None,
    metrics: Sequence[str] | None = None,
    variables: Sequence[str] | None = None,
    figsize: tuple[float, float] | None = None,
):
  """Plots scalar metrics over training/evaluation steps.

  Args:
    data: Dataset containing history or evaluation results.
    task: The SyntheticTask instance.
    task_names: List of task names to plot (e.g., ['train', 'generalization']).
      If None, inferred from task.
    metrics: List of metrics to plot (e.g., ['mse', 'loss']). If None, inferred.
    variables: List of target variables to plot. If None, inferred from task.
    figsize: Figure size.
  """
  if task_names is None:
    task_names = ['train'] + list(task.generalization_tasks.keys())
  if variables is None:
    variables = list(task.target_split_axes.keys())
  if metrics is None:
    if (
        hasattr(task, 'metrics_evaluator')
        and task.metrics_evaluator is not None
    ):
      metrics = list(task.metrics_evaluator.metrics.keys())
    else:
      metrics = ['loss', 'mse']  # Default common metrics

  n_rows = len(task_names)
  n_cols = len(variables) + 1  # +1 for loss

  if figsize is None:
    figsize = (6 * n_cols, 5 * n_rows)

  _, axarray = plt.subplots(n_rows, n_cols, figsize=figsize, sharex='col')
  if n_rows == 1:
    axarray = axarray[np.newaxis, :]

  for i, task_name in enumerate(task_names):
    # Plot Loss
    ax = axarray[i, 0]
    loss_key = f'{task_name}/loss/total'
    if loss_key in data:
      loss_val = data[loss_key]
      if loss_val.ndim == 1:
        loss_val.plot(x='eval_step', marker='o', label='Loss', ax=ax)
      elif loss_val.ndim == 2:
        loss_val.plot.line(x='eval_step', ax=ax)

    if task_name == 'train' and 'train_loss' in data:
      train_loss = data['train_loss']
      if train_loss.ndim == 1:
        train_loss.plot(x='train_step', label='Train loss', ax=ax, zorder=1)
      elif train_loss.ndim == 2:
        train_loss.plot.line(x='train_step', ax=ax)

    ax.set_title(f'{task_name} Loss')
    ax.legend()

    # Plot variables
    for j, var in enumerate(variables):
      ax = axarray[i, j + 1]
      for metric in metrics:
        if metric == 'loss' or metric.endswith('_binned'):
          continue
        key = f'{task_name}/{metric}/{var}'
        if key in data:
          val = data[key]
          # Skip zonal and map metrics
          if 'latitude' in val.dims or 'longitude' in val.dims:
            continue
          # Handle binned metrics if present
          binned_key = f'{task_name}/{metric}_binned/{var}'
          binned_val = data[binned_key] if binned_key in data else None

          extra_dims = [d for d in val.dims if d != 'eval_step']
          if extra_dims:
            val.plot(
                x='eval_step',
                hue=extra_dims[0],
                marker='o',
                label=metric,
                ax=ax,
            )
          else:
            val.plot(x='eval_step', marker='o', label=metric, ax=ax)

          if binned_val is not None and 'latitude_bin' in binned_val.dims:
            for lat_bin in binned_val.latitude_bin:
              sliced_val = binned_val.sel(latitude_bin=lat_bin)
              if sliced_val.ndim == 1:
                sliced_val.plot(
                    x='eval_step',
                    marker='s',
                    label=f'lat_bin={lat_bin.values}',
                    ax=ax,
                    zorder=1,
                )
      ax.set_title(f'{task_name} {var.capitalize()}')
      _, labels = ax.get_legend_handles_labels()
      if labels:
        ax.legend()

  plt.tight_layout()
  plt.show()


def plot_zonal_metrics(
    data: xarray.Dataset,
    task: Any,
    task_names: Sequence[str] | None = None,
    metrics: Sequence[str] | None = None,
    variables: Sequence[str] | None = None,
    step_idx: int = -1,
    figsize: tuple[float, float] | None = None,
):
  """Plots zonal metrics against latitude."""
  if task_names is None:
    task_names = ['train'] + list(task.generalization_tasks.keys())
  if variables is None:
    variables = list(task.target_split_axes.keys())
  if metrics is None:
    if (
        hasattr(task, 'metrics_evaluator')
        and task.metrics_evaluator is not None
    ):
      metrics = [
          k for k in task.metrics_evaluator.metrics.keys() if 'zonal' in k
      ]
    else:
      metrics = ['mse_zonal', 'bias_zonal']

  n_rows = len(task_names)
  n_cols = len(metrics) * len(variables)

  if figsize is None:
    figsize = (6 * n_cols, 5 * n_rows)

  _, axarray = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True)
  if n_rows == 1:
    axarray = axarray[np.newaxis, :]
  if n_cols == 1:
    axarray = axarray[:, np.newaxis]

  for i, task_name in enumerate(task_names):
    col_idx = 0
    for metric, var in itertools.product(metrics, variables):
      ax = axarray[i, col_idx]
      key = f'{task_name}/{metric}/{var}'
      if key in data:
        val = data[key]
        if 'eval_step' in val.dims and step_idx is not None:
          val = val.isel(eval_step=step_idx)
        val.plot(x='latitude', marker='o', ax=ax)

      if i == 0:
        ax.set_title(f'{var.capitalize()} {metric}')
      else:
        ax.set_title('')

      if col_idx == n_cols - 1:
        ax.set_ylabel(task_name, rotation=270, labelpad=20)
        ax.yaxis.set_label_position('right')
      col_idx += 1

  plt.tight_layout()
  plt.show()


def plot_map_metrics(
    data: xarray.Dataset,
    task: synthetic_tasks.SyntheticTask,
    task_names: Sequence[str] | None = None,
    metrics: Sequence[str] | None = None,
    variables: Sequence[str] | None = None,
    step_idx: int = -1,
    figsize: tuple[float, float] | None = None,
):
  """Plots 2D maps for metrics."""
  if task_names is None:
    task_names = ['train'] + list(task.generalization_tasks.keys())
  if variables is None:
    variables = list(task.target_split_axes.keys())
  if metrics is None:
    if (
        hasattr(task, 'metrics_evaluator')
        and task.metrics_evaluator is not None
    ):
      metrics = [k for k in task.metrics_evaluator.metrics.keys() if 'map' in k]
    else:
      metrics = ['mse_map', 'bias_map']

  n_rows = len(task_names)
  n_cols = len(metrics) * len(variables)

  if figsize is None:
    figsize = (6 * n_cols, 5 * n_rows)

  _, axarray = plt.subplots(
      n_rows, n_cols, figsize=figsize, sharex=True, sharey=True
  )
  if n_rows == 1:
    axarray = axarray[np.newaxis, :]
  if n_cols == 1:
    axarray = axarray[:, np.newaxis]

  for i, task_name in enumerate(task_names):
    col_idx = 0
    for metric, var in itertools.product(metrics, variables):
      ax = axarray[i, col_idx]
      key = f'{task_name}/{metric}/{var}'
      if key in data:
        val = data[key]
        if 'eval_step' in val.dims and step_idx is not None:
          val = val.isel(eval_step=step_idx)
        val.plot(x='longitude', y='latitude', ax=ax)

      if i == 0:
        ax.set_title(f'{var.capitalize()} {metric}')
      else:
        ax.set_title('')

      if col_idx == n_cols - 1:
        ax.set_ylabel(task_name, rotation=270, labelpad=20)
        ax.yaxis.set_label_position('right')
      col_idx += 1

  plt.tight_layout()
  plt.show()


def plot_snapshots(
    eval_results: xarray.Dataset,
    task: Any,
    task_names: Sequence[str] | None = None,
    variables: Sequence[str] | None = None,
    sample_idx: int = 0,
    figsize: tuple[float, float] | None = None,
):
  """Plots inputs, targets, predictions, and errors for a sample."""
  if task_names is None:
    task_names = ['train'] + list(task.generalization_tasks.keys())
  if variables is None:
    variables = list(task.target_split_axes.keys())

  row_configs = list(itertools.product(task_names, variables))
  n_rows = len(row_configs)
  n_cols = 4  # Inputs, Targets, Predictions, Errors

  if figsize is None:
    figsize = (18, 2.5 * n_rows)

  fig, axarray = plt.subplots(
      n_rows, n_cols, figsize=figsize, sharex=True, sharey=True
  )
  if n_rows == 1:
    axarray = axarray[np.newaxis, :]

  def add_aligned_colorbar(plot_obj, ax):
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    fig.colorbar(plot_obj, cax=cax)

  for i, (task_name, var) in enumerate(row_configs):
    ax1, ax2, ax3, ax4 = axarray[i]

    # 1. Inputs (Try to plot wind quiver if u and v are present)
    u_key = f'{task_name}/inputs/u'
    v_key = f'{task_name}/inputs/v'
    if u_key in eval_results and v_key in eval_results:
      u_arr = eval_results[u_key]
      v_arr = eval_results[v_key]
      u = u_arr.isel(batch=sample_idx)
      v = v_arr.isel(batch=sample_idx)
      wind_ds = xarray.Dataset({'u': u, 'v': v}).thin(longitude=2, latitude=2)
      wind_ds['wind_speed'] = np.sqrt(wind_ds['u'] ** 2 + wind_ds['v'] ** 2)
      wind_ds.plot.quiver(
          x='longitude',
          y='latitude',
          u='u',
          v='v',
          hue='wind_speed',
          ax=ax1,
          add_guide=False,
      )
    else:
      # Just plot the first input as a map if available
      first_input_key = next(
          (k for k in eval_results if f'{task_name}/inputs/' in str(k)), None
      )
      if first_input_key:
        val_arr = eval_results[first_input_key]
        val = val_arr.isel(batch=sample_idx)
        val.plot(x='longitude', y='latitude', ax=ax1)
      else:
        ax1.text(0.5, 0.5, 'No plottable inputs', ha='center', va='center')
    ax1.set_aspect('equal')

    # 2. Targets, 3. Predictions, 4. Errors
    target_key = f'{task_name}/targets/{var}'
    pred_key = f'{task_name}/predictions/{var}'
    target_arr = (
        eval_results[target_key] if target_key in eval_results else None
    )
    pred_arr = eval_results[pred_key] if pred_key in eval_results else None

    if target_arr is not None and pred_arr is not None:
      target = target_arr.isel(batch=sample_idx)
      pred = pred_arr.isel(batch=sample_idx)

      # 2. Targets
      im2 = target.plot(x='longitude', y='latitude', ax=ax2, add_colorbar=False)
      t_vmin, t_vmax = im2.get_clim()
      cmap = im2.get_cmap()
      add_aligned_colorbar(im2, ax2)
      ax2.set_aspect('equal')

      # 3. Predictions
      im3 = pred.plot(
          x='longitude',
          y='latitude',
          ax=ax3,
          add_colorbar=False,
          vmin=t_vmin,
          vmax=t_vmax,
          cmap=cmap,
      )
      add_aligned_colorbar(im3, ax3)
      ax3.set_aspect('equal')

      # 4. Errors
      scale = np.abs(target).quantile(0.99).values
      error_pct = 100 * np.abs(pred - target) / (scale + 1e-8)
      im4 = error_pct.plot(
          x='longitude',
          y='latitude',
          ax=ax4,
          vmin=0,
          vmax=20,
          cmap='Reds',
          add_colorbar=False,
      )
      add_aligned_colorbar(im4, ax4)
      ax4.set_aspect('equal')
    else:
      ax2.text(0.5, 0.5, 'Missing data', ha='center', va='center')
      ax3.text(0.5, 0.5, 'Missing data', ha='center', va='center')
      ax4.text(0.5, 0.5, 'Missing data', ha='center', va='center')

    if i == 0:
      ax1.set_title('Inputs')
      ax2.set_title('Target')
      ax3.set_title('Prediction')
      ax4.set_title('Error (% of scale)')
    else:
      for ax in [ax1, ax2, ax3, ax4]:
        ax.set_title('')

    row_label = f'{var.capitalize()}\n({task_name})'
    ax4.set_ylabel(row_label, rotation=270, labelpad=30)
    ax4.yaxis.set_label_position('right')

  plt.tight_layout()
  plt.show()


def plot_task_samples(
    task: Any, rng: Any, task_names: Sequence[str] | None = None
):
  """Visualizes task inputs and targets for different generalizations.

  Args:
    task: The SyntheticTask instance.
    rng: JAX random key.
    task_names: List of task names to plot. If None, inferred from task.
  """
  if task_names is None:
    task_names = ['train'] + list(task.generalization_tasks.keys())

  sub_tasks = [task] + [task.generalization_tasks[k] for k in task_names[1:]]
  n_rows = len(task_names)

  # Inspect one sample to determine columns
  inputs, targets = sub_tasks[0].sample_batch(rng)
  inputs_ds = xarray_utils.fields_to_xarray(inputs)

  input_vars = list(inputs_ds.data_vars)
  has_wind = 'u' in input_vars and 'v' in input_vars

  if has_wind:
    col_vars = ['wind_quiver'] + [v for v in input_vars if v not in ['u', 'v']]
  else:
    col_vars = list(input_vars)

  target_vars = list(targets.keys())
  col_vars += target_vars

  n_cols = len(col_vars)

  _, axarray = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
  if n_rows == 1:
    axarray = axarray[np.newaxis, :]
  if n_cols == 1:
    axarray = axarray[:, np.newaxis]

  for i, (row_name, sub_task) in enumerate(zip(task_names, sub_tasks)):
    ax_row = axarray[i]
    task_rng = jax.random.fold_in(rng, i)
    inputs, targets = sub_task.sample_batch(task_rng)
    inputs_ds = xarray_utils.fields_to_xarray(inputs)

    current_has_wind = 'u' in inputs_ds.data_vars and 'v' in inputs_ds.data_vars
    if current_has_wind:
      inputs_ds['wind_speed'] = np.sqrt(
          inputs_ds['u'] ** 2 + inputs_ds['v'] ** 2
      )

    for j, col_var in enumerate(col_vars):
      ax = ax_row[j]

      if col_var == 'wind_quiver':
        if current_has_wind:
          inputs_ds.thin(longitude=2, latitude=2).plot.quiver(
              x='longitude', y='latitude', u='u', v='v', hue='wind_speed', ax=ax
          )
        else:
          ax.text(0.5, 0.5, 'Wind not available', ha='center', va='center')
        if i == 0:
          ax.set_title('Wind vector')
      elif col_var in inputs_ds.data_vars:
        val = inputs_ds[col_var]
        slicing = {
            d: -1 for d in val.dims if d not in ['latitude', 'longitude']
        }
        if slicing:
          val = val.isel(slicing)
        val.plot(x='longitude', y='latitude', ax=ax)
        if i == 0:
          ax.set_title(col_var)
      elif col_var in targets:
        val = targets[col_var]
        if isinstance(val, cx.Field):
          val = val.to_xarray()
        slicing = {
            d: -1 for d in val.dims if d not in ['latitude', 'longitude']
        }
        if slicing:
          val = val.isel(slicing)
        val.plot(x='longitude', y='latitude', ax=ax)
        if i == 0:
          ax.set_title(col_var)

      if j == n_cols - 1:
        ax.set_ylabel(row_name, rotation=270, labelpad=80)
        ax.yaxis.set_label_position('right')

  plt.tight_layout(rect=[0, 0, 0.95, 1], h_pad=3.0)
  plt.show()
