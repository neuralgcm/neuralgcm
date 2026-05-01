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
"""Tests for plot_utils."""

from absl.testing import absltest
from absl.testing import parameterized
import coordax as cx
import jax
import matplotlib
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.synthetic_learning_tasks import plot_utils
import numpy as np
import xarray


matplotlib.use('Agg')  # Use non-interactive backend
# pylint: disable=broad-exception-caught


class DummyTask:

  def __init__(self):
    self.generalization_tasks = {'gen1': self}
    self.target_split_axes = {'v1': None, 'v2': None}

  def sample_batch(self, rng):
    grid = coordinates.LonLatGrid.TL31()
    lats = grid.fields['latitude'].data
    lons = grid.fields['longitude'].data
    n_lats = lats.shape[0]
    n_lons = lons.shape[0]
    inputs = {
        'u': cx.field(np.random.rand(n_lons, n_lats), grid),
        'v': cx.field(np.random.rand(n_lons, n_lats), grid),
    }
    targets = {
        'vorticity': cx.field(np.random.rand(n_lons, n_lats), grid),
        'divergence': cx.field(np.random.rand(n_lons, n_lats), grid),
    }
    return inputs, targets


class PlotUtilsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.task = DummyTask()
    grid = coordinates.LonLatGrid.TL63()
    lats = grid.fields['latitude'].data
    lons = grid.fields['longitude'].data
    n_lats = lats.shape[0]
    n_lons = lons.shape[0]
    self.data = xarray.Dataset({
        'train/loss/total': xarray.DataArray(
            np.random.rand(5), dims=['eval_step']
        ),
        'train_loss': xarray.DataArray(np.random.rand(10), dims=['train_step']),
        'train/mse/v1': xarray.DataArray(np.random.rand(5), dims=['eval_step']),
        'gen1/loss/total': xarray.DataArray(
            np.random.rand(5), dims=['eval_step']
        ),
        'train/mse_zonal/v1': xarray.DataArray(
            np.random.rand(5, n_lats),
            dims=['eval_step', 'latitude'],
            coords={'latitude': lats},
        ),
        'train/mse_map/v1': xarray.DataArray(
            np.random.rand(5, n_lats, n_lons),
            dims=['eval_step', 'latitude', 'longitude'],
            coords={'latitude': lats, 'longitude': lons},
        ),
        'train/inputs/u': xarray.DataArray(
            np.random.rand(4, n_lats, n_lons),
            dims=['batch', 'latitude', 'longitude'],
            coords={'latitude': lats, 'longitude': lons},
        ),
        'train/inputs/v': xarray.DataArray(
            np.random.rand(4, n_lats, n_lons),
            dims=['batch', 'latitude', 'longitude'],
            coords={'latitude': lats, 'longitude': lons},
        ),
        'train/targets/v1': xarray.DataArray(
            np.random.rand(4, n_lats, n_lons),
            dims=['batch', 'latitude', 'longitude'],
            coords={'latitude': lats, 'longitude': lons},
        ),
        'train/predictions/v1': xarray.DataArray(
            np.random.rand(4, n_lats, n_lons),
            dims=['batch', 'latitude', 'longitude'],
            coords={'latitude': lats, 'longitude': lons},
        ),
    })

  def test_plot_scalar_metrics(self):
    try:
      plot_utils.plot_scalar_metrics(self.data, self.task)
    except Exception as e:
      self.fail(f'plot_scalar_metrics raised {e} unexpectedly!')

  def test_plot_zonal_metrics(self):
    try:
      plot_utils.plot_zonal_metrics(self.data, self.task, step_idx=0)
    except Exception as e:
      self.fail(f'plot_zonal_metrics raised {e} unexpectedly!')

  def test_plot_map_metrics(self):
    try:
      plot_utils.plot_map_metrics(self.data, self.task, step_idx=0)
    except Exception as e:
      self.fail(f'plot_map_metrics raised {e} unexpectedly!')

  def test_plot_snapshots(self):
    try:
      plot_utils.plot_snapshots(self.data, self.task, sample_idx=0)
    except Exception as e:
      self.fail(f'plot_snapshots raised {e} unexpectedly!')

  def test_plot_task_samples(self):
    try:
      rng = jax.random.key(0)
      plot_utils.plot_task_samples(self.task, rng)
    except Exception as e:
      self.fail(f'plot_task_samples raised {e} unexpectedly!')


if __name__ == '__main__':
  absltest.main()
