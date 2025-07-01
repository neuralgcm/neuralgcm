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

"""Tests that atmospheric transforms produce outputs with expected structure."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import coordax as cx
import jax
import jax_datetime as jdt
from neuralgcm.experimental.atmosphere import transforms as atmos_transforms
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import pytree_utils
from neuralgcm.experimental.core import spherical_transforms
from neuralgcm.experimental.core import transforms
from neuralgcm.experimental.core import typing
import numpy as np


class AtmosphereTransformsTest(parameterized.TestCase):
  """Tests atmospheric transforms."""

  def _test_feature_module(
      self,
      feature_module: transforms.Transform,
      inputs: typing.Pytree,
  ):
    features = feature_module(inputs)
    expected = pytree_utils.shape_structure(features)
    input_shapes = pytree_utils.shape_structure(inputs)
    actual = feature_module.output_shapes(input_shapes)
    chex.assert_trees_all_equal(actual, expected)

  def test_velocity_and_prognostics_with_modal_gradients(self):
    sigma = coordinates.SigmaLevels.equidistant(4)
    ylm_transform = spherical_transforms.SphericalHarmonicsTransform(
        lon_lat_grid=coordinates.LonLatGrid.T21(),
        ylm_grid=coordinates.SphericalHarmonicGrid.T21(),
        partition_schema_key=None,
        mesh=parallelism.Mesh(),
    )
    with_gradients_transform = transforms.ToModalWithFilteredGradients(
        ylm_transform,
        filter_attenuations=[2.0],
    )
    features_grads = atmos_transforms.VelocityAndPrognosticsWithModalGradients(
        ylm_transform,
        volume_field_names=(
            'u',
            'v',
            'vorticity',
        ),
        surface_field_names=('lsp',),
        compute_gradients_transform=with_gradients_transform,
    )
    modal_grid = ylm_transform.modal_grid
    shape_3d = sigma.shape + modal_grid.shape
    inputs = {
        'u': cx.wrap(np.ones(shape_3d), sigma, modal_grid),
        'v': cx.wrap(np.ones(shape_3d), sigma, modal_grid),
        'vorticity': cx.wrap(np.ones(shape_3d), sigma, modal_grid),
        'divergence': cx.wrap(np.ones(shape_3d), sigma, modal_grid),
        'lsp': cx.wrap(np.ones(modal_grid.shape), modal_grid),
        'time': cx.wrap(jdt.to_datetime('2025-01-09T15:00')),
    }
    self._test_feature_module(features_grads, inputs)

  def test_pressure_features(self):
    sigma = coordinates.SigmaLevels.equidistant(8)
    ylm_grid = coordinates.SphericalHarmonicGrid.T21()
    ylm_transform = spherical_transforms.SphericalHarmonicsTransform(
        lon_lat_grid=coordinates.LonLatGrid.T21(),
        ylm_grid=ylm_grid,
        partition_schema_key=None,
        mesh=parallelism.Mesh(None),
    )
    pressure_features = atmos_transforms.PressureOnSigmaFeatures(
        ylm_transform=ylm_transform,
        sigma=sigma,
    )
    inputs = {
        'log_surface_pressure': cx.wrap(np.ones(ylm_grid.shape), ylm_grid),
    }
    self._test_feature_module(pressure_features, inputs)


if __name__ == '__main__':
  jax.config.update('jax_traceback_filtering', 'off')
  absltest.main()
