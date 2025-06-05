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

"""Tests that feature transforms produce outputs with expected structure."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
from flax import nnx
import jax
import jax_datetime as jdt
from neuralgcm.experimental import coordax as cx
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import dynamic_io
from neuralgcm.experimental.core import feature_transforms
from neuralgcm.experimental.core import orographies
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import pytree_utils
from neuralgcm.experimental.core import random_processes
from neuralgcm.experimental.core import spherical_transforms
from neuralgcm.experimental.core import transforms
from neuralgcm.experimental.core import typing
from neuralgcm.experimental.core import units
import numpy as np


class FeatureTransformsTest(parameterized.TestCase):
  """Tests feature transforms."""

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

  @parameterized.named_parameters(
      dict(testcase_name='t21', grid=coordinates.LonLatGrid.T21()),
      dict(testcase_name='tl31', grid=coordinates.LonLatGrid.TL31()),
  )
  def test_radiation_features(self, grid):
    radiation_features = feature_transforms.RadiationFeatures(grid=grid)
    self._test_feature_module(
        radiation_features,
        {'time': cx.wrap(jdt.to_datetime('2025-01-09T15:00'))},
    )

  def test_latitude_features(self):
    grid = coordinates.LonLatGrid.T21()
    latitude_features = feature_transforms.LatitudeFeatures(grid=grid)
    self._test_feature_module(latitude_features, None)

  def test_orography_features(self):
    ylm_transform = spherical_transforms.SphericalHarmonicsTransform(
        lon_lat_grid=coordinates.LonLatGrid.T21(),
        ylm_grid=coordinates.SphericalHarmonicGrid.T21(),
        partition_schema_key=None,
        mesh=parallelism.Mesh(),
    )
    orography = orographies.ModalOrography(
        ylm_transform=ylm_transform,
        rngs=None,
    )
    orography_features = feature_transforms.OrographyFeatures(
        orography_module=orography,
    )
    self._test_feature_module(orography_features, None)

  def test_orography_with_grads_features(self):
    ylm_transform = spherical_transforms.SphericalHarmonicsTransform(
        lon_lat_grid=coordinates.LonLatGrid.T21(),
        ylm_grid=coordinates.SphericalHarmonicGrid.T21(),
        partition_schema_key=None,
        mesh=parallelism.Mesh(),
    )
    orography = orographies.ModalOrography(
        ylm_transform=ylm_transform,
        rngs=None,
    )
    orography_features = feature_transforms.OrographyWithGradsFeatures(
        orography_module=orography,
        compute_gradients_transform=transforms.ToModalWithFilteredGradients(
            ylm_transform,
            filter_attenuations=[2.0],
        ),
    )
    self._test_feature_module(orography_features, None)

  def test_dynamic_input_features(self):
    grid = coordinates.LonLatGrid.T21()
    dynamic_input = dynamic_io.DynamicInputSlice(
        keys_to_coords={'a': grid, 'b': grid, 'c': grid},
        observation_key='abc',
    )
    expand_dims = lambda x: np.expand_dims(x, axis=(1, 2))
    data = {
        'abc': {
            'a': expand_dims(np.arange(2)) * np.ones(grid.shape),
            'b': expand_dims(np.arange(2)) * np.zeros(grid.shape),
            'c': expand_dims(np.arange(2)) * np.ones(grid.shape),
        }
    }
    timedelta = coordinates.TimeDelta(np.arange(2, dtype='timedelta64[h]'))
    grid_trajectory = cx.compose_coordinates(timedelta, grid)
    time = jdt.to_datetime('2000-01-01') + jdt.to_timedelta(
        12, 'h'
    ) * np.arange(timedelta.shape[0])
    in_data = jax.tree.map(lambda x: cx.wrap(x, grid_trajectory), data)
    in_data['abc']['time'] = cx.wrap(time, timedelta)
    dynamic_input.update_dynamic_inputs(in_data)
    with self.subTest('two_keys'):
      dynamic_input_features = feature_transforms.DynamicInputFeatures(
          ('a', 'b'), dynamic_input
      )
      self._test_feature_module(
          dynamic_input_features,
          {'time': cx.wrap(jdt.to_datetime('2000-01-01T06'))},
      )

  def test_dynamic_input_features_inder_jit(self):
    grid = coordinates.LonLatGrid.T21()
    dynamic_input = dynamic_io.DynamicInputSlice(
        keys_to_coords={'a': grid, 'b': grid, 'c': grid},
        observation_key='abc',
    )
    expand_dims = lambda x: np.expand_dims(x, axis=(1, 2))
    data = {
        'abc': {
            'a': expand_dims(np.arange(2)) * np.ones(grid.shape),
            'b': expand_dims(np.arange(2)) * np.zeros(grid.shape),
            'c': expand_dims(np.arange(2)) * np.ones(grid.shape),
        }
    }
    timedelta = coordinates.TimeDelta(np.arange(2, dtype='timedelta64[h]'))
    grid_trajectory = cx.compose_coordinates(timedelta, grid)
    time = jdt.to_datetime('2000-01-01') + jdt.to_timedelta(
        12, 'h'
    ) * np.arange(timedelta.shape[0])
    in_data = jax.tree.map(lambda x: cx.wrap(x, grid_trajectory), data)
    in_data['abc']['time'] = cx.wrap(time, timedelta)

    @nnx.jit
    def run(module, inputs, dynamic_inputs):
      module.dynamic_input_module.update_dynamic_inputs(dynamic_inputs)
      return module(inputs)

    dynamic_input_features = feature_transforms.DynamicInputFeatures(
        ('a', 'b'), dynamic_input
    )
    run(
        dynamic_input_features,
        {'time': cx.wrap(jdt.to_datetime('2000-01-01T06'))},
        in_data,
    )

  def test_coord_features(self):
    z = cx.SizedAxis('z', 8)
    grid = coordinates.LonLatGrid.T21()
    coords = {
        'surface_embedings': cx.compose_coordinates(z, grid),
        'land_sea_mask': grid,
    }
    coord_features = feature_transforms.CoordFeatures(coords, rngs=nnx.Rngs(1))
    self._test_feature_module(coord_features, None)

  @parameterized.named_parameters(
      dict(
          testcase_name='T21_grid',
          ylm_transform=spherical_transforms.SphericalHarmonicsTransform(
              lon_lat_grid=coordinates.LonLatGrid.T21(),
              ylm_grid=coordinates.SphericalHarmonicGrid.T21(),
              partition_schema_key=None,
              mesh=parallelism.Mesh(None),
          ),
      ),
  )
  def test_randomness_features(self, ylm_transform):
    with self.subTest('gaussian_random_field'):
      random_process = random_processes.GaussianRandomField(
          ylm_transform=ylm_transform,
          dt=1.0,
          sim_units=units.DEFAULT_UNITS,
          correlation_time=1.0,
          correlation_length=1.0,
          variance=1.0,
          rngs=nnx.Rngs(0),
      )
      random_process.unconditional_sample(jax.random.key(0))
      randomness_features = feature_transforms.RandomnessFeatures(
          random_process=random_process,
          grid=ylm_transform.nodal_grid,
      )
      self._test_feature_module(randomness_features, None)

    with self.subTest('batched_gaussian_random_fields'):
      random_process = random_processes.BatchGaussianRandomField(
          ylm_transform=ylm_transform,
          dt=1.0,
          sim_units=units.DEFAULT_UNITS,
          correlation_times=[1.0, 2.0],
          correlation_lengths=[0.6, 0.9],
          variances=[1.0, 1.0],
          rngs=nnx.Rngs(0),
      )
      random_process.unconditional_sample(jax.random.key(0))
      randomness_features = feature_transforms.RandomnessFeatures(
          random_process=random_process,
          grid=ylm_transform.nodal_grid,
      )
      self._test_feature_module(randomness_features, None)


if __name__ == '__main__':
  jax.config.update('jax_traceback_filtering', 'off')
  absltest.main()
