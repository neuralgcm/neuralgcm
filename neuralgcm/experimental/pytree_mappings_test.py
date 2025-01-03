# Copyright 2024 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests that pytree mappings produce outputs with expected shapes."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import chex
import dinosaur
from flax import nnx
import jax
from neuralgcm.experimental import coordinates
from neuralgcm.experimental import pytree_mappings
from neuralgcm.experimental import pytree_transforms
from neuralgcm.experimental import pytree_utils
from neuralgcm.experimental import standard_layers
from neuralgcm.experimental import towers
from neuralgcm.experimental import typing
import numpy as np


class EmbeddingsTest(parameterized.TestCase):
  """Tests embedding modules."""

  def _test_embedding_module(
      self,
      embedding_module: nnx.Module,
      inputs: typing.Pytree,
  ):
    embedded_features = embedding_module(inputs)
    actual = embedding_module.output_shapes
    expected = pytree_utils.shape_structure(embedded_features)
    chex.assert_trees_all_equal(actual, expected)

  def test_embedding(self):
    coords = coordinates.DinosaurCoordinates(
        horizontal=coordinates.LonLatGrid.T21(),
        vertical=coordinates.SigmaLevels.equidistant(4),
    )
    input_names = ('u', 'v')
    feature_module = pytree_transforms.PrognosticFeatures(
        coords, volume_field_names=input_names
    )
    tower_factory = functools.partial(
        towers.ColumnTower,
        column_net_factory=functools.partial(
            standard_layers.MlpUniform, hidden_size=6, n_hidden_layers=2
        ),
    )
    mapping_factory = functools.partial(
        pytree_mappings.ChannelMapping,
        tower_factory=tower_factory,
    )
    output_shapes = {
        'a': typing.ShapeFloatStruct((7,) + coords.horizontal.shape),
        'b': typing.ShapeFloatStruct((3,) + coords.horizontal.shape),
    }
    embedding = pytree_mappings.Embedding(
        output_shapes=output_shapes,
        feature_module=feature_module,
        mapping_factory=mapping_factory,
        rngs=nnx.Rngs(0),
    )
    test_inputs = {k: np.ones(coords.shape) for k in input_names}
    self._test_embedding_module(embedding, test_inputs)

  def test_coordinate_state_mapping(self):
    """Checks that CoordsStateMapping produces outputs with expected shapes."""
    coords = coordinates.DinosaurCoordinates(
        horizontal=coordinates.LonLatGrid.T21(),
        vertical=coordinates.SigmaLevels.equidistant(4),
    )
    input_names = ('u', 'v')
    feature_module = pytree_transforms.PrognosticFeatures(
        coords, volume_field_names=input_names
    )
    tower_factory = functools.partial(
        towers.ColumnTower,
        column_net_factory=functools.partial(
            standard_layers.MlpUniform, hidden_size=6, n_hidden_layers=2
        ),
    )
    mapping_factory = functools.partial(
        pytree_mappings.ChannelMapping,
        tower_factory=tower_factory,
    )
    embedding_factory = functools.partial(
        pytree_mappings.Embedding,
        feature_module=feature_module,
        mapping_factory=mapping_factory,
    )
    volume_field_names = ('u', 'div')
    surface_field_names = ('pressure',)
    state_mapping = pytree_mappings.CoordsStateMapping(
        coords=coords,
        surface_field_names=surface_field_names,
        volume_field_names=volume_field_names,
        embedding_factory=embedding_factory,
        rngs=nnx.Rngs(0),
    )
    test_inputs = {k: np.ones(coords.shape) for k in input_names}
    out = state_mapping(test_inputs)
    out_shape = pytree_utils.shape_structure(out)
    expected_shape = {
        'u': typing.ShapeFloatStruct(coords.shape),
        'div': typing.ShapeFloatStruct(coords.shape),
        'pressure': typing.ShapeFloatStruct((1,) + coords.horizontal.shape),
    }
    chex.assert_trees_all_equal(out_shape, expected_shape)

  def test_masked_volume_embedding(self):
    n_levels = 2
    coords = coordinates.DinosaurCoordinates(
        horizontal=coordinates.LonLatGrid.T21(),
        vertical=coordinates.LayerLevels(n_layers=n_levels),
    )
    data_shape = coords.shape
    sub_feature_module_mock = pytree_transforms.PrognosticFeatures(
        coords=coords,
        volume_field_names=('u', 'v'),
    )
    feature_names = {'u': 'embedded_a', 'v': 'embedded_b'}
    initializers = {'u': np.ones(coords.shape), 'v': np.ones(coords.shape)}
    masks = {
        'u': ((np.arange(np.prod(data_shape)) % 15) / 15).reshape(data_shape),
        'v': ((np.arange(np.prod(data_shape)) % 15) / 15).reshape(data_shape),
    }
    masks = {k: np.where(v > 0.5, 1.0, 0.0) for k, v in masks.items()}
    embedding_module = pytree_mappings.MaskedVolumeFeatureEmbedding(
        variables_to_embed=['u', 'v'],
        mask_values=masks,
        coords=coords,
        feature_names=feature_names,
        feature_module=sub_feature_module_mock,
        initializer=initializers,
        rngs=nnx.Rngs(0),
    )
    test_inputs = {
        'u': 0.5 * np.ones(coords.shape),
        'v': 0.5 * np.ones(coords.shape),
    }
    self._test_embedding_module(embedding_module, test_inputs)

    with self.subTest('output_masked_appropriately'):
      expected = {
          feature_names[k]: np.where(v > 0.5, 0.5, 1.5)
          for k, v in masks.items()
      }
      outputs = embedding_module(test_inputs)
      chex.assert_trees_all_close(outputs, expected)

  def test_masked_surface_embedding(self):
    n_levels = 1
    coords = coordinates.DinosaurCoordinates(
        horizontal=coordinates.LonLatGrid.T21(),
        vertical=coordinates.LayerLevels(n_layers=n_levels),
    )
    ylm_impl = dinosaur.spherical_harmonic.FastSphericalHarmonics
    grid = coordinates.LonLatGrid.T21(spherical_harmonics_impl=ylm_impl)
    data_shape = coords.shape
    mask_shape = coords.horizontal.shape
    feature_module = pytree_transforms.PrognosticFeatures(
        coords=coords,
        volume_field_names=('u', 'v'),
    )
    feature_names = {'u': 'embedded_a', 'v': 'embedded_b'}
    initializers = {'u': np.ones(mask_shape), 'v': np.ones(mask_shape)}
    masks = {
        'u': ((np.arange(np.prod(mask_shape)) % 15) / 15).reshape(data_shape),
        'v': ((np.arange(np.prod(mask_shape)) % 15) / 15).reshape(data_shape),
    }
    masks = {k: np.where(v > 0.5, 1.0, 0.0) for k, v in masks.items()}
    embedding_module = pytree_mappings.MaskedSurfaceFeatureEmbedding(
        variables_to_embed=['u', 'v'],
        mask_values=masks,
        grid=grid,
        feature_names=feature_names,
        feature_module=feature_module,
        initializer=initializers,
        rngs=nnx.Rngs(0),
    )
    test_inputs = {
        'u': 0.5 * np.ones(data_shape) * masks['u'],
        'v': 0.5 * np.ones(data_shape) * masks['v'],
    }
    self._test_embedding_module(embedding_module, test_inputs)


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
