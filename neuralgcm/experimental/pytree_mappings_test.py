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

"""Tests that pytree mappings produce outputs with expected shapes."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import chex
from flax import nnx
import jax
from neuralgcm.experimental import pytree_mappings
from neuralgcm.experimental import pytree_transforms
from neuralgcm.experimental import towers
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import pytree_utils
from neuralgcm.experimental.core import standard_layers
from neuralgcm.experimental.core import typing
import numpy as np


class EmbeddingsTest(parameterized.TestCase):
  """Tests embedding modules."""

  def setUp(self):
    """Set up common parameters and configurations for tests."""
    super().setUp()
    self.coords = coordinates.DinosaurCoordinates(
        horizontal=coordinates.LonLatGrid.T21(),
        vertical=coordinates.SigmaLevels.equidistant(4),
    )
    self.tower_factory = functools.partial(
        towers.ColumnTower,
        column_net_factory=functools.partial(
            standard_layers.MlpUniform, hidden_size=6, n_hidden_layers=2
        ),
    )
    self.mapping_factory = functools.partial(
        pytree_mappings.ChannelMapping,
        tower_factory=self.tower_factory,
    )
    self.output_shapes = {
        'a': typing.ShapeFloatStruct((7,) + self.coords.horizontal.shape),
        'b': typing.ShapeFloatStruct((3,) + self.coords.horizontal.shape),
    }

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
    input_names = ('u', 'v')
    test_inputs = {k: np.ones(self.coords.shape) for k in input_names}
    input_state_shapes = pytree_utils.shape_structure(test_inputs)
    feature_module = pytree_transforms.PrognosticFeatures(input_names)
    embedding = pytree_mappings.Embedding(
        output_shapes=self.output_shapes,
        feature_module=feature_module,
        mapping_factory=self.mapping_factory,
        input_state_shapes=input_state_shapes,
        rngs=nnx.Rngs(0),
        mesh=parallelism.Mesh(None),
    )
    self._test_embedding_module(embedding, test_inputs)

  def test_masked_embedding(self):
    """Tests MaskedEmbedding's handling of all-NaN inputs with/without mask."""
    input_names = ('u',)
    test_inputs = {}
    test_inputs['u'] = np.ones(self.coords.shape)
    test_inputs['u'][0, 0, :] = np.nan
    test_inputs['u'][1, :, 2] = np.nan
    input_state_shapes = pytree_utils.shape_structure(test_inputs)
    feature_module = pytree_transforms.PrognosticFeatures(input_names)

    embedding = pytree_mappings.MaskedEmbedding(
        output_shapes=self.output_shapes,
        feature_module=feature_module,
        mapping_factory=self.mapping_factory,
        input_state_shapes=input_state_shapes,
        rngs=nnx.Rngs(0),
        mesh=parallelism.Mesh(None),
    )


    embedded_features_with_nans = embedding(test_inputs)
    has_nans_tree = jax.tree.map(
        lambda x: np.any(np.isnan(x)), embedded_features_with_nans
    )
    some_leaves_have_nans = any(jax.tree_util.tree_leaves(has_nans_tree))
    self.assertTrue(some_leaves_have_nans, 'Outputs should contain some NaNs.')

    mask_for_nan_input = np.isnan(test_inputs['u'])
    embedded_features_without_nans = embedding(test_inputs, mask_for_nan_input)
    no_nans_tree = jax.tree.map(
        lambda x: np.all(~np.isnan(x)), embedded_features_without_nans
    )
    all_leaves_are_no_nans = all(jax.tree_util.tree_leaves(no_nans_tree))
    self.assertTrue(all_leaves_are_no_nans, 'Outputs should not contain NaNs.')

  def test_coordinate_state_mapping(self):
    """Checks that CoordsStateMapping produces outputs with expected shapes."""
    input_names = ('u', 'v')
    test_inputs = {k: np.ones(self.coords.shape) for k in input_names}
    input_state_shapes = pytree_utils.shape_structure(test_inputs)
    feature_module = pytree_transforms.PrognosticFeatures(input_names)
    embedding_factory = functools.partial(
        pytree_mappings.Embedding,
        feature_module=feature_module,
        mapping_factory=self.mapping_factory,
        input_state_shapes=input_state_shapes,
        mesh=parallelism.Mesh(None),
    )
    volume_field_names = ('u', 'div')
    surface_field_names = ('pressure',)
    state_mapping = pytree_mappings.CoordsStateMapping(
        coords=self.coords,
        surface_field_names=surface_field_names,
        volume_field_names=volume_field_names,
        embedding_factory=embedding_factory,
        rngs=nnx.Rngs(0),
        mesh=parallelism.Mesh(None),
    )
    out = state_mapping(test_inputs)
    out_shape = pytree_utils.shape_structure(out)
    expected_shape = {
        'u': typing.ShapeFloatStruct(self.coords.shape),
        'div': typing.ShapeFloatStruct(self.coords.shape),
        'pressure': typing.ShapeFloatStruct(self.coords.horizontal.shape),
    }
    chex.assert_trees_all_equal(out_shape, expected_shape)


class ChannelMappingTest(parameterized.TestCase):
  """Tests ChannelMapping."""

  def test_channel_mapping(self):
    """Checks that ChannelMapping produces outputs with expected shapes."""
    coords = coordinates.DinosaurCoordinates(
        horizontal=coordinates.LonLatGrid.T21(),
        vertical=coordinates.SigmaLevels.equidistant(4),
    )
    tower_factory = functools.partial(
        towers.ColumnTower,
        column_net_factory=functools.partial(
            standard_layers.MlpUniform, hidden_size=6, n_hidden_layers=2
        ),
    )
    inputs = {
        'full_a': np.ones(coords.shape),
        'full_b': np.ones(coords.shape),
        'surface_a': np.ones(coords.horizontal.shape),
    }
    input_shapes = pytree_utils.shape_structure(inputs)
    output_shapes = {
        'out_full': typing.ShapeFloatStruct((7,) + coords.horizontal.shape),
        'out_surface': typing.ShapeFloatStruct(coords.horizontal.shape),
    }
    mapping = pytree_mappings.ChannelMapping(
        input_shapes=input_shapes,
        output_shapes=output_shapes,
        tower_factory=tower_factory,
        rngs=nnx.Rngs(0),
    )
    actual = mapping(inputs)
    chex.assert_trees_all_equal_shapes(actual, output_shapes)


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
