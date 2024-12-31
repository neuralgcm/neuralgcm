"""Tests that embedding modules work as expected."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import dinosaur
from flax import nnx
import jax
from neuralgcm.experimental import coordinates
from neuralgcm.experimental import embeddings
from neuralgcm.experimental import pytree_transforms
from neuralgcm.experimental import pytree_utils
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
    input_shapes = pytree_utils.shape_structure(inputs)
    actual = embedding_module.output_shapes(input_shapes)
    expected = pytree_utils.shape_structure(embedded_features)
    chex.assert_trees_all_equal(actual, expected)

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
    embedding_module = embeddings.MaskedDataVolumeFeatureEmbedding(
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
    grid = coordinates.LonLatGrid.TL63(spherical_harmonics_impl=ylm_impl)
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
    embedding_module = embeddings.MaskedDataSurfaceFeatureEmbedding(
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

    with self.subTest('output_masked_appropriately'):
      expected = {
          feature_names[k]: np.where(v > 0.5, 0.5, 1) for k, v in masks.items()
      }
      outputs = embedding_module(test_inputs)
      chex.assert_trees_all_close(outputs, expected)


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
