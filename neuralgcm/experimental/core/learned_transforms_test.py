# Copyright 2025 Google LLC
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

"""Tests that learned transforms produce outputs with expected shapes."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import chex
from flax import nnx
import jax
import jax_datetime as jdt
from neuralgcm.experimental import coordax as cx
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import feature_transforms
from neuralgcm.experimental.core import field_utils
from neuralgcm.experimental.core import learned_transforms
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import pytree_utils
from neuralgcm.experimental.core import standard_layers
from neuralgcm.experimental.core import towers
from neuralgcm.experimental.core import transforms
import numpy as np


# Aliases for readability.
ForwardTowerTransform = learned_transforms.ForwardTowerTransform


def ones_field_for_coord(coord: cx.Coordinate):
  return cx.wrap(np.ones(coord.shape), coord)


class ForwardTowerTransformTest(parameterized.TestCase):
  """Tests different instantiations of ForwardTowerTransform."""

  def setUp(self):
    """Set up common parameters and configurations for tests."""
    super().setUp()
    self.grid = coordinates.LonLatGrid.T21()
    self.levels = coordinates.SigmaLevels.equidistant(12)
    self.coord = cx.compose_coordinates(self.levels, self.grid)
    self.tower_factory = functools.partial(
        towers.ForwardTower.build_using_factories,
        inputs_in_dims=('d',),
        out_dims=('d',),
        neural_net_factory=functools.partial(
            standard_layers.Mlp.uniform, hidden_size=6, hidden_layers=2
        ),
    )
    self.mesh = parallelism.Mesh()

  def test_tower_transform_as_surface_embeddings(self):
    """Tests that ForwardTowerTransform can work as surface embeddings."""
    test_inputs = {
        'u': ones_field_for_coord(self.coord),
        'v': ones_field_for_coord(self.coord),
    }
    input_shapes = pytree_utils.shape_structure(test_inputs)
    az, bz = cx.SizedAxis('a', 7), cx.SizedAxis('b', 3)
    embedding_coords = {  # will create embeddings of multiple sizes for fun.
        'a': cx.compose_coordinates(az, self.grid),
        'b': cx.compose_coordinates(bz, self.grid),
    }
    embedding = ForwardTowerTransform.build_using_factories(
        input_shapes=input_shapes,
        targets=embedding_coords,
        tower_factory=self.tower_factory,
        dims_to_align=(self.grid,),
        mesh=self.mesh,
        rngs=nnx.Rngs(0),
    )

    with self.subTest('output_shapes'):
      actual = pytree_utils.shape_structure(embedding(test_inputs))
      expected = field_utils.shape_struct_fields_from_coords(embedding_coords)
      chex.assert_trees_all_equal(actual, expected)

    with self.subTest('output_shapes_method'):
      actual = embedding.output_shapes(input_shapes)
      expected = field_utils.shape_struct_fields_from_coords(embedding_coords)
      chex.assert_trees_all_equal(actual, expected)

  def test_tower_transform_as_volume_embeddings(self):
    """Tests that ForwardTowerTransform can work as volume embeddings."""
    features_coords = cx.compose_coordinates(
        cx.SizedAxis('in_features', 13), self.coord
    )
    test_inputs = {
        'features': ones_field_for_coord(features_coords),
    }
    input_shapes = pytree_utils.shape_structure(test_inputs)
    z = cx.SizedAxis('embedding', 8)
    embedding_coords = {
        'atm_embedding': cx.compose_coordinates(z, self.coord),
    }
    v_embedding = ForwardTowerTransform.build_using_factories(
        input_shapes=input_shapes,
        targets=embedding_coords,
        tower_factory=self.tower_factory,
        dims_to_align=(self.levels, self.grid,),
        mesh=self.mesh,
        rngs=nnx.Rngs(0),
    )

    with self.subTest('output_shapes'):
      actual = pytree_utils.shape_structure(v_embedding(test_inputs))
      expected = field_utils.shape_struct_fields_from_coords(embedding_coords)
      chex.assert_trees_all_equal(actual, expected)

    with self.subTest('output_shapes_method'):
      actual = v_embedding.output_shapes(input_shapes)
      expected = field_utils.shape_struct_fields_from_coords(embedding_coords)
      chex.assert_trees_all_equal(actual, expected)

  def test_tower_transform_maps_to_surface_and_volume_targets(self):
    """Tests that ForwardTowerTransform predicts surface & volume targets."""
    test_inputs = {
        'u': ones_field_for_coord(self.coord),
        'v': ones_field_for_coord(self.coord),
        'time': cx.wrap(jdt.to_datetime('2025-05-21T00')),
    }
    input_shapes = pytree_utils.shape_structure(test_inputs)
    target_coords = {  # will create embeddings of multiple sizes for fun.
        'du_dt': self.coord,
        'dv_dt': self.coord,
        'd_p_surface_dt': self.grid,
    }
    features = transforms.Merge({
        'radiation': feature_transforms.RadiationFeatures(self.grid),
        'latitude': feature_transforms.LatitudeFeatures(self.grid),
        'prognostics': transforms.Select(r'(?!time).*'),
    })
    parameterization = ForwardTowerTransform.build_using_factories(
        input_shapes=input_shapes,
        targets=target_coords,
        tower_factory=self.tower_factory,
        dims_to_align=(self.grid,),
        in_transform=features,
        mesh=self.mesh,
        rngs=nnx.Rngs(0),
    )

    with self.subTest('output_shapes'):
      out = parameterization(test_inputs)
      actual = pytree_utils.shape_structure(out)
      expected = field_utils.shape_struct_fields_from_coords(target_coords)
      chex.assert_trees_all_equal(actual, expected)

    with self.subTest('output_shapes_method'):
      actual = parameterization.output_shapes(input_shapes)
      expected = field_utils.shape_struct_fields_from_coords(target_coords)
      chex.assert_trees_all_equal(actual, expected)

  def test_weighted_land_sea_ice_tower_transform(self):
    """Tests that WeightedLandSeaIceTowersTransform can be used."""
    grid = self.grid
    latent_coord = cx.SizedAxis('latent', 3)
    embedding_coord = cx.compose_coordinates(latent_coord, grid)
    output_coords = {'surface_embedding': embedding_coord}

    # Create mock data with nans for sst + masks.
    lon, lat = grid.fields['longitude'], grid.fields['latitude']
    atm_2m_temp = cx.wrap(288 * np.ones(grid.shape), grid)
    land_sea_mask = (lon < 120) * (lon > 30) * (lat < 70)
    sst = cx.wrap(np.where(land_sea_mask.data, np.nan, 279), grid)
    sic_vals = (lat >= 70).broadcast_like(atm_2m_temp)
    sea_ice_cover = cx.wrap(
        np.where(land_sea_mask.data, np.nan, sic_vals.data), grid
    )

    mask_nans_transform = transforms.Mask(
        mask_key='sea_ice_cover',
        compute_mask_method='isnan',
        apply_mask_method='nan_to_0',
    )
    land_mask_transform = transforms.Select('land_sea_mask')
    sea_ice_mask_transform = transforms.Select('sea_ice_cover')

    lat_features = feature_transforms.LatitudeFeatures(grid)
    land_features = transforms.Merge({
        'lats': lat_features,
        'atm_t_and_mask': transforms.Select('2m_temp|sea_ice_cover'),
    })
    land_features = transforms.Sequential([land_features, mask_nans_transform])
    sea_features = transforms.Merge({
        'lats': lat_features,
        'sst_and_mask': transforms.Select('sst|sea_ice_cover'),
    })
    sea_features = transforms.Sequential([sea_features, mask_nans_transform])
    ice_features = transforms.Merge(
        {'lats': lat_features, 'mask': transforms.Select('sea_ice_cover')}
    )
    ice_features = transforms.Sequential([ice_features, mask_nans_transform])

    inputs = {
        'land_sea_mask': land_sea_mask.astype(np.float32),
        'sea_ice_cover': sea_ice_cover.astype(np.float32),
        'sst': sst.astype(np.float32),
        '2m_temp': atm_2m_temp,
    }
    input_shapes = pytree_utils.shape_structure(inputs)
    rngs = nnx.Rngs(0)

    ice_transform = ForwardTowerTransform.build_using_factories(
        input_shapes=input_shapes,
        targets=output_coords,
        tower_factory=self.tower_factory,
        dims_to_align=(grid,),
        in_transform=ice_features,
        mesh=self.mesh,
        rngs=rngs,
    )
    land_transform = ForwardTowerTransform.build_using_factories(
        input_shapes=input_shapes,
        targets=output_coords,
        tower_factory=self.tower_factory,
        dims_to_align=(grid,),
        in_transform=land_features,
        mesh=self.mesh,
        rngs=rngs,
    )
    sea_transform = ForwardTowerTransform.build_using_factories(
        input_shapes=input_shapes,
        targets=output_coords,
        tower_factory=self.tower_factory,
        dims_to_align=(grid,),
        in_transform=sea_features,
        mesh=self.mesh,
        rngs=rngs,
    )
    land_sea_ice = learned_transforms.WeightedLandSeaIceTowersTransform(
        land_transform=land_transform,
        sea_transform=sea_transform,
        sea_ice_transform=ice_transform,
        land_sea_mask_transform=land_mask_transform,
        sea_ice_value_transform=sea_ice_mask_transform,
        mesh=self.mesh,
    )
    out = land_sea_ice(inputs)
    self.assertEqual(
        cx.get_coordinate(out['surface_embedding']), embedding_coord
    )
    self.assertFalse(np.isnan(out['surface_embedding'].data).any())


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
