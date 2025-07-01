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
import collections
from typing import Callable

from absl.testing import absltest
from absl.testing import parameterized
import coordax as cx
from coordax import testing as coordax_testing
import jax
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import parallelism
import numpy as np


class CoordinatesTest(parameterized.TestCase):
  """Tests that coordinate have expected shapes and dims."""

  @parameterized.named_parameters(
      dict(
          testcase_name='spherical_harmonic',
          coords=coordinates.SphericalHarmonicGrid.TL31(),
          expected_dims=('longitude_wavenumber', 'total_wavenumber'),
          expected_shape=(64, 33),
      ),
      dict(
          testcase_name='lon_lat',
          coords=coordinates.LonLatGrid.T21(),
          expected_dims=('longitude', 'latitude'),
          expected_shape=(64, 32),
      ),
      dict(
          testcase_name='product_of_levels',
          coords=cx.compose_coordinates(
              coordinates.SigmaLevels.equidistant(4),
              coordinates.PressureLevels([50, 100, 200, 800, 1000]),
              coordinates.LayerLevels(3),
          ),
          expected_dims=('sigma', 'pressure', 'layer_index'),
          expected_shape=(4, 5, 3),
      ),
      dict(
          testcase_name='sigma_spherical_harmonic_product',
          coords=cx.compose_coordinates(
              coordinates.SigmaLevels.equidistant(4),
              coordinates.SphericalHarmonicGrid.T21(),
          ),
          expected_dims=('sigma', 'longitude_wavenumber', 'total_wavenumber'),
          expected_shape=(4, 44, 23),
      ),
      dict(
          testcase_name='dinosaur_primitive_equation_coords',
          coords=coordinates.DinosaurCoordinates(
              horizontal=coordinates.SphericalHarmonicGrid.T21(),
              vertical=coordinates.SigmaLevels.equidistant(4),
          ),
          expected_dims=('sigma', 'longitude_wavenumber', 'total_wavenumber'),
          expected_shape=(4, 44, 23),
      ),
      dict(
          testcase_name='batched_trajectory',
          coords=cx.compose_coordinates(
              cx.SizedAxis('batch', 7),
              coordinates.TimeDelta(np.arange(5) * np.timedelta64(1, 'h')),
              coordinates.PressureLevels([50, 200, 800, 1000]),
              coordinates.LonLatGrid.T21(),
          ),
          expected_dims=(
              'batch',
              'timedelta',
              'pressure',
              'longitude',
              'latitude',
          ),
          expected_shape=(7, 5, 4, 64, 32),
          expected_field_transform=lambda f: f.untag('batch').tag('batch'),
      ),
      dict(
          testcase_name='coordinate_shard_none',
          coords=parallelism.CoordinateShard(
              coordinate=coordinates.LonLatGrid.T42(),
              spmd_mesh_shape=collections.OrderedDict(x=2, y=1, z=2),
              dimension_partitions={'longitude': None, 'latitude': None}
          ),
          expected_dims=('longitude', 'latitude'),
          expected_shape=(128, 64),  # unchanged.
          supports_xarray_roundtrip=False,
      ),
      dict(
          testcase_name='coordinate_shard_longitude',
          coords=parallelism.CoordinateShard(
              coordinate=coordinates.LonLatGrid.T42(),
              spmd_mesh_shape=collections.OrderedDict(x=2, y=1, z=2),
              dimension_partitions={'longitude': ('x', 'z'), 'latitude': None}
          ),
          expected_dims=('longitude', 'latitude'),
          expected_shape=(32, 64),  # unchanged.
          supports_xarray_roundtrip=False,
      ),
      dict(
          testcase_name='coordinate_shard_longitude_and_latitude',
          coords=parallelism.CoordinateShard(
              coordinate=coordinates.LonLatGrid.T42(),
              spmd_mesh_shape=collections.OrderedDict(x=2, y=4, z=2),
              dimension_partitions={'longitude': 'x', 'latitude': ('y', 'z')}
          ),
          expected_dims=('longitude', 'latitude'),
          expected_shape=(64, 8),  # unchanged.
          supports_xarray_roundtrip=False,
      ),
  )
  def test_coordinates(
      self,
      coords: cx.Coordinate,
      expected_dims: tuple[str, ...],
      expected_shape: tuple[int, ...],
      expected_field_transform: Callable[[cx.Field], cx.Field] = lambda x: x,
      supports_xarray_roundtrip: bool = True,
  ):
    """Tests that coordinates are pytrees and have expected shape and dims."""
    with self.subTest('pytree_roundtrip'):
      leaves, tree_def = jax.tree.flatten(coords)
      reconstructed = jax.tree.unflatten(tree_def, leaves)
      self.assertEqual(reconstructed, coords)

    with self.subTest('dims'):
      self.assertEqual(coords.dims, expected_dims)

    with self.subTest('shape'):
      self.assertEqual(coords.shape, expected_shape)

    if supports_xarray_roundtrip:
      with self.subTest('xarray_roundtrip'):
        field = cx.wrap(np.zeros(coords.shape), coords)
        data_array = field.to_xarray()
        reconstructed = coordinates.field_from_xarray(data_array)
        expected = expected_field_transform(field)
        coordax_testing.assert_fields_equal(reconstructed, expected)


if __name__ == '__main__':
  absltest.main()
