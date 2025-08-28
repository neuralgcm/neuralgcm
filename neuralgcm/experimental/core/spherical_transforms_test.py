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

from absl.testing import absltest
from absl.testing import parameterized
import chex
import coordax as cx
from dinosaur import spherical_harmonic
import jax
from jax import config  # pylint: disable=g-importing-member
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import spherical_transforms
from neuralgcm.experimental.core import typing
import numpy as np


class SphericalTransformsTest(parameterized.TestCase):
  """Tests SphericalHarmonicsTransform methods."""

  @parameterized.parameters(
      dict(
          modal_input_array=np.arange(44 * 23).reshape((44, 23)),
          ylm_grid=coordinates.SphericalHarmonicGrid.T21(),
          lon_lat_grid=coordinates.LonLatGrid.T21(),
      ),
      dict(
          modal_input_array=np.arange(128 * 65).reshape((128, 65)),
          ylm_grid=coordinates.SphericalHarmonicGrid.TL63(),
          lon_lat_grid=coordinates.LonLatGrid.TL63(),
      ),
      dict(
          modal_input_array=np.arange(128 * 65).reshape((128, 65)),
          ylm_grid=coordinates.SphericalHarmonicGrid.TL63(
              spherical_harmonics_method='fast'
          ),
          lon_lat_grid=coordinates.LonLatGrid.TL63(),
      ),
  )
  def test_array_transforms(
      self,
      modal_input_array: typing.Array,
      ylm_grid: coordinates.SphericalHarmonicGrid,
      lon_lat_grid: coordinates.LonLatGrid,
  ):
    """Tests that SphericalHarmonicsTransform is equivalent to dinosaur."""
    mesh = parallelism.Mesh(spmd_mesh=None)
    transform = spherical_transforms.FixedYlmMapping(
        lon_lat_grid=lon_lat_grid,
        ylm_grid=ylm_grid,
        mesh=mesh,
        partition_schema_key='spatial',  # unused.
    )
    method = coordinates.SPHERICAL_HARMONICS_METHODS[
        ylm_grid.spherical_harmonics_method
    ]
    dinosaur_grid = spherical_harmonic.Grid(
        longitude_wavenumbers=ylm_grid.longitude_wavenumbers,
        total_wavenumbers=ylm_grid.total_wavenumbers,
        longitude_nodes=lon_lat_grid.longitude_nodes,
        latitude_nodes=lon_lat_grid.latitude_nodes,
        longitude_offset=lon_lat_grid.longitude_offset,
        latitude_spacing=lon_lat_grid.latitude_spacing,
        radius=1.0,
        spherical_harmonics_impl=method,
        spmd_mesh=None,
    )
    nodal_array = transform.to_nodal_array(modal_input_array)
    expected_nodal_array = dinosaur_grid.to_nodal(modal_input_array)
    np.testing.assert_allclose(nodal_array, expected_nodal_array)
    # back to modal transform.
    modal_array = transform.to_modal_array(nodal_array)
    expected_modal_array = dinosaur_grid.to_modal(expected_nodal_array)
    np.testing.assert_allclose(modal_array, expected_modal_array)

  def test_modal_grid_property_is_padded(self):
    ylm_grid = coordinates.SphericalHarmonicGrid.T21()
    lon_lat_grid = coordinates.LonLatGrid.T21()
    spmd_mesh = jax.sharding.Mesh(
        devices=np.array(jax.devices()).reshape((2, 2, 2)),
        axis_names=['z', 'x', 'y'],
    )
    mesh = parallelism.Mesh(
        spmd_mesh=spmd_mesh,
        field_partitions={'spatial': {'longitude': 'x', 'latitude': 'y'}},
    )
    transform = spherical_transforms.FixedYlmMapping(
        lon_lat_grid=lon_lat_grid,
        ylm_grid=ylm_grid,
        mesh=mesh,
        partition_schema_key='spatial',
    )
    padded_modal_grid = transform.modal_grid
    self.assertIsInstance(padded_modal_grid, coordinates.SphericalHarmonicGrid)
    self.assertEqual(transform.modal_grid.total_wavenumbers, 23)
    self.assertEqual(transform.modal_grid.shape, (64, 32))

  @parameterized.parameters(
      dict(
          f=cx.wrap(np.ones((64, 32)), coordinates.LonLatGrid.T21()),
          truncation_rule='cubic',
          spherical_harmonics_method='fast',
          expected_modal_shape=(44, 23),
          expected_nodal_shape=(64, 32),
      ),
      dict(
          f=cx.wrap(np.ones((64, 32)), coordinates.LonLatGrid.T21()),
          truncation_rule='linear',
          spherical_harmonics_method='fast',
          expected_modal_shape=(64, 33),
          expected_nodal_shape=(64, 32),
      ),
      dict(
          f=cx.wrap(np.ones((64, 32)), coordinates.LonLatGrid.T21()),
          truncation_rule='linear',
          spherical_harmonics_method='real',
          expected_modal_shape=(63, 33),
          expected_nodal_shape=(64, 32),
      ),
  )
  def test_ylm_mapper(
      self,
      f: cx.Field,
      truncation_rule: spherical_transforms.TruncationRules,
      spherical_harmonics_method: spherical_transforms.SphericalHarmonicMethods,
      expected_modal_shape: tuple[int, ...],
      expected_nodal_shape: tuple[int, ...],
  ):
    ylm_mapper = spherical_transforms.YlmMapper(
        truncation_rule=truncation_rule,
        spherical_harmonics_method=spherical_harmonics_method,
        partition_schema_key=None,
        mesh=parallelism.Mesh(spmd_mesh=None),
    )
    modal = ylm_mapper.to_modal(f)
    nodal = ylm_mapper.to_nodal(modal)
    self.assertEqual(modal.shape, expected_modal_shape)
    self.assertEqual(nodal.shape, expected_nodal_shape)

  def test_ylm_mapper_with_padding(self):
    spmd_mesh = jax.sharding.Mesh(
        devices=np.array(jax.devices()).reshape((2, 2, 2)),
        axis_names=['z', 'x', 'y'],
    )
    field_partitions = {
        'spatial': {
            'longitude': 'x',
            'longitude_wavenumber': 'x',
            'total_wavenumber': 'y',
            'latitude': 'y',
            'level': 'z',
        }
    }
    mesh = parallelism.Mesh(spmd_mesh, field_partitions=field_partitions)
    with self.subTest('cubic_truncation'):
      ylm_mapper = spherical_transforms.YlmMapper(
          truncation_rule='cubic',
          partition_schema_key='spatial',
          mesh=mesh,
      )
      grid = coordinates.LonLatGrid.T21(
          mesh=mesh, partition_schema_key='spatial'
      )
      f = cx.wrap(np.ones(grid.shape), grid)
      modal_f = ylm_mapper.to_modal(f)
      restored_nodal_f = ylm_mapper.to_nodal(modal_f)
      expected_modal_grid = coordinates.SphericalHarmonicGrid.T21(
          mesh=mesh, partition_schema_key='spatial'
      )
      self.assertEqual(cx.get_coordinate(modal_f), expected_modal_grid)
      self.assertEqual(cx.get_coordinate(restored_nodal_f), grid)

    with self.subTest('linear_truncation'):
      ylm_mapper = spherical_transforms.YlmMapper(
          truncation_rule='linear',
          partition_schema_key='spatial',
          mesh=mesh,
      )
      grid = coordinates.LonLatGrid.T21(
          mesh=mesh, partition_schema_key='spatial'
      )
      f = cx.wrap(np.ones(grid.shape), grid)
      modal_f = ylm_mapper.to_modal(f)
      restored_nodal_f = ylm_mapper.to_nodal(modal_f)
      expected_modal_grid = coordinates.SphericalHarmonicGrid.TL31(
          mesh=mesh, partition_schema_key='spatial'
      )
      self.assertEqual(cx.get_coordinate(modal_f), expected_modal_grid)
      self.assertEqual(cx.get_coordinate(restored_nodal_f), grid)


class GeometryMethodsTest(chex.TestCase):
  """Tests other geometric transforms on FixedYlmMapping and YlmMapper."""

  def setUp(self):
    super().setUp()
    self.lon_lat_grid = coordinates.LonLatGrid.T21()
    self.ylm_grid = coordinates.SphericalHarmonicGrid.T21()
    spherical_harmonics_impl = spherical_harmonic.FastSphericalHarmonics
    self.dinosaur_grid = spherical_harmonic.Grid.T21(
        spherical_harmonics_impl=spherical_harmonics_impl
    )
    mesh = parallelism.Mesh(spmd_mesh=None)
    self.fixed_ylm_mapping = spherical_transforms.FixedYlmMapping(
        lon_lat_grid=self.lon_lat_grid,
        ylm_grid=self.ylm_grid,
        mesh=mesh,
        partition_schema_key=None,
    )
    self.ylm_mapper = spherical_transforms.YlmMapper(
        truncation_rule='cubic',
        partition_schema_key=None,
        mesh=mesh,
    )
    key = jax.random.PRNGKey(42)
    # modal data needs to respect the mask
    mask = self.dinosaur_grid.mask
    modal_data = jax.random.normal(key, self.ylm_grid.shape) * mask
    nodal_data = jax.random.normal(key, self.lon_lat_grid.shape)
    self.modal_field = cx.wrap(modal_data, self.ylm_grid)
    self.nodal_field = cx.wrap(nodal_data, self.lon_lat_grid)
    # For vector fields
    u_key, v_key = jax.random.split(key, 2)
    modal_data_u = jax.random.normal(u_key, self.ylm_grid.shape) * mask
    modal_data_v = jax.random.normal(v_key, self.ylm_grid.shape) * mask
    self.modal_field_u = cx.wrap(modal_data_u, self.ylm_grid)
    self.modal_field_v = cx.wrap(modal_data_v, self.ylm_grid)
    nodal_data_u = jax.random.normal(u_key, self.lon_lat_grid.shape)
    nodal_data_v = jax.random.normal(v_key, self.lon_lat_grid.shape)
    self.nodal_field_u = cx.wrap(nodal_data_u, self.lon_lat_grid)
    self.nodal_field_v = cx.wrap(nodal_data_v, self.lon_lat_grid)

  def test_laplacian(self):
    expected_data = self.dinosaur_grid.laplacian(self.modal_field.data)
    actual_data = self.fixed_ylm_mapping.laplacian(self.modal_field).data
    np.testing.assert_allclose(actual_data, expected_data)
    actual_data = self.ylm_mapper.laplacian(self.modal_field).data
    np.testing.assert_allclose(actual_data, expected_data)

  def test_inverse_laplacian(self):
    expected_data = self.dinosaur_grid.inverse_laplacian(self.modal_field.data)
    actual_data = self.fixed_ylm_mapping.inverse_laplacian(
        self.modal_field
    ).data
    np.testing.assert_allclose(actual_data, expected_data)
    actual_data = self.ylm_mapper.inverse_laplacian(self.modal_field).data
    np.testing.assert_allclose(actual_data, expected_data)

  def test_d_dlon(self):
    expected_data = self.dinosaur_grid.d_dlon(self.modal_field.data)
    actual_data = self.fixed_ylm_mapping.d_dlon(self.modal_field).data
    np.testing.assert_allclose(actual_data, expected_data)
    actual_data = self.ylm_mapper.d_dlon(self.modal_field).data
    np.testing.assert_allclose(actual_data, expected_data)

  def test_cos_lat_d_dlat(self):
    expected_data = self.dinosaur_grid.cos_lat_d_dlat(self.modal_field.data)
    actual_data = self.fixed_ylm_mapping.cos_lat_d_dlat(self.modal_field).data
    np.testing.assert_allclose(actual_data, expected_data)
    actual_data = self.ylm_mapper.cos_lat_d_dlat(self.modal_field).data
    np.testing.assert_allclose(actual_data, expected_data)

  def test_sec_lat_d_dlat_cos2(self):
    expected_data = self.dinosaur_grid.sec_lat_d_dlat_cos2(
        self.modal_field.data
    )
    actual_data = self.fixed_ylm_mapping.sec_lat_d_dlat_cos2(
        self.modal_field
    ).data
    np.testing.assert_allclose(actual_data, expected_data)
    actual_data = self.ylm_mapper.sec_lat_d_dlat_cos2(self.modal_field).data
    np.testing.assert_allclose(actual_data, expected_data)

  def test_cos_lat_grad(self):
    expected_u, expected_v = self.dinosaur_grid.cos_lat_grad(
        self.modal_field.data
    )
    actual_u, actual_v = (
        x.data for x in self.fixed_ylm_mapping.cos_lat_grad(self.modal_field)
    )
    np.testing.assert_allclose(actual_u, expected_u)
    np.testing.assert_allclose(actual_v, expected_v)
    actual_u, actual_v = (
        x.data for x in self.ylm_mapper.cos_lat_grad(self.modal_field)
    )
    np.testing.assert_allclose(actual_u, expected_u)
    np.testing.assert_allclose(actual_v, expected_v)

  def test_k_cross(self):
    expected_i, expected_j = self.dinosaur_grid.k_cross(
        (self.modal_field_u.data, self.modal_field_v.data)
    )
    actual_i, actual_j = (
        x.data
        for x in self.fixed_ylm_mapping.k_cross(
            self.modal_field_u, self.modal_field_v
        )
    )
    np.testing.assert_allclose(actual_i, expected_i)
    np.testing.assert_allclose(actual_j, expected_j)
    actual_i, actual_j = (
        x.data
        for x in self.ylm_mapper.k_cross(self.modal_field_u, self.modal_field_v)
    )
    np.testing.assert_allclose(actual_i, expected_i)
    np.testing.assert_allclose(actual_j, expected_j)

  def test_div_cos_lat(self):
    expected_data = self.dinosaur_grid.div_cos_lat(
        (self.modal_field_u.data, self.modal_field_v.data)
    )
    actual_data = self.fixed_ylm_mapping.div_cos_lat(
        self.modal_field_u, self.modal_field_v
    ).data
    np.testing.assert_allclose(actual_data, expected_data)
    actual_data = self.ylm_mapper.div_cos_lat(
        self.modal_field_u, self.modal_field_v
    ).data
    np.testing.assert_allclose(actual_data, expected_data)

  def test_curl_cos_lat(self):
    expected_data = self.dinosaur_grid.curl_cos_lat(
        (self.modal_field_u.data, self.modal_field_v.data)
    )
    actual_data = self.fixed_ylm_mapping.curl_cos_lat(
        self.modal_field_u, self.modal_field_v
    ).data
    np.testing.assert_allclose(actual_data, expected_data)
    actual_data = self.ylm_mapper.curl_cos_lat(
        self.modal_field_u, self.modal_field_v
    ).data
    np.testing.assert_allclose(actual_data, expected_data)

  def test_integrate(self):
    expected_data = self.dinosaur_grid.integrate(self.nodal_field.data)
    actual_data = self.fixed_ylm_mapping.integrate(self.nodal_field).data
    np.testing.assert_allclose(actual_data, expected_data)
    actual_data = self.ylm_mapper.integrate(self.nodal_field).data
    np.testing.assert_allclose(actual_data, expected_data)

  def test_get_cos_lat_vector(self):
    expected_u, expected_v = spherical_harmonic.get_cos_lat_vector(
        self.modal_field_u.data, self.modal_field_v.data, self.dinosaur_grid
    )
    # With FixedYlmMapping
    actual_u, actual_v = (
        x.data
        for x in spherical_transforms.get_cos_lat_vector(
            self.modal_field_u, self.modal_field_v, self.fixed_ylm_mapping
        )
    )
    np.testing.assert_allclose(actual_u, expected_u)
    np.testing.assert_allclose(actual_v, expected_v)
    # With YlmMapper
    actual_u, actual_v = (
        x.data
        for x in spherical_transforms.get_cos_lat_vector(
            self.modal_field_u, self.modal_field_v, self.ylm_mapper
        )
    )
    np.testing.assert_allclose(actual_u, expected_u)
    np.testing.assert_allclose(actual_v, expected_v)

  def test_uv_nodal_to_vor_div_modal(self):
    expected_vort, expected_div = spherical_harmonic.uv_nodal_to_vor_div_modal(
        self.dinosaur_grid, self.nodal_field_u.data, self.nodal_field_v.data
    )
    # With FixedYlmMapping
    actual_vort, actual_div = (
        x.data
        for x in spherical_transforms.uv_nodal_to_vor_div_modal(
            self.nodal_field_u, self.nodal_field_v, self.fixed_ylm_mapping
        )
    )
    np.testing.assert_allclose(actual_vort, expected_vort)
    np.testing.assert_allclose(actual_div, expected_div)
    # With YlmMapper
    actual_vort, actual_div = (
        x.data
        for x in spherical_transforms.uv_nodal_to_vor_div_modal(
            self.nodal_field_u, self.nodal_field_v, self.ylm_mapper
        )
    )
    np.testing.assert_allclose(actual_vort, expected_vort)
    np.testing.assert_allclose(actual_div, expected_div)

  def test_vor_div_to_uv_nodal(self):
    expected_u, expected_v = spherical_harmonic.vor_div_to_uv_nodal(
        self.dinosaur_grid, self.modal_field_u.data, self.modal_field_v.data
    )
    # With FixedYlmMapping
    actual_u, actual_v = (
        x.data
        for x in spherical_transforms.vor_div_to_uv_nodal(
            self.modal_field_u, self.modal_field_v, self.fixed_ylm_mapping
        )
    )
    np.testing.assert_allclose(actual_u, expected_u)
    np.testing.assert_allclose(actual_v, expected_v)
    # With YlmMapper
    actual_u, actual_v = (
        x.data
        for x in spherical_transforms.vor_div_to_uv_nodal(
            self.modal_field_u, self.modal_field_v, self.ylm_mapper
        )
    )
    np.testing.assert_allclose(actual_u, expected_u)
    np.testing.assert_allclose(actual_v, expected_v)


if __name__ == '__main__':
  chex.set_n_cpu_devices(8)
  config.update('jax_traceback_filtering', 'off')
  absltest.main()
