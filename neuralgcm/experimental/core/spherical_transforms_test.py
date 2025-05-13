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
    transform = spherical_transforms.SphericalHarmonicsTransform(
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
    transform = spherical_transforms.SphericalHarmonicsTransform(
        lon_lat_grid=lon_lat_grid,
        ylm_grid=ylm_grid,
        mesh=mesh,
        partition_schema_key='spatial',
    )
    padded_modal_grid = transform.modal_grid
    self.assertIsInstance(padded_modal_grid, coordinates.SphericalHarmonicGrid)
    self.assertEqual(transform.modal_grid.total_wavenumbers, 23)
    self.assertEqual(transform.modal_grid.shape, (64, 32))


if __name__ == '__main__':
  chex.set_n_cpu_devices(8)
  config.update('jax_traceback_filtering', 'off')
  absltest.main()
