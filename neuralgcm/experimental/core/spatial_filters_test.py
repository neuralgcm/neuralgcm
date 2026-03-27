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

from absl.testing import absltest
import jax

from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import spatial_filters
from neuralgcm.experimental.core import spherical_harmonics


class SpatialFiltersTest(absltest.TestCase):

  def test_exponential_modal_filter_is_pytree(self):
    mapping = spherical_harmonics.FixedYlmMapping(
        lon_lat_grid=coordinates.LonLatGrid.T21(),
        ylm_grid=coordinates.SphericalHarmonicGrid.T21(),
    )
    filter_mod = spatial_filters.ExponentialModalFilter(
        ylm_map=mapping,
        attenuation=16.0,
        order=18,
        cutoff=0.0,
        skip_missing=False,
    )
    leaves, treedef = jax.tree.flatten(filter_mod)
    restored = jax.tree.unflatten(treedef, leaves)
    self.assertEqual(filter_mod.attenuation, restored.attenuation)
    self.assertEqual(filter_mod.order, restored.order)
    self.assertIsInstance(restored, spatial_filters.ExponentialModalFilter)

  def test_sequential_modal_filter_is_pytree(self):
    mapping = spherical_harmonics.FixedYlmMapping(
        lon_lat_grid=coordinates.LonLatGrid.T21(),
        ylm_grid=coordinates.SphericalHarmonicGrid.T21(),
    )
    filter_1 = spatial_filters.ExponentialModalFilter(ylm_map=mapping)
    filter_2 = spatial_filters.ExponentialModalFilter(ylm_map=mapping)
    seq_filter = spatial_filters.SequentialModalFilter(
        filters=[filter_1, filter_2],
        ylm_map=mapping,
    )
    leaves, treedef = jax.tree.flatten(seq_filter)
    restored = jax.tree.unflatten(treedef, leaves)
    self.assertLen(restored.filters, 2)
    self.assertIsInstance(
        restored.filters[0], spatial_filters.ExponentialModalFilter
    )
    self.assertIsInstance(restored, spatial_filters.SequentialModalFilter)


if __name__ == '__main__':
  absltest.main()
