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
from neuralgcm.experimental.core import step_filters


class StepFiltersTest(absltest.TestCase):

  def test_no_filter_is_pytree(self):
    filter_mod = step_filters.NoFilter()
    leaves, treedef = jax.tree.flatten(filter_mod)
    restored = jax.tree.unflatten(treedef, leaves)
    self.assertIsInstance(restored, step_filters.NoFilter)

  def test_modal_fixed_global_mean_filter_is_pytree(self):
    ylm_grid = coordinates.SphericalHarmonicGrid.T21()
    filter_mod = step_filters.ModalFixedGlobalMeanFilter(
        ylm_grid=ylm_grid, keys=('temperature', 'wind')
    )
    leaves, treedef = jax.tree.flatten(filter_mod)
    restored = jax.tree.unflatten(treedef, leaves)
    self.assertEqual(filter_mod.keys, restored.keys)
    self.assertIsInstance(restored, step_filters.ModalFixedGlobalMeanFilter)


if __name__ == '__main__':
  absltest.main()
