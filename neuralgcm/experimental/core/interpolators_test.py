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

"""Tests for interpolation routines."""

from absl.testing import absltest
from absl.testing import parameterized
import coordax as cx
import jax
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import interpolators
import numpy as np


class InterpolatorsTest(parameterized.TestCase):
  """Tests interpolator modules."""

  @parameterized.named_parameters(
      dict(
          testcase_name='spherical_coords',
          target_grid=coordinates.SphericalHarmonicGrid.T21(),
          input_coords=coordinates.SphericalHarmonicGrid.TL31(),
      ),
  )
  def test_spectral_regridder(self, target_grid, input_coords):
    regridder = interpolators.SpectralRegridder(target_grid)
    inputs = cx.wrap(np.ones(input_coords.shape), input_coords)
    outputs = regridder(inputs)
    output_coords = cx.get_coordinate(outputs)
    self.assertEqual(output_coords, target_grid)


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
