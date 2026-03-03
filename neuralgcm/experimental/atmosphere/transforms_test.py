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

"""Tests that atmospheric transforms produce outputs with expected structure."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import coordax as cx
import jax
from neuralgcm.experimental.atmosphere import transforms as atmos_transforms
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import diagnostics
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import pytree_utils
from neuralgcm.experimental.core import spherical_harmonics
from neuralgcm.experimental.core import transforms
from neuralgcm.experimental.core import typing
import numpy as np


class AtmosphereTransformsTest(parameterized.TestCase):
  """Tests atmospheric transforms."""

  def _test_feature_module(
      self,
      feature_module: typing.Transform,
      inputs: typing.Pytree,
  ):
    features = feature_module(inputs)
    expected = pytree_utils.shape_structure(features)
    input_shapes = pytree_utils.shape_structure(inputs)
    actual = feature_module.output_shapes(input_shapes)
    chex.assert_trees_all_equal(actual, expected)

  def test_velocity_and_prognostics_with_modal_gradients(self):
    sigma = coordinates.SigmaLevels.equidistant(4)
    ylm_map = spherical_harmonics.FixedYlmMapping(
        lon_lat_grid=coordinates.LonLatGrid.T21(),
        ylm_grid=coordinates.SphericalHarmonicGrid.T21(),
        partition_schema_key=None,
        mesh=parallelism.Mesh(),
    )
    with_gradients_transform = transforms.ToModalWithDerivatives(
        ylm_map,
        filter_attenuations=[2.0],
    )
    features_grads = atmos_transforms.VelocityAndPrognosticsWithModalGradients(
        ylm_map,
        volume_field_names=(
            'u',
            'v',
            'vorticity',
        ),
        surface_field_names=('lsp',),
        compute_gradients_transform=with_gradients_transform,
    )
    modal_grid = ylm_map.modal_grid
    shape_3d = sigma.shape + modal_grid.shape
    inputs = {
        'u': cx.field(np.ones(shape_3d), sigma, modal_grid),
        'v': cx.field(np.ones(shape_3d), sigma, modal_grid),
        'vorticity': cx.field(np.ones(shape_3d), sigma, modal_grid),
        'divergence': cx.field(np.ones(shape_3d), sigma, modal_grid),
        'lsp': cx.field(np.ones(modal_grid.shape), modal_grid),
    }
    self._test_feature_module(features_grads, inputs)

  def test_pressure_features(self):
    sigma = coordinates.SigmaLevels.equidistant(8)
    ylm_grid = coordinates.SphericalHarmonicGrid.T21()
    ylm_map = spherical_harmonics.FixedYlmMapping(
        lon_lat_grid=coordinates.LonLatGrid.T21(),
        ylm_grid=ylm_grid,
        partition_schema_key=None,
        mesh=parallelism.Mesh(None),
    )
    pressure_features = atmos_transforms.PressureFeatures(
        ylm_map=ylm_map,
        levels=sigma,
    )
    inputs = {
        'log_surface_pressure': cx.wrap(np.ones(ylm_grid.shape), ylm_grid),
    }
    self._test_feature_module(pressure_features, inputs)

  def test_pressure_features_hybrid(self):
    hybrid = coordinates.HybridLevels(
        a_boundaries=np.zeros(9),
        b_boundaries=coordinates.SigmaLevels.equidistant(8).boundaries,
    )
    ylm_grid = coordinates.SphericalHarmonicGrid.T21()
    ylm_map = spherical_harmonics.FixedYlmMapping(
        lon_lat_grid=coordinates.LonLatGrid.T21(),
        ylm_grid=ylm_grid,
        partition_schema_key=None,
        mesh=parallelism.Mesh(None),
    )
    pressure_features = atmos_transforms.PressureFeatures(
        ylm_map=ylm_map,
        levels=hybrid,
    )
    inputs = {
        'log_surface_pressure': cx.field(np.ones(ylm_grid.shape), ylm_grid),
    }
    self._test_feature_module(pressure_features, inputs)

  def test_constrain_precipitation_and_evaporation(self):
    x = cx.SizedAxis('x', 4)
    precip_key = 'precipitation'
    evap_key = 'evaporation'

    with self.subTest('diagnose_precip'):
      p_plus_e_val = cx.field(np.array([-2.0, -1.0, 0.5, 1.0]), x)
      evap_in = cx.field(np.array([-3.0, -0.5, -1.0, 0.2]), x)
      p_plus_e_diag = diagnostics.InstantDiagnostic(
          extract=lambda *args, **kwargs: {
              'precipitation_plus_evaporation_rate': p_plus_e_val
          },
          extract_coords={'precipitation_plus_evaporation_rate': x},
      )
      p_plus_e_diag({}, prognostics={})  # call to compute and store values
      inputs = {evap_key: evap_in}
      transform = atmos_transforms.ConstrainWaterBudget(
          p_plus_e_diagnostic=p_plus_e_diag,
          var_to_constrain=evap_key,
          precipitation_key=precip_key,
          evaporation_key=evap_key,
      )
      actual = transform(inputs)
      # in evap case, result should be min(evap_in, min(p_plus_e, 0))
      # constrained_evap=min([-3,-2],[-0.5,-1],[-1,0],[0.2,0])
      expected_evap_val = np.array([-3.0, -1.0, -1.0, 0.0])
      cx.testing.assert_fields_allclose(
          actual[evap_key], cx.field(expected_evap_val, x)
      )
      # diagnosed precip = p_plus_e - constrained_evap
      expected_precip_val = p_plus_e_val.data - expected_evap_val
      cx.testing.assert_fields_allclose(
          actual[precip_key], cx.field(expected_precip_val, x)
      )

    with self.subTest('diagnose_evap'):
      p_plus_e_val = cx.field(np.array([-1.0, 0.5, 2.0, 3.0]), x)
      precip_in = cx.field(np.array([-2.0, -1.0, 1.0, 4.0]), x)
      p_plus_e_diag = diagnostics.InstantDiagnostic(
          extract=lambda *args, **kwargs: {
              'precipitation_plus_evaporation_rate': p_plus_e_val
          },
          extract_coords={'precipitation_plus_evaporation_rate': x},
      )
      p_plus_e_diag({}, prognostics={})  # call to compute and store values
      inputs = {precip_key: precip_in}
      transform = atmos_transforms.ConstrainWaterBudget(
          p_plus_e_diagnostic=p_plus_e_diag,
          var_to_constrain=precip_key,
          precipitation_key=precip_key,
          evaporation_key=evap_key,
      )
      actual = transform(inputs)
      # in precip case, result should be max(precip_in, max(p_plus_e, 0))
      # constrained_precip = max([-2,0],[-1,0.5],[1,2],[4,3]) = [0, 0.5, 2, 4]
      expected_precip_val = np.array([0.0, 0.5, 2.0, 4.0])
      cx.testing.assert_fields_allclose(
          actual[precip_key], cx.field(expected_precip_val, x)
      )
      expected_evap_val = p_plus_e_val.data - expected_precip_val
      cx.testing.assert_fields_allclose(
          actual[evap_key], cx.field(expected_evap_val, x)
      )


if __name__ == '__main__':
  jax.config.update('jax_traceback_filtering', 'off')
  absltest.main()
