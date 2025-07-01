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

import functools

from absl.testing import absltest
from absl.testing import parameterized
import coordax as cx
from flax import nnx
import jax
from neuralgcm.experimental.core import boundaries
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import spherical_transforms
from neuralgcm.experimental.core import standard_layers
from neuralgcm.experimental.core import transformer_layers
import numpy as np


class TransformerLayersTest(parameterized.TestCase):
  """Tests transformer layers instantiate and produce expected outputs."""

  def setUp(self):
    super().setUp()
    self.grid = coordinates.LonLatGrid.T21()
    self.levels = coordinates.SigmaLevels.equidistant(12)
    self.hidden_size = 8
    self.per_head_kv_size = 3
    self.dense_factory = functools.partial(
        standard_layers.Mlp.uniform,
        hidden_layers=1,
        hidden_size=self.hidden_size,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='no_gating',
          input_size=6,
          output_size=3,
          num_heads=2,
          gating=lambda skip, x: x,
      ),
      dict(
          testcase_name='gating_where_possible',
          input_size=5,
          output_size=2,
          num_heads=3,
          gating=None,
      ),
  )
  def test_transformer_blocks_self_attention(
      self,
      input_size: int,
      output_size: int,
      num_heads: int,
      gating: transformer_layers.Gating,
  ):
    """Tests output_shape of TransformerBlocks."""
    n_levels = self.levels.shape[0]
    net = transformer_layers.TransformerBlocks.build_using_factories(
        input_size,
        output_size,
        intermediate_sizes=[self.hidden_size, self.hidden_size],
        num_heads=num_heads,
        dense_factory=self.dense_factory,
        qkv_features=(num_heads * self.per_head_kv_size),
        gating=gating,
        rngs=nnx.Rngs(0),
    )
    rng = jax.random.key(0)
    inputs = jax.random.uniform(rng, (input_size, n_levels))
    out = net(inputs)
    self.assertEqual(out.shape, (output_size, n_levels))

  @parameterized.named_parameters(
      dict(
          testcase_name='no_gating',
          output_size=3,
          num_heads=2,
          gating=lambda skip, x: x,
      ),
      dict(
          testcase_name='gating_where_possible',
          output_size=2,
          num_heads=3,
          gating=None,
      ),
      dict(
          testcase_name='with_norm_layer',
          output_size=2,
          num_heads=3,
          gating=None,
          normalization=nnx.LayerNorm,
      ),
      dict(
          testcase_name='with_dynamic_tanh',
          output_size=2,
          num_heads=3,
          gating=None,
          normalization=transformer_layers.DyT,
      ),
  )
  def test_transformer_blocks_decode(
      self,
      output_size: int,
      num_heads: int,
      gating: transformer_layers.Gating,
      normalization: transformer_layers.NormalizationFactory | None = None,
  ):
    """Tests output_shape of TransformerBlocks."""
    n_levels = self.levels.shape[0]
    # NOTE: In this case we apply both pre and post normalization layers,
    # this is not a recommended configuration, but works as an effective test.
    net = transformer_layers.TransformerBlocks.build_using_factories(
        self.hidden_size,
        output_size,
        intermediate_sizes=[self.hidden_size, self.hidden_size],
        num_heads=num_heads,
        dense_factory=self.dense_factory,
        qkv_features=(num_heads * self.per_head_kv_size),
        gating=gating,
        pre_normalization_factory=normalization,
        post_normalization_factory=normalization,
        rngs=nnx.Rngs(0),
    )
    rng = jax.random.key(0)
    inputs = jax.random.uniform(rng, (self.hidden_size, n_levels))
    levels_in_latents = 13
    latents = jax.random.uniform(rng, (self.hidden_size, levels_in_latents))
    out = net(inputs, latents=latents)
    self.assertEqual(out.shape, (output_size, n_levels))

  @parameterized.named_parameters(
      dict(
          testcase_name='no_shifts_even_window_shape_no_gating',
          input_size=6,
          output_size=3,
          n_layers=2,
          num_heads=2,
          inputs_window_shape=(2, 2),
          shift_windows=False,
          gating=lambda skip, x: x,
      ),
      dict(
          testcase_name='no_shifts_odd_window_shape_gating_where_possible',
          input_size=5,
          output_size=2,
          n_layers=1,
          inputs_window_shape=(3, 3),
          shift_windows=False,
          num_heads=3,
          gating=None,
      ),
      dict(
          testcase_name='shifts_even_window_shape_no_gating',
          input_size=6,
          output_size=3,
          n_layers=2,
          num_heads=2,
          inputs_window_shape=(2, 2),
          shift_windows=True,
          gating=lambda skip, x: x,
      ),
      dict(
          testcase_name='shifts_odd_window_shape_gating_where_possible',
          input_size=5,
          output_size=2,
          n_layers=1,
          inputs_window_shape=(3, 3),
          shift_windows=True,
          num_heads=3,
          gating=None,
      ),
  )
  def test_window_transformer_blocks_self_attention(
      self,
      input_size: int,
      output_size: int,
      n_layers: int,
      inputs_window_shape: tuple[int, int],
      shift_windows: bool,
      num_heads: int,
      gating: transformer_layers.Gating,
  ):
    """Tests output_shape of TransformerBlocks."""
    ylm_mapper = spherical_transforms.YlmMapper(
        mesh=parallelism.Mesh(),
        partition_schema_key=None,
    )
    ylm_pe = transformer_layers.spherical_harmonic_lon_lat_encodings(
        ylm_mapper.ylm_transform(self.grid), 4
    )
    rngs = nnx.Rngs(0)
    relative_bias_net = nnx.Linear(ylm_pe.shape[0], num_heads, rngs=rngs)
    net = transformer_layers.WindowTransformerBlocks.build_using_factories(
        input_size,
        output_size,
        intermediate_sizes=([self.hidden_size] * n_layers),
        num_heads=num_heads,
        relative_bias_net=relative_bias_net,
        inputs_window_shape=inputs_window_shape,
        qkv_features=num_heads * self.per_head_kv_size,
        shift_windows=shift_windows,
        dense_factory=self.dense_factory,
        gating=gating,
        inputs_bc=boundaries.LonLatBoundary(),
        rngs=rngs,
    )
    rng = jax.random.key(0)
    inputs = jax.random.uniform(rng, (input_size,) + self.grid.shape)
    out = net(inputs, inputs_pos_encoding=ylm_pe)
    self.assertEqual(out.shape, (output_size,) + self.grid.shape)

  def test_spherical_positional_encoder(self):
    """Tests output_shape of SphericalPositionalEncoder."""
    ylm_mapper = spherical_transforms.YlmMapper(
        mesh=parallelism.Mesh(),
        partition_schema_key=None,
    )
    lmax = 21
    lmin = 0
    pe = transformer_layers.SphericalPositionalEncoder(ylm_mapper, lmax, lmin)
    grid = coordinates.LonLatGrid.T21()
    inputs = cx.wrap(np.ones(grid.shape), grid)
    pos_encoding = pe(inputs, ('longitude', 'latitude'))
    self.assertEqual(pos_encoding.shape, (lmax**2 - lmin**2,) + grid.shape)


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
