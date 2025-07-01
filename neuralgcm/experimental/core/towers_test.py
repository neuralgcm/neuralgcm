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

"""Tests that Towers can be instantiated with different networks."""
import functools

from absl.testing import absltest
from absl.testing import parameterized
import coordax as cx
from flax import nnx
import jax
import jax.numpy as jnp
from neuralgcm.experimental.core import boundaries
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import spherical_transforms
from neuralgcm.experimental.core import standard_layers
from neuralgcm.experimental.core import towers
from neuralgcm.experimental.core import transformer_layers
import numpy as np


class ForwardTowerTest(parameterized.TestCase):
  """Tests ForwardTower implementation."""

  def setUp(self):
    super().setUp()
    self.grid = coordinates.LonLatGrid.T21()
    self.levels = coordinates.SigmaLevels.equidistant(12)
    self.coord = cx.compose_coordinates(self.levels, self.grid)

  def test_mlp_over_grid_default_constructor(self):
    rngs = nnx.Rngs(0)
    mlp = standard_layers.Mlp.uniform(
        input_size=7, output_size=13, hidden_size=8, hidden_layers=4, rngs=rngs
    )
    tower = towers.ForwardTower(
        inputs_in_dims=('din',),
        out_dims=('dout',),
        apply_remat=False,
        neural_net=mlp,
    )
    # Note: inputs must be compatible with the `mlp`.
    inputs = cx.wrap(jnp.ones((7,) + self.grid.shape), 'din', self.grid)
    out = tower(inputs)
    self.assertEqual(out.shape, (13,) + self.grid.shape)
    self.assertEqual(out.dims, ('dout',) + self.grid.dims)
    self.assertEqual(cx.get_coordinate(out, missing_axes='skip'), self.grid)

  def test_mlp_over_grid(self):
    mlp_factory = functools.partial(
        standard_layers.Mlp.uniform,
        hidden_size=8,
        hidden_layers=4,
    )
    tower = towers.ForwardTower.build_using_factories(
        input_size=7,
        output_size=13,
        inputs_in_dims=('din',),
        out_dims=('dout',),
        apply_remat=False,
        neural_net_factory=mlp_factory,
        rngs=nnx.Rngs(0),
    )
    inputs = cx.wrap(jnp.ones((7,) + self.grid.shape), 'din', self.grid)
    out = tower(inputs)
    self.assertEqual(out.shape, (13,) + self.grid.shape)
    self.assertEqual(out.dims, ('dout',) + self.grid.dims)
    self.assertEqual(cx.get_coordinate(out, missing_axes='skip'), self.grid)

  def test_cnn_levels_over_coords(self):
    cnn_level_factory = functools.partial(
        standard_layers.CnnLevel,
        channels=[8, 8],
        kernel_sizes=5,
    )
    tower = towers.ForwardTower.build_using_factories(
        input_size=6,
        output_size=4,
        inputs_in_dims=('din', self.levels),
        out_dims=('dout', self.levels),
        apply_remat=False,
        neural_net_factory=cnn_level_factory,
        rngs=nnx.Rngs(0),
    )
    inputs = cx.wrap(jnp.ones((6,) + self.coord.shape), 'din', self.coord)
    out = tower(inputs)
    self.assertEqual(out.shape, (4,) + self.coord.shape)
    self.assertEqual(out.dims, ('dout',) + self.coord.dims)
    self.assertEqual(cx.get_coordinate(out, missing_axes='skip'), self.coord)

  def test_cnn_lon_lat_over_coords(self):
    cnn_lon_lat_factory = functools.partial(
        standard_layers.CnnLonLat.build_using_factories,
        hidden_size=7,
        hidden_layers=2,
        kernel_size=(3, 3),
    )
    level_bounds = cx.LabeledAxis('sigma_bound', self.levels.boundaries[1:-1])
    tower = towers.ForwardTower.build_using_factories(
        input_size=self.levels.shape[0],  # since mapping over levels and grid.
        output_size=level_bounds.shape[0],
        inputs_in_dims=(self.coord,),
        out_dims=(level_bounds, self.grid),
        apply_remat=True,
        neural_net_factory=cnn_lon_lat_factory,
        rngs=nnx.Rngs(0),
    )
    inputs = cx.wrap(jnp.ones(self.coord.shape), self.coord)
    out = tower(inputs)
    self.assertEqual(out.shape, level_bounds.shape + self.grid.shape)
    expected_coord = cx.compose_coordinates(level_bounds, self.grid)
    self.assertEqual(cx.get_coordinate(out), expected_coord)

  def test_epd_over_batch_coords(self):
    mlp_factory = functools.partial(
        standard_layers.Mlp.uniform,
        hidden_size=8,
        hidden_layers=4,
    )
    epd_factory = functools.partial(
        standard_layers.Epd.build_using_factories,
        encode_factory=mlp_factory,
        process_factory=mlp_factory,
        decode_factory=mlp_factory,
        latent_size=8,
        num_process_blocks=2,
        post_encode_activation=jax.nn.relu,
        pre_decode_activation=jax.nn.gelu,
    )
    tower = towers.ForwardTower.build_using_factories(
        input_size=6,
        output_size=2,
        inputs_in_dims=('d',),
        out_dims=('dout',),
        apply_remat=False,
        neural_net_factory=epd_factory,
        rngs=nnx.Rngs(0),
    )
    inputs = cx.wrap(jnp.ones((2, 6) + self.grid.shape), 'b', 'd', self.grid)
    out = tower(inputs)
    self.assertEqual(out.shape, (2, 2) + self.grid.shape)
    self.assertEqual(out.dims, ('b', 'dout') + self.grid.dims)
    self.assertEqual(cx.get_coordinate(out, missing_axes='skip'), self.grid)

  def test_raises_on_wrong_alignment(self):
    cnn_level_factory = functools.partial(
        standard_layers.CnnLevel,
        channels=[8, 8],
        kernel_sizes=5,
    )
    tower = towers.ForwardTower.build_using_factories(
        input_size=8,
        output_size=4,
        inputs_in_dims=('d', self.levels),
        out_dims=('d', self.levels),
        apply_remat=False,
        neural_net_factory=cnn_level_factory,
        rngs=nnx.Rngs(0),
    )
    transposed_inputs = cx.wrap(  # despite suitable shape, axes are misaligned.
        jnp.ones(self.levels.shape + (8,) + self.grid.shape),
        self.levels,
        'd',
        self.grid,
    )
    with self.assertRaises(ValueError):
      tower(transposed_inputs)


class TransformerTowerTest(parameterized.TestCase):
  """Tests TransformerTower implementation."""

  def setUp(self):
    super().setUp()
    self.grid = coordinates.LonLatGrid.T21()
    self.levels = coordinates.SigmaLevels.equidistant(12)
    self.coord = cx.compose_coordinates(self.levels, self.grid)

  def test_transformer_blocks_over_levels(self):
    input_size, output_size, num_heads = 6, 3, 2
    dense_factory = functools.partial(
        standard_layers.Mlp.uniform, hidden_layers=1, hidden_size=8
    )
    din = cx.DummyAxis('din', input_size)
    dout = cx.DummyAxis('dout', output_size)
    vectorized_x = cx.SizedAxis('x', 3)  # tower should vectorize over x.
    neural_net_factory = functools.partial(
        transformer_layers.TransformerBlocks.build_using_factories,
        intermediate_sizes=[8, 8],
        num_heads=num_heads,
        dense_factory=dense_factory,
        qkv_features=(num_heads * 3),
        gating=lambda skip, x: x,
    )
    tower = towers.TransformerTower.build_using_factories(
        input_size=input_size,
        output_size=output_size,
        neural_net_factory=neural_net_factory,
        inputs_in_dims=('din', self.levels),
        out_dims=('dout', self.levels),
        positional_encoder=None,
        rngs=nnx.Rngs(0),
    )
    inputs = cx.wrap(
        jnp.ones(din.shape + self.levels.shape + vectorized_x.shape),
        din,
        self.levels,
        vectorized_x,
    )
    out = tower(inputs)
    expected_out_coord = cx.compose_coordinates(dout, self.levels, vectorized_x)
    self.assertEqual(cx.get_coordinate(out), expected_out_coord)
    x_slice_0 = cx.cmap(lambda x: x[0])(out.untag(vectorized_x)).data
    x_slice_1 = cx.cmap(lambda x: x[1])(out.untag(vectorized_x)).data
    np.testing.assert_allclose(x_slice_0, x_slice_1)

  def test_transformer_blocks_decode_over_levels(self):
    input_size, output_size, num_heads = 8, 3, 2
    dense_factory = functools.partial(
        standard_layers.Mlp.uniform, hidden_layers=1, hidden_size=8
    )
    neural_net_factory = functools.partial(
        transformer_layers.TransformerBlocks.build_using_factories,
        intermediate_sizes=[8, 8],
        num_heads=num_heads,
        dense_factory=dense_factory,
        qkv_features=(num_heads * 3),
        gating=lambda skip, x: x,
    )
    latents_levels = coordinates.SigmaLevels.equidistant(5)
    tower = towers.TransformerTower.build_using_factories(
        input_size=input_size,
        output_size=output_size,
        neural_net_factory=neural_net_factory,
        inputs_in_dims=('din', self.levels),
        out_dims=('dout', self.levels),
        positional_encoder=None,
        latents_in_dims=('din', latents_levels),
        rngs=nnx.Rngs(0),
    )
    inputs = cx.wrap(
        jnp.ones((input_size,) + self.levels.shape), 'din', self.levels
    )
    latents = cx.wrap(
        jnp.ones((input_size,) + latents_levels.shape), 'din', latents_levels
    )
    out = tower(inputs, latents=latents)
    dout = cx.DummyAxis('dout', output_size)
    expected_out_coord = cx.compose_coordinates(dout, self.levels)
    self.assertEqual(cx.get_coordinate(out), expected_out_coord)

  def test_transformer_blocks_raises_on_inconsistent_vectorized_dims(self):
    input_size, output_size, num_heads = 8, 3, 2
    dense_factory = functools.partial(
        standard_layers.Mlp.uniform, hidden_layers=1, hidden_size=8
    )
    neural_net_factory = functools.partial(
        transformer_layers.TransformerBlocks.build_using_factories,
        intermediate_sizes=[8, 8],
        num_heads=num_heads,
        dense_factory=dense_factory,
        qkv_features=(num_heads * 3),
        gating=lambda skip, x: x,
    )
    latents_levels = coordinates.SigmaLevels.equidistant(5)
    tower = towers.TransformerTower.build_using_factories(
        input_size=input_size,
        output_size=output_size,
        neural_net_factory=neural_net_factory,
        inputs_in_dims=('din', self.levels),
        out_dims=('dout', self.levels),
        positional_encoder=None,
        latents_in_dims=('din', latents_levels),
        rngs=nnx.Rngs(0),
    )
    vectorize_axis_1 = cx.SizedAxis('x', 3)
    inputs = cx.wrap(
        jnp.ones((input_size,) + self.levels.shape + vectorize_axis_1.shape),
        'din',
        self.levels,
        vectorize_axis_1,
    )
    vectorize_axis_2 = cx.SizedAxis('y', 3)
    latents = cx.wrap(
        jnp.ones((input_size,) + latents_levels.shape + vectorize_axis_2.shape),
        'din',
        latents_levels,
        vectorize_axis_2,
    )
    with self.assertRaisesRegex(
        ValueError,
        'Vectorized dimensions on .* do not match',
    ):
      tower(inputs, latents=latents)

  def test_window_transformer_over_grid(self):
    input_size, output_size, num_heads = 6, 3, 2
    dense_factory = functools.partial(
        standard_layers.Mlp.uniform, hidden_layers=1, hidden_size=8
    )
    ylm_mapper = spherical_transforms.YlmMapper(
        mesh=parallelism.Mesh(), partition_schema_key=None
    )
    lmax = 4
    positional_encoder = transformer_layers.SphericalPositionalEncoder(
        ylm_mapper, lmax
    )
    pe_channels = lmax**2
    rngs = nnx.Rngs(0)
    relative_bias_net = nnx.Linear(pe_channels, num_heads, rngs=rngs)
    neural_net_factory = functools.partial(
        transformer_layers.WindowTransformerBlocks.build_using_factories,
        intermediate_sizes=[8, 8],
        num_heads=num_heads,
        relative_bias_net=relative_bias_net,
        inputs_window_shape=(4, 4),
        qkv_features=(num_heads * 3),
        shift_windows=False,
        dense_factory=dense_factory,
        gating=lambda skip, x: x,
        inputs_bc=boundaries.LonLatBoundary(),
    )
    tower = towers.TransformerTower.build_using_factories(
        input_size=input_size,
        output_size=output_size,
        neural_net_factory=neural_net_factory,
        inputs_in_dims=('din', self.grid),
        out_dims=('dout', self.grid),
        positional_encoder=positional_encoder,
        rngs=rngs,
    )
    inputs = cx.wrap(
        jnp.ones((input_size,) + self.grid.shape), 'din', self.grid
    )
    out = tower(inputs)
    dout = cx.DummyAxis('dout', output_size)
    expected_out_coord = cx.compose_coordinates(dout, self.grid)
    self.assertEqual(cx.get_coordinate(out), expected_out_coord)


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
