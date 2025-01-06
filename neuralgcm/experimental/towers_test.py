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
"""Tests for towers.py."""
import functools
from typing import Callable

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from neuralgcm.experimental import standard_layers
from neuralgcm.experimental import towers
import numpy as np


class TowersTest(parameterized.TestCase):
  """Tests Tower primitives."""

  @parameterized.named_parameters(
      dict(
          testcase_name='_mlp_uniform',
          nn_input_shape=(12,),
          spatial_shape=(10, 20),
          output_size=4,
          nn_factory=functools.partial(
              standard_layers.MlpUniform,
              hidden_size=3,
              n_hidden_layers=4,
              use_bias=True,
          ),
      ),
      dict(
          testcase_name='_mlp_uniform_with_remat',
          nn_input_shape=(3,),
          spatial_shape=(16, 7),
          output_size=3,
          nn_factory=functools.partial(
              standard_layers.MlpUniform,
              hidden_size=1,
              n_hidden_layers=13,
              use_bias=False,
          ),
          apply_remat=True,
      ),
      dict(
          testcase_name='_cnn',
          nn_input_shape=(4, 12),  # here 12 corresponds to level axis.
          spatial_shape=(6, 6),
          output_size=2,
          nn_factory=functools.partial(
              standard_layers.CnnLevel,
              channels=(3, 5, 7),
              kernel_sizes=3,
              use_bias=False,
          ),
      ),
  )
  def test_column_tower(
      self,
      nn_input_shape: tuple[int, ...],
      spatial_shape: tuple[int, ...],
      output_size: int,
      nn_factory: standard_layers.UnaryLayerFactory,
      apply_remat: bool = False,
  ):
    input_size = nn_input_shape[0]
    tower = towers.ColumnTower(
        input_size,
        output_size,
        column_net_factory=nn_factory,
        rngs=nnx.Rngs(0),
        apply_remat=apply_remat,
    )
    inputs = jax.random.uniform(
        jax.random.key(42), nn_input_shape + spatial_shape
    )
    outputs = tower(inputs)
    with self.subTest('output_shape'):
      expected_shape = (output_size,) + nn_input_shape[1:] + spatial_shape
      self.assertEqual(outputs.shape, expected_shape)
    with self.subTest('same_params_count_as_column'):
      net = nn_factory(input_size, output_size, rngs=nnx.Rngs(1))
      expected = sum(
          [np.prod(x.shape) for x in jax.tree.leaves(nnx.state(net, nnx.Param))]
      )
      actual = sum([
          np.prod(x.shape) for x in jax.tree.leaves(nnx.state(tower, nnx.Param))
      ])
      self.assertEqual(actual, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name='_mlp_uniform',
          nn_input_shape=(12,),
          spatial_shape=(10, 20),
          output_size=4,
          tower_factory=functools.partial(
              towers.ColumnTower,
              column_net_factory=functools.partial(
                  standard_layers.MlpUniform, hidden_size=3, n_hidden_layers=4
              ),
          ),
      ),
      dict(
          testcase_name='_cnn',
          nn_input_shape=(4, 12),  # here 12 corresponds to level axis.
          spatial_shape=(6, 6),
          output_size=2,
          tower_factory=functools.partial(
              towers.ColumnTower,
              column_net_factory=functools.partial(
                  standard_layers.CnnLevel,
                  channels=(3, 5, 7),
                  kernel_sizes=3,
                  use_bias=False,
              ),
          ),
      ),
  )
  def test_epd_tower(
      self,
      nn_input_shape: tuple[int, ...],
      spatial_shape: tuple[int, ...],
      output_size: int,
      tower_factory: towers.UnaryTowerFactory,
  ):
    input_size = nn_input_shape[0]
    tower = towers.EpdTower(
        input_size,
        output_size,
        latent_size=8,
        num_process_blocks=2,
        encode_tower_factory=tower_factory,
        process_tower_factory=tower_factory,
        decode_tower_factory=tower_factory,
        rngs=nnx.Rngs(0),
    )
    inputs = jax.random.uniform(
        jax.random.key(42), nn_input_shape + spatial_shape
    )
    outputs = tower(inputs)
    with self.subTest('output_shape'):
      expected_shape = (output_size,) + nn_input_shape[1:] + spatial_shape
      self.assertEqual(outputs.shape, expected_shape)

  @parameterized.parameters(
      dict(
          input_size=1,
          output_size=1,
          kernel_size=(5, 5),
          num_hidden_units=12,
          num_hidden_layers=2,
          dilation=1,
          activation=jax.nn.relu,
          activate_final=True,
          use_bias=True,
          apply_remat=False,
      ),
      dict(
          input_size=2,
          output_size=1,
          kernel_size=(3, 3),
          num_hidden_units=24,
          num_hidden_layers=2,
          dilation=4,
          activation=jax.nn.gelu,
          activate_final=True,
          use_bias=False,
          apply_remat=False,
      ),
      dict(
          input_size=6,
          output_size=5,
          kernel_size=(3, 3),
          num_hidden_units=12,
          num_hidden_layers=0,
          dilation=2,
          activation=jax.nn.gelu,
          activate_final=False,
          use_bias=True,
          apply_remat=True,
      ),
  )
  def test_conv_lon_lat_tower(
      self,
      input_size: int,
      output_size: int,
      kernel_size: tuple[int, int],
      num_hidden_units: int,
      num_hidden_layers: int,
      dilation: int,
      activation: Callable[[jnp.ndarray], jnp.ndarray],
      activate_final: bool,
      use_bias: bool,
      apply_remat: bool,
  ):
    conv_tower = towers.ConvLonLatTower(
        input_size=input_size,
        output_size=output_size,
        kernel_size=kernel_size,
        num_hidden_units=num_hidden_units,
        num_hidden_layers=num_hidden_layers,
        dilation=dilation,
        activation=activation,
        activate_final=activate_final,
        use_bias=use_bias,
        apply_remat=apply_remat,
        rngs=nnx.Rngs(0),
    )

    params = nnx.state(conv_tower, nnx.Param)

    with self.subTest('output_shape'):
      inputs = jnp.ones((input_size, 56, 45))
      outputs = conv_tower(inputs)
      self.assertEqual(outputs.shape, (output_size,) + inputs.shape[1:])

    with self.subTest('total_params_count'):
      expected = count_conv_tower_params(
          input_size,
          output_size,
          kernel_size,
          num_hidden_units,
          num_hidden_layers,
          use_bias,
      )
      actual = sum([np.prod(x.shape) for x in jax.tree.leaves(params)])
      self.assertEqual(actual, expected)

  @parameterized.parameters(
      dict(
          input_size=1,
          output_size=1,
          hidden_sizes=(1, 1),
          conv_block_hidden_layers=2,
          use_bias=True,
          kernel_size=(5, 5),
          dilations=(1, 2, 4),
          activation=jax.nn.gelu,
          apply_remat=False,
          num_hidden_layers_skip_residual=0,
      ),
      dict(
          input_size=1,
          output_size=1,
          hidden_sizes=(1, 1),
          conv_block_hidden_layers=2,
          use_bias=True,
          kernel_size=(5, 5),
          dilations=1,
          activation=jax.nn.relu,
          apply_remat=False,
          num_hidden_layers_skip_residual=0,
      ),
      dict(
          input_size=3,
          output_size=1,
          hidden_sizes=(1, 2, 9),
          conv_block_hidden_layers=2,
          use_bias=True,
          kernel_size=(3, 3),
          dilations=(1, 2, 4, 1),
          activation=jax.nn.tanh,
          apply_remat=False,
          num_hidden_layers_skip_residual=1,
      ),
      dict(
          input_size=10,
          output_size=5,
          hidden_sizes=(1, 1),
          conv_block_hidden_layers=2,
          use_bias=True,
          kernel_size=(3, 3),
          dilations=(1, 2, 1),
          activation=jax.nn.relu,
          apply_remat=True,
          num_hidden_layers_skip_residual=2,
      ),
  )
  def test_resnet(
      self,
      input_size: int,
      output_size: int,
      hidden_sizes: tuple[int, int],
      conv_block_hidden_layers: int,
      use_bias: bool,
      kernel_size: tuple[int, int],
      dilations: tuple[int, ...] | int,
      activation: Callable[[jnp.ndarray], jnp.ndarray],
      apply_remat: bool,
      num_hidden_layers_skip_residual: int,
  ):

    resnet_tower = towers.ResNet(
        input_size=input_size,
        output_size=output_size,
        conv_block_num_channels=hidden_sizes,
        conv_block_num_hidden_layers=conv_block_hidden_layers,
        use_bias=use_bias,
        kernel_size=kernel_size,
        dilations=dilations,
        activation=activation,
        num_hidden_layers_skip_residual=num_hidden_layers_skip_residual,
        apply_remat=apply_remat,
        rngs=nnx.Rngs(0),
    )
    with self.subTest('output_shape'):
      inputs = jnp.ones((input_size, 128, 64))
      outputs = resnet_tower(inputs)
      self.assertEqual(outputs.shape, (output_size,) + inputs.shape[1:])
      inputs = jnp.ones((input_size, 127, 61))
      outputs = resnet_tower(inputs)
      self.assertEqual(outputs.shape, (output_size,) + inputs.shape[1:])


def count_conv_tower_params(
    input_size: int,
    output_size: int,
    kernel_size: tuple[int, int],
    num_hidden_units: int,
    num_hidden_layers: int,
    use_bias: bool,
):

  def _count_conv_params(input_size, kernel_size, output_size, use_bias):
    n_w_params = np.prod(kernel_size) * input_size * output_size
    n_b_params = output_size
    return n_w_params + n_b_params * use_bias

  param_count = 0
  param_count += _count_conv_params(
      input_size, kernel_size, num_hidden_units, use_bias
  )
  for _ in range(num_hidden_layers):
    param_count += _count_conv_params(
        num_hidden_units, kernel_size, num_hidden_units, use_bias
    )
  param_count += _count_conv_params(
      num_hidden_units, kernel_size, output_size, use_bias
  )
  return param_count


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
