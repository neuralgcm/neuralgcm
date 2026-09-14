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
"""Basic neural network layers for whirl/gcm codebase."""

from typing import Callable, Optional, Sequence, Tuple
from dinosaur import typing
import gin
import haiku as hk
import jax
import jax.numpy as jnp

from neuralgcm.legacy import initializers  # pylint: disable=unused-import

Array = typing.Array
GatingFactory = typing.GatingFactory
TowerFactory = typing.TowerFactory
MLP = gin.external_configurable(hk.nets.MLP)

# nonlinearities
relu = gin.external_configurable(jax.nn.relu)
gelu = gin.external_configurable(jax.nn.gelu)
silu = gin.external_configurable(jax.nn.silu)


@gin.register(denylist=['output_size'])
class MlpUniform(hk.nets.MLP):
  """MLP network with same output size in each hidden layer."""

  def __init__(
      self,
      output_size: int,
      num_hidden_units: int = gin.REQUIRED,
      num_hidden_layers: int = gin.REQUIRED,
      w_init: Optional[hk.initializers.Initializer] = None,
      b_init: Optional[hk.initializers.Initializer] = None,
      with_bias: bool = True,
      activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
      activate_final: bool = False,
      w_init_final: Optional[hk.initializers.Initializer] = None,
      b_init_final: Optional[hk.initializers.Initializer] = None,
      name: Optional[str] = None,
  ):
    hidden_output_sizes = [num_hidden_units] * num_hidden_layers
    super().__init__(
        hidden_output_sizes,
        w_init=w_init,
        b_init=b_init,
        with_bias=with_bias,
        activation=activation,
        activate_final=True,  # last layer added explicitly.
        name=name,
    )
    self.linear_final = hk.Linear(
        output_size=output_size,
        w_init=w_init_final,
        b_init=b_init_final,
        with_bias=with_bias,
        name='linear_%d' % num_hidden_layers,
    )
    self.activate_linear_final = activate_final

  def __call__(
      self,
      inputs: jax.Array,
      dropout_rate: Optional[float] = None,
      rng: Optional[jax.Array] = None,
  ) -> jax.Array:
    out = super().__call__(inputs, dropout_rate=dropout_rate, rng=rng)
    out = self.linear_final(out)
    if self.activate_linear_final:
      out = self.activation(out)
    return out


@gin.register(denylist=['output_size'])
class ConvLonLat(hk.Module):
  """Two dimensional convolutional neural network."""

  def __init__(
      self,
      output_size: int,
      kernel_shape: Tuple[int, int] = gin.REQUIRED,
      with_bias: bool = True,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self._padding = []
    for kernel_size in kernel_shape:
      pad_left = kernel_size // 2
      self._padding.append((pad_left, kernel_size - pad_left - 1))
    # Use padding='VALID': since padding is done in call, haiku trims
    self._conv_module = hk.Conv2D(
        output_channels=output_size,
        kernel_shape=kernel_shape,
        with_bias=with_bias,
        padding='VALID',
        data_format='NCHW',
    )
    # NCHW = batch (ignored), channels (sigma), height (lon), width (lat)

  def __call__(self, inputs: Array) -> Array:
    """Applies convolution to inputs."""
    # Padding order is z, x, y
    # Periodic padding in longitude (x)
    # Zero padding in latitude (y)
    inputs = jnp.pad(inputs, [(0, 0), self._padding[0], (0, 0)], mode='wrap')
    # TODO(pnorgaard): consider rotated mirror padding to simulate wrapping
    # around the N/S poles.
    inputs = jnp.pad(
        inputs, [(0, 0), (0, 0), self._padding[1]], mode='constant'
    )
    return self._conv_module(inputs)


@gin.register
class ConvLevel(hk.Conv1D):
  """1D convolution in the vertical (convolution on atmospheric columns)."""

  def __init__(
      self,
      output_channels: int,
      kernel_shape: int,
      dilation_rate: int = 1,
      padding: str = 'SAME',
      with_bias: bool = True,
      w_init: Optional[hk.initializers.Initializer] = None,
      b_init: Optional[hk.initializers.Initializer] = None,
      data_format: str = 'NCW',
      name: Optional[str] = None,
  ):
    super().__init__(
        output_channels=output_channels,
        kernel_shape=kernel_shape,
        rate=dilation_rate,
        padding=padding,
        with_bias=with_bias,
        w_init=w_init,
        b_init=b_init,
        data_format=data_format,
        name=name,
    )


@gin.register
class VerticalConvNet(hk.Module):
  """1D CNN in the vertical (convolution on atmospheric columns)."""

  def __init__(
      self,
      output_size: int,
      channels: Sequence[int],
      kernel_shapes: int | Sequence[int],
      dilation_rates: int | Sequence[int],
      padding: str = 'SAME',
      with_bias: bool = True,
      w_init: Optional[hk.initializers.Initializer] = None,
      b_init: Optional[hk.initializers.Initializer] = None,
      data_format: str = 'NCW',
      activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
      activate_final: bool = False,
      w_init_final: Optional[hk.initializers.Initializer] = None,
      b_init_final: Optional[hk.initializers.Initializer] = None,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    n_hidden = len(channels)
    if isinstance(kernel_shapes, int):
      kernel_shapes = [kernel_shapes] * (n_hidden + 1)  # +1 for output layer.
    if isinstance(dilation_rates, int):
      dilation_rates = [dilation_rates] * (n_hidden + 1)  # +1 for output layer.
    channels = list(channels) + [output_size]
    if len(set([len(channels), len(kernel_shapes), len(dilation_rates)])) != 1:
      raise ValueError(
          f'Missing kernel|dilation specs for {n_hidden + 1} '
          f'layers, got {kernel_shapes=}, {dilation_rates=}.'
      )
    w_inits = [w_init] * n_hidden + [w_init_final]
    b_inits = [b_init] * n_hidden + [b_init_final]
    params = zip(channels, kernel_shapes, dilation_rates, w_inits, b_inits)
    self.layers = []
    for c, kernel, dilation, w_init_i, b_init_i in params:
      self.layers.append(
          ConvLevel(
              output_channels=c,
              kernel_shape=kernel,
              dilation_rate=dilation,
              padding=padding,
              with_bias=with_bias,
              w_init=w_init_i,
              b_init=b_init_i,
              data_format=data_format,
          )
      )
    self.activation = activation
    self.activate_final = activate_final

  def __call__(self, inputs: Array) -> Array:
    out = inputs
    num_layers = len(self.layers)
    for i, layer in enumerate(self.layers):
      out = layer(out)
      if i < (num_layers - 1) or self.activate_final:
        out = self.activation(out)
    return out


@gin.register
class LevelTransformer(hk.Module):
  """Network that uses attention mechanism across vertical levels.

  This network is a simple variation of a transformer architecture. It is
  configurable to represent either the encoder and decoder blocks. Contrary to
  other layers, this module accepts additional optional arguments: `latents` and
  `positional_encoding` that enable it to represent computations with more
  complex dependency structure. By default these arguments have value `None`, in
  which case the network uses `inputs` and performs self-attention calculation
  throughout. If `latents` are provided, then they are used for key and value
  calculations for all attention blocks. If `positional_encoding` is provided,
  then it is used to produce the first set of queries in an attention block.
  Additionally this module supports extension with gating mechanism, generally
  resembling GTrXL transformer from https://arxiv.org/pdf/1910.06764.pdf.

  Attributes:
    output_size: desired number of channels in the output of the module.
    latent_size: latent representation size. Must be divisible by `num_heads`.
    n_layers: number of transformer blocks in the network.
    num_heads: number of attention heads in each attention layer.
    key_size: size of key/query vectors to use for computing attention scores.
    widening_factor: widening factor in dense layer at the end of each block.
    activation: activation function to apply between linear transforms.
    input_projection_net: network or layer to be used to project inputs into
      initial latent representation. If set to `None`, then input projection is
      skipped entirely (only possible if input size == latent_size).
    skip_final_projection: whether to skip final projection layer. If set to
      `True`, then requested `output_size` must be equal to `latent_size`.
    gating_module: gating mechanism to use to combine residual connection and
      dense updates. Defaults to residual connections.
    name: optional name for the module.
  """

  def __init__(
      self,
      output_size: int,
      latent_size: int,
      n_layers: int,
      num_heads: int,
      key_size: int,
      widening_factor: int = 2,
      activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.gelu,
      input_projection_net: TowerFactory = hk.Linear,
      skip_final_projection: bool = False,
      gating_module: GatingFactory = lambda: lambda x, y: x + y,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    value_size, reminder = divmod(latent_size, num_heads)
    if reminder != 0:
      raise ValueError(f'{latent_size=} is not divisible by {num_heads=}.')

    self.output_size = output_size
    self.latent_size = latent_size
    self.n_layers = n_layers
    self.num_heads = num_heads
    self.key_size = key_size
    self.value_size = value_size
    self.wide_latent_size = widening_factor * latent_size
    self.activation = activation
    self.w_init = hk.initializers.VarianceScaling(2 / self.n_layers)
    self.gating_fn = gating_module()

    if input_projection_net is not None:
      self.project_input_fn = input_projection_net(latent_size)
    else:

      def skip_with_check_fn(inputs):
        _, d = inputs.shape
        if d != latent_size:
          raise ValueError(
              f'{inputs.shape=} not compatible with {latent_size=}'
              ' Specify projection module in the transformer.'
          )
        return inputs

      self.project_input_fn = skip_with_check_fn
    if skip_final_projection:
      if output_size != self.latent_size:
        raise ValueError(
            f'Unable to skip projection for {output_size=}, '
            f'{self.latent_size=}.'
        )
      self.final_projection = lambda x: x
    else:
      self.final_projection = hk.Linear(output_size)

  @hk.transparent
  def layer_norm(self, x: jnp.ndarray) -> jnp.ndarray:
    """Applies a unique LayerNorm to x with default settings."""
    ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
    return ln(x)

  def __call__(
      self,
      inputs: Array,
      latents: Optional[Array] = None,
      positional_encoding: Optional[Array] = None,
  ) -> Array:
    """Applies transformer layer to inputs. See class docstring for details."""
    inputs = jnp.transpose(inputs)  # transpose to [levels, channels].
    h = self.project_input_fn(inputs)
    if latents is not None:
      latents = jnp.transpose(latents)
    if positional_encoding is not None:
      init_query_input = jnp.transpose(positional_encoding)
      special_query_stage = 0  # uses `positional_encoding` for first query.
    else:
      special_query_stage = -1  # ensures we pass `h_norm` to query in h_attn.
    h_dense = None  # not used in the first layer.
    last_layer_id = self.n_layers - 1
    for layer_id in range(self.n_layers - 1):
      # connects residual updates from the previous layer; skipped first time.
      h = self.gating_fn(h, h_dense) if h_dense is not None else h
      # apply layer norm before the attention block, as in GTrXL.
      h_norm = self.layer_norm(h)  # pyrefly: ignore[bad-argument-type]
      attn_block = hk.MultiHeadAttention(
          num_heads=self.num_heads,
          key_size=self.key_size,
          value_size=self.value_size,
          model_size=self.latent_size,
          w_init=self.w_init,
      )
      # attend to `latents` if in decoding stage, otherwise use self-attention.
      h_attn = attn_block(
          query=init_query_input if layer_id == special_query_stage else h_norm,  # pyrefly: ignore[unbound-name]
          key=latents if latents is not None else h_norm,
          value=latents if latents is not None else h_norm,
      )
      # connects residual updates from attention layer.
      h = self.gating_fn(h, h_attn)
      if layer_id != last_layer_id:
        dense_block = hk.Sequential([
            hk.Linear(self.wide_latent_size, w_init=self.w_init),
            self.activation,
            hk.Linear(self.latent_size, w_init=self.w_init),
        ])
        h_dense = dense_block(self.layer_norm(h))  # pyrefly: ignore[bad-argument-type]

    h_dense = self.final_projection(h)
    h_dense = jnp.transpose(h_dense)  # transpose back to [channels, levels].
    return h_dense


@gin.register(denylist=['output_size'])
class LevelBiLSTM(hk.Module):
  """Applies a bidirectional LSTM to inputs.

    This network is a bi-directional LSTM. This module accepts additional
    optional argument, window_size which determines the number of positional
    features the LSTM will use at each step. By default this argument have
    value `1`, in which case the network uses features from a single level at
    each step.

  Attributes:
    output_size: desired number of channels in the output of the module.
    hidden_size: size of the hidden state in the LSTM.
    n_layers: number of bi-directional LSTM layers in the network.
    final_activation: optional activation to be applied to the output.
    window_size: number of (local) features the LSTM will use at each step.
    name: optional name for the module.
  """
  def __init__(
      self,
      output_size: int,
      hidden_size: int = gin.REQUIRED,
      n_layers: int = gin.REQUIRED,
      final_activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
      window_size: int = 1,
      name='lstm'):
    super().__init__(name=name)
    self.hidden_size = hidden_size
    self.n_layers = n_layers
    self.final_projection = hk.Linear(output_size)
    self.final_activation = final_activation
    self.window_size = window_size

    self.fw_lstms = []
    self.bw_lstms = []
    for i in range(n_layers):
      self.fw_lstms.append(hk.LSTM(hidden_size, name=f"{name}_fw_{i}"))
      self.bw_lstms.append(hk.LSTM(hidden_size, name=f"{name}_bw_{i}"))

  def sliding_window_reshape(self, data):
    """Reshapes data to include local vertical features."""
    levels_num = data.shape[0]
    pad_start = (self.window_size - 1) // 2
    pad_end = self.window_size - 1 - pad_start
    padded_data = jnp.pad(data, [(pad_start, pad_end)] + [(0, 0)])
    feature_indices = (
        jnp.arange(self.window_size)[jnp.newaxis, :]
        + jnp.arange(levels_num)[:, jnp.newaxis]
    )
    windowed_data = padded_data[feature_indices, ...]
    windowed_data = jnp.reshape(
        windowed_data,
        [
            windowed_data.shape[0],
            windowed_data.shape[2] * windowed_data.shape[1],
        ],
    )
    return windowed_data

  def __call__(self, inputs):
    inputs = jnp.transpose(inputs)  # transpose to [levels, channels].
    if self.window_size > 1:
      inputs = self.sliding_window_reshape(inputs)
    for i in range(self.n_layers):
      #TODO(janniyuval): initializing from previous hidden state?
      fw_initial_state = self.fw_lstms[i].initial_state(None)
      bw_initial_state = self.bw_lstms[i].initial_state(None)

      fw_outputs, _ = hk.dynamic_unroll(
          self.fw_lstms[i], inputs, fw_initial_state
      )
      bw_outputs, _ = hk.dynamic_unroll(
          self.bw_lstms[i], inputs, bw_initial_state, reverse=True
      )
      outputs = jnp.concatenate([fw_outputs, bw_outputs], axis=-1)
      inputs = outputs
    h_dense = self.final_projection(outputs)  # pyrefly: ignore[unbound-name]
    if self.final_activation is not None:
      h_dense = self.final_activation(h_dense)
    h_dense = jnp.transpose(h_dense)  # transpose back to [channels, levels].
    return h_dense
