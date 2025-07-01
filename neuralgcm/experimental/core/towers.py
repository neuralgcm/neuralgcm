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
"""Modules that define stacks of neural-net layers acting on spatial arrays."""

from __future__ import annotations

from typing import Callable

import coordax as cx
from flax import nnx
from neuralgcm.experimental.core import nnx_compat
from neuralgcm.experimental.core import standard_layers
from neuralgcm.experimental.core import transformer_layers
from neuralgcm.experimental.core import typing


@nnx_compat.dataclass
class ForwardTower(nnx.Module):
  """Applies `neural_net` to a single input Field `f` over `inputs_in_dims`.

  The output Field is computed by applying vectorized `neural_net` to inputs
  followed by an optional `final_activation` performed element-wise.
  `inputs_in_dims` determines the dimensions that are processed by the
  `neural_net` while all other axes are vectorized over. `out_dims` specified
  dimension or coordinates to be used for the axes produced by the `neural_net`.

  Attributes:
    neural_net: The neural network to be applied to the input.
    inputs_in_dims: Dims or coordinates over which `neural_net` is applied.
    out_dims: Dims or coordinates to attach to non-vectorized axes.
    apply_remat: Whether to apply nnx.remat to the neural network.
    final_activation: The activation function to be applied to the output.
  """

  neural_net: nnx.Module
  inputs_in_dims: tuple[str | cx.Coordinate, ...]
  out_dims: tuple[str | cx.Coordinate, ...]
  apply_remat: bool = False
  final_activation: Callable[[typing.Array], typing.Array] = lambda x: x

  def __call__(self, field: cx.Field) -> cx.Field:
    def apply_net(net, x):
      return net(x)

    field = field.untag(*self.inputs_in_dims)
    cmap_apply_net = cx.cmap(apply_net, field.named_axes, vmap=nnx.vmap)
    if self.apply_remat:
      cmap_apply_net = nnx.remat(cmap_apply_net)

    out = cmap_apply_net(self.neural_net, field)
    return cx.cmap(self.final_activation)(out.tag(*self.out_dims))

  @classmethod
  def build_using_factories(
      cls,
      input_size: int,
      output_size: int,
      *,
      inputs_in_dims: tuple[str | cx.Coordinate, ...],
      out_dims: tuple[str | cx.Coordinate, ...],
      apply_remat: bool = False,
      neural_net_factory: standard_layers.UnaryLayerFactory,
      rngs: nnx.Rngs,
  ):
    network = neural_net_factory(input_size, output_size, rngs=rngs)
    return cls(network, inputs_in_dims, out_dims, apply_remat)


# Factory typically expects input_size, output_size args, and rngs kwarg.
ForwardTowerFactory = Callable[..., ForwardTower]


@nnx_compat.dataclass
class TransformerTower(nnx.Module):
  """Applies transformer NN to a input fields over specified dimensions.

  This tower is similar to FieldTower, but includes optional `latents` and
  `mask` input to support decoding mode of the transformer nets and
  performing attention masking. The transformer `neural_net` is applied to
  `inputs` over `inputs_in_dims` and `latents` (if provided) over
  `latents_in_dims`, while vectorizing over other dimensions. The vectorized
  dimensions must match across all of the inputs. Additional positional
  encodings that are used by the transformer are computed using
  `positional_encoder` and `latents_positional_encoder` modules. As FieldTower
  this module assumes that the leading dimension processed by layers corresponds
  to "channels".

  Attributes:
    neural_net: The transformer neural network to be applied to inputs/latents.
    inputs_in_dims: Dims or coordinates over which `neural_net` is applied.
    out_dims: Dims or coordinates to attach to non-vectorized axes.
    positional_encoder: Module for generating positional encodings for inputs.
    latents_in_dims: Dims or coordinates over which to perform latent attention.
    latents_positional_encoder: Same as positional_encoder but for latents.
    apply_remat: Whether to apply nnx.remat to the neural network.
    final_activation: The activation function to be applied to the output.
  """

  neural_net: transformer_layers.TransformerBase
  inputs_in_dims: tuple[str | cx.Coordinate, ...]
  out_dims: tuple[str | cx.Coordinate, ...]
  positional_encoder: transformer_layers.PositionalEncoder
  latents_in_dims: tuple[str | cx.Coordinate, ...] | None = None
  latents_positional_encoder: transformer_layers.PositionalEncoder | None = None
  apply_remat: bool = False
  final_activation: Callable[[typing.Array], typing.Array] = lambda x: x

  def _prep_field(
      self,
      f: cx.Field | None,
      pos_enc: nnx.Module,
      dims: tuple[str | cx.Coordinate, ...],
  ) -> tuple[cx.Field | None, cx.Field | None]:
    if f is None:
      return None, None
    axis = cx.tmp_axis_name(f)
    pe = pos_enc(f, dims[1:], axis).untag(axis, *dims[1:]) if pos_enc else None
    return f.untag(*dims), pe

  def __call__(
      self,
      inputs: cx.Field,
      latents: cx.Field | None = None,
      mask: cx.Field | None = None,
  ) -> cx.Field:
    x_dims = self.inputs_in_dims
    z_dims = self.latents_in_dims if self.latents_in_dims else x_dims
    x, x_pe = self._prep_field(inputs, self.positional_encoder, x_dims)
    z, z_pe = self._prep_field(latents, self.latents_positional_encoder, z_dims)
    assert isinstance(x, cx.Field)  # make pytype happy.
    if mask is not None:
      mask = mask.untag(*z_dims)

    named_shape = (
        lambda y: () if y is None else tuple(sorted(y.named_shape.items()))
    )
    named_shapes = set([named_shape(x), named_shape(z), named_shape(mask)])
    if len(named_shapes - set([()])) > 1:
      raise ValueError(
          f'Vectorized dimensions on {x, z, mask} do not match'
          f' {[named_shape(x), named_shape(z), named_shape(mask)]}'
      )

    def apply_net(net, x, x_pe, z, z_pe, mask):
      return net(x, z, x_pe, z_pe, mask=mask)

    cmap_apply_net = cx.cmap(apply_net, x.named_axes, vmap=nnx.vmap)
    if self.apply_remat:
      cmap_apply_net = nnx.remat(cmap_apply_net)

    out = cmap_apply_net(self.neural_net, x, x_pe, z, z_pe, mask)
    return cx.cmap(self.final_activation)(out.tag(*self.out_dims))

  @classmethod
  def build_using_factories(
      cls,
      input_size: int,
      output_size: int,
      neural_net_factory: Callable[..., transformer_layers.TransformerLayer],
      inputs_in_dims: tuple[str | cx.Coordinate, ...],
      out_dims: tuple[str | cx.Coordinate, ...],
      positional_encoder: transformer_layers.PositionalEncoder,
      latents_in_dims: tuple[str | cx.Coordinate, ...] | None = None,
      latents_positional_encoder: (
          transformer_layers.PositionalEncoder | None
      ) = None,
      apply_remat: bool = False,
      final_activation: Callable[[typing.Array], typing.Array] = lambda x: x,
      *,
      rngs,
  ):
    transformer = neural_net_factory(input_size, output_size, rngs=rngs)
    return cls(
        neural_net=transformer,
        inputs_in_dims=inputs_in_dims,
        out_dims=out_dims,
        positional_encoder=positional_encoder,
        latents_in_dims=latents_in_dims,
        latents_positional_encoder=latents_positional_encoder,
        apply_remat=apply_remat,
        final_activation=final_activation,
    )


TransformerTowerFactory = Callable[..., TransformerTower]
