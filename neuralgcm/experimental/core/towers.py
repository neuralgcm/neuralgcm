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

from flax import nnx
from neuralgcm.experimental import coordax as cx
from neuralgcm.experimental.core import nnx_compat
from neuralgcm.experimental.core import standard_layers
from neuralgcm.experimental.core import typing


@nnx_compat.dataclass
class ForwardTower(nnx.Module):
  """Applies `neural_net` to a single input Field `f` over `net_in_dims`.

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
