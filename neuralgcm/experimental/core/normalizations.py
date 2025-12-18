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
"""Modules that perform custom normalization transformations on arrays."""

import dataclasses
import math
from flax import nnx
import jax
import jax.numpy as jnp
from neuralgcm.experimental.core import pytree_utils
from neuralgcm.experimental.core import typing


class StreamingValue(nnx.Intermediate):
  ...


class StreamingCounter(nnx.Intermediate):
  ...


@dataclasses.dataclass
class StreamNorm(nnx.Module):
  """Streaming normalization module.

  Normalizes input values along the feature axes using streaming estimate of
  mean and variance. This type of normalization is helpful as an initialization
  step during which the statistics of the input distribution is collected.
  Arguments `feature_shape` and `feature_axes` control which entries in the
  inputs are interpreted as samples. The streaming mean and variance is computed
  using parallel online algorithm [1]. At initialization time, the estimates are
  set to zero. To avoid division by zero, a small epsilon is added to the
  variance, similar to batch normalization. If no statistics has been collected,
  the inputs are returned unchanged.

  References:
    [1]
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

  Attributes:
    feature_shape: shape of the feature dimensions.
    feature_axes: axis indices in inputs that correspond to feature dimensions.
    epsilon: a small float added to variance to avoid dividing by zero.
  """

  feature_shape: tuple[int, ...] = ()
  feature_axes: tuple[int, ...] = ()
  epsilon: float = 1e-6

  def __post_init__(self):
    self.counter = StreamingCounter(jnp.array(0, dtype=jnp.uint32))
    self.mean = StreamingValue(jnp.zeros(self.feature_shape))
    self.m2 = StreamingValue(jnp.zeros(self.feature_shape))

  def stats(self, ddof: float = 1) -> tuple[typing.Array, typing.Array]:
    counter = self.counter[...] - ddof
    mean = self.mean[...]
    var = self.m2[...] / counter
    var = jnp.where(self.counter[...] > 0, var, jnp.ones_like(var))
    return mean, var

  def _batch_axes(self, inputs: typing.Array) -> tuple[int, ...]:
    # pylint: disable=protected-access
    feature_axes = tuple(
        pytree_utils._normalize_axis(i, inputs.ndim) for i in self.feature_axes
    )
    # pylint: enable=protected-access
    return tuple([i for i in range(inputs.ndim) if i not in feature_axes])

  def update_stats(self, inputs: typing.Array):
    """Updates the streaming statistics estimates using parallel algorithm."""
    batch_axes = self._batch_axes(inputs)
    original_counter = self.counter[...]
    batch_shape = [inputs.shape[i] for i in batch_axes]
    batch_size = math.prod(batch_shape)
    counter = original_counter + batch_size
    delta = inputs.mean(batch_axes) - self.mean[...]
    m2 = self.m2[...]
    del_m2 = inputs.var(batch_axes) * batch_size
    m2 += del_m2 + delta * delta * batch_size * original_counter / counter
    self.counter[...] = counter
    self.mean[...] += delta * batch_size / counter
    self.m2[...] = m2

  def __call__(
      self,
      inputs: typing.Array,
      update_stats: bool = True,
  ) -> typing.Array:
    """Transforms `inputs` by subtracting mean and normalizing by std.

    Args:
      inputs: array to normalize using current statistics.
      update_stats: whether to update the normalization statistics.

    Returns:
      Normalized inputs.
    """
    if update_stats:
      self.update_stats(inputs)

    batch_axes = self._batch_axes(inputs)
    mean, var = self.stats()
    mean = jnp.expand_dims(mean, batch_axes)
    var = jnp.expand_dims(var, batch_axes)
    return (inputs - mean) * jax.lax.rsqrt(var + self.epsilon)
