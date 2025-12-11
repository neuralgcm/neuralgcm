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

import coordax as cx
from flax import nnx
import jax
import jax.numpy as jnp
from neuralgcm.experimental.core import nnx_compat
from neuralgcm.experimental.core import pytree_utils
from neuralgcm.experimental.core import typing
import numpy as np


class StreamingValue(nnx.Variable):
  ...


class StreamingCounter(nnx.Variable):
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
    self.counter = StreamingCounter(0, dtype=jnp.uint32)
    self.mean = StreamingValue(jnp.zeros(self.feature_shape))
    self.m2 = StreamingValue(jnp.zeros(self.feature_shape))

  def stats(self, ddof: float = 1) -> tuple[typing.Array, typing.Array]:
    counter = self.counter.value - ddof
    mean = self.mean.value
    var = self.m2.value / counter
    var = jnp.where(self.counter.value > 0, var, jnp.ones_like(var))
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
    original_counter = self.counter.value
    batch_shape = [inputs.shape[i] for i in batch_axes]
    batch_size = math.prod(batch_shape)
    counter = original_counter + batch_size
    delta = inputs.mean(batch_axes) - self.mean.value
    m2 = self.m2.value
    del_m2 = inputs.var(batch_axes) * batch_size
    m2 += del_m2 + delta * delta * batch_size * original_counter / counter
    self.counter.value = counter
    self.mean.value += delta * batch_size / counter
    self.m2.value = m2

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


@nnx_compat.dataclass
class StreamingNormalizer(nnx.Module):
  """Streaming normalization module.

  Normalizes input values using mean and variance statistics computed via
  a parallel online algorithm [1] over dimensions specified in `stat_coords`.
  This type of normalization is helpful as an initialization
  step during which the statistics of the input distribution is collected.
  At initialization time, the estimates are set to zero. To avoid division by
  zero, a small epsilon is added to the variance, similar to batch norm.
  If no statistics has been collected, the inputs are returned unchanged.

  References:
    [1]
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
  """

  stat_coords: dict[str, cx.Coordinate]
  counters: dict[str, StreamingCounter]
  means: dict[str, StreamingValue]
  m2: dict[str, StreamingValue]
  epsilon: float

  def __init__(
      self, stat_coords: dict[str, cx.Coordinate], epsilon: float = 1e-8
  ):
    """Initializes MeanAndStd.

    Args:
      stat_coords: A dictionary mapping variable names to their coordinates.
        These coordinates define the shape of the statistics to be computed.
      epsilon: a small float added to variance to avoid dividing by zero.
    """
    self.stat_coords = stat_coords
    self.epsilon = epsilon
    self.counters = StreamingCounter({k: 0 for k in stat_coords})
    self.means = StreamingValue(
        {k: cx.wrap(jnp.zeros(c.shape), c) for k, c in stat_coords.items()}
    )
    self.m2 = StreamingValue(
        {k: cx.wrap(jnp.zeros(c.shape), c) for k, c in stat_coords.items()}
    )

  def _update_stats(
      self, inputs: dict[str, cx.Field], mask: cx.Field | None = None
  ):
    """Updates the statistics using the given inputs."""
    if inputs.keys() != self.stat_coords.keys():
      raise ValueError(
          'Input keys must match the keys provided in stat_coords during'
          ' initialization.'
      )

    counters = self.counters.get_value()
    means = self.means.get_value()
    m2s = self.m2.get_value()
    for k, f in inputs.items():
      counter = counters[k]
      mean = means[k]
      sum_squares = m2s[k]
      stat_coord = self.stat_coords[k]
      batch_dims = tuple(d for d in f.dims if d not in stat_coord.dims)
      x = f.untag(*batch_dims)
      if cx.get_coordinate(x, missing_axes='skip') != stat_coord:
        raise ValueError(f'wrong coord on {k=}')

      if mask is None:
        batch_size = np.prod([f.named_shape[d] for d in batch_dims])
        count_inc = batch_size
        w = 1.0
      else:
        mask_batch_dims = tuple(d for d in batch_dims if d in mask.dims)
        m = mask.untag(*mask_batch_dims)
        w = cx.cmap(lambda m_arr: 1.0 - m_arr.astype(jnp.float32))(m)
        # Broadcast w to x to count valid entries correctly.
        ones = cx.cmap(jnp.ones_like)(x)
        count_inc = cx.cmap(jnp.sum)(w * ones)

      counter += count_inc
      delta = x - mean
      inv_counter = cx.cmap(lambda c: jnp.where(c > 0, 1.0 / c, 0.0))(counter)
      mean_inc = cx.cmap(jnp.sum)(delta * w)
      mean += mean_inc * inv_counter
      delta2 = x - mean
      sum_squares += cx.cmap(jnp.sum)(delta * delta2 * w)
      counters[k] = counter
      means[k] = mean
      m2s[k] = sum_squares
    self.counters.set_value(counters)
    self.means.set_value(means)
    self.m2.set_value(m2s)

  def stats(
      self, ddof: float = 1
  ) -> tuple[dict[str, cx.Field], dict[str, cx.Field]]:
    means = {}
    variances = {}
    for k in self.stat_coords:
      means[k] = self.means.get_value()[k]
      counter = self.counters.get_value()[k]
      divisor = counter - ddof
      var = self.m2.get_value()[k] / divisor
      ones = cx.wrap(jnp.ones(var.shape), var.coordinate)
      variances[k] = cx.cmap(jnp.where)(divisor > 0, var, ones)
    return means, variances

  def __call__(
      self,
      inputs: dict[str, cx.Field],
      update_stats: bool = True,
      mask: cx.Field | None = None,
  ) -> dict[str, cx.Field]:
    """Returns the current mean & std and updates state if update_stats=True."""
    if update_stats:
      self._update_stats(inputs, mask)

    means, variances = self.stats()
    rsqrt = cx.cmap(jax.lax.rsqrt)
    normalize = lambda x, m, var: (x - m) * rsqrt(var + self.epsilon)
    return jax.tree.map(
        normalize, inputs, means, variances, is_leaf=cx.is_field
    )
