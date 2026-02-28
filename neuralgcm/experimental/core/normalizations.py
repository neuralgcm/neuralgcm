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

import coordax as cx
from flax import nnx
import jax
import jax.numpy as jnp
from neuralgcm.experimental.core import nnx_compat
from neuralgcm.experimental.core import pytree_utils
import numpy as np


class StreamingValue(nnx.Variable):
  ...


class StreamingCounter(nnx.Variable):
  ...


@nnx_compat.dataclass
class StreamNorm(nnx.Module):
  """Streaming normalization module.

  Normalizes inputs using mean and variance statistics. Mean and variance values
  are stored on the module and updated using a parallel online algorithm [1].
  The dimensions over which the normalization is performed is parameterized by
  the associated `cx.Coordinate` object in `coords` attribute.

  StreamNorm resembles BatchNorm approach, but instead of using a using batch
  dependent statistics, StreamNorm uses pre-computed statistics evaluated over
  a fixed set of batches (typically done at initialization time). This removes
  the dependence on the batch size, but misses on the stochastic adjustments
  present in BatchNorm. Similar to BatchNorm, to avoid division by zero, a small
  epsilon is added to the variance. If no statistics has been collected, the
  inputs are returned unchanged. Additionally, StreamNorm supports masking that
  marks entries to be used when updating the statistics. Value `True` in the
  mask corresponds to the entry being included in the statistics computation.

  References:
    [1] https://tinyurl.com/mve7d45a  # Online variance algorithm (wikipedia).

  Attributes:
    coords: Coordinates for keys over which normalization is done independently.
    counters: Dictionary storing "counter" values for different keys.
    means: Dictionary storing mean statistics for different keys.
    m2: Dictionary storing squared sum statistics for different keys.
    epsilon: A small float added to variance to avoid dividing by zero.
    skip_unspecified: Whether to skip normalization for inputs not in `coords`.
    allow_missing: Whether to allow inputs to be missing keys in `coords`.
  """

  coords: dict[str, cx.Coordinate]
  counters: dict[str, StreamingCounter]
  means: dict[str, StreamingValue]
  m2: dict[str, StreamingValue]
  epsilon: float
  skip_unspecified: bool
  skip_nans: bool
  allow_missing: bool

  def __init__(
      self,
      coords: dict[str, cx.Coordinate],
      epsilon: float = 1e-11,
      skip_unspecified: bool = False,
      skip_nans: bool = False,
      allow_missing: bool = True,
  ):
    """Initializes StreamingNormalizer.

    Args:
      coords: A dictionary mapping variable names to the coordinates over which
        the normalization is done independently.
      epsilon: A small float added to variance to avoid dividing by zero.
      skip_unspecified: If True, inputs with keys not in `coords` are passed
        through. If False, an error is raised for such keys.
      skip_nans: If True, ignores NaNs when updating the statistics.
      allow_missing: If True, `inputs` can be missing keys that are in `coords`.
        If False, an error is raised if `inputs` is missing keys from `coords`.
    """
    self.coords = coords
    self.epsilon = epsilon
    self.skip_unspecified = skip_unspecified
    self.skip_nans = skip_nans
    self.allow_missing = allow_missing
    self.counters = StreamingCounter({
        k: cx.field(jnp.zeros(c.shape, dtype=jnp.int32), c)
        for k, c in coords.items()
    })
    self.means = StreamingValue(
        {k: cx.field(jnp.zeros(c.shape), c) for k, c in coords.items()}
    )
    self.m2 = StreamingValue(
        {k: cx.field(jnp.zeros(c.shape), c) for k, c in coords.items()}
    )

  def _update_stats(
      self, inputs: dict[str, cx.Field], mask: cx.Field | None = None
  ):
    """Updates the statistics using the given inputs."""
    counters = self.counters.get_value()
    means = self.means.get_value()
    m2s = self.m2.get_value()
    for k, f in inputs.items():
      if k not in self.coords:
        continue
      counter = counters[k]
      mean = means[k]
      sum_squares = m2s[k]
      stat_coord = self.coords[k]
      batch_dims = tuple(d for d in f.dims if d not in stat_coord.dims)
      x = f.untag(*batch_dims)
      if cx.get_coordinate(x, missing_axes='skip') != stat_coord:
        raise ValueError(f'wrong coord on {k=}')

      w = None
      if mask is not None:
        mask_batch_dims = tuple(d for d in batch_dims if d in mask.dims)
        w = cx.cmap(lambda m: m.astype(jnp.int32))(mask.untag(*mask_batch_dims))
        # Broadcast w to x to count valid entries correctly.
        ones = cx.cmap(lambda x: jnp.ones_like(x, dtype=jnp.int32))(x)
        w = w * ones

      if self.skip_nans:
        is_not_nan = cx.cmap(lambda arr: (~jnp.isnan(arr)).astype(jnp.int32))(x)
        if w is None:
          w = is_not_nan
        else:
          w = w * is_not_nan

      if w is None:
        batch_size = np.prod([f.named_shape[d] for d in batch_dims])
        count_inc = batch_size
        batch_mean = cx.cmap(jnp.mean)(x)
        # Note: we need population variance here (sum of squared deviations).
        batch_m2 = cx.cmap(lambda x: jnp.var(x) * x.size)(x)
      else:
        count_inc = cx.cmap(jnp.sum)(w)

        inv_count_inc = cx.cmap(lambda c: jnp.where(c > 0, 1.0 / c, 0.0))(
            count_inc
        )
        safe_x = cx.cmap(lambda x, w: jnp.where(w > 0, x, 0.0))(x, w)
        batch_mean = cx.cmap(jnp.sum)(safe_x) * inv_count_inc
        delta_batch = safe_x - batch_mean
        batch_m2 = cx.cmap(jnp.sum)(delta_batch * delta_batch * w)

      delta = batch_mean - mean
      new_counter = counter + count_inc
      inv_new_counter = cx.cmap(lambda c: jnp.where(c > 0, 1.0 / c, 0.0))(
          new_counter
      )
      mean += delta * count_inc * inv_new_counter
      term = delta * delta * counter * count_inc * inv_new_counter
      sum_squares += batch_m2 + term
      counters[k] = new_counter
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
    for k in self.coords:
      means[k] = self.means.get_value()[k]
      counter = self.counters.get_value()[k]
      divisor = counter - ddof
      var = self.m2.get_value()[k] / divisor
      ones = cx.field(jnp.ones(var.shape), var.coordinate)
      variances[k] = cx.cmap(jnp.where)(divisor > 0, var, ones)
    return means, variances

  def __call__(
      self,
      inputs: dict[str, cx.Field],
      update_stats: bool = True,
      mask: cx.Field | None = None,
  ) -> dict[str, cx.Field]:
    """Returns the current mean & std and updates state if update_stats=True."""
    input_keys = set(inputs.keys())
    coords_keys = set(self.coords.keys())

    if not self.skip_unspecified:
      unspecified_keys = input_keys - coords_keys
      if unspecified_keys:
        raise ValueError(
            f'Inputs contain keys not in coords: {unspecified_keys}'
        )

    if not self.allow_missing:
      missing_keys = coords_keys - input_keys
      if missing_keys:
        raise ValueError(
            f'Inputs are missing keys in coords: {missing_keys}'
        )
    if update_stats:
      self._update_stats(inputs, mask)

    means, variances = self.stats()
    if self.skip_unspecified or self.allow_missing:
      means = pytree_utils.replace_with_matching_or_default(
          inputs, means, check_used_all_replace_keys=False
      )
      variances = pytree_utils.replace_with_matching_or_default(
          inputs, variances, check_used_all_replace_keys=False
      )

    eps = self.epsilon
    rsqrt = cx.cmap(jax.lax.rsqrt)
    normalize = lambda x, m, var: x if m is None else (x - m) * rsqrt(var + eps)
    return jax.tree.map(
        normalize, inputs, means, variances, is_leaf=cx.is_field
    )
