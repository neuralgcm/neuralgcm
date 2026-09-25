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
"""LinearTransforms for use in Metrics."""
import dataclasses
import functools
from typing import Callable, Mapping, Optional, Sequence
from dinosaur import coordinate_systems
from dinosaur import filtering
from dinosaur import horizontal_interpolation
from dinosaur import pytree_utils
from dinosaur import spherical_harmonic
from dinosaur import typing
import gin
import jax
import jax.numpy as jnp
import numpy as np

import metrics_util


Pytree = typing.Pytree
TrajectoryRepresentations = typing.TrajectoryRepresentations

tree_leaves = jax.tree_util.tree_leaves
tree_map = jax.tree_util.tree_map


@dataclasses.dataclass
class LinearTransform:
  """A linear transformation, for TransformedL2Loss."""

  trajectory_spec: metrics_util.TrajectorySpec

  def __call__(self, errors: Pytree, targets: Pytree) -> Pytree:
    raise NotImplementedError


LinearTransformConstructor = Callable[
    [metrics_util.TrajectorySpec], LinearTransform
]


@dataclasses.dataclass
class ComposedTransformForLoss(LinearTransform):
  """Composition of multiple linear transformations for computation of loss.

  Attributes:
    components: components[i](self.trajectory_spec) initializes the i + 1 member
      of self.transforms.
    transforms: errors are transformed as error --> transforms[0](error) -->
      transforms[1](error) --> ⋯. The 0th transform is inserted by this class as
      TruncateToTrajectoryLength.
  """

  components: Sequence[LinearTransformConstructor]
  transforms: Sequence[LinearTransform] = dataclasses.field(init=False)

  def __post_init__(self):
    # Insert TruncateToTrajectoryLength first in all cases. It's okay if it was
    # already inserted... it is idempotent. This ensures that
    #   len(self.transforms) = len(self.components) + 1
    # in all cases.
    components = [TruncateToTrajectoryLength] + list(self.components)
    self.transforms = [
        constructor(self.trajectory_spec) for constructor in components
    ]

  def __call__(self, errors: Pytree, targets: Pytree) -> Pytree:
    for transform in self.transforms:
      errors = transform(errors, targets)
    return errors


@gin.register
@dataclasses.dataclass
class LegacyTimeRescaling(LinearTransform):
  """Time scaling from WeightedL2CumulativeLoss."""

  def __call__(self, errors: Pytree, targets: Pytree) -> Pytree:
    del targets  # unused.
    n = self.trajectory_spec.trajectory_length
    steps_per_save = self.trajectory_spec.steps_per_save
    scale = 1 if n == 1 else 1 / np.sqrt((n - 1) * steps_per_save)
    return tree_map(lambda x: x * scale, errors)


@gin.register
@dataclasses.dataclass
class TimeRescaling(LinearTransform):
  """Time scaling that assumes error grows like a random walk.

  This rescales errors like
    errors --> errors / σ(T),
    σ(T) := sqrt( sum(variance) / variance(T) )
  where variance(T) is the assumed variance. A random walk has variance ∝ T.
  This function uses similar scaling.

  See also:
  * Climatology vs. ENS CRPS values indicate skill difficult after 240 hrs
    http://screen/8sVodqThEk6o693
  * Plotting this function for various parameter values
    http://screen/AubXNomsgm7g92o and http://gpaste/6727081386835968

  Attributes:
    base_squared_error_in_hours: Number of hours before assumed variance starts
      growing (almost) linearly.
    asymptotic_squared_error_in_hours: Number of hours before assumed variance
      slows its growth. Set to None (the default) if variance grows indefinitely
  """

  base_squared_error_in_hours: float
  asymptotic_squared_error_in_hours: Optional[float] = None

  def __call__(self, errors: Pytree, targets: Pytree) -> Pytree:
    del targets  # unused.
    time_sizes = np.unique([x.shape[0] for x in tree_leaves(errors)])
    if time_sizes.size != 1:
      raise ValueError(f'Expected unique time dimension size. {time_sizes=}')
    time_size = time_sizes[0]
    if self.trajectory_spec.trajectory_length != time_size:
      logging.info(
          f'errors has {time_size=} !='
          f' {self.trajectory_spec.trajectory_length=}. This is probably due to'
          ' the Loss slicing via the time_step kwarg. Will use {time_size=}'
          ' to compute scaling.'
      )

    steps_per_save = self.trajectory_spec.steps_per_save
    t = np.arange(time_size) * steps_per_save
    if self.asymptotic_squared_error_in_hours is not None:
      # Rescale "time" `t`, so it stops growing when
      #   t >> asymptotic_squared_error_in_hours.
      t = t / (1 + t / self.asymptotic_squared_error_in_hours)

    inv_variance = 1 / (1 + t / self.base_squared_error_in_hours)
    scale = np.sqrt(inv_variance / inv_variance.sum())
    scale = scale.reshape(-1, 1, 1, 1)

    return tree_map(lambda x: x * scale, errors)


@gin.register
@dataclasses.dataclass
class CustomTimeRescaling(LinearTransform):
  """Custom time scaling that uses pre-specified values."""

  scaling_weights: Sequence[float]

  def __call__(self, errors: Pytree, targets: Pytree) -> Pytree:
    del targets  # unused.
    n = self.trajectory_spec.trajectory_length
    scale = np.asarray(self.scaling_weights)[:n].reshape(-1, 1, 1, 1)
    return tree_map(lambda x: x * scale, errors)


@gin.register
@dataclasses.dataclass
class DelayedTimeRescaling(LinearTransform):
  """Time scaling with smooth delay that transitions into hyperbolic decay."""

  base_squared_error_in_hours: float
  delay_power: float = 1.0
  decay_power: float = 1.0

  def __call__(self, errors: Pytree, targets: Pytree) -> Pytree:
    del targets  # unused.
    n = self.trajectory_spec.trajectory_length
    steps_per_save = self.trajectory_spec.steps_per_save
    t = np.arange(n) * steps_per_save

    a = 1 / self.base_squared_error_in_hours
    inv_variance = 1 / (
        (1 + (a * t) ** self.delay_power) ** (1/self.decay_power))
    scale = np.sqrt(inv_variance / inv_variance.sum())
    scale = scale.reshape(-1, 1, 1, 1)

    return tree_map(lambda x: x * scale, errors)


@gin.register
@dataclasses.dataclass
class TruncateToTrajectoryLength(LinearTransform):
  """Truncate errors to self.trajectory_spec.trajectory_length.

  To ensure loss is computed over the correct trajectory length, this transform
  should be used as the first step in any ComposedTransformForLoss.
  """

  def __call__(self, errors: Pytree, targets: Pytree) -> Pytree:
    del targets  # unused.
    n = self.trajectory_spec.trajectory_length
    return metrics_util.extract_time_slice(errors, slice(0, n))


@gin.register
@dataclasses.dataclass
class TotalWavenumberMasking(LinearTransform):
  """Transform that masks out wavenumbers greater than `max_wavenumber`."""

  max_wavenumber: int
  is_encoded: bool = False

  def __call__(self, errors: Pytree, targets: Pytree) -> Pytree:
    del targets  # unused.
    if self.is_encoded:
      grid = self.trajectory_spec.coords.horizontal
    else:
      grid = self.trajectory_spec.data_coords.horizontal

    modal_shape = grid.modal_shape
    mask = np.arange(modal_shape[-1]) < self.max_wavenumber
    mask = mask.astype(float)
    return tree_map(lambda x: x * mask, errors)


@gin.register
@dataclasses.dataclass
class ConservativeRegridder(LinearTransform):
  """Linear transform that regrids."""

  def __init__(
      self,
      trajectory_spec: metrics_util.TrajectorySpec,
      target_grid: spherical_harmonic.Grid,
  ):
    super().__init__(trajectory_spec=trajectory_spec)
    self.regridder = horizontal_interpolation.ConservativeRegridder(
        source_grid=trajectory_spec.coords.horizontal, target_grid=target_grid
    )

  def __call__(self, errors: Pytree, targets: Pytree) -> Pytree:
    del targets  # Unused
    return tree_map(self.regridder, errors)


@gin.register
@dataclasses.dataclass
class PerVariableRescaling(LinearTransform):
  """Transform that reweights contribution per variable."""
  weights: Pytree
  scale: float = 1.0

  def __call__(self, errors: Pytree, targets: Pytree) -> Pytree:
    del targets  # unused.
    weights = self.weights
    if weights is None:
      weights = tree_map(lambda x: 1.0, errors)
    else:
      weights = pytree_utils.replace_with_matching_or_default(
          errors, weights, default=None,
          check_used_all_replace_keys=True,
      )
    root_weights = tree_map(lambda w: np.sqrt(w * self.scale), weights)
    return tree_map(jnp.multiply, errors, root_weights)


@gin.register
class ExponentialFilteringByLeadtime(LinearTransform):
  """Applied leadtime dependent exponential filters to errors."""

  def __init__(
      self,
      trajectory_spec: metrics_util.TrajectorySpec,
      filter_attenuations: typing.Pytree,
      filter_orders: typing.Pytree,
      is_encoded: bool = False,
  ):
    super().__init__(trajectory_spec=trajectory_spec)
    n = trajectory_spec.trajectory_length
    if is_encoded:
      grid = trajectory_spec.coords.horizontal
    else:
      grid = trajectory_spec.data_coords.horizontal
    # expand dims for `level, lon, total wavenumbers` so that filter parameters
    # are applied to different time values.
    to_array_fn = lambda x: np.expand_dims(np.array(x)[:n], axis=(1, 2, 3))
    is_leaf = lambda x: isinstance(x, Sequence)
    attenuations = tree_map(to_array_fn, filter_attenuations, is_leaf=is_leaf)
    orders = tree_map(to_array_fn, filter_orders, is_leaf=is_leaf)
    self.filter_fns = tree_map(
        lambda a, p: filtering.exponential_filter(grid, a, p),
        attenuations,
        orders,
    )

  def __call__(self, errors: Pytree, targets: Pytree) -> Pytree:
    del targets  # unused.
    filter_fns = pytree_utils.replace_with_matching_or_default(
        errors, self.filter_fns, default=None, check_used_all_replace_keys=True)
    return tree_map(lambda fn, err: fn(err), filter_fns, errors)


@gin.register
class LevelRescaling(LinearTransform):
  """Linear transform that scales values with vertical levels."""

  def __init__(
      self,
      trajectory_spec: metrics_util.TrajectorySpec,
      scale: Sequence[float],
      keys_to_scale: Sequence[str] = tuple(),
  ):
    super().__init__(trajectory_spec)
    self.scale_fn = functools.partial(
        coordinate_systems.scale_levels_for_matching_keys,
        scales=np.asarray(scale),
        keys_to_scale=keys_to_scale,
    )

  def __call__(self, errors: Pytree, targets: Pytree) -> Pytree:
    del targets  # unused.
    return self.scale_fn(errors)


@gin.register
class LevelRemoval(LinearTransform):
  """Linear transform that removes vertical levels."""

  def __init__(
      self,
      trajectory_spec: metrics_util.TrajectorySpec,
      keep_levels: Sequence[float],
  ):
    super().__init__(trajectory_spec)
    n_levels = trajectory_spec.data_coords.vertical.layers
    indices = jnp.array([i for i in range(n_levels) if keep_levels[i]])
    self.take_arr = lambda x: jnp.take(x, indices, axis=metrics_util.LEVEL_AXIS)

  def __call__(self, errors: Pytree, targets: Pytree) -> Pytree:
    del targets  # unused.
    return tree_map(self.take_arr, errors)
