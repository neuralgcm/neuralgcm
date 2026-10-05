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
"""Defines `filtering` that aim to improve stability of integration."""

from typing import Any, Callable, Dict, Optional, Sequence, Union
from dinosaur import coordinate_systems
from dinosaur import filtering
from dinosaur import pytree_utils
from dinosaur import scales
from dinosaur import time_integration
from dinosaur import typing
import gin
import haiku as hk
import jax
import numpy as np


QuantityOrStr = Union[str, scales.Quantity]

StepFilterModule = Callable[..., typing.PyTreeStepFilterFn]
TransformModule = typing.TransformModule


#  =============================================================================
#  Step filters that attenuate modal components between time steps.
#  =============================================================================


@gin.register
class NoFilter(hk.Module):
  """Filter module that performs no filtering."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      name: Optional[str] = None,
  ):
    del coords, dt, physics_specs, aux_features  # unused.
    super().__init__(name=name)

  def __call__(
      self,
      u: typing.PyTreeState,
      u_next: typing.PyTreeState
  ) -> typing.PyTreeState:
    del u  # unused.
    return u_next


@gin.register
class ClipFilter(hk.Module):
  """Filter that clips highest total wavenumber in the next state."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      wavenumbers_to_clip: int = 1,
      name: Optional[str] = None,
  ):
    """See `time_integration.exponential_filter` for details."""
    del dt, physics_specs, aux_features  # unused.
    super().__init__(name=name)
    self.coords = coords
    self.wavenumbers_to_clip = wavenumbers_to_clip

  def __call__(
      self,
      u: typing.PyTreeState,
      u_next: typing.PyTreeState
  ) -> typing.PyTreeState:
    del u  # unused.
    return self.coords.horizontal.clip_wavenumbers(
        u_next, self.wavenumbers_to_clip
    )


@gin.register
class ExponentialLeapfrogFilter(hk.Module):
  """Filter that removes high frequency components from a spectral state."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      tau: QuantityOrStr = '0.010938',
      order: int = 18,
      cutoff: float = 0,
      name: Optional[str] = None,
  ):
    """See `time_integration.exponential_filter` for details."""
    del aux_features  # unused.
    super().__init__(name=name)
    tau = physics_specs.nondimensionalize(scales.Quantity(tau))
    self.filter_fn = time_integration.exponential_leapfrog_step_filter(
        coords.horizontal, dt, tau, order, cutoff)  # pyrefly: ignore[bad-argument-type]

  def __call__(
      self,
      u: typing.PyTreeState,
      u_next: typing.PyTreeState
  ) -> typing.PyTreeState:
    return self.filter_fn(u, u_next)


@gin.register
class ExponentialFilter(hk.Module):
  """Filter that removes high frequency components from a spectral state."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      tau: QuantityOrStr = '0.010938',
      order: int = 18,
      cutoff: float = 0,
      name: Optional[str] = None,
  ):
    """See `time_integration.exponential_step_filter` for details."""
    del aux_features  # unused.
    super().__init__(name=name)
    tau = physics_specs.nondimensionalize(scales.Quantity(tau))
    self.filter_fn = time_integration.exponential_step_filter(
        coords.horizontal, dt, tau, order, cutoff)  # pyrefly: ignore[bad-argument-type]

  def __call__(
      self,
      u: typing.PyTreeState,
      u_next: typing.PyTreeState
  ) -> typing.PyTreeState:
    return self.filter_fn(u, u_next)


@gin.register
class HorizontalDiffusionFilter(hk.Module):
  """Filter that applies implicit diffusion operator to a spectral state."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      tau: QuantityOrStr = '',
      order: int = 1,
      name: Optional[str] = None,
  ):
    """See `time_integration.horizontal_diffusion_filter` for details."""
    del aux_features  # unused.
    super().__init__(name=name)
    tau = physics_specs.nondimensionalize(scales.Quantity(tau))
    self.filter_fn = time_integration.horizontal_diffusion_step_filter(
        coords.horizontal, dt, tau, order)  # pyrefly: ignore[bad-argument-type]

  def __call__(
      self,
      u: typing.PyTreeState,
      u_next: typing.PyTreeState
  ) -> typing.PyTreeState:
    del u  # unused
    return self.filter_fn(u_next)


@gin.register
class RobertAsselinLeapfrogFilter(hk.Module):
  """Time smoothing filter."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      strength: float = 0.05,
      name: Optional[str] = None,
  ):
    """See `time_integration.robert_asselin_leapfrog_filter` for details."""
    del dt, coords, physics_specs, aux_features  # unused.
    super().__init__(name=name)
    self.filter_fn = time_integration.robert_asselin_leapfrog_filter(strength)

  def __call__(
      self,
      u: typing.PyTreeState,
      u_next: typing.PyTreeState
  ) -> typing.PyTreeState:
    return self.filter_fn(u, u_next)


@gin.register
class LearnedExponentialFilter(hk.Module):
  """Low pass filter with learned parameters."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      name: Optional[str] = None,
  ):
    del dt, physics_specs, aux_features  # unused.
    self.coords = coords
    self.a_init = hk.initializers.Constant(16)
    self.p_init = hk.initializers.Constant(18)
    self.c_init = hk.initializers.Constant(0)
    super().__init__(name=name)

  def __call__(
      self,
      u: typing.PyTreeState,
      u_next: typing.PyTreeState
  ) -> typing.PyTreeState:
    del u  # unused.
    a_logit = hk.get_parameter('attenuation_logit', shape=(), init=self.a_init)
    p_logit = hk.get_parameter('order_logit', shape=(), init=self.p_init)
    c_logit = hk.get_parameter('threshold_logit', shape=(), init=self.c_init)
    a = jax.nn.softplus(a_logit)
    p = jax.nn.softplus(p_logit)
    c = jax.nn.sigmoid(c_logit)
    filter_fn = filtering.exponential_filter(self.coords.horizontal, a, p, c)  # pyrefly: ignore[bad-argument-type]
    return filter_fn(u_next)


@gin.register
class SequentialStepFilter(hk.Module):
  """Filter module that combines multiple step filters applied sequentially."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      filter_modules: Sequence[StepFilterModule],
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.filter_fns = [module(coords, dt, physics_specs, aux_features)
                       for module in filter_modules]

  def __call__(
      self,
      u: typing.PyTreeState,
      u_next: typing.PyTreeState
  ) -> typing.PyTreeState:
    for filter_fn in self.filter_fns:
      u_next = filter_fn(u, u_next)
    return u_next


@gin.register
class LayeredStepFilter(hk.Module):
  """Filter decorator that uses varying time-scales at different levels."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      filter_module: Union[HorizontalDiffusionFilter, ExponentialFilter],
      tau_vals: Union[Sequence[float], np.ndarray],
      tau_units: QuantityOrStr,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    tau = (np.asarray(tau_vals) * scales.Quantity(tau_units))
    tau = tau[:, np.newaxis, np.newaxis]  # add spatial axes.
    self.filter_fn = filter_module(
        coords, dt, physics_specs, aux_features, tau=tau)  # pyrefly: ignore[bad-argument-count, unexpected-keyword]

  def __call__(
      self,
      u: typing.PyTreeState,
      u_next: typing.PyTreeState
  ) -> typing.PyTreeState:
    return self.filter_fn(u, u_next)  # pyrefly: ignore[not-callable]


@gin.register
class MaskedFilter(hk.Module):
  """Filter that is only applied to a part of the state."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      filter_module: StepFilterModule,
      mask: typing.Pytree,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.filter_fn = filter_module(coords, dt, physics_specs, aux_features)
    self.mask = mask

  def __call__(
      self,
      u: typing.PyTreeState,
      u_next: typing.PyTreeState
  ) -> typing.PyTreeState:
    mask = type(u_next)(**self.mask)  # convert to same structure.
    return jax.tree_util.tree_map(
        lambda x, y, b: self.filter_fn(x, y) if b else y,
        u, u_next, mask)


@gin.register
class FilterFromTransform(hk.Module):
  """Filter module that wraps a transform module."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      transform_module: TransformModule,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.transform_fn = transform_module(
        coords, dt, physics_specs, aux_features)

  def __call__(
      self,
      u: typing.PyTreeState,
      u_next: typing.PyTreeState
  ) -> typing.PyTreeState:
    return self.transform_fn(u_next)


@gin.register
class FixGlobalMeanFilter(hk.Module):
  """Filter that removes the change in the global mean of certain keys."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      keys: tuple[str, ...] = ('log_surface_pressure',),
      name: Optional[str] = None,
  ):
    del aux_features  # unused.
    super().__init__(name=name)
    self.keys = keys

  def __call__(
      self, u: typing.PyTreeState, u_next: typing.PyTreeState
  ) -> typing.PyTreeState:
    u_dict, _ = pytree_utils.as_dict(u)
    u_dict, _ = pytree_utils.flatten_dict(u_dict)
    u_next_dict, from_dict_fn = pytree_utils.as_dict(u_next)
    u_next_dict, _ = pytree_utils.flatten_dict(u_next_dict)
    for key in self.keys:
      global_mean = u_dict[key][..., 0]
      u_next_dict[key] = u_next_dict[key].at[..., 0].set(global_mean)
    u_next_dict = pytree_utils.unflatten_dict(u_next_dict)
    return from_dict_fn(u_next_dict)


#  =============================================================================
#  Filters that act on modal variables without time-step context.
#  =============================================================================


@gin.register
class DataNoFilter(hk.Module):
  """Filter module that performs no filtering."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      name: Optional[str] = None,
  ):
    del coords, dt, physics_specs, aux_features  # unused.
    super().__init__(name=name)

  def __call__(self, inputs: typing.PyTreeState) -> typing.PyTreeState:
    return inputs


@gin.register
class DataExponentialFilter(hk.Module):
  """Filter that removes high frequency components from a modal data."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      attenuation: float = 16,
      order: int = 18,
      cutoff: float = 0,
      name: Optional[str] = None,
  ):
    """See `filtering.exponential_filter` for details."""
    del dt, physics_specs, aux_features  # unused.
    super().__init__(name=name)
    self.filter_fn = filtering.exponential_filter(
        coords.horizontal, attenuation, order, cutoff)

  def __call__(self, inputs: typing.PyTreeState) -> typing.PyTreeState:
    return self.filter_fn(inputs)


@gin.register
class PerVariableDataFilter(hk.Module):
  """Filter module that applies different filters for each variable."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      per_variable_filters: Dict[str, Any],
      name: Optional[str] = None,
  ):
    """See `filtering.exponential_filter` for details."""
    super().__init__(name=name)
    self.filter_fns = jax.tree_util.tree_map(
        lambda m: m(coords, dt, physics_specs, aux_features),
        per_variable_filters)

  def __call__(self, inputs: typing.PyTreeState) -> typing.PyTreeState:
    inputs_dict, from_dict_fn = pytree_utils.as_dict(inputs)
    return from_dict_fn(jax.tree_util.tree_map(
        lambda x, fn: fn(x), inputs_dict, self.filter_fns))
