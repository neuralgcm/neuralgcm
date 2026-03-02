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
"""Transformation modules that convert or pre/post process data structures."""

import dataclasses
import functools
import re
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple
from dinosaur import coordinate_systems
from dinosaur import pytree_utils
from dinosaur import sigma_coordinates
from dinosaur import typing
import gin
import haiku as hk
import jax
import jax.numpy as jnp
from neuralgcm.legacy import filters
import numpy as np


KeyWithCosLatFactor = typing.KeyWithCosLatFactor
TransformModule = typing.TransformModule


@gin.register
class EmptyTransform(hk.Module):
  """Transform returns an empty dict."""

  def __init__(self, *args, name: Optional[str] = None):
    del args  # unused.
    super().__init__(name=name)

  def __call__(self, inputs) -> typing.Pytree:
    return {}


@gin.register
class IdentityTransform(hk.Module):
  """Transform does not modify inputs."""

  def __init__(self, *args, name: Optional[str] = None, **kwargs):
    del args, kwargs  # unused.
    super().__init__(name=name)

  def __call__(self, inputs: typing.PyTreeState) -> typing.PyTreeState:
    return inputs


@gin.register
class ShiftAndNormalize(hk.Module):
  """Transforms inputs by shifting and normalizing values by `shifts/scales`."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      shifts: typing.Pytree,
      scales: typing.Pytree,
      features_to_exclude: Sequence[str] = tuple(),
      global_scale: Optional[float] = None,
      name: Optional[str] = None,
  ):
    del coords, dt, physics_specs, aux_features  # unused.
    super().__init__(name=name)
    self.shifts = shifts
    if global_scale is not None:
      scales = jax.tree_util.tree_map(lambda x: x * global_scale, scales)
    self.scales = scales

  def __call__(self, inputs):
    inputs, from_dict_fn = pytree_utils.as_dict(inputs)
    shifts = pytree_utils.replace_with_matching_or_default(
        inputs, self.shifts, default=None, check_used_all_replace_keys=False)
    scales = pytree_utils.replace_with_matching_or_default(
        inputs, self.scales, default=None, check_used_all_replace_keys=False)
    # if shifts/scales have missing values present in `inputs`, we insert `None`
    # for the default. If corresponding `inputs` is not `None`, this will raise
    # an error, as expected. This works because tree_map skips `None` values in
    # the first argument, as long as all dictionary keys match.
    result = jax.tree_util.tree_map(
        lambda x, y, z: (x - y) / z, inputs, shifts, scales)
    return from_dict_fn(result)


@gin.register
class InverseShiftAndNormalize(hk.Module):
  """Inverse of the `ShiftAndNormalize` for the same `shifts/scales`."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      shifts: typing.Pytree,
      scales: typing.Pytree,
      global_scale: Optional[float] = None,
      name: Optional[str] = None,
  ):
    del coords, dt, physics_specs, aux_features  # unused.
    super().__init__(name=name)
    self.shifts = shifts
    if global_scale is not None:
      scales = jax.tree_util.tree_map(lambda x: x * global_scale, scales)
    self.scales = scales

  def __call__(self, inputs):
    inputs, from_dict_fn = pytree_utils.as_dict(inputs)
    shifts = pytree_utils.replace_with_matching_or_default(
        inputs, self.shifts, default=None, check_used_all_replace_keys=False)
    scales = pytree_utils.replace_with_matching_or_default(
        inputs, self.scales, default=None, check_used_all_replace_keys=False)
    # if shifts/scales have missing values present in `inputs`, we insert `None`
    # for the default. If corresponding `inputs` is not `None`, this will raise
    # an error, as expected. This works because tree_map skips `None` values in
    # the first argument, as long as all dictionary keys match.
    result = jax.tree_util.tree_map(
        lambda x, y, z: (None if x is None else x * z + y),
        inputs,
        shifts,
        scales,
        is_leaf=lambda x: x is None,
    )
    return from_dict_fn(result)


@gin.register
class ToModalWithDivCurlTransform(hk.Module):
  """Module that converts inputs to modal replacing velocity with div/curl."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      name: Optional[str] = None,
  ):
    del dt, physics_specs, aux_features  # unused.
    super().__init__(name=name)
    self.coords = coords

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    if 'u' not in inputs or 'v' not in inputs:
      raise ValueError('Inputs to ToModalWithDivCurlTransform must include `u, '
                       f'v`, got keys: {inputs.keys()}')
    sec_lat = 1 / self.coords.horizontal.cos_lat
    u, v = inputs.pop('u'), inputs.pop('v')
    # here u,v stand for velocity / cos(lat), but the cos(lat) is cancelled in
    # divergence and curl operators below.
    inputs['u'] = u * sec_lat
    inputs['v'] = v * sec_lat
    to_modal_fn = lambda x: (self.coords.horizontal.to_modal(x)  # pylint: disable=g-long-lambda
                             if x is not None else None)
    modal_outputs = jax.tree_util.tree_map(to_modal_fn, inputs)
    u, v = modal_outputs.pop('u'), modal_outputs.pop('v')
    modal_outputs['divergence'] = self.coords.horizontal.div_cos_lat((u, v))
    modal_outputs['vorticity'] = self.coords.horizontal.curl_cos_lat((u, v))
    return modal_outputs


@gin.register
class ToModalDiffOperators(hk.Module):
  """Module that returns grad and laplacian features of inputs fields.

  To avoid accidental accumulation of the cos(lat) factors, features must be
  keyed using typing.KeyWithCosLatFactor namedtuple.
  """

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      name: Optional[str] = None,
  ):
    del dt, physics_specs, aux_features  # unused.
    super().__init__(name=name)
    self.coords = coords

  def __call__(
      self,
      inputs: Mapping[typing.KeyWithCosLatFactor, typing.Array],
  ) -> Mapping[typing.KeyWithCosLatFactor, typing.Array]:
    features = {}
    for k, value in inputs.items():
      name, cos_lat_order = k.name, k.factor_order
      d_value_dlon, d_value_dlat = self.coords.horizontal.cos_lat_grad(value)
      laplacian_value = self.coords.horizontal.laplacian(value)
      dlon_key = typing.KeyWithCosLatFactor(name + '_dlon', cos_lat_order + 1)
      dlat_key = typing.KeyWithCosLatFactor(name + '_dlat', cos_lat_order + 1)
      del2_key = typing.KeyWithCosLatFactor(name + '_del2', cos_lat_order)
      features[dlon_key] = d_value_dlon
      features[dlat_key] = d_value_dlat
      features[del2_key] = laplacian_value
    return features


@gin.register
class ModalToNodalTransform(hk.Module):
  """Transform that converts modal inputs to nodal representation."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      name: Optional[str] = None,
  ):
    del dt, physics_specs, aux_features  # unused.
    super().__init__(name=name)
    self.coords = coords

  def __call__(self, inputs: typing.PyTreeState) -> typing.PyTreeState:
    return self.coords.horizontal.to_nodal(inputs)


@gin.register
class NodalToModalTransform(hk.Module):
  """Transform that converts nodal inputs to modal representation."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      name: Optional[str] = None,
  ):
    del dt, physics_specs, aux_features  # unused.
    super().__init__(name=name)
    self.coords = coords

  def __call__(self, inputs: typing.PyTreeState) -> typing.PyTreeState:
    return self.coords.horizontal.to_modal(inputs)


@gin.register
class ClipTransform(hk.Module):
  """Transform that clips highest total wavenumber in inputs."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      wavenumbers_to_clip: int = 1,
      name: Optional[str] = None,
  ):
    """See `time_integration.exponential_filter` for details."""
    del dt, physics_specs, aux_features  # unused.
    super().__init__(name=name)
    self.coords = coords
    self.wavenumbers_to_clip = wavenumbers_to_clip

  def __call__(self, inputs: typing.PyTreeState) -> typing.PyTreeState:
    return self.coords.horizontal.clip_wavenumbers(
        inputs, self.wavenumbers_to_clip
    )


@gin.register
class NondimensionalizeTransform(hk.Module):
  """Transform that nondimensionalizes inputs."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      input_coords: coordinate_systems.CoordinateSystem,
      inputs_to_units_mapping: Dict[str, str],
      name: Optional[str] = None,
  ):
    """See `time_integration.exponential_filter` for details."""
    del coords, dt, aux_features, input_coords  # unused.
    super().__init__(name=name)
    self.inputs_to_units_mapping = inputs_to_units_mapping
    self.nondimensionalize = physics_specs.nondimensionalize

  def __call__(self, inputs: typing.PyTreeState) -> typing.PyTreeState:
    inputs, from_dict_fn = pytree_utils.as_dict(inputs)
    inputs_to_units_mapping = pytree_utils.replace_with_matching_or_default(
        inputs, self.inputs_to_units_mapping, default=None,
        check_used_all_replace_keys=False,
    )
    nondim_fn = lambda x, y: self.nondimensionalize(x * typing.Quantity(y))
    result = jax.tree_util.tree_map(nondim_fn, inputs, inputs_to_units_mapping)
    return from_dict_fn(result)


@gin.register
class RedimensionalizeTransform(hk.Module):
  """Transform that redimensionalizes inputs."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      output_coords: coordinate_systems.CoordinateSystem,
      inputs_to_units_mapping: Dict[str, str],
      name: Optional[str] = None,
  ):
    """See `time_integration.exponential_filter` for details."""
    del coords, dt, aux_features, output_coords  # unused.
    super().__init__(name=name)
    self.inputs_to_units_mapping = inputs_to_units_mapping
    self.dimensionalize = physics_specs.dimensionalize

  def __call__(self, inputs: typing.PyTreeState) -> typing.PyTreeState:
    inputs, from_dict_fn = pytree_utils.as_dict(inputs)
    inputs_to_units_mapping = pytree_utils.replace_with_matching_or_default(
        inputs, self.inputs_to_units_mapping, default=None,
        check_used_all_replace_keys=False,
    )
    dim_fn = lambda x, y: self.dimensionalize(x, typing.Quantity(y)).m
    result = jax.tree_util.tree_map(dim_fn, inputs, inputs_to_units_mapping)
    return from_dict_fn(result)


@gin.register
class SequentialTransform(hk.Module):
  """Transform module that combines multiple transforms applied sequentially."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      transform_modules: Sequence[TransformModule],
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.transform_fns = [module(coords, dt, physics_specs, aux_features)
                          for module in transform_modules]

  def __call__(self, inputs: typing.PyTreeState) -> typing.PyTreeState:
    for transform_fn in self.transform_fns:
      inputs = transform_fn(inputs)
    return inputs


@gin.register
class LevelScale(hk.Module):
  """Transforms inputs by scaling different vertical levels."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      scales: Sequence[float],
      keys_to_scale: Sequence[str] = tuple(),
      name: Optional[str] = None,
  ):
    del coords, dt, physics_specs, aux_features  # unused.
    super().__init__(name=name)
    self.scale_fn = functools.partial(
        coordinate_systems.scale_levels_for_matching_keys,
        scales=np.asarray(scales),
        keys_to_scale=keys_to_scale)

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    return self.scale_fn(inputs)


@gin.register
class InverseLevelScale(hk.Module):
  """Transforms inputs by inverse scaling different vertical levels."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      scales: Sequence[float],
      keys_to_scale: Sequence[str] = tuple(),
      name: Optional[str] = None,
  ):
    del coords, dt, physics_specs, aux_features  # unused.
    super().__init__(name=name)
    self.scale_fn = functools.partial(
        coordinate_systems.scale_levels_for_matching_keys,
        scales=1/np.asarray(scales),
        keys_to_scale=keys_to_scale)

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    return self.scale_fn(inputs)


@gin.register
class HardClip(hk.Module):
  """Transforms inputs by hard clipping inputs to (-max_value, max_value)."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      max_value: float,
      name: Optional[str] = None,
  ):
    del coords, dt, physics_specs, aux_features  # unused.
    super().__init__(name=name)
    self.clip_fn = functools.partial(
        jnp.clip, min=-max_value, max=max_value)

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    return jax.tree_util.tree_map(self.clip_fn, inputs)


@gin.register
class SoftClip(hk.Module):
  """Transforms inputs by clipping values to a range with smooth boundaries.

  Attributes:
    coords: horizontal and vertical descritization.
    dt: time step of the model.
    physics_specs: object describing the scales and physical constants.
    aux_features: dictionary holding static features that the model may use.
    max_value: specifies the range (-max_value, max_value) of return values.
    hinge_softness: controls the softness of the smoothing at the boundaries;
      values outside of the max_value range are mapped into intervals of width
      approximately `log(2) * hinge_softness` on the interior of each boundary.
    name: optional name of the module.
  """

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      max_value: float,
      hinge_softness: float = 1.0,
      name: Optional[str] = None,
  ):
    del coords, dt, physics_specs, aux_features  # unused.
    if max_value < 0 or hinge_softness < 0:
      raise ValueError('max_value and hinge_softness must be positive, '
                       f'{max_value=}, {hinge_softness=}')
    super().__init__(name=name)
    low = -max_value
    high = max_value
    hinge = hinge_softness
    softplus_fn = lambda x: hinge * jax.nn.softplus(x / hinge)
    self.clip_fn = lambda x: (  # pylint: disable=g-long-lambda
        -softplus_fn(high - low - softplus_fn(x - low)) *
        (high - low) / (softplus_fn(high - low)) + high)

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    return jax.tree_util.tree_map(self.clip_fn, inputs)


@gin.register
class ToModalDiffOperatorsWithFiltering(hk.Module):
  """Module that returns filtered grad and laplacian features of inputs fields.

  To avoid accidental accumulation of the cos(lat) factors, features must be
  keyed using typing.KeyWithCosLatFactor namedtuple.
  """

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      filter_attenuations: Tuple[float, ...] = tuple(),
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.coords = coords
    self.attenuations = filter_attenuations
    feature_filters = []
    for attenuation in filter_attenuations:
      feature_filters.append(
          filters.DataExponentialFilter(
              coords, dt, physics_specs, aux_features,
              order=1, attenuation=attenuation))
    self.feature_filters = feature_filters

  def __call__(
      self,
      inputs: Mapping[KeyWithCosLatFactor, typing.Array],
  ) -> Mapping[KeyWithCosLatFactor, typing.Array]:
    features = {}
    for k, value in inputs.items():
      name, cos_lat_order = k.name, k.factor_order
      for filter_fn, att in zip(self.feature_filters, self.attenuations):
        filtered_value = filter_fn(value)
        d_value_dlon, d_value_dlat = self.coords.horizontal.cos_lat_grad(
            filtered_value)
        laplacian_value = self.coords.horizontal.laplacian(filtered_value)
        # since gradient values picked up cos_lat factor we increment the
        # corresponding key. This factor is adjusted at the caller level.
        dlon_key = KeyWithCosLatFactor(
            name + f'_dlon_{att}', cos_lat_order + 1, att)
        dlat_key = KeyWithCosLatFactor(
            name + f'_dlat_{att}', cos_lat_order + 1, att)
        del2_key = KeyWithCosLatFactor(
            name + f'_del2_{att}', cos_lat_order, att)
        features[dlon_key] = d_value_dlon
        features[dlat_key] = d_value_dlat
        features[del2_key] = laplacian_value
    return features


@gin.register
class TruncateSigmaLevels(hk.Module):
  """Transform module that truncates vertical levels for specified variables."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      sigma_ranges: dict[str, Tuple[float, float]],
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    del dt, physics_specs, aux_features  # unused.
    self.sigma_ranges = sigma_ranges
    self.sigma_levels = coords.vertical.centers

  def __call__(self, inputs: typing.PyTreeState) -> typing.PyTreeState:
    """Returns `inputs` where only specified levels are retained."""

    def _slice_fn(x, sigma_range):
      """Returns `x` sliced to include values in `sigma_range`."""
      sigma_min_slice, sigma_max_slice = sigma_range
      lower_index = np.argmax((self.sigma_levels - sigma_min_slice) > 0)
      if sigma_max_slice > np.max(self.sigma_levels):
        upper_index = len(self.sigma_levels)
      else:
        upper_index = np.argmin((self.sigma_levels - sigma_max_slice) < 0)
      return x[slice(lower_index, upper_index), ...]

    def recurse_and_replace(x: dict[str, Any],
                            y: dict[str, Any],
                            default=None) -> dict[str, Any]:
      """Copy x, setting leaf values to `default` or value from y if keys match."""
      return {
          k: (
              y.get(k, default)
              if not isinstance(v, dict)
              else recurse_and_replace(v, y, default)
          )
          for k, v in x.items()
      }

    inputs, from_dict_fn = pytree_utils.as_dict(inputs)
    sigma_ranges_extended = recurse_and_replace(
        inputs, self.sigma_ranges, default=(0, 1)
    )
    outputs = jax.tree_util.tree_map(_slice_fn, inputs, sigma_ranges_extended)
    return from_dict_fn(outputs)


@gin.register
class TakeSurfaceAdjacentSigmaLevel(hk.Module):
  """Transform module that retains only the vertical level nearest to Earth surface for all variables."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    del coords, dt, physics_specs, aux_features  # unused.

  def __call__(self, inputs: typing.PyTreeState) -> typing.PyTreeState:
    """Returns `inputs` where only last sigma level is retained."""

    def _slice_fn(x):
      return x[slice(-1, None), ...]

    inputs, from_dict_fn = pytree_utils.as_dict(inputs)
    outputs = jax.tree_util.tree_map(_slice_fn, inputs)
    return from_dict_fn(outputs)


@gin.register
@dataclasses.dataclass
class FeatureSelector:
  """Features transform that retains items whose keys match against regex.

  Attributes:
    regex_patterns: regular expression pattern that specifies the set of keys
      from `inputs` that will be returned by __call__ method.
  """
  regex_patterns: str

  def __call__(
      self,
      inputs: Dict[str, typing.Array],
  ) -> Dict[str, typing.Array]:
    outputs = {}
    for k, v in inputs.items():
      if re.fullmatch(self.regex_patterns, k):
        outputs[k] = v
    return outputs


@gin.register
class BroadcastTransform:
  """Features transform that broadcasts all features."""

  def __init__(self, *args, **kwargs):
    del args, kwargs  # unused.

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    leaves, tree_def = jax.tree_util.tree_flatten(inputs)
    leaves = jnp.broadcast_arrays(*leaves)
    return jax.tree_util.tree_unflatten(tree_def, leaves)


@gin.register
class SquashLevelsTransform:
  """Transform that "squashes" values of inputs depending on their sigma level.

  Multiplies inputs by the piecewise linear values used to "squash" inputs
  by sigma level. See function χ definition at: http://screen/5V3jzU7ZFA4vVJP

  The squash is paramtereizaed by low_cutoffs and high_cutoffs.
  On Palmer 2009 (http://shortn/_56HCcQwmSS) page 4, the cutoffs for
  perturbations are given. Below are translated to sigma levels values:
    low_cutoffs: (100hPa, 50hPa)
    low_cutoffs: (0.05, 0.1),
    high_cutoffs: (1300m, 300m)
    high_cutoffs: (0.86, 0.965)

  Inputs that have a singleton or no level dimension are assumed defined at
  the highest value of sigma ("surface level").

  Attributes:
    coords: horizontal and vertical descritization.
    dt: time step of the model.
    physics_specs: object describing the scales and physical constants.
    aux_features: dictionary holding static features that the model may use.
    low_cutoffs: σ=low_cutoffs[0] is when χ starts linearly increasing from 0.
      σ=low_cutoffs[1] is when χ levels out at 1
    high_cutoffs: σ=high_cutoffs[0] is when χ starts linearly decreasing from 1.
      σ=high_cutoffs[1] is when χ reaches 0.
  """

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      low_cutoffs: Sequence[float] = (0.05, 0.1),
      high_cutoffs: Sequence[float] = (0.86, 0.965),
  ):
    del dt, physics_specs, aux_features  # unused.
    if not isinstance(coords.vertical, sigma_coordinates.SigmaCoordinates):
      raise ValueError(f'Cannot apply sigma_squash on {coords.vertical=}')
    sigma = coords.vertical.centers
    if len(low_cutoffs) != 2:
      raise ValueError(f'{len(low_cutoffs)=} but should have been 2.')
    if len(high_cutoffs) != 2:
      raise ValueError(f'{len(high_cutoffs)=} but should have been 2.')

    low_func = (sigma - low_cutoffs[0]) / (low_cutoffs[1] - low_cutoffs[0])
    high_func = (high_cutoffs[1] - sigma) / (high_cutoffs[1] - high_cutoffs[0])

    # lower_bound is a function equal to the squasher between
    # low_cutoffs[0] and high_cutoffs[1].
    # It becomes negative outside that range.
    lower_bound = np.minimum(1., np.minimum(low_func, high_func))
    self._sigma_squash = np.maximum(0., lower_bound)[:, np.newaxis, np.newaxis]

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    def squash_per_level_only(x):
      shape = jnp.shape(x)
      ndim = len(shape)
      if ndim >= 3 and shape[-3] > 1:  # If defined per-level
        return x * self._sigma_squash
      elif ndim in {2, 3}:
        return x * self._sigma_squash[-1]   # If defined at surface level
      else:
        return x
    return jax.tree_util.tree_map(squash_per_level_only, inputs)


@gin.register
def add_prefix(features: dict[str, Any], prefix: str) -> dict[str, Any]:
  """Adds prefix to keys in features."""
  return {prefix + k: v for k, v in features.items()}


def straight_through(
    f: Callable[[typing.Array], typing.Array],
) -> Callable[[typing.Array], typing.Array]:
  """Straight-through estimator of `func`.

  The "straight-through" estimator is a trick that fools auto-diff into
  assigning a constant gradient (≡ 1) to a function.
  See http://shortn/_kRQjMbF2QF

  Args:
    f: Callable mapping arrays to arrays. May be non-differentiable.

  Returns:
    g: Function g such that g(x) ≡ f(x) and g'(x) ≡ 1.
  """
  def straight_through_f(x):
    zero = x - jax.lax.stop_gradient(x)
    return zero + jax.lax.stop_gradient(f(x))
  return straight_through_f
