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

"""Modules that implement transformations between dicts of coordax.Fields.

Transforms are mappings from dict[str, Field] --> dict[str, Field].
These transformations are most often used in two different settings:
  1. To transform individual fields (subset of) within a dict.
     [e.g. rescaling, reshaping, broadcasting, changing coordinates, etc.]
  2. To generate new Fields that will be used as input features downstream.
     [e.g. input featurization, injection of staticly known features etc.]
"""

from __future__ import annotations

import abc
import re
from typing import Callable, Literal, Protocol, Sequence

import coordax as cx
from flax import nnx
import jax.numpy as jnp
import jax_datetime as jdt
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import nnx_compat
from neuralgcm.experimental.core import normalizations
from neuralgcm.experimental.core import pytree_utils
from neuralgcm.experimental.core import spatial_filters
from neuralgcm.experimental.core import spherical_transforms
from neuralgcm.experimental.core import typing
from neuralgcm.experimental.core import units


class TransformParams(nnx.Variable):
  """Custom variable class for transform parameters."""


class Transform(Protocol):
  """Protocol for pytree transforms."""

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    ...

  def output_shapes(
      self, input_shapes: dict[str, cx.Field]
  ) -> dict[str, cx.Field]:
    ...


TransformFactory = Callable[..., Transform]


class TransformABC(nnx.Module, abc.ABC):
  """Abstract base class for pytree transforms."""

  @abc.abstractmethod
  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    raise NotImplementedError()

  def output_shapes(
      self, input_shapes: dict[str, cx.Field]
  ) -> dict[str, cx.Field]:
    call_dispatch = lambda transform, inputs: transform(inputs)
    return nnx.eval_shape(call_dispatch, self, input_shapes)


def filter_fields_by_coordinate(
    f: dict[str, cx.Field], coord: cx.Coordinate
) -> dict[str, cx.Field]:
  """Returns a subset of fields in `f` that fully contain `coord`."""
  return {
      k: v
      for k, v in f.items()
      if set(coord.axes).issubset(set(cx.get_coordinate(v).axes))
  }


def _masked_nan_to_num(
    x: cx.Field, mask: cx.Field, num: float = 0.0
) -> cx.Field:
  """Replaces NaNs in `x` with `num` where mask is True."""

  mask_coord = cx.get_coordinate(mask)
  masked_nan_to_num = lambda x, m: jnp.where(m, jnp.nan_to_num(x, nan=num), x)
  [x, mask] = cx.untag([x, mask], mask_coord)
  masked_nan_to_num = cx.cmap(masked_nan_to_num, out_axes=x.named_axes)
  result = masked_nan_to_num(x, mask)
  return result.tag(mask_coord)


ApplyMaskMethods = Literal['multiply', 'nan_to_0']
ComputeMaskMethods = Literal['isnan', 'isinf', 'above', 'below', 'take']
APPLY_MASK_FNS = {
    'multiply': lambda x, mask: x * mask,
    'nan_to_0': _masked_nan_to_num,
}
COMPUTE_MASK_FNS = {
    'isnan': lambda x, t: cx.cmap(jnp.isnan)(x),
    'isinf': lambda x, t: cx.cmap(jnp.isinf)(x),
    'above': lambda x, t: cx.cmap(jnp.where)(x > t, 1, 0),
    'below': lambda x, t: cx.cmap(jnp.where)(x < t, 1, 0),
    'take': lambda x, t: x,
}


class Identity(TransformABC):
  """Returns inputs as they are."""

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    return inputs


class Empty(TransformABC):
  """Returns an empty dict."""

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    return {}


class Broadcast(TransformABC):
  """Broadcasts all fields in `inputs` to the same coordinates."""

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    # when broadcasting, fields either maintain or increase ndim. Hence it is
    # safe to attempt broadcasting to a field with largest ndim. If coordinates
    # do not align, an error will be raised during broadcasting.
    ndims = {k: v.ndim for k, v in inputs.items()}
    ref = inputs[max(ndims, key=ndims.get)]  # get key of the largest ndim.
    return {k: v.broadcast_like(ref) for k, v in inputs.items()}


@nnx_compat.dataclass
class Select(TransformABC):
  """Selects only fields whose keys match against regex.

  Attributes:
    regex_patterns: regular expression pattern that specifies the set of keys
      from `inputs` that will be returned by __call__ method.
  """

  regex_patterns: str

  def __call__(
      self,
      inputs: dict[str, cx.Field],
  ) -> dict[str, cx.Field]:
    outputs = {}
    for k, v in inputs.items():
      if re.fullmatch(self.regex_patterns, k):
        outputs[k] = v
    return outputs


@nnx_compat.dataclass
class Sequential(TransformABC):
  """Applies sequence of transforms in order."""

  transforms: Sequence[Transform]

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    for transform in self.transforms:
      inputs = transform(inputs)
    return inputs


@nnx_compat.dataclass
class Merge(TransformABC):
  """Merges outputs of multiple transforms into a single dictionary.

  Transforms that will be combined are specified as dictionary values where
  keys indicate optional feature prefix. This helps with: (1) disambiguating
  multiple differently configured features; (2) accessing feature modules of a
  configured model. By default, prefix is only added if `always_add_prefix` is
  set to True or if there's a conflict in feature names.
  """

  feature_modules: dict[str, Transform]
  always_add_prefix: bool = False

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    all_features = {}
    for name_prefix, feature_module in self.feature_modules.items():
      features = feature_module(inputs)
      for k, v in features.items():
        if k not in all_features and not self.always_add_prefix:
          feature_key = k
        else:
          feature_key = '_'.join([name_prefix, k])
          if feature_key in all_features:
            raise ValueError(f'Encountered duplicate {feature_key=}')
        all_features[feature_key] = v
    return all_features


@nnx_compat.dataclass
class Islice(TransformABC):
  """Slices all fields along `dim` at index `idx`."""

  dim: str | cx.Coordinate
  idx: int

  def __post_init__(self):
    if isinstance(self.dim, cx.Coordinate):
      assert isinstance(self.dim, cx.Coordinate)
      if self.dim.ndim != 1:  # pytype: disable=attribute-error
        raise ValueError('Islice only supports 1d slice.')

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    """Returns `inputs` where all fields are sliced along `dim` at `idx`."""
    slice_fn = lambda x: x[self.idx]
    return {k: cx.cmap(slice_fn)(v.untag(self.dim)) for k, v in inputs.items()}


class ShiftAndNormalize(TransformABC):
  """Applies (x - shift) / scale to all input fields when reverse is False.

  Attributes:
    shift: The shift to use for centering input fields/
    scale: The scale to use for normalization.
    reverse: Whether to perform the inverse transformation.
  """

  def __init__(
      self,
      shift: cx.Field,
      scale: cx.Field,
      reverse: bool = False,
  ):
    self.shift = TransformParams(shift)
    self.scale = TransformParams(scale)
    self.reverse = reverse

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    if self.reverse:
      scale_fn = lambda x, shift, scale: x * scale + shift
    else:
      scale_fn = lambda x, shift, scale: (x - shift) / scale
    shift, scale = self.shift.value, self.scale.value
    return {k: scale_fn(v, shift, scale) for k, v in inputs.items()}


class ShiftAndNormalizePerKey(TransformABC):
  """Shifts and then scales inputs per key.

  This transform applies shifts and scales on a per key basis. The specified
  `shifts` and `scales` can be a superset of input values, but if a key in the
  inputs is not present in the shifts/scales, then an error is raised.
  """

  def __init__(
      self,
      shifts: dict[str, cx.Field],
      scales: dict[str, cx.Field],
      global_scale: float | None = None,  # TODO(dkochkov): deprecate this.
      reverse: bool = False,
  ):
    if global_scale is not None:
      scales = {k: global_scale * v for k, v in scales.items()}
    self.shifts = TransformParams(shifts)
    self.scales = TransformParams(scales)
    self.reverse = reverse

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    shifts = pytree_utils.replace_with_matching_or_default(
        inputs,
        self.shifts.value,
        default=None,
        check_used_all_replace_keys=False,
    )
    scales = pytree_utils.replace_with_matching_or_default(
        inputs,
        self.scales.value,
        default=None,
        check_used_all_replace_keys=False,
    )
    if self.reverse:
      scale_fn = lambda x, shift, scale: x * scale + shift
    else:
      scale_fn = lambda x, shift, scale: (x - shift) / scale
    return {k: scale_fn(v, shifts[k], scales[k]) for k, v in inputs.items()}


@nnx_compat.dataclass
class ClipWavenumbers(TransformABC):
  """Sets top `wavenumbers_to_clip` total wavenumbers to zero in the input."""

  grid: coordinates.SphericalHarmonicGrid
  wavenumbers_to_clip: int = 1

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    """Returns `inputs` where only last sigma level is retained."""
    return self.grid.clip_wavenumbers(inputs, self.wavenumbers_to_clip)


@nnx_compat.dataclass
class Mask(TransformABC):
  """Masks input Fields with a static mask."""

  mask_key: str
  compute_mask_method: ComputeMaskMethods = 'take'
  apply_mask_method: ApplyMaskMethods = 'multiply'
  threshold_value: float | None = None
  drop_mask: bool = True

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    compute_mask = COMPUTE_MASK_FNS[self.compute_mask_method]
    mask = compute_mask(inputs[self.mask_key], self.threshold_value)
    apply_mask = APPLY_MASK_FNS[self.apply_mask_method]
    if self.drop_mask:
      inputs = {k: v for k, v in inputs.items() if k != self.mask_key}
    return {k: apply_mask(v, mask) for k, v in inputs.items()}


class Nondimensionalize(TransformABC):
  """Transform that nondimensionalizes inputs."""

  def __init__(
      self,
      sim_units: units.SimUnits,
      inputs_to_units_mapping: dict[str, str],
  ):
    self.inputs_to_units_mapping = inputs_to_units_mapping
    self.sim_units = sim_units

  def _nondim_numeric(self, x: typing.Numeric | jdt.Datetime, k: str):
    if isinstance(x, jdt.Datetime):
      return x  # Datetime is always in days/seconds units.
    if k not in self.inputs_to_units_mapping:
      raise ValueError(
          f'Key {k!r} not found in {self.inputs_to_units_mapping=}'
      )
    quantity = typing.Quantity(self.inputs_to_units_mapping[k])
    return self.sim_units.nondimensionalize(quantity * x)

  def _nondim_field(self, x: cx.Field, k: str):
    nondim_value = self._nondim_numeric(x.data, k)
    return cx.wrap_like(nondim_value, x)

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    result = {}
    for k, v in inputs.items():
      result[k] = self._nondim_field(v, k)
    return result


class Redimensionalize(TransformABC):
  """Transform that redimensionalizes inputs."""

  def __init__(
      self,
      sim_units: units.SimUnits,
      inputs_to_units_mapping: dict[str, str],
  ):
    self.inputs_to_units_mapping = inputs_to_units_mapping
    self.sim_units = sim_units

  def _redim_numeric(self, x: typing.Numeric | jdt.Datetime, k: str):
    if isinstance(x, jdt.Datetime):
      return x  # Datetime is always in days/seconds units.
    if k not in self.inputs_to_units_mapping:
      raise ValueError(f'Key {k} not found in {self.inputs_to_units_mapping=}')
    unit = typing.Quantity(self.inputs_to_units_mapping[k])
    return self.sim_units.dimensionalize(x, unit, as_quantity=False)

  def _redim_field(self, x: cx.Field, k: str):
    dim_value = self._redim_numeric(x.data, k)
    return cx.wrap_like(dim_value, x)

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    result = {}
    for k, v in inputs.items():
      result[k] = self._redim_field(v, k)
    return result


@nnx_compat.dataclass
class TanhClip(TransformABC):
  """Clips inputs to (-scale, scale) range via tanh function.

  Attributes:
    scale: A positive float that determines the range of the outputs.
  """

  scale: float

  def __post_init__(self):
    if self.scale <= 0:
      raise ValueError(f'scale must be positive, got scale={self.scale}')

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    clip_fn = cx.cmap(lambda x: self.scale * jnp.tanh(x / self.scale))
    return {k: clip_fn(v) for k, v in inputs.items()}


class StreamingStatsNormalization(TransformABC):
  """Normalizes inputs using values from streaming mean and variances."""

  def __init__(
      self,
      feature_shapes: dict[str, tuple[int, ...]],
      feature_axes: tuple[int, ...],
      update_stats: bool = False,
      epsilon: float = 1e-5,
      skip_unspecified: bool = False,
      rngs: nnx.Rngs | None = None,
  ):
    # TODO(dkochkov): Consider removing rngs from constructors if we can
    # instantiate normalization modules in the config. Currently these modules
    # are initialized via factory pattern and receive rngs as an argument.
    del rngs  # unused.
    # TODO(dkochkov): Update StreamNorm to work directly with cx.Fields to
    # avoid the need to tweak feature axes manually.
    stream_norm_transforms = {}
    for k, v in feature_shapes.items():
      if not v:  # if there's no shape, do not provide feature axis.
        stream_norm_transforms[k] = normalizations.StreamNorm(
            tuple(), tuple(), epsilon=epsilon
        )
      else:
        stream_norm_transforms[k] = normalizations.StreamNorm(
            v, feature_axes, epsilon=epsilon
        )
    self.stream_norm_transforms = stream_norm_transforms
    self.skip_unspecified = skip_unspecified
    self.update_stats = update_stats

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    transforms = pytree_utils.replace_with_matching_or_default(
        inputs,
        self.stream_norm_transforms,
        default=lambda x, _: x if self.skip_unspecified else None,
        check_used_all_replace_keys=False,
    )
    results = {}
    for k, v in inputs.items():
      results[k] = cx.wrap_like(transforms[k](v.data, self.update_stats), v)
    return results

  @classmethod
  def for_input_shapes(
      cls,
      input_shapes: dict[str, typing.ShapeDtypeStruct],
      feature_axes: tuple[int, ...],
      exclude_regex: str | None = None,
      update_stats: bool = False,
      epsilon: float = 1e-6,
      skip_unspecified: bool = False,
      rngs: nnx.Rngs | None = None,
  ):
    """Custom constructor based on input shapes that should be normalized."""
    feature_shapes = {
        k: tuple(v.shape[i] for i in feature_axes) if v.ndim > 2 else tuple()
        for k, v in input_shapes.items()
    }
    if exclude_regex is not None:
      feature_shapes = {
          k: v
          for k, v in feature_shapes.items()
          if not re.search(exclude_regex, k)
      }
    return cls(
        feature_shapes=feature_shapes,
        feature_axes=feature_axes,
        update_stats=update_stats,
        epsilon=epsilon,
        skip_unspecified=skip_unspecified,
        rngs=rngs,
    )


@nnx_compat.dataclass
class ToModal(TransformABC):
  """Transforms inputs from nodal to modal space."""

  ylm_transform: spherical_transforms.SphericalHarmonicsTransform

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    modal_outputs = {}
    for k, v in inputs.items():
      modal_outputs[k] = self.ylm_transform.to_modal(v)
    return modal_outputs


@nnx_compat.dataclass
class ToNodal(TransformABC):
  """Transforms inputs from modal to nodal space."""

  ylm_transform: spherical_transforms.SphericalHarmonicsTransform

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    nodal_outputs = {}
    for k, v in inputs.items():
      nodal_outputs[k] = self.ylm_transform.to_nodal(v)
    return nodal_outputs


class ToModalWithFilteredGradients:
  """Helper module that returns filtered grads and laplacians of inputs fields.

  Gradients are filtered with an exponential filter of order 1 and provided
  attentuations. If no attentuations are provided, then this transform returns
  no gradient features. To avoid accidental accumulation of the cos(lat)
  factors, features must be keyed using typing.KeyWithCosLatFactor namedtuple.
  """

  def __init__(
      self,
      ylm_transform: spherical_transforms.SphericalHarmonicsTransform,
      filter_attenuations: tuple[float, ...] = tuple(),
  ):
    self.ylm_transform = ylm_transform
    self.attenuations = filter_attenuations
    modal_filters = [
        spatial_filters.ExponentialModalFilter(
            ylm_transform,
            attenuation=a,
            order=1,
        )
        for a in filter_attenuations
    ]
    self.modal_filters = modal_filters

  def __call__(
      self,
      inputs: dict[typing.KeyWithCosLatFactor, cx.Field],
  ) -> dict[typing.KeyWithCosLatFactor, cx.Field]:
    ylm_grid = self.ylm_transform.modal_grid
    dinosaur_grid = self.ylm_transform.dinosaur_grid
    features = {}
    for k, x in inputs.items():
      name, cos_lat_order = k.name, k.factor_order
      x = x.untag(ylm_grid)  # will be processed by grad/laplacian functions.
      cos_lat_grad = cx.cmap(dinosaur_grid.cos_lat_grad, out_axes=x.named_axes)
      laplacian = cx.cmap(dinosaur_grid.laplacian, out_axes=x.named_axes)
      for filter_module, att in zip(self.modal_filters, self.attenuations):
        d_x_dlon, d_x_dlat = cx.tag(cos_lat_grad(x), ylm_grid)
        laplacian = laplacian(x).tag(ylm_grid)
        # since gradient values picked up cos_lat factor we increment the
        # corresponding key. This factor is adjusted at the caller level.
        dlon_key = typing.KeyWithCosLatFactor(
            name + f'_dlon_{att}', cos_lat_order + 1
        )
        dlat_key = typing.KeyWithCosLatFactor(
            name + f'_dlat_{att}', cos_lat_order + 1
        )
        del2_key = typing.KeyWithCosLatFactor(
            name + f'_del2_{att}', cos_lat_order
        )
        features[dlon_key] = filter_module.filter_modal(d_x_dlon)
        features[dlat_key] = filter_module.filter_modal(d_x_dlat)
        features[del2_key] = filter_module.filter_modal(laplacian)
    return features

  def output_shapes(
      self, input_shapes: dict[typing.KeyWithCosLatFactor, cx.Field]
  ) -> dict[typing.KeyWithCosLatFactor, cx.Field]:
    return nnx.eval_shape(self.__call__, input_shapes)
