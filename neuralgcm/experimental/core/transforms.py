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
import itertools
import re
from typing import Callable, Literal, Protocol, Sequence

import coordax as cx
from flax import nnx
import jax.nn
import jax.numpy as jnp
import jax_datetime as jdt
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import interpolators
from neuralgcm.experimental.core import nnx_compat
from neuralgcm.experimental.core import normalizations
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import pytree_utils
from neuralgcm.experimental.core import spatial_filters
from neuralgcm.experimental.core import spherical_harmonics
from neuralgcm.experimental.core import typing
from neuralgcm.experimental.core import units
import numpy as np


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

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(pytree=False, **kwargs)


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
  result = cx.cpmap(masked_nan_to_num)(*cx.untag([x, mask], mask_coord))
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


@nnx_compat.dataclass
class Sel(TransformABC):
  """Selects a slice and an index along a specified dimensions."""

  sel_arg: dict[str, slice | float | int | np.ndarray | None]
  method: Literal['nearest'] | None = None

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    """Applies sel operation to all Fields in inputs."""
    if len(self.sel_arg) != 1:
      raise ValueError('Sel only supports 1d slice.')
    # TODO(dkochkov): use .sel method of Field once added.
    dim, selection = tuple(*self.sel_arg.items())
    slices = {}
    for k, field in inputs.items():
      if dim in field.dims and selection is not None:
        if isinstance(selection, slice):
          raise NotImplementedError('Sel with slice is not supported yet')
        if self.method is None:
          matches = field.coord_fields[dim].data == selection
          indices = np.argwhere(matches).astype(int).ravel()
        elif self.method == 'nearest':
          assert np.size(selection) == 1, (
              'With method="nearest", selection must be a single value.'
          )
          indices = np.array(
              [np.argmin(np.abs(field.coord_fields[dim].data - selection))]
          )
        else:
          raise ValueError(f'Unsupported method: {self.method}')
        f = field.untag(dim)
        # pylint: disable=cell-var-from-loop
        if indices.size > 1:
          raise ValueError('Currently only single value slices are supported.')
        elif indices.size == 0:
          raise ValueError(
              f'No match found for {selection} in dim {dim} with'
              f' method={self.method}'
          )
        elif indices.size == 1:
          slices[k] = cx.cmap(lambda x: x[indices.squeeze()])(f)
          assert not slices[k].positional_shape  # should never happen.
        # pylint: enable=cell-var-from-loop
      else:
        slices[k] = field
    return slices


def _get_shared_axis(
    inputs: dict[str, cx.Field], axis: str | cx.Coordinate
) -> cx.Coordinate | str:
  """Returns shared coordinate or axis_name corresponding to `axis`."""
  # TODO(dkochkov): Always return cx.Coordinate for consistency?
  if isinstance(axis, cx.Coordinate) and axis.ndim != 1:
    raise ValueError(f'shared axis must be 1d, got {axis.ndim=}')
  ax_name = axis if isinstance(axis, str) else axis.dims[0]
  candidates = set(
      v.axes.get(ax_name, ax_name if ax_name in v.dims else None)
      for v in inputs.values()
  )
  candidates = candidates | set([ax_name])  # add fallback to ax_name.
  if None in candidates:
    raise ValueError(
        f'Cannot get shared axis for dim {ax_name} in {inputs=} because it is '
        'not present in all fields.'
    )
  ax = ax_name
  candidates.remove(ax_name)  # guaranteed to be present since added explicitly.
  if len(candidates) > 1:
    raise ValueError(f'Encountered multiple {candidates=} for axis {ax_name}')
  if len(candidates) == 1:
    ax = candidates.pop()
  return ax


@nnx_compat.dataclass
class ApplyToKeys(TransformABC):
  """Wrapper transform that is applied to a subset of keys.

  This is a helper transform that applies `transform` to `keys` and keeps the
  rest of the inputs unchanged. It is equivalent to:
  merge(select(inputs, !keys), transform(select(inputs, keys)))
  """

  transform: Transform
  keys: Sequence[str]

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    to_transform = {k: v for k, v in inputs.items() if k in self.keys}
    keep_as_is = {k: v for k, v in inputs.items() if k not in self.keys}
    return self.transform(to_transform) | keep_as_is


@nnx_compat.dataclass
class ApplyFnToKeys(TransformABC):
  """Applies a Field -> Field function to a subset of keys.

  This is a helper transform that applies `fn` to `keys`. If `include_remaining`
  is set to True, outputs include the rest of the inptus unchanged.
  """

  fn: Callable[[cx.Field], cx.Field]
  keys: Sequence[str]
  include_remaining: bool = False

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    transformed = {k: self.fn(v) for k, v in inputs.items() if k in self.keys}
    if self.include_remaining:
      transformed |= {k: v for k, v in inputs.items() if k not in self.keys}
    return transformed


@nnx_compat.dataclass
class ApplyOverAxisWithScan(TransformABC):
  """Wrapper transform that applies `transform` over `axis` using scan."""

  transform: Transform
  axis: str | cx.Coordinate
  apply_remat: bool = False

  def _out_dims_order(
      self, in_dims: tuple[str, ...], out_dims: tuple[str, ...]
  ) -> tuple[str, ...]:
    """Returns new dimensions order that aligns with in_dims where possible."""
    backfill_dims = [d for d in out_dims if d not in in_dims]
    backfill_iter = iter(backfill_dims)
    merged_dims = (
        d if d in out_dims else next(backfill_iter, None) for d in in_dims
    )
    full_iterator = itertools.chain(merged_dims, backfill_iter)
    return tuple(x for x in full_iterator if x is not None)

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    ax = _get_shared_axis(inputs, self.axis)  # raises if ax is not 1d.
    original_order = {k: v.dims for k, v in inputs.items()}
    inputs = {k: v.order_as(self.axis, ...) for k, v in inputs.items()}
    inputs = cx.untag(
        inputs, ax.dims[0] if isinstance(ax, cx.Coordinate) else ax
    )  # already checked ax.ndim == 1.

    def _process(transform, x):
      if self.apply_remat:
        processed = nnx.remat(transform)(x)
      else:
        processed = transform(x)
      return transform, processed

    scan_over_axis = nnx.scan(
        _process,
        in_axes=(nnx.Carry, 0),
        out_axes=(nnx.Carry, 0),
    )
    self.transform, scanned = scan_over_axis(self.transform, inputs)
    scanned = cx.tag(scanned, ax)
    scanned = {
        k: v.order_as(*self._out_dims_order(original_order[k], v.dims))
        for k, v in scanned.items()
    }
    return scanned


@nnx_compat.dataclass
class AddShardingConstraint(TransformABC):
  """Adds a sharding constraint to all fields in `inputs`."""

  mesh: parallelism.Mesh
  schema: str | tuple[str, ...]

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    return self.mesh.with_sharding_constraint(inputs, self.schema)


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
  """Clips wavenumbers in inputs for grids matching `wavenumbers_for_grid`.

  Attributes:
    wavenumbers_for_grid: A dictionary mapping grids to the number of wavenumber
      to clip for that grid.
    skip_missing: If True, grids without a matching coordinate will be skipped,
      otherwise an error is raised.
  """

  wavenumbers_for_grid: dict[coordinates.SphericalHarmonicGrid, int]
  skip_missing: bool = False

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    """Returns `inputs` with top `wavenumbers_to_clip` set to zero."""
    result = {}
    for k, v in inputs.items():
      for grid, n_clip in self.wavenumbers_for_grid.items():
        if all(ax in v.axes.values() for ax in grid.axes):
          result[k] = grid.clip_wavenumbers(v, n_clip)
          break
      else:
        if self.skip_missing:
          result[k] = v
        else:
          raise ValueError(
              f'No matching grid found for {k=} in {self.wavenumbers_for_grid=}'
          )
    return result

  @classmethod
  def for_grids(
      cls,
      grids: (
          Sequence[coordinates.SphericalHarmonicGrid]
          | coordinates.SphericalHarmonicGrid
      ),
      wavenumbers_to_clip: Sequence[int] | int,
      skip_missing: bool = False,
  ):
    """Custom constructor based grids and wavenumbers to clip sequence."""
    if isinstance(grids, coordinates.SphericalHarmonicGrid):
      grids = [grids]
    if isinstance(wavenumbers_to_clip, int):
      wavenumbers_to_clip = [wavenumbers_to_clip] * len(grids)
    wavenumbers_for_grid = {
        grid: n for grid, n in zip(grids, wavenumbers_to_clip, strict=True)
    }
    return cls(
        wavenumbers_for_grid=wavenumbers_for_grid,
        skip_missing=skip_missing,
    )


@nnx_compat.dataclass
class Regrid(TransformABC):
  """Applies `self.regridder` to `inputs`."""

  regridder: interpolators.BaseRegridder

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    """Returns `inputs` regridded with `self.regridder`."""
    return self.regridder(inputs)


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
class RemovePrefix(TransformABC):
  """Transforms inputs by removing `prefix` from dictionary keys."""

  prefix: str

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    return {k.removeprefix(self.prefix): v for k, v in inputs.items()}


@nnx_compat.dataclass
class Rename(TransformABC):
  """Renames keys in inputs based on rename_dict."""

  rename_dict: dict[str, str]

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    outputs = {}
    for k, v in inputs.items():
      new_key = self.rename_dict.get(k, k)
      if new_key in outputs:
        raise ValueError(f'Duplicate key after renaming: {new_key}')
      outputs[new_key] = v
    return outputs


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


@nnx_compat.dataclass
class Relu(TransformABC):
  """Applies relu to fields specified in keys_to_apply_relu, or to all."""

  keys_to_apply_relu: Sequence[str] | None = None

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    if self.keys_to_apply_relu is None:
      return {k: cx.cmap(jax.nn.relu)(v) for k, v in inputs.items()}
    outputs = {}
    for k, v in inputs.items():
      if k in self.keys_to_apply_relu:
        outputs[k] = cx.cmap(jax.nn.relu)(v)
      else:
        outputs[k] = v
    return outputs


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

  ylm_map: (
      spherical_harmonics.FixedYlmMapping | spherical_harmonics.YlmMapper
  )

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    modal_outputs = {}
    for k, v in inputs.items():
      modal_outputs[k] = self.ylm_map.to_modal(v)
    return modal_outputs


@nnx_compat.dataclass
class ToNodal(TransformABC):
  """Transforms inputs from modal to nodal space."""

  ylm_map: (
      spherical_harmonics.FixedYlmMapping | spherical_harmonics.YlmMapper
  )

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    nodal_outputs = {}
    for k, v in inputs.items():
      nodal_outputs[k] = self.ylm_map.to_nodal(v)
    return nodal_outputs


class ToModalWithFilteredGradients:
  """Helper module that returns filtered grads and laplacians of inputs fields.

  Gradients are filtered with an exponential filter of order 1 and provided
  attentuations. If no attentuations are provided, then this transform returns
  no gradient features. To avoid accidental accumulation of the cos(lat)
  factors, features must be keyed using typing.KeyWithCosLatFactor namedtuple.
  Gradient features are scaled by R^{m} where m is the diffentiation order.
  """

  def __init__(
      self,
      ylm_map: spherical_harmonics.FixedYlmMapping,
      filter_attenuations: tuple[float, ...] = tuple(),
  ):
    self.ylm_map = ylm_map
    self.attenuations = filter_attenuations
    modal_filters = [
        spatial_filters.ExponentialModalFilter(
            ylm_map,
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
    features = {}
    for k, x in inputs.items():
      name, cos_lat_order = k.name, k.factor_order
      r = self.ylm_map.radius
      for filter_module, att in zip(self.modal_filters, self.attenuations):
        d_x_dlon, d_x_dlat = self.ylm_map.cos_lat_grad(x)
        laplacian = self.ylm_map.laplacian(x)
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
        features[dlon_key] = filter_module.filter_modal(r * d_x_dlon)
        features[dlat_key] = filter_module.filter_modal(r * d_x_dlat)
        features[del2_key] = filter_module.filter_modal(r * r * laplacian)
    return features

  def output_shapes(
      self, input_shapes: dict[typing.KeyWithCosLatFactor, cx.Field]
  ) -> dict[typing.KeyWithCosLatFactor, cx.Field]:
    return nnx.eval_shape(self.__call__, input_shapes)


@nnx_compat.dataclass
class ScaleToMatchCoarseFields(TransformABC):
  """Scales hres fields s.t. spatially-averaged values match coarse ones.

  This transform applies scaling to high-resolution fields such that their
  spatial average, when regridded to a coarse grid, matches the average of
  corresponding coarse-resolution fields. This is useful for enforcing
  conservation laws across different resolutions.

  Note: due to regridding, conservation is not exact.

  Attributes:
    raw_hres_transform: Transform that generates high-resolution fields.
    ref_coarse_transform: Transform that generates coarse-resolution fields.
    coarse_grid: The coarse grid to which high-resolution fields are
      downsampled for comparison.
    hres_grid: The high-resolution grid.
    keys: A sequence of keys for which to apply conservation.
    epsilon: A small value to avoid division by zero.
  """
  raw_hres_transform: TransformABC
  ref_coarse_transform: TransformABC
  coarse_grid: coordinates.LonLatGrid
  hres_grid: coordinates.LonLatGrid
  keys: Sequence[str]
  epsilon: float = 1e-6

  def __post_init__(self):
    self.regrid_to_coarse = Regrid(
        regridder=interpolators.ConservativeRegridder(self.coarse_grid)
    )
    self.regrid_to_hres = Regrid(
        regridder=interpolators.ConservativeRegridder(self.hres_grid)
    )

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    hres_outputs = self.raw_hres_transform(inputs)
    coarse_outputs = self.ref_coarse_transform(inputs)

    for key in self.keys:
      if key not in hres_outputs or key not in coarse_outputs:
        raise ValueError(
            f'Key {key} not found in {hres_outputs.keys()} or in'
            f' {coarse_outputs.keys()}'
        )

      coarse_field = coarse_outputs[key]
      hres_field = hres_outputs[key]
      downsampled_hres = self.regrid_to_coarse({key: hres_field})[key]
      ratio = coarse_field / (
          downsampled_hres
          + self.epsilon
          * cx.cmap(lambda x: jnp.where(x >= 0, 1.0, -1.0))(downsampled_hres)
      )
      upsampled_ratio = self.regrid_to_hres({key: ratio})[key]
      conserved_hres_field = hres_field * upsampled_ratio
      hres_outputs[key] = conserved_hres_field
    return hres_outputs


@nnx_compat.dataclass
class VelocityFromModalDivCurl(TransformABC):
  """Transform divergence and vorticity in inputs to 2D velocity components."""

  ylm_map: spherical_harmonics.FixedYlmMapping | spherical_harmonics.YlmMapper
  divergence_key: str = 'divergence'
  vorticity_key: str = 'vorticity'
  u_key: str = 'u_component_of_wind'
  v_key: str = 'v_component_of_wind'
  clip: bool = True

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    divergence = inputs[self.divergence_key]
    vorticity = inputs[self.vorticity_key]
    u, v = spherical_harmonics.vor_div_to_uv_nodal(
        vorticity, divergence, self.ylm_map, clip=self.clip
    )
    return {self.u_key: u, self.v_key: v}


@nnx_compat.dataclass
class DivCurlFromNodalVelocity(TransformABC):
  """Transform 2D velocity components to divergence and vorticity."""

  ylm_map: spherical_harmonics.FixedYlmMapping | spherical_harmonics.YlmMapper
  divergence_key: str = 'divergence'
  vorticity_key: str = 'vorticity'
  u_key: str = 'u_component_of_wind'
  v_key: str = 'v_component_of_wind'
  clip: bool = True

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    u, v = inputs[self.u_key], inputs[self.v_key]
    vorticity, div = spherical_harmonics.uv_nodal_to_vor_div_modal(
        u, v, self.ylm_map, clip=self.clip
    )
    return {self.divergence_key: div, self.vorticity_key: vorticity}
