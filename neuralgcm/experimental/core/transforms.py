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
import collections
import dataclasses
import itertools
import re
from typing import Any, Callable, Literal, Sequence, TypeAlias

import coordax as cx
from flax import nnx
import jax
import jax.nn
import jax.numpy as jnp
import jax_datetime as jdt
from neuralgcm.experimental.core import boundaries
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import interpolators
from neuralgcm.experimental.core import nnx_compat
from neuralgcm.experimental.core import normalizations
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import spatial_filters
from neuralgcm.experimental.core import spherical_harmonics
from neuralgcm.experimental.core import typing
from neuralgcm.experimental.core import units


Transform: TypeAlias = typing.Transform


class TransformParams(nnx.Variable):
  """Custom variable class for transform parameters."""


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


def _masked_nan_to_num(
    x: cx.Field, mask: cx.Field, num: float = 0.0
) -> cx.Field:
  """Replaces NaN entries over `True` mask with `num`."""
  mask_coord = cx.get_coordinate(mask)
  masked_nan_to_num = lambda x, m: jnp.where(m, jnp.nan_to_num(x, nan=num), x)
  result = cx.cpmap(masked_nan_to_num)(*cx.untag([x, mask], mask_coord))
  return result.tag(mask_coord)


def _masked_to_mean(x: cx.Field, mask: cx.Field) -> cx.Field:
  """Replaces NaN entries over `True` mask with the mean of the complement."""
  mask_coord = cx.get_coordinate(mask)
  mean_over_complement = lambda x, m: jnp.mean(x, where=~m.astype(jnp.bool))
  mean_values = cx.cmap(mean_over_complement)(*cx.untag([x, mask], mask_coord))
  replace_over_mask = lambda x, means, mask: jnp.where(mask, means, x)
  result = cx.cpmap(replace_over_mask)(
      x.untag(mask_coord), mean_values, mask.untag(mask_coord)
  )
  return result.tag(mask_coord)


ApplyMaskMethods = (
    Literal['zero_multiply', 'nan_to_0', 'nan_to_mean', 'set_nan']
    | Callable[[cx.Field, cx.Field], cx.Field]
)
ComputeMaskMethods = Literal[
    'isnan',
    'notnan',
    'isinf',
    'notinf',
    'above',
    'below',
    'as_bool',
]
APPLY_MASK_FNS = {
    'zero_multiply': lambda x, mask: x * (~mask).astype(jnp.float32),
    'nan_to_0': _masked_nan_to_num,
    'nan_to_mean': _masked_to_mean,
    'set_nan': cx.cmap(lambda x, mask: jnp.where(mask, jnp.nan, x)),
}
COMPUTE_MASK_FNS = {
    'isnan': lambda x, t: cx.cmap(jnp.isnan)(x),
    'notnan': lambda x, t: cx.cmap(lambda x: ~jnp.isnan(x))(x),
    'isinf': lambda x, t: cx.cmap(jnp.isinf)(x),
    'notinf': lambda x, t: cx.cmap(lambda x: ~jnp.isinf(x))(x),
    'above': lambda x, t: cx.cmap(lambda x: x > t)(x),
    'below': lambda x, t: cx.cmap(lambda x: x < t)(x),
    'as_bool': lambda x, t: x.astype(jnp.bool),
}


class Identity(TransformABC):
  """Returns inputs as they are."""

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    return inputs


class Empty(TransformABC):
  """Returns an empty dict."""

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    return {}


@nnx_compat.dataclass
class Prescribed(TransformABC):
  """Returns a prescribed dict of fields."""

  prescribed_fields: dict[str, cx.Field]

  def __init__(self, prescribed_fields: dict[str, cx.Field]):
    self.prescribed_fields = prescribed_fields

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    return self.prescribed_fields


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
class RavelDims(TransformABC):
  """Ravels specified dimensions into another."""

  dims_to_ravel: tuple[str | cx.Coordinate, ...]
  out_dim: str | cx.Coordinate

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    untagged = cx.untag(inputs, *self.dims_to_ravel, allow_missing=True)
    ravel_fn = cx.cmap(jnp.ravel, out_axes='leading')
    return {k: ravel_fn(v).tag(self.out_dim) for k, v in untagged.items()}


class Select(TransformABC):
  """Selects subset of keys in the input dictionary.

  Attributes:
    keys: Keys to select. Can be a string, sequence of strings, or a callable.
    invert: If True, invert the selection.
    mode: Selection mode for string ``keys``. Can be 'match' or 'regex'.
    strict: If True, raise an error if any of the keys are not found in inputs.
  """

  def __init__(
      self,
      keys: str | Sequence[str] | Callable[[str], bool],
      invert: bool = False,
      mode: Literal['match', 'regex'] = 'match',
      strict: bool = True,
  ):
    if isinstance(keys, str):
      self.keys = tuple([keys])
    elif isinstance(keys, Sequence):
      self.keys = tuple(keys)
    elif callable(keys):
      if mode == 'regex':
        raise TypeError(f'mode must be "match" when using {type(keys)=}.')
      self.keys = keys
    else:
      raise TypeError(
          "The 'keys' parameter must be a str, sequence of str, or a callable."
      )
    if mode not in ['match', 'regex']:
      raise ValueError(f'Unknown mode: {mode}')
    self.invert = invert
    self.mode = mode
    self.strict = strict

  def __call__(self, data: dict[str, cx.Field]) -> dict[str, cx.Field]:
    selected_keys = set()
    if callable(self.keys):
      selected_keys = {k for k in data.keys() if self.keys(k)}
    elif self.mode == 'regex':
      for pattern in self.keys:
        matched = {k for k in data.keys() if re.fullmatch(pattern, k)}
        if self.strict and not matched:
          raise KeyError(
              f'Select strict mode failed. Regex pattern {pattern!r} '
              'did not match any keys.'
          )
        selected_keys.update(matched)
    elif self.mode == 'match':
      selected_keys = set(self.keys)
      if self.strict:
        missing = selected_keys - data.keys()
        if missing:
          raise KeyError(f'Select strict mode failed. Missing keys: {missing}')
      selected_keys = selected_keys.intersection(data.keys())
    else:
      raise ValueError(f'Unknown mode: {self.mode}')

    # Apply inversion logic if needed.
    if self.invert:
      final_keys = set(data.keys()) - selected_keys
    else:
      final_keys = selected_keys
    return {k: v for k, v in data.items() if k in final_keys}


@nnx_compat.dataclass
class FilterByCoord(TransformABC):
  """Selects subset of fields that match specified coordinate or type.

  Attributes:
    coords: A coordinate or a sequence of coordinates. If provided, the field
      must contain at least one of these coordinates to be selected.
    types: A type of coordinate or sequence of types. If provided, the field
      must contain a coordinate of at least one of these types to be selected.
    invert: If True, invert the selection.
  """

  coords: cx.Coordinate | Sequence[cx.Coordinate] | None = None
  types: type[cx.Coordinate] | Sequence[type[cx.Coordinate]] | None = None
  invert: bool = False

  def __post_init__(self):
    if self.coords is None and self.types is None:
      raise ValueError('At least one of `coords` or `types` must be provided.')
    # Standardize `coords` and `types` representation.
    coords = () if self.coords is None else self.coords
    self.coords = (coords,) if cx.is_coord(coords) else tuple(coords)
    types = () if self.types is None else self.types
    self.types = (types,) if isinstance(types, type) else tuple(types)

  def __call__(self, data: dict[str, cx.Field]) -> dict[str, cx.Field]:
    selected_keys = set()
    for k, v in data.items():
      match = False
      v_axes = set(v.coordinate.axes)
      v_components = set(cx.coords.canonicalize(*v.coordinate.axes))
      for c in self.coords:
        if set(c.axes).issubset(v_axes):
          match = True
          break
      if not match:
        for t in self.types:
          if any(isinstance(ax, t) for ax in v_components):
            match = True
            break
      if match:
        selected_keys.add(k)

    if self.invert:
      final_keys = set(data.keys()) - selected_keys
    else:
      final_keys = selected_keys

    return {k: v for k, v in data.items() if k in final_keys}


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
class Isel(TransformABC):
  """Slices all fields using indexers."""

  indexers: dict[str | cx.Coordinate, Any]

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    """Returns `inputs` where all fields are sliced using indexers."""
    outputs = {}
    used_indexers = set()
    for k, field in inputs.items():
      valid_indexers = {}
      for dim, idx in self.indexers.items():
        if cx.contains_dims(field, dim):
          valid_indexers[dim] = idx
          used_indexers.add(dim)
      outputs[k] = field.isel(valid_indexers)
    unused_indexers = set(self.indexers) - used_indexers
    if unused_indexers:
      raise ValueError(f'Dimensions {unused_indexers} not found in inputs.')
    return outputs


@nnx_compat.dataclass
class MeanOverAxes(TransformABC):
  """Computes mean over specified dims, accounting for grid geometry."""

  dims: tuple[str | cx.Coordinate, ...]

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    outputs = {}
    for k, v in inputs.items():
      dim_names = set()
      for d in self.dims:
        if isinstance(d, cx.Coordinate):
          dim_names.update(d.dims)
        else:
          dim_names.add(d)

      if not all(cx.fields.contains_dims(v, d) for d in self.dims):
        raise ValueError(f'Dims {self.dims} not found in {v=} for key={k}')

      try:
        grid = cx.coords.extract(v.coordinate, coordinates.LonLatGrid)
      except ValueError:
        grid = None

      if grid is not None:
        spatial_dims = dim_names & {'latitude', 'longitude'}
        if spatial_dims:
          v = grid.mean(v, dims=tuple(spatial_dims))
          dim_names -= spatial_dims

      if dim_names:
        # Sort dimensions to untag based on their order in the field.
        ordered_dims = [d for d in v.dims if d in dim_names]
        outputs[k] = cx.cmap(jnp.mean)(v.untag(*ordered_dims))
      else:
        outputs[k] = v

    return outputs


@nnx_compat.dataclass
class Sel(TransformABC):
  """Selects values using label indexers."""

  indexers: dict[str | cx.Coordinate, Any]
  method: Literal['nearest'] | None = None

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    """Applies sel operation to all Fields in inputs."""
    outputs = {}
    used_indexers = set()
    for k, field in inputs.items():
      valid_indexers = {}
      for dim, idx in self.indexers.items():
        if cx.contains_dims(field, dim):
          valid_indexers[dim] = idx
          used_indexers.add(dim)
      outputs[k] = field.sel(valid_indexers, method=self.method)
    unused_indexers = set(self.indexers) - used_indexers
    if unused_indexers:
      raise ValueError(f'Dimensions {unused_indexers} not found in inputs.')
    return outputs


@nnx_compat.dataclass
class InsertAxis(TransformABC):
  """Inserts `self.axis` at `self.loc` in all fields in `inputs`."""

  axis: cx.Coordinate
  loc: int | str | cx.Coordinate

  def _expand_axis(self, field: cx.Field) -> cx.Field:
    if isinstance(self.loc, int):
      loc = self.loc
    else:
      if cx.contains_dims(field, self.loc):
        axis_to_the_right = cx.get_coordinate_part(field, self.loc)
        loc = field.named_axes[axis_to_the_right.dims[0]]
      else:
        raise ValueError(f'Axis {self.loc} not present in {field=}')
    out_coord = cx.coords.insert_axes(field.coordinate, {loc: self.axis})
    return field.broadcast_like(out_coord)

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    return {k: self._expand_axis(v) for k, v in inputs.items()}


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
class WithExemptKeys(TransformABC):
  """Wrapper transform that is excludes `keys` from the wrapped transform.

  This is a helper transform that passes through `keys` unchanged and applies
  `transform` to the rest of the keys. This is equivalent to:

      merge(select(inputs, keys), transform(select(inputs, keys, invert=True)))
  """

  transform: Transform
  keys: str | Sequence[str]

  def __post_init__(self):
    keys = self.keys
    self.keys = tuple([keys]) if isinstance(keys, str) else tuple(keys)

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    to_transform = {k: v for k, v in inputs.items() if k not in self.keys}
    keep_as_is = {k: v for k, v in inputs.items() if k in self.keys}
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


class Scale(TransformABC):
  """Applies x * `self.scale` to all fields in inputs."""

  def __init__(self, scale: cx.Field):
    self.scale = TransformParams(scale)

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    scale = self.scale.get_value()
    scale_fn = lambda x: x * scale
    return {k: scale_fn(v) for k, v in inputs.items()}


class StandardizeFields(TransformABC):
  """Shifts and scales inputs.

  This transform applies shifts and scales. It can be applied globally
  or on a per key basis.

  Attributes:
    shifts: The shifts to use for centering input fields.
    scales: The scales to use for normalization.
    from_normalized: Whether to invert the transform normalized --> original.
  """

  def __init__(
      self,
      shifts: cx.Field | dict[str, cx.Field],
      scales: cx.Field | dict[str, cx.Field],
      from_normalized: bool = False,
  ):
    self.shifts = TransformParams(shifts)
    self.scales = TransformParams(scales)
    self.from_normalized = from_normalized

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    shifts = self.shifts.get_value()
    scales = self.scales.get_value()
    if self.from_normalized:
      scale_fn = lambda x, shift, scale: x * scale + shift
    else:
      scale_fn = lambda x, shift, scale: (x - shift) / scale

    if isinstance(shifts, dict) or isinstance(scales, dict):
      same_value_dict = lambda x: collections.defaultdict(lambda: x)
      is_dict = lambda x: isinstance(x, dict)
      shifts = shifts if is_dict(shifts) else same_value_dict(shifts)
      scales = scales if is_dict(scales) else same_value_dict(scales)
      return {k: scale_fn(v, shifts[k], scales[k]) for k, v in inputs.items()}
    else:
      return {k: scale_fn(v, shifts, scales) for k, v in inputs.items()}


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
              f'No matching grid for {k=}, {v=} in {self.wavenumbers_for_grid=}'
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
class InpaintMaskForHarmonics(TransformABC):
  """Inpaints `inputs` over `mask` with smoothed spherical harmonics.

  A variation of PGA (Papoulis-Gerchberg Algorithm) that iteratively inpaints,
  spatial signal imposing limited spherical harmonics bandwidth prior. It works
  by repeatedly transforming to modal space, applying a low-pass filter,
  transforming back to nodal space and restoring non masked entries to their
  original values. At the start, masked values are initialized with
  the mean of the valid values.

  Attributes:
    ylm_map: Mapping between nodal and modal representations.
    compute_masks: Transform that returns a mask for each input field. The True
      mask values indicate entries to be inpainted.
    lowpass_filter: Filter to apply in modal space to smooth the inpainting.
    n_iter: Number of iterations to perform.
  """

  ylm_map: spherical_harmonics.YlmMapper | spherical_harmonics.FixedYlmMapping
  compute_masks: Transform
  lowpass_filter: spatial_filters.ModalSpatialFilter
  n_iter: int = 2
  default_mask_key: str | None = None

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    masks = self.compute_masks(inputs)
    masks = {k: masks.get(k, masks.get(self.default_mask_key)) for k in inputs}
    for k, mask in masks.items():
      if mask is None:
        raise ValueError(f'No mask found for {k=}')
    # Initialize masked values with mean of valid data.
    current_guess = {k: _masked_to_mean(v, masks[k]) for k, v in inputs.items()}

    # Helper to reset valid pixels: where(mask, current, original)
    # mask is True for pixels to be inpainted, False is to be reset with inputs.
    update_guess_fn = cx.cmap(lambda c, o, m: jnp.where(m, c, o))

    for _ in range(self.n_iter):
      modal = {k: self.ylm_map.to_modal(v) for k, v in current_guess.items()}
      filtered_modal = self.lowpass_filter.filter_modal(modal)
      filtered_nodal = self.ylm_map.to_nodal(filtered_modal)
      current_guess = {
          k: update_guess_fn(filtered_nodal[k], inputs[k], masks[k])
          for k in inputs
      }

    return current_guess


@nnx_compat.dataclass
class Regrid(TransformABC):
  """Applies `self.regridder` to `inputs`."""

  regridder: interpolators.BaseRegridder

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    """Returns `inputs` regridded with `self.regridder`."""
    return self.regridder(inputs)


@nnx_compat.dataclass
class ComputeMasks(TransformABC):
  """Transforms inputs to boolean masks for keys in `mask_keys`."""

  compute_mask_method: ComputeMaskMethods = 'as_bool'
  threshold_value: float | None = None
  mask_keys: str | tuple[str, ...] | None = None

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    is_str = lambda x: isinstance(x, str)
    mask_keys = (self.mask_keys,) if is_str(self.mask_keys) else self.mask_keys
    if mask_keys is None:
      mask_keys = inputs.keys()
    compute_mask_fn = COMPUTE_MASK_FNS[self.compute_mask_method]
    return {
        k: compute_mask_fn(inputs[k], self.threshold_value) for k in mask_keys
    }


@nnx_compat.dataclass
class ApplyOverMasks(TransformABC):
  """Applies masking to `transform(inputs)` using `compute_masks(inputs)`.

  Attributes:
    transform: Transform to apply to inputs.
    compute_masks: Transform that returns a mask for each input field.
    apply_mask_method: Method to apply the mask to the input fields.
    default_mask_key: Key of the default mask to use when compute_masks does not
      produce a key matching mask for field in inputs.
  """

  compute_masks: TransformABC
  apply_mask_method: ApplyMaskMethods = 'zero_multiply'
  transform: TransformABC | None = None
  default_mask_key: str | None = None

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    masks = self.compute_masks(inputs)
    inputs = self.transform(inputs) if self.transform is not None else inputs
    apply_mask_fn = (
        self.apply_mask_method
        if callable(self.apply_mask_method)
        else APPLY_MASK_FNS[self.apply_mask_method]
    )
    outputs = {}
    for k, v in inputs.items():
      mask = masks.get(k, masks.get(self.default_mask_key))
      if mask is None:
        raise ValueError(f'No mask found for {k=}')
      outputs[k] = apply_mask_fn(v, mask)
    return outputs


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
    return cx.field(nondim_value, x.coordinate)

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
    return cx.field(dim_value, x.coordinate)

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


class StreamingStatsNorm(TransformABC):
  """Normalizes inputs using values from streaming mean and variances.

  Attributes:
    streaming_norm: StreamingNormalizer that performs the normalization.
    update_stats: Whether to update the normalization statistics.
    make_masks_transform: Optional transform that produces a mask indicating
      which entries should contribute to the statistics updates.
  """

  def __init__(
      self,
      norm_coords: dict[str, cx.Coordinate],
      update_stats: bool = False,
      epsilon: float = 1e-11,
      make_masks_transform: Transform | None = None,
      skip_nans: bool = True,
      skip_unspecified: bool = False,
      allow_missing: bool = True,
  ):
    """Initializes StreamingStatsNorm.

    Args:
      norm_coords: A dictionary mapping variable names to the coordinates over
        which the normalization is done independently.
      update_stats: Whether to update the normalization statistics.
      epsilon: A small float added to variance to avoid dividing by zero.
      make_masks_transform: Optional transform that produces a mask indicating
        which entries should contribute to the statistics updates.
      skip_nans: If True, ignores NaNs when updating the statistics.
      skip_unspecified: If True, input fields for which no normalization is
        specified (i.e., keys not in `norm_coords`) are passed through
        unchanged. If False, presence of such fields in `inputs` raises.
      allow_missing: If True, `inputs` may omit fields for which normalization
        is specified (i.e., keys in `norm_coords`). If False, all keys in
        `norm_coords` must be present in `inputs`, otherwise an error is raised.
    """
    self.streaming_norm = normalizations.StreamingNormalizer(
        norm_coords,
        epsilon=epsilon,
        skip_unspecified=skip_unspecified,
        allow_missing=allow_missing,
        skip_nans=skip_nans,
    )
    self.make_masks_transform = make_masks_transform
    self.update_stats = update_stats

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    """Normalizes inputs and optionally updates statistics."""
    if self.make_masks_transform is not None:
      mask = self.make_masks_transform(inputs)
      if len(mask) > 1:
        raise ValueError(
            f'Mask transform must contain at most 1 Field, got {len(mask)=}.'
        )
      if not mask:
        mask = None
      else:
        [mask] = list(mask.values())
    else:
      mask = None
    return self.streaming_norm(inputs, self.update_stats, mask=mask)

  @classmethod
  def for_inputs_struct(
      cls,
      inputs_struct: dict[str, cx.Field],
      independent_axes: tuple[cx.Coordinate, ...],
      exclude_regex: str | None = None,
      scalar_regex: str | None = None,
      update_stats: bool = False,
      make_masks_transform: Transform | None = None,
      epsilon: float = 1e-11,
      skip_unspecified: bool = False,
      skip_nans: bool = False,
      allow_missing: bool = True,
  ):
    """Custom constructor based on inputs struct that should be normalized."""
    norm_coords = {
        k: cx.coords.compose(
            *[c for c in independent_axes if c in v.coordinate.axes]
        )
        for k, v in inputs_struct.items()
    }
    if exclude_regex is not None:
      norm_coords = {
          k: v
          for k, v in norm_coords.items()
          if not re.search(exclude_regex, k)
      }
    if scalar_regex is not None:
      norm_coords = {
          k: cx.Scalar() if re.search(scalar_regex, k) else v
          for k, v in norm_coords.items()
      }
    return cls(
        norm_coords=norm_coords,
        update_stats=update_stats,
        epsilon=epsilon,
        make_masks_transform=make_masks_transform,
        skip_unspecified=skip_unspecified,
        skip_nans=skip_nans,
        allow_missing=allow_missing,
    )


@nnx_compat.dataclass
class ToModal(TransformABC):
  """Transforms inputs from nodal to modal space."""

  ylm_map: spherical_harmonics.FixedYlmMapping | spherical_harmonics.YlmMapper

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    modal_outputs = {}
    for k, v in inputs.items():
      modal_outputs[k] = self.ylm_map.to_modal(v)
    return modal_outputs


@nnx_compat.dataclass
class ToNodal(TransformABC):
  """Transforms inputs from modal to nodal space."""

  ylm_map: spherical_harmonics.FixedYlmMapping | spherical_harmonics.YlmMapper

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
        # pytype: disable=wrong-arg-types
        grads = {
            dlon_key: r * d_x_dlon,
            dlat_key: r * d_x_dlat,
            del2_key: r * r * laplacian,
        }
        features |= filter_module.filter_modal(grads)
        # pytype: enable=wrong-arg-types
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
    coarse_grid: The coarse grid to which high-resolution fields are downsampled
      for comparison.
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


@nnx_compat.dataclass
class PointNeighborsFromGrid(TransformABC):
  """Extracts a patch of neighbors from gridded fields around query points.

  This transform splits inputs into two parts: query longitude/latitude fields
  and gridded inputs. The query fields are used to determine the patch to
  extract from the remaining gridded data. The gridded inputs are expected
  to include longitude and latitude coordinate fields in degrees and be in the
  ascending order, with longitudes in the range [-180, 180] and latitudes in
  the range [-90, 90].

  Attributes:
    width: The width of the square patch to extract. Default is 1.
    lon_query_key: Key for query longitudes in inputs.
    lat_query_key: Key for query latitudes in inputs.
  """

  width: int = 1
  lon_query_key: str = 'longitude'
  lat_query_key: str = 'latitude'
  include_offset: bool = True
  boundary_condition: boundaries.BoundaryCondition = dataclasses.field(
      init=False
  )
  padding: int = dataclasses.field(init=False)

  def __post_init__(self):
    self.boundary_condition = boundaries.LonLatBoundary()
    self.padding = (self.width + 1) // 2  # Safe padding amount.
    self.pad_sizes = {
        'longitude': (self.padding, self.padding),
        'latitude': (self.padding, self.padding),
    }

  def _1d_indices(
      self,
      query: typing.Array,
      grid_vals: typing.Array,
      period: float | None = None,
  ) -> typing.Array:
    """Returns indices of neighbors for `query` in `grid_vals`."""
    diff = grid_vals - query
    if period is not None:
      diff = jnp.mod(diff + period / 2, period) - period / 2
    idx = jnp.argmin(jnp.abs(diff))
    # Returned indices must be computed relative to the padded array.
    radius = (self.width - 1) // 2  # accounts for patch centering.
    start = idx + self.padding - radius

    if self.width % 2 == 0:
      # Bias towards the interval containing the point for even widths.
      start = jnp.where(diff[idx] > 0, start - 1, start)

    if self.width == 1:
      return start
    return start + jnp.arange(self.width)

  def _indices_and_patch(
      self,
      lon_grid: cx.Field,
      lat_grid: cx.Field,
      lon_query: cx.Field,
      lat_query: cx.Field,
  ) -> tuple[cx.Field, cx.Field, cx.Coordinate]:
    """Returns longitude and latitude indices and patch coordinate."""
    get_lon_indices = lambda q: self._1d_indices(q, lon_grid.data, 360.0)
    get_lat_indices = lambda q: self._1d_indices(q, lat_grid.data)
    lon_indices = cx.cmap(get_lon_indices)(lon_query)
    lat_indices = cx.cmap(get_lat_indices)(lat_query)

    if self.width == 1:
      patch_coord = cx.Scalar()
    else:
      patch_lon = cx.SizedAxis('longitude', self.width)
      patch_lat = cx.SizedAxis('latitude', self.width)
      patch_coord = cx.coords.compose(patch_lon, patch_lat)

    return lon_indices, lat_indices, patch_coord

  @nnx.jit
  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    inputs = inputs.copy()
    if self.lon_query_key not in inputs or self.lat_query_key not in inputs:
      raise ValueError(
          f'Inputs must contain {self.lon_query_key} and {self.lat_query_key}.'
      )
    lon_query = inputs.pop(self.lon_query_key)
    lat_query = inputs.pop(self.lat_query_key)
    if lon_query.positional_shape or lat_query.positional_shape:
      raise ValueError(
          f'Query fields {lon_query} and {lat_query} must be fully labeled.'
      )
    lon_lat_dims = ('longitude', 'latitude')
    padded_lon_lat = tuple(f'padded_{d}' for d in lon_lat_dims)

    if self.width == 1:
      extract_fn = lambda lo, la, f: f[lo, la]
    else:
      # Broadcast indices so that we extract a patch rather than the diagonal.
      extract_fn = lambda lo, la, f: f[lo[:, None], la]
    extract_fn = cx.cmap(extract_fn, 'leading')

    cache = {}
    outputs = {}
    for k, f in inputs.items():
      if any(d not in f.dims for d in lon_lat_dims):
        raise ValueError(f'{k=} -> {f} does not have {lon_lat_dims=} axes.')

      grid = cx.coords.compose(f.axes['longitude'], f.axes['latitude'])

      if grid not in cache:
        lon_1d, lat_1d = f.coord_fields['longitude'], f.coord_fields['latitude']
        lon_indices, lat_indices, patch_coord = self._indices_and_patch(
            lon_1d, lat_1d, lon_query, lat_query
        )
        cache[grid] = (lon_indices, lat_indices, patch_coord, lon_1d, lat_1d)

      lon_indices, lat_indices, patch_coord, _, _ = cache[grid]
      padded_f = self.boundary_condition.pad(f, self.pad_sizes)
      untagged_f = padded_f.untag(*padded_lon_lat)
      patch = extract_fn(lon_indices, lat_indices, untagged_f).tag(patch_coord)
      outputs[k] = patch

    if self.include_offset:
      for grid_idx, (grid, cached_values) in enumerate(cache.items()):
        lon_indices, lat_indices, patch_coord, lon_1d, lat_1d = cached_values
        lons = lon_1d.broadcast_like(grid)
        lats = lat_1d.broadcast_like(grid)
        padded_lon = self.boundary_condition.pad(lons, self.pad_sizes)
        padded_lat = self.boundary_condition.pad(lats, self.pad_sizes)
        patch_lon = extract_fn(
            lon_indices, lat_indices, padded_lon.untag(*padded_lon_lat)
        ).tag(patch_coord)
        patch_lat = extract_fn(
            lon_indices, lat_indices, padded_lat.untag(*padded_lon_lat)
        ).tag(patch_coord)

        norm_lon = cx.cmap(lambda x: jnp.mod(x + 180.0, 360.0) - 180.0)
        outputs[f'delta_longitude_{grid_idx}'] = norm_lon(patch_lon - lon_query)
        outputs[f'delta_latitude_{grid_idx}'] = patch_lat - lat_query

    return outputs


def _sanitize_values(
    values: typing.Pytree,
    masks: typing.Pytree,
) -> dict[str, cx.Field]:
  """Masks values where mask is True, preserving structure."""
  _clean = lambda x, m: jnp.where(m, jnp.zeros((), x.dtype), x)
  return jax.tree.map(_clean, values, masks)


@nnx_compat.dataclass
class SanitizeNanGradTransform(TransformABC):
  """Wraps a transform to provide NaN-safe gradients."""

  transform: TransformABC

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    @nnx.custom_vjp
    def _sanitized_transform(t, inputs):
      return t(inputs)

    def _sanitized_fwd(t, inputs):
      # we convert any arrays to jax arrays in residual to avoid jax converting
      # them to an internal type of numpy arrays that is not supported in `cx`.
      t_def, t_state_before = nnx.split(t)
      to_array = lambda x: jnp.asarray(x, dtype=x.dtype)
      residuals = (jax.tree.map(to_array, inputs), t_def, t_state_before)
      return _sanitized_transform(t, inputs), residuals

    def _sanitized_bwd(res, g):
      input_updates_g, out_g = g
      inputs, t_def, t_state_before = res
      t_updates_g, _ = input_updates_g  # inputs are not stateful -> ignored.
      is_nan = jax.tree.map(jnp.isnan, inputs)
      safe_inputs = _sanitize_values(inputs, is_nan)  # Sanitize inputs.
      grad_nans = jax.tree.map(jnp.isnan, out_g)  # Sanitize gradients.
      safe_out_g = _sanitize_values(out_g, grad_nans)

      def _apply_transform(s, x):
        t = nnx.merge(t_def, s, copy=True)
        out = t(x)
        _, new_s = nnx.split(t)
        return out, new_s

      _, vjp_fn = jax.vjp(_apply_transform, t_state_before, safe_inputs)
      state_g, inputs_g = vjp_fn((safe_out_g, t_updates_g))
      return state_g, inputs_g

    _sanitized_transform.defvjp(_sanitized_fwd, _sanitized_bwd)
    return _sanitized_transform(self.transform, inputs)


class NestedTransform(nnx.Module, pytree=False):
  """Wrapper that applies transforms to values of a nested dict[dict[Field]].

  This module simplifies application of transforms to nested fields by assigning
  a transform to each outer key in the nested dict. If provided with a single
  transform, it will be applied to all keys. An ellipsis can be used to specify
  a default transform for all keys that do not have an explicitly assigned
  transform.

  If no transform is assigned to a key and no default transform is provided, a
  `ValueError` is raised upon attempting to transform that key.
  """

  def __init__(
      self,
      transform: typing.Transform | dict[str | type(...), typing.Transform],
  ):
    if isinstance(transform, dict):
      transforms_map = transform.copy()
      self.default_transform = transforms_map.pop(..., None)
      self.transforms = transforms_map
    else:
      self.default_transform = transform
      self.transforms = {}

  def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
    outputs = {}
    for k, v in inputs.items():
      if k in self.transforms:
        outputs[k] = self.transforms[k](v)
      elif self.default_transform is not None:
        outputs[k] = self.default_transform(v)
      else:
        raise ValueError(f'No default or key-specific transform for {k=}.')
    return outputs

  def output_shapes(self, input_shapes: dict[str, Any]) -> dict[str, Any]:
    call_dispatch = lambda nested_transform, inputs: nested_transform(inputs)
    return nnx.eval_shape(call_dispatch, self, input_shapes)
