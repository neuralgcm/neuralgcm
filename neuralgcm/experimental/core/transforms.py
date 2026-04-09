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
import functools
import itertools
import operator
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
from neuralgcm.experimental.core import normalizations
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import spatial_filters
from neuralgcm.experimental.core import spherical_harmonics
from neuralgcm.experimental.core import typing
from neuralgcm.experimental.core import units
from neuralgcm.experimental.core import xarray_utils
import tree_math
import xarray


_SUPPORTED_BINARY_OPS = {
    'subtract': operator.sub,
    'divide': operator.truediv,
    'add': operator.add,
    'multiply': operator.mul,
}

# pylint: disable=g-classes-have-attributes
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

  def __init_subclass__(cls, pytree: bool = False, **kwargs):
    super().__init_subclass__(pytree=pytree, **kwargs)


class PytreeTransformABC(TransformABC, pytree=True):
  """Base class for transforms that are safe to be treated as pytrees."""

  def __init_subclass__(cls, pytree: bool = True, **kwargs):
    super().__init_subclass__(pytree=pytree, **kwargs)


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


def get_partial_jnp_fn(name, **kwargs):
  """Returns a function from jax.numpy with a given name."""
  fn = getattr(jnp, name)  # allows specifying fiddle serializable fns.
  return functools.partial(fn, **kwargs)


class Identity(PytreeTransformABC):
  """Returns inputs as they are.

  Examples:
    >>> import coordax as cx
    >>> import jax.numpy as jnp
    >>> from neuralgcm.experimental.core import transforms
    >>> inputs = {'a': cx.field(jnp.zeros(2))}
    >>> transforms.Identity()(inputs)
    {'a': <Field dims=(None,) shape=(2,) axes={} >}
  """

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    return inputs


class Empty(PytreeTransformABC):
  """Returns an empty dict, regardless of inputs.

  Examples:
    >>> import coordax as cx
    >>> import jax.numpy as jnp
    >>> from neuralgcm.experimental.core import transforms
    >>> inputs = {'a': cx.field(jnp.zeros(2))}
    >>> transforms.Empty()(inputs)
    {}
  """

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    return {}


@nnx.dataclass
class PrescribedFields(PytreeTransformABC):
  """Returns a prescribed dict of fields, regardless of inputs.

  Args:
    prescribed_fields: A dictionary of predefined coordax Fields to return.

  Examples:
    >>> import coordax as cx
    >>> import jax.numpy as jnp
    >>> from neuralgcm.experimental.core import transforms
    >>> prescribed = {'b': cx.field(jnp.ones(2))}
    >>> transforms.PrescribedFields(prescribed)({'a': cx.field(jnp.zeros(2))})
    {'b': <Field dims=(None,) shape=(2,) axes={} >}
  """

  prescribed_fields: dict[str, TransformParams] = nnx.data()

  def __init__(self, prescribed_fields: dict[str, cx.Field]):
    self.prescribed_fields = {
        k: TransformParams(v) for k, v in prescribed_fields.items()
    }

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    return {k: v.get_value() for k, v in self.prescribed_fields.items()}

  def update_from_xarray(
      self,
      dataset: xarray.Dataset,
      sim_units: units.SimUnits,
      strict_matches: bool = True,
      **kwargs,
  ):
    """Updates `self.prescribed_fields` with data from dataset."""
    del kwargs  # Unused.
    extra_types = (cx.LabeledAxis,) if not strict_matches else ()
    for key, f in self.prescribed_fields.items():
      if key in dataset:
        da = dataset[key]
        data_units = units.parse_units(da.attrs['units'])
        da = da.copy(data=sim_units.nondimensionalize(da.values * data_units))
        candidate = xarray_utils.field_from_xarray(da, extra_types)
        candidate = candidate.order_as(*f.dims)
        if strict_matches:
          candidate = candidate.untag(f.coordinate).tag(f.coordinate)
        else:
          candidate = candidate.untag(*f.dims).tag(f.coordinate)
        f.set_value(candidate)

  @classmethod
  def zeros_like(
      cls,
      coords: dict[str, cx.Coordinate],
  ):
    """Returns a PrescribedFields instance with given shapes and zero values."""
    zeros_like = lambda c: cx.field(jnp.zeros(c.shape), c)
    return cls({k: zeros_like(c) for k, c in coords.items()})


class Broadcast(PytreeTransformABC):
  """Broadcasts fields in ``inputs`` of congruent dimensions to the same coords.

  This transform ensures that all input fields are broadcasted to share the same
  dimensions of the field with the highest number of dimensions. All other
  fields are expected to feature a subset of these dimensions to make the order
  of broadcasting dimensions well-defined.

  Examples:
    >>> import coordax as cx
    >>> import jax.numpy as jnp
    >>> from neuralgcm.experimental.core import transforms
    >>> x = cx.SizedAxis('x', 3)
    >>> y = cx.SizedAxis('y', 2)
    >>> inputs = {'a': cx.field(1.5), 'b': cx.field(jnp.ones((3, 2)), x, y)}
    >>> transforms.Broadcast()(inputs)['a'].dims
    ('x', 'y')
  """

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    # when broadcasting, fields either maintain or increase ndim. Hence it is
    # safe to attempt broadcasting to a field with largest ndim. If coordinates
    # do not align, an error will be raised during broadcasting.
    ndims = {k: v.ndim for k, v in inputs.items()}
    ref = inputs[max(ndims, key=ndims.get)]  # get key of the largest ndim.
    return {k: v.broadcast_like(ref) for k, v in inputs.items()}


@nnx.dataclass
class RavelDims(PytreeTransformABC):
  """Ravels specified dimensions into a new single dimension.

  Flattens a sequence of dims in each field into a single output dimension.
  If a field in ``inputs`` lacks some of the dimensions and `allow_missing` is
  ``False``, an error is raised. Otherwise the field is raveled only over
  `dims_to_ravel` that are present.

  Args:
    dims_to_ravel: Sequence of dimensions or coordinates to flatten.
    out_dim: The name or coordinate for the resulting flattened dimension.
    allow_missing: If False, raises if missing any dim in `dims_to_ravel`.

  Examples:
    >>> import coordax as cx
    >>> import jax.numpy as jnp
    >>> from neuralgcm.experimental.core import transforms
    >>> x, y = cx.SizedAxis('x', 2), cx.SizedAxis('y', 3)
    >>> inputs = {'a': cx.field(jnp.ones((2, 3)), x, y)}
    >>> out = transforms.RavelDims(('x', 'y'), 'z')(inputs)
    >>> out['a'].shape
    (6,)
    >>> out['a'].dims
    ('z',)
  """

  dims_to_ravel: tuple[str | cx.Coordinate, ...]
  out_dim: str | cx.Coordinate
  allow_missing: bool = True

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    untagged = cx.untag(
        inputs, *self.dims_to_ravel, allow_missing=self.allow_missing
    )
    ravel_fn = cx.cmap(jnp.ravel, out_axes='leading')
    return {k: ravel_fn(v).tag(self.out_dim) for k, v in untagged.items()}


class SelectKeys(PytreeTransformABC):
  """Selects subset of keys from the ``inputs``.

  This transform filters the inputs dictionary based on key matching, regex, or
  a custom callable.

  Args:
    keys: Keys to select. Can be a string, sequence of strings, or a callable.
    invert: If True, invert the selection (returns keys NOT matching).
    mode: Selection mode for string ``keys``. Can be 'match' or 'regex'.
    strict: If True, raise an error if any of the string keys are not found.

  Examples:
    >>> import coordax as cx
    >>> import jax.numpy as jnp
    >>> from neuralgcm.experimental.core import transforms
    >>> inputs = {'a': cx.field(jnp.ones(2)), 'b': cx.field(jnp.zeros(2))}
    >>> list(transforms.SelectKeys('a')(inputs).keys())
    ['a']
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


@nnx.dataclass
class FilterByCoord(PytreeTransformABC):
  """Selects subset of fields in ``inputs`` that match specified coord or type.

  Args:
    coords: A coordinate or sequence of coordinates that must be present.
    types: A type or sequence of types that the field's coordinates must match.
    invert: If True, invert the selection (returns fields that do NOT match).

  Examples:
    >>> import coordax as cx
    >>> import jax.numpy as jnp
    >>> from neuralgcm.experimental.core import transforms
    >>> x = cx.SizedAxis('x', 3)
    >>> inputs = {'a': cx.field(jnp.ones(3), x), 'b': cx.field(jnp.ones(2))}
    >>> list(transforms.FilterByCoord(coords=x)(inputs).keys())
    ['a']
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


@nnx.dataclass
class Sequential(TransformABC):
  """Applies sequence of transforms in order.

  Args:
    transforms: A sequence of Transforms to apply sequentially.

  Examples:
    >>> import coordax as cx
    >>> import jax.numpy as jnp
    >>> from neuralgcm.experimental.core import transforms
    >>> inputs = {'a': cx.field(jnp.ones(2)), 'b': cx.field(jnp.zeros(2))}
    >>> seq = transforms.Sequential([
    ...     transforms.SelectKeys('a'),
    ...     transforms.RenameKeys({'a': 'c'}),
    ... ])
    >>> list(seq(inputs).keys())
    ['c']
  """

  transforms: Sequence[Transform] = nnx.data()

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    for transform in self.transforms:
      inputs = transform(inputs)
    return inputs


@nnx.dataclass
class Merge(TransformABC):
  """Merges outputs of multiple transforms into a single dictionary.

  Applies multiple feature modules to the inputs and merges their outputs.
  Prefixes are added based on the feature module keys to disambiguate features.

  Args:
    feature_modules: Dictionary mapping prefix strings to Transform objects.
    always_add_prefix: If True, always prepends the prefix to the output keys.

  Examples:
    >>> import coordax as cx
    >>> import jax.numpy as jnp
    >>> from neuralgcm.experimental.core import transforms
    >>> inputs = {'a': cx.field(jnp.ones(2))}
    >>> t1, t2 = transforms.SelectKeys('a'), transforms.RenameKeys({'a': 'b'})
    >>> merged = transforms.Merge({'t1': t1, 't2': t2})(inputs)
    >>> sorted(merged.keys())
    ['a', 'b']
  """

  feature_modules: dict[str, Transform] = nnx.data()
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


@nnx.dataclass
class Isel(PytreeTransformABC):
  """Applies ``isel(indexers)`` to all fields in ``inputs``.

  Args:
    indexers: A mapping from dimension names or coordinates to indices/slices.

  Examples:
    >>> import coordax as cx
    >>> import jax.numpy as jnp
    >>> from neuralgcm.experimental.core import transforms
    >>> x = cx.SizedAxis('x', 4)
    >>> inputs = {'a': cx.field(jnp.arange(4), x)}
    >>> transforms.Isel({'x': slice(0, 2)})(inputs)['a'].shape
    (2,)
  """

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


@nnx.dataclass
class ReduceMean(PytreeTransformABC):
  """Computes mean over specified dims, using coordinate methods if available.

  Takes the mean over the specified dimensions. If reducing over a coordinate
  with specialized `mean` method, such as LonLatGrid, uses that method.

  Args:
    dims: Sequence of dimension names or coordinates to reduce over.

  Examples:
    >>> import coordax as cx
    >>> import jax.numpy as jnp
    >>> from neuralgcm.experimental.core import transforms
    >>> x = cx.SizedAxis('x', 4)
    >>> inputs = {'a': cx.field(jnp.array([1., 2., 3., 4.]), x)}
    >>> transforms.ReduceMean(('x',))(inputs)['a'].data
    Array(2.5, dtype=float32)
  """

  dims: tuple[str | cx.Coordinate, ...]

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    # TODO(dkochkov): Add more flexible support for custom reductions.
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


@nnx.dataclass
class Sel(PytreeTransformABC):
  """Applies ``sel(indexers, method=method)`` to all fields in ``inputs``.

  Args:
    indexers: A mapping from dimension names or coordinates to values/slices.
    method: Optional method name (e.g., 'nearest') to use for inexact matching.

  Examples:
    >>> import coordax as cx
    >>> import jax.numpy as jnp
    >>> import numpy as np
    >>> from neuralgcm.experimental.core import transforms
    >>> x = cx.LabeledAxis('x', np.array([0.1, 0.5, 1.0]))
    >>> inputs = {'a': cx.field(jnp.array([10., 20., 30.]), x)}
    >>> transforms.Sel({'x': 0.5})(inputs)['a'].data
    Array(20., dtype=float32)
  """

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


@nnx.dataclass
class ExpandDims(PytreeTransformABC):
  """Returns inputs with expanded `axis` at `loc`, broadcasting if necessary.

  Inserts a new axis into all input fields at the specified location.

  Args:
    axis: The coordinate of the axis to insert.
    loc: The position to insert at. Can be an integer or string/coordinate.

  Examples:
    >>> import coordax as cx
    >>> import jax.numpy as jnp
    >>> from neuralgcm.experimental.core import transforms
    >>> x = cx.SizedAxis('x', 2)
    >>> inputs = {'a': cx.field(jnp.ones((3,)))}
    >>> transforms.ExpandDims(x, 0)(inputs)['a'].dims
    ('x', None)
  """

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


@nnx.dataclass
class SplitAxis(PytreeTransformABC):
  """Splits fields along an axis and assigns new names to the split fields.

  Args:
    axis: The coordinate or a dimension name of the axis to split along.
    keys: Optional argument to ``SelectKeys`` to select fields to split. If
      ``None``, all fields in inputs are selected. Defaults to ``None``.
    split_names: Optional sequence of names to assign to the resulting split
      fields. If provided, its length must match the total number of splits
      across all selected fields. If ``None``, new names are generated by
      appending `_{i}` to the original field name. Defaults to ``None``.
    include_remaining: If ``True``, fields not selected for splitting are
      included in the output unchanged. Defaults to ``False``.

  Examples:
    >>> import coordax as cx
    >>> import jax.numpy as jnp
    >>> from neuralgcm.experimental.core import transforms
    >>> x = cx.SizedAxis('x', 2)
    >>> inputs = {'data': cx.field(jnp.array([1., 2.]), x)}
    >>> out = transforms.SplitAxis(axis=x, split_names=['a_0', 'a_1'])(inputs)
    >>> sorted(out.keys())
    ['a_0', 'a_1']
  """

  axis: cx.Coordinate | str
  keys: str | Sequence[str] | Callable[[str], bool] | None = None
  split_names: Sequence[str] | None = None
  include_remaining: bool = False

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    to_split = inputs if self.keys is None else SelectKeys(self.keys)(inputs)
    outputs = {}
    split_name_idx = 0
    for k, f in to_split.items():
      f = f.untag(self.axis)
      size = f.positional_shape[0]
      splits = cx.cmap(tuple)(f)  # tuple unpacking is split + squeeze.
      for j, split_field in enumerate(splits):
        if self.split_names is not None:
          new_name = self.split_names[split_name_idx + j]
        else:
          new_name = '_'.join([k, str(j)])
        outputs[new_name] = split_field
      split_name_idx += size

    if self.include_remaining:
      outputs |= {k: v for k, v in inputs.items() if k not in to_split}
    return outputs


@nnx.dataclass
class ApplyToKeys(TransformABC):
  """Applies `transform` to a subset of keys and merges with unchanged keys.

  Applies the given transform to specific keys and merges the results with the
  remaining keys.

  Args:
    transform: The ``Transform`` to apply to the selected keys.
    keys: An arg to ``SelectKeys`` that selects keys to apply the transform to.
    invert: If True, apply the transform to all keys except those in `keys`.
    include_remaining: If ``True``, pass through fields not in `keys` to output.

  Examples:
    >>> import coordax as cx
    >>> import jax.numpy as jnp
    >>> from neuralgcm.experimental.core import transforms
    >>> inputs = {'a': cx.field(jnp.ones(2)), 'b': cx.field(jnp.ones(2))}
    >>> double = transforms.ScaleFields(cx.field(2.0))
    >>> out = transforms.ApplyToKeys(double, ['a'])(inputs)
    >>> out['a'].data[0], out['b'].data[0]
    (Array(2., dtype=float32), Array(1., dtype=float32))
  """

  transform: Transform = nnx.data()
  keys: str | Sequence[str] | Callable[[str], bool]
  invert: bool = False
  include_remaining: bool = True

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    to_transform = SelectKeys(self.keys, self.invert)(inputs)
    if self.include_remaining:
      keep_as_is = {k: v for k, v in inputs.items() if k not in to_transform}
    else:
      keep_as_is = {}
    return self.transform(to_transform) | keep_as_is


@nnx.dataclass
class ApplyToFilteredKeys(TransformABC):
  """Applies `transform` to a subset of fields matching coord or type.

  Applies the given transform to fields selected by ``FilterByCoord`` and merges
  the results with the remaining fields if `include_remaining` is True.

  Args:
    transform: The ``Transform`` to apply to the selected fields.
    coords: A coordinate or sequence of coordinates that must be present.
    types: A type or sequence of types that the field's coordinates must match.
    invert: If True, apply the transform to fields that do NOT match.
    include_remaining: If ``True``, pass through unselected fields to output.

  Examples:
    >>> import coordax as cx
    >>> import jax.numpy as jnp
    >>> from neuralgcm.experimental.core import transforms
    >>> x = cx.SizedAxis('x', 3)
    >>> inputs = {'a': cx.field(jnp.ones(3), x), 'b': cx.field(jnp.ones(2))}
    >>> double = transforms.ScaleFields(cx.field(2.0))
    >>> out = transforms.ApplyToFilteredKeys(double, coords=x)(inputs)
    >>> out['a'].data[0], out['b'].data[0]
    (Array(2., dtype=float32), Array(1., dtype=float32))
  """

  transform: Transform = nnx.data()
  coords: cx.Coordinate | Sequence[cx.Coordinate] | None = None
  types: type[cx.Coordinate] | Sequence[type[cx.Coordinate]] | None = None
  invert: bool = False
  include_remaining: bool = True

  def __post_init__(self):
    self._filter = FilterByCoord(
        coords=self.coords, types=self.types, invert=self.invert
    )

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    to_transform = self._filter(inputs)
    if self.include_remaining:
      keep_as_is = {k: v for k, v in inputs.items() if k not in to_transform}
    else:
      keep_as_is = {}
    return self.transform(to_transform) | keep_as_is


@nnx.dataclass
class ApplyFnToKeys(PytreeTransformABC):
  """Applies a Field -> Field function to a subset of keys.

  Transforms specific fields using a provided function.

  Args:
    fn: A callable taking a coordax Field and returning a coordax Field.
    keys: An arg to ``SelectKeys`` that selects keys to apply the function to.
    invert: If True, apply the function to all keys except those in `keys`.
    include_remaining: If ``True``, pass through fields not in `keys` to output.

  Examples:
    >>> import coordax as cx
    >>> import jax.numpy as jnp
    >>> from neuralgcm.experimental.core import transforms
    >>> inputs = {'a': cx.field(jnp.ones(2)), 'b': cx.field(jnp.ones(2))}
    >>> fn = lambda f: f * 2.0
    >>> out = transforms.ApplyFnToKeys(fn, ['a'], True)(inputs)
    >>> out['a'].data[0], out['b'].data[0]
    (Array(2., dtype=float32), Array(1., dtype=float32))
  """

  fn: Callable[[cx.Field], cx.Field]
  keys: str | Sequence[str] | Callable[[str], bool]
  invert: bool = False
  include_remaining: bool = False

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    to_transform = SelectKeys(self.keys, self.invert)(inputs)
    if self.include_remaining:
      keep_as_is = {k: v for k, v in inputs.items() if k not in to_transform}
    else:
      keep_as_is = {}
    return {k: self.fn(v) for k, v in to_transform.items()} | keep_as_is


@nnx.dataclass
class WrapFn(TransformABC):
  """Wraps a callable to conform to the Transform API.

  Enables using functions that take individual fields and/or a dict of fields
  as a `Transform`.

  Args:
    fn: The function or module to wrap.
    out_keys: Specifies how to map the function's output to the output dict. If
      `fn` returns a single `cx.Field`, this must be a string. If `fn` returns a
      sequence of `cx.Field`s, this must be a sequence of strings of the same
      length. If `fn` returns a dict, this can be `None` to return it as-is, or
      it can be a dictionary mapping the returned keys to new keys.
    pass_inputs: Whether to pass the entire `inputs` dict to `fn`. Can be a
      boolean (True to pass as the first positional argument) or a string (to
      pass as a keyword argument).
    bind_inputs: A dictionary mapping argument names of `fn` to keys in the
      `inputs` dictionary that are passed as kwargs and removed from `inputs`.

  Examples:
    >>> import coordax as cx
    >>> import jax.numpy as jnp
    >>> from neuralgcm.experimental.core import transforms
    >>> inputs = {'a': cx.field(jnp.array([1.0, 2.0]))}
    >>> def get_sum(inputs, scale=1.0):
    ...   return inputs['a'] + scale
    >>> tx = transforms.WrapFn(get_sum, out_keys='b', kwargs={'scale': 2.0})
    >>> out = tx(inputs)
    >>> out['b'].data
    (Array([3., 4.], dtype=float32)
  """

  fn: Callable[..., Any]
  out_keys: str | Sequence[str] | dict[str, str] | None = None
  pass_inputs: bool | str = True
  bind_inputs: dict[str, str] | None = None
  kwargs: dict[str, Any] | None = None

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    inputs = inputs.copy()  # avoid mutating inputs when splitting keys.
    fn_kwargs = dict(self.kwargs) if self.kwargs is not None else {}
    if self.bind_inputs is not None:
      for arg_name, input_key in self.bind_inputs.items():
        if input_key not in inputs:
          raise KeyError(f'Key {input_key} not found in inputs')
        fn_kwargs[arg_name] = inputs.pop(input_key)

    if isinstance(self.pass_inputs, str):
      fn_kwargs[self.pass_inputs] = inputs
      result = self.fn(**fn_kwargs)
    elif self.pass_inputs:
      result = self.fn(inputs, **fn_kwargs)
    else:
      result = self.fn(**fn_kwargs)

    out_keys = self.out_keys
    if isinstance(result, cx.Field):
      if not isinstance(out_keys, str):
        raise ValueError(
            'out_keys must be a string when fn returns a single Field.'
        )
      outputs = {out_keys: result}
    elif isinstance(result, dict):
      if isinstance(out_keys, dict):
        outputs = {out_keys.get(k, k): v for k, v in result.items()}
      elif out_keys is None:
        outputs = dict(result)
      else:
        raise ValueError(
            'out_keys must be None or a dict when fn returns a dict, got'
            f' {out_keys=}'
        )
    elif isinstance(result, Sequence):
      if not isinstance(out_keys, Sequence) or len(out_keys) != len(result):
        raise ValueError(
            'out_keys must be a sequence of the same length as the returned'
            ' sequence.'
        )
      outputs = dict(zip(self.out_keys, result))
    else:
      raise TypeError(f'Unexpected return type {type(result)} from {self.fn}')
    return outputs


@nnx.dataclass
class ScanOverAxis(TransformABC):
  """Wrapper transform that applies `transform` over `axis` using scan.

  Runs a transform sequentially over slices along `axis` using ``nnx.scan``.

  Args:
    transform: The Transform to apply at each slice.
    axis: The dimension name or coordinate to scan over.
    apply_remat: If True, applies gradient rematerialization to the transform.

  Examples:
    >>> import coordax as cx
    >>> import jax.numpy as jnp
    >>> from neuralgcm.experimental.core import transforms
    >>> x = cx.SizedAxis('x', 3)
    >>> inputs = {'a': cx.field(jnp.ones((3, 2)), x, 'y')}
    >>> out = transforms.ScanOverAxis(transforms.Identity(), 'x')(inputs)
    >>> out['a'].dims
    ('x', 'y')
  """

  transform: Transform = nnx.data()
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


@nnx.dataclass
class ShardFields(PytreeTransformABC):
  """Adds `mesh.field_partitiotions[schema]` sharding constraint to ``inputs``.

  Args:
    mesh: The parallelism mesh defining device layout and sharding schemas.
    schema: The partitioning schema name (or sequence) to apply.

  Examples:
    >>> import coordax as cx
    >>> import jax.numpy as jnp
    >>> from neuralgcm.experimental.core import parallelism, transforms
    >>> mesh = parallelism.default_mesh()
    >>> inputs = {'a': cx.field(jnp.ones(2))}
    >>> transforms.ShardFields(mesh, 'data')(inputs)
    {'a': <Field dims=(None,) shape=(2,) axes={} >}
  """

  mesh: parallelism.Mesh
  schema: str | tuple[str, ...]

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    return self.mesh.with_sharding_constraint(inputs, self.schema)


class ScaleFields(PytreeTransformABC):
  """Applies x * `self.scale` to all fields in ``inputs``.

  Args:
    scale: The coordax Field used to scale the inputs.

  Examples:
    >>> import coordax as cx
    >>> import jax.numpy as jnp
    >>> from neuralgcm.experimental.core import transforms
    >>> inputs = {'a': cx.field(jnp.ones(2))}
    >>> out = transforms.ScaleFields(cx.field(2.0))(inputs)
    >>> out['a'].data[0]
    Array(2., dtype=float32)
  """

  def __init__(self, scale: cx.Field):
    self.scale = TransformParams(scale)

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    scale = self.scale.get_value()
    scale_fn = lambda x: x * scale
    return {k: scale_fn(v) for k, v in inputs.items()}


class StandardizeFields(PytreeTransformABC):
  """Shifts and scales inputs.

  Normalizes fields using provided shift (mean) and scale (std). Can be
  applied globally or uniquely per key. Also supports inverse normalization.

  Args:
    shifts: The shifts (or dict of shifts) used for centering input fields.
    scales: The scales (or dict of scales) used for normalization.
    from_normalized: If True, reverses the transformation (x * scale + shift).

  Examples:
    >>> import coordax as cx
    >>> import jax.numpy as jnp
    >>> from neuralgcm.experimental.core import transforms
    >>> inputs = {'a': cx.field(jnp.ones(2))}
    >>> out = transforms.StandardizeFields(cx.field(1.0), cx.field(2.0))(inputs)
    >>> out['a'].data[0]
    Array(0., dtype=float32)
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


@nnx.dataclass
class ClipWavenumbers(PytreeTransformABC):
  """Clips wavenumbers in inputs for grids matching `wavenumbers_for_grid`.

  Removes the highest wavenumbers (sets them to zero) for fields represented on
  spherical harmonic grids.

  Args:
    wavenumbers_for_grid: Mapping from grids to the # of wavenumbers to clip.
    skip_missing: If False, raises an error if encounters a grid that does not
      have a corresponding entry in `wavenumbers_for_grid`.

  Examples:
    >>> import coordax as cx
    >>> import jax.numpy as jnp
    >>> from neuralgcm.experimental.core import coordinates, transforms
    >>> grid = coordinates.SphericalHarmonicGrid.T21()
    >>> inputs = {'u': cx.field(jnp.ones(grid.shape), grid)}
    >>> out = transforms.ClipWavenumbers({grid: 2})(inputs)
    >>> out['u'].shape == grid.shape
    True
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


@nnx.dataclass
class InpaintHarmonics(TransformABC):
  """Inpaints `inputs` over `mask` with smoothed spherical harmonics.

  Iteratively fills masked regions of fields using a low-pass spherical
  harmonics filter, similar to the Papoulis-Gerchberg Algorithm.

  Args:
    ylm_map: Mapping between nodal and modal representations.
    compute_masks: Transform returning masks indicating entries to inpaint.
    lowpass_filter: Modal spatial filter to smooth the inpainting.
    n_iter: Number of PGA iterations to perform.
    default_mask_key: Optional fallback key for selecting mask for entries in
      ``inputs`` that do not have a key-matching mask in `compute_masks` output.
  """

  ylm_map: spherical_harmonics.YlmMapper | spherical_harmonics.FixedYlmMapping
  compute_masks: Transform = nnx.data()
  lowpass_filter: spatial_filters.ModalSpatialFilter = nnx.data()
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


@nnx.dataclass
class Regrid(PytreeTransformABC):
  """Applies `regridder` to all fields in `inputs`.

  Args:
    regridder: An interpolator object defining the regridding procedure.

  Examples:
    >>> import coordax as cx
    >>> import jax.numpy as jnp
    >>> from neuralgcm.experimental.core import coordinates, interpolators
    >>> from neuralgcm.experimental.core import transforms
    >>> grid_in = coordinates.LonLatGrid.TL63()
    >>> grid_out = coordinates.LonLatGrid.T21()
    >>> inputs = {'u': cx.field(jnp.ones(grid_in.shape), grid_in)}
    >>> regridder = interpolators.ConservativeRegridder(grid_out)
    >>> out = transforms.Regrid(regridder)(inputs)
    >>> out['u'].shape == grid_out.shape
    True
  """

  regridder: interpolators.BaseRegridder

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    return self.regridder(inputs)


@nnx.dataclass
class ComputeMasks(PytreeTransformABC):
  """Transforms inputs to boolean masks for keys in `mask_keys`.

  Computes a boolean mask for specified fields based on conditions like
  `isnan`, `above` threshold, or simply casting to boolean.

  Args:
    compute_mask_method: String specifying the method to compute bool mask.
    threshold_value: Value used by threshold-based `compute_mask_method`s.
    mask_keys: Keys to generate masks for. Defaults to all key in ``inputs``.

  Examples:
    >>> import coordax as cx
    >>> import jax.numpy as jnp
    >>> from neuralgcm.experimental.core import transforms
    >>> inputs = {'a': cx.field(jnp.array([1.0, jnp.nan, 3.0]))}
    >>> out = transforms.ComputeMasks('isnan')(inputs)
    >>> out['a'].data.tolist()
    [False, True, False]
  """

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


@nnx.dataclass
class ApplyOverMasks(TransformABC):
  """Applies masking to `transform(inputs)` using `compute_masks(inputs)`.

  Conditionally alters input fields based on computed masks. Can zero out,
  replace with mean, or apply custom functions over the masked elements.

  Args:
    compute_masks: Transform that generates the masks.
    apply_mask_method: String or callable specifying how to apply the mask.
    transform: Optional transform to apply to inputs before masking.
    default_mask_key: Fallback key for mask lookup.

  Examples:
    >>> import coordax as cx
    >>> import jax.numpy as jnp
    >>> from neuralgcm.experimental.core import transforms
    >>> inputs = {'a': cx.field(jnp.array([1., jnp.nan, 3.]))}
    >>> mask_tx = transforms.ComputeMasks('isnan')
    >>> out = transforms.ApplyOverMasks(mask_tx, 'nan_to_0')(inputs)
    >>> out['a'].data.tolist()
    [1.0, 0.0, 3.0]
  """

  compute_masks: Transform = nnx.data()
  apply_mask_method: ApplyMaskMethods = 'zero_multiply'
  transform: Transform | None = nnx.data(default=None)
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


class Nondimensionalize(PytreeTransformABC):
  """Nondimensionalizes ``inputs`` based on `sim_units` and expected units.

  Converts ``inputs`` to nondimensional quantities by associating each key with
  units specified in `inputs_to_units_mapping` and nondimensionalizing.

  Args:
    sim_units: Reference simulation units for nondimensionalization.
    inputs_to_units_mapping: Dict mapping keys to their physical units.

  Examples:
    >>> from neuralgcm.experimental.core import transforms, units
    >>> nondim = transforms.Nondimensionalize(units.SI_UNITS, {'x': 'inch'})
    >>> inputs = {'x': cx.field(np.array([1.0, 2.0]), 'idx')}
    >>> nondim(inputs)['x'].data.tolist()
    [0.0254, 0.0508]
  """

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


class Redimensionalize(PytreeTransformABC):
  """Redimensionalizes nondimensional ``inputs`` to expected units.

  Converts nondimensional quantities back into dimensional quantities using
  the provided reference simulation units and key-to-units mapping.

  Args:
    sim_units: Reference simulation units for dimensionalization.
    inputs_to_units_mapping: Dict mapping keys to their physical units.

  Examples:
    >>> from neuralgcm.experimental.core import transforms, units
    >>> redim = transforms.Redimensionalize(units.SI_UNITS, {'x': 'inch'})
    >>> inputs = {'x': cx.field(np.array([0.0254, 0.0508]), 'idx')}
    >>> redim(inputs)['x'].data.tolist()
    [1.0, 2.0]
  """

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


@nnx.dataclass
class StripPrefix(PytreeTransformABC):
  """Strips `prefix` from the dictionary keys, leaving values unchanged.

  Args:
    prefix: The string prefix to remove from keys.

  Examples:
    >>> import coordax as cx
    >>> import jax.numpy as jnp
    >>> from neuralgcm.experimental.core import transforms
    >>> inputs = {'pre_a': cx.field(jnp.ones(2))}
    >>> list(transforms.StripPrefix('pre_')(inputs).keys())
    ['a']
  """

  prefix: str

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    return {k.removeprefix(self.prefix): v for k, v in inputs.items()}


@nnx.dataclass
class RenameKeys(PytreeTransformABC):
  """Renames keys in inputs based on rename_dict.

  Args:
    rename_dict: A dictionary mapping current keys to new keys.

  Examples:
    >>> import coordax as cx
    >>> import jax.numpy as jnp
    >>> from neuralgcm.experimental.core import transforms
    >>> inputs = {'a': cx.field(jnp.ones(2))}
    >>> list(transforms.RenameKeys({'a': 'b'})(inputs).keys())
    ['b']
  """

  rename_dict: dict[str, str]

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    outputs = {}
    for k, v in inputs.items():
      new_key = self.rename_dict.get(k, k)
      if new_key in outputs:
        raise ValueError(f'Duplicate key after renaming: {new_key}')
      outputs[new_key] = v
    return outputs


@nnx.dataclass
class TanhClip(PytreeTransformABC):
  """Clips inputs to ``(-scale, scale)`` range via tanh function.

  Args:
    scale: A positive float determining the maximum and minimum output values.

  Examples:
    >>> import coordax as cx
    >>> import jax.numpy as jnp
    >>> from neuralgcm.experimental.core import transforms
    >>> inputs = {'a': cx.field(jnp.array([10.0, -10.0]))}
    >>> out = transforms.TanhClip(1.0)(inputs)
    >>> jnp.all(jnp.abs(out['a'].data) < 1.0).item()
    True
  """

  scale: float

  def __post_init__(self):
    if self.scale <= 0:
      raise ValueError(f'scale must be positive, got scale={self.scale}')

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    clip_fn = cx.cmap(lambda x: self.scale * jnp.tanh(x / self.scale))
    return {k: clip_fn(v) for k, v in inputs.items()}


@nnx.dataclass
class Relu(PytreeTransformABC):
  """Equivalent to ApplyFnToKeys with `jax.nn.relu` for `fn`.

  Args:
    keys: An arg to ``SelectKeys`` that selects keys to apply ReLU to.
    invert: If True, apply the function to all keys except those in `keys`.
    include_remaining: If ``True``, pass through fields not in `keys` to output.

  Examples:
    >>> import coordax as cx
    >>> import jax.numpy as jnp
    >>> from neuralgcm.experimental.core import transforms
    >>> inputs = {'a': cx.field(jnp.array([-1.0, 2.0]))}
    >>> transforms.Relu()(inputs)['a'].data.tolist()
    [0.0, 2.0]
  """

  keys: str | Sequence[str] | Callable[[str], bool] | None = None
  invert: bool = False
  include_remaining: bool = True

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    return ApplyFnToKeys(
        fn=cx.cpmap(jax.nn.relu),
        keys=self.keys,
        invert=self.invert,
        include_remaining=self.include_remaining,
    )(inputs)


class StreamNorm(TransformABC):
  """Normalizes inputs using values from streaming mean and variances.

  Maintains and applies streaming statistics for normalization.

  Args:
    norm_coords: A dict mapping variable names to their normalization axes.
    update_stats: Whether to update the normalization statistics on each call.
    epsilon: A small float added to variance to avoid division by zero.
    compute_masks: Optional transform to select entries for stat updates.
    skip_nans: If True, ignores NaNs when updating statistics.
    skip_unspecified: If True, passes through entries not in `norm_coords`.
    allow_missing: If True, allows `norm_coords` be a superset of input keys.

  Examples:
    >>> import coordax as cx
    >>> import jax.numpy as jnp
    >>> from neuralgcm.experimental.core import transforms
    >>> x = cx.SizedAxis('x', 2)
    >>> norm = transforms.StreamNorm({'a': x}, epsilon=0.0)
  """

  def __init__(
      self,
      norm_coords: dict[str, cx.Coordinate],
      update_stats: bool = False,
      epsilon: float = 1e-11,
      compute_masks: Transform | None = None,
      skip_nans: bool = True,
      skip_unspecified: bool = False,
      allow_missing: bool = True,
  ):
    """Initializes StreamNorm.

    Args:
      norm_coords: A dictionary mapping variable names to the coordinates over
        which the normalization is done independently.
      update_stats: Whether to update the normalization statistics.
      epsilon: A small float added to variance to avoid dividing by zero.
      compute_masks: Optional transform that produces a mask indicating which
        entries should contribute to the statistics updates.
      skip_nans: If True, ignores NaNs when updating the statistics.
      skip_unspecified: If True, input fields for which no normalization is
        specified (i.e., keys not in `norm_coords`) are passed through
        unchanged. If False, presence of such fields in `inputs` raises.
      allow_missing: If True, `inputs` may omit fields for which normalization
        is specified (i.e., keys in `norm_coords`). If False, all keys in
        `norm_coords` must be present in `inputs`, otherwise an error is raised.
    """
    self.stream_norm = normalizations.StreamNorm(
        norm_coords,
        epsilon=epsilon,
        skip_unspecified=skip_unspecified,
        allow_missing=allow_missing,
        skip_nans=skip_nans,
    )
    self.compute_masks = compute_masks
    self.update_stats = update_stats

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    """Normalizes inputs and optionally updates statistics."""
    if self.compute_masks is not None:
      mask = self.compute_masks(inputs)
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
    return self.stream_norm(inputs, self.update_stats, mask=mask)

  @classmethod
  def for_inputs_struct(
      cls,
      inputs_struct: dict[str, cx.Field],
      independent_axes: tuple[cx.Coordinate, ...],
      exclude_regex: str | None = None,
      scalar_regex: str | None = None,
      update_stats: bool = False,
      compute_masks: Transform | None = None,
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
        compute_masks=compute_masks,
        skip_unspecified=skip_unspecified,
        skip_nans=skip_nans,
        allow_missing=allow_missing,
    )


@nnx.dataclass
class ToModal(PytreeTransformABC):
  """Transforms inputs from nodal to modal space.

  Args:
    ylm_map: ``FixedYlmMapping`` or ``YlmMapper`` from `spherical_harmonics`.
  """

  ylm_map: spherical_harmonics.FixedYlmMapping | spherical_harmonics.YlmMapper

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    modal_outputs = {}
    for k, v in inputs.items():
      modal_outputs[k] = self.ylm_map.to_modal(v)
    return modal_outputs


@nnx.dataclass
class ToNodal(PytreeTransformABC):
  """Transforms inputs from modal to nodal space.

  Args:
    ylm_map: ``FixedYlmMapping`` or ``YlmMapper`` from `spherical_harmonics`.
  """

  ylm_map: spherical_harmonics.FixedYlmMapping | spherical_harmonics.YlmMapper

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    nodal_outputs = {}
    for k, v in inputs.items():
      nodal_outputs[k] = self.ylm_map.to_nodal(v)
    return nodal_outputs


@nnx.dataclass
class ToModalWithDerivatives(nnx.Pytree):
  """Helper module that returns filtered grads and laplacians of input fields.

  Computes spatial derivatives in modal space, applying exponential
  filters at specified attenuations.

  Args:
    ylm_map: The fixed spherical harmonics mapping configuration.
    filter_attenuations: Tuple of attenuations to apply for the modal filters.
  """

  ylm_map: spherical_harmonics.FixedYlmMapping
  attenuations: tuple[float, ...] = tuple()
  modal_filters: tuple[spatial_filters.ModalSpatialFilter, ...] = nnx.data(
      init=False
  )

  def __post_init__(self):
    self.modal_filters = tuple(
        spatial_filters.ExponentialModalFilter(
            self.ylm_map,
            attenuation=a,
            order=1,
        )
        for a in self.attenuations
    )

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


@nnx.dataclass
class ConstrainToCoarse(TransformABC):
  """Constrains fields so that spatially-averaged values match coarse reference.

  Applies scaling to high-resolution fields such that their spatial average,
  when regridded to a coarse grid, matches the coarse-resolution fields.

  Args:
    raw_hres_transform: Transform generating high-resolution fields.
    ref_coarse_transform: Transform generating coarse-resolution fields.
    coarse_grid: Coarse target grid for comparison.
    hres_grid: High-resolution source grid.
    keys: Sequence of keys to apply conservation scaling to.
    epsilon: Small value to avoid division by zero.

  Examples:
    >>> # Detailed example omitted for brevity.
    >>> pass
  """

  raw_hres_transform: Transform = nnx.data()
  ref_coarse_transform: Transform = nnx.data()
  coarse_grid: coordinates.LonLatGrid
  hres_grid: coordinates.LonLatGrid
  keys: Sequence[str]
  epsilon: float = 1e-6
  regrid_to_coarse: Regrid = nnx.static(init=False)
  regrid_to_hres: Regrid = nnx.static(init=False)

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


@nnx.dataclass
class VelocityFromDivCurl(PytreeTransformABC):
  """Transform divergence and vorticity in inputs to 2D velocity components.

  Computes u and v wind components from divergence and vorticity fields
  using spherical harmonics.

  Args:
    ylm_map: ``FixedYlmMapping`` or ``YlmMapper`` from `spherical_harmonics`.
    divergence_key: Key for divergence field in inputs.
    vorticity_key: Key for vorticity field in inputs.
    u_key: Key for the output u velocity field.
    v_key: Key for the output v velocity field.
    clip: Whether to apply clipping in the spectral transform.

  Examples:
    >>> # Requires SphericalHarmonics mapping object.
    >>> pass
  """

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


@nnx.dataclass
class VelocityToDivCurl(PytreeTransformABC):
  """Transform 2D velocity components to divergence and vorticity.

  Computes divergence and vorticity from u and v wind components
  using spherical harmonics.

  Args:
    ylm_map: ``FixedYlmMapping`` or ``YlmMapper`` from `spherical_harmonics`.
    divergence_key: Key for the output divergence field.
    vorticity_key: Key for the output vorticity field.
    u_key: Key for the input u velocity field.
    v_key: Key for the input v velocity field.
    clip: Whether to apply clipping in the spectral transform.

  Examples:
    >>> # Requires SphericalHarmonics mapping object.
    >>> pass
  """

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


@nnx.dataclass
class ExtractLocalPatchFromGrid(PytreeTransformABC):
  """Extracts a patch of neighbors from gridded fields around query points.

  Splits inputs into query points and gridded data, then extracts local
  patches from the gridded data centered around each query point.

  Args:
    width: The width of the square patch to extract. Defaults to 1.
    lon_query_key: Key for query longitudes in inputs.
    lat_query_key: Key for query latitudes in inputs.
    include_offset: Whether to include coordinates offset relative to queries.
  """

  width: int = 1
  lon_query_key: str = 'longitude'
  lat_query_key: str = 'latitude'
  include_offset: bool = True
  boundary_condition: boundaries.BoundaryCondition = nnx.data(init=False)
  padding: int = nnx.static(init=False)

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


@nnx.dataclass
class SanitizeGradients(TransformABC):
  """Wraps a transform to provide NaN-safe gradients.

  Uses a custom VJP to sanitize inputs and gradients by replacing NaNs with
  zeros during the backward pass, ensuring stable learning, while preserving
  the original transform's behavior in the forward pass to correctly inform NaN
  locations for weighting, collecting statistics etc. This effectively results
  in an additional forward pass in the backward propagation, similar to remat.

  Args:
    transform: The Transform to wrap.

  Examples:
    >>> import coordax as cx
    >>> import jax.numpy as jnp
    >>> from neuralgcm.experimental.core import transforms
    >>> safe_tx = transforms.SanitizeGradients(transforms.Identity())
  """

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


#
# Helper transforms that combine frequently used patterns.
#


@nnx.dataclass
class FillNaNs(PytreeTransformABC):
  """Replaces NaN values with a specified value for selected keys.

  Combines `SelectKeys`, `ComputeMask` and `ApplyOverMasks` transforms to fill
  `nan` values in Fields selected by `keys` with `value`.

  Args:
    keys: Argument to `SelectKeys` transform. If None, applies to all keys.
    invert: Whether to invert the selection of keys.
    value: The numeric value or 'mean' to replace NaNs with.
    include_remaining: Whether to include remaining keys not selected by `keys`.

  Examples:
    >>> import coordax as cx
    >>> import jax.numpy as jnp
    >>> from neuralgcm.experimental.core import transforms
    >>> inputs = {'a': cx.field(jnp.array([jnp.nan, 2.0]))}
    >>> transforms.FillNaNs(0.0)(inputs)['a'].data
    Array([0., 2.], dtype=float32)
  """

  keys: str | Sequence[str] | Callable[[str], bool] | None = None
  invert: bool = False
  value: float | Literal['mean'] = 0.0
  include_remaining: bool = True

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    keys = tuple(inputs.keys()) if self.keys is None else self.keys
    to_transform = SelectKeys(keys, self.invert)(inputs)

    def fill_nans(field: cx.Field) -> cx.Field:
      mask = cx.cmap(jnp.isnan)(field)
      if self.value == 'mean':
        return _masked_to_mean(field, mask)
      return _masked_nan_to_num(field, mask, num=self.value)

    outputs = {k: fill_nans(v) for k, v in to_transform.items()}
    if self.include_remaining:
      outputs.update({k: v for k, v in inputs.items() if k not in to_transform})
    return outputs


@nnx.dataclass
class ApplyMask(PytreeTransformABC):
  """Applies a boolean mask field to other fields.

  Args:
    mask_key: The key of the boolean mask field in ``inputs``.
    apply_method: Method specifying how to apply the mask. Can be one of:
      ('zero_multiply', 'nan_to_0', 'nan_to_mean', 'set_nan') or a callable.
    keys: Argument to ``SelectKeys`` transform. If None, applies to all keys.
    invert: Whether to invert the selection of keys.
    include_remaining: Whether to include remaining keys not selected by `keys`.

  Examples:
    >>> import coordax as cx
    >>> import jax.numpy as jnp
    >>> from neuralgcm.experimental.core import transforms
    >>> mask = cx.field(jnp.array([True, False]))
    >>> inputs = {'a': cx.field(jnp.array([1.0, 2.0])), 'mask': mask}
    >>> transforms.ApplyMask('mask', 'zero_multiply')(inputs)['a'].data
    Array([0., 2.], dtype=float32)
  """

  mask_key: str
  apply_method: ApplyMaskMethods = 'zero_multiply'
  keys: str | Sequence[str] | Callable[[str], bool] | None = None
  invert: bool = False
  include_remaining: bool = True

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    if self.mask_key not in inputs:
      raise KeyError(f'Mask key {self.mask_key} not found in inputs.')
    mask = inputs[self.mask_key]
    keys = tuple(inputs.keys()) if self.keys is None else self.keys
    to_transform = SelectKeys(keys, self.invert)(inputs)

    apply_fn = (
        self.apply_method
        if callable(self.apply_method)
        else APPLY_MASK_FNS[self.apply_method]
    )

    outputs = {k: apply_fn(v, mask) for k, v in to_transform.items()}
    if self.include_remaining:
      outputs.update({k: v for k, v in inputs.items() if k not in to_transform})
    return outputs


@nnx.dataclass
class Where(TransformABC):
  """Applies one of two transforms based on a boolean mask.

  Condition is evaluated using `jnp.where` on the fields output by the two
  transforms.
  Outputs of both transforms must have compatible shapes and coordinates with
  the mask.

  Args:
    mask_key: The key of the boolean mask field in inputs.
    true_transform: Transform to apply where the mask is True.
    false_transform: Transform to apply where the mask is False.

  Examples:
    >>> import coordax as cx
    >>> import jax.numpy as jnp
    >>> from neuralgcm.experimental.core import transforms
    >>> mask = cx.field(jnp.array([True, False]))
    >>> inputs = {'a': cx.field(jnp.array([1.0, 2.0])), 'mask': mask}
    >>> true_fn = transforms.ScaleFields(cx.field(10.0))
    >>> false_fn = transforms.ScaleFields(cx.field(100.0))
    >>> transforms.Where('mask', true_fn, false_fn)(inputs)['a'].data
    Array([ 10., 200.], dtype=float32)
  """

  mask_key: str
  true_transform: TransformABC = nnx.data()
  false_transform: TransformABC = nnx.data()

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    if self.mask_key not in inputs:
      raise KeyError(f'Mask key {self.mask_key} not found in inputs.')
    mask = inputs[self.mask_key]

    true_outputs = self.true_transform(inputs)
    false_outputs = self.false_transform(inputs)
    if set(true_outputs.keys()) != set(false_outputs.keys()):
      raise ValueError('True and false transforms must have the same keys.')

    where_fn = cx.cmap(jnp.where)
    results = {}
    for k in set(true_outputs.keys()) - {self.mask_key}:
      results[k] = where_fn(mask, true_outputs[k], false_outputs[k])
    return results


@nnx.dataclass
class EntrywiseBinaryOp(TransformABC):
  """Applies binary operation entrywise between fields and operand fields.

  Applies `op` entrywise to the pairs of Fields `inputs_transform(inputs)[key]`,
  `operand_transform(inputs)[key]` for all keys present in the latter.

  Args:
    op: A name of a supported operator ('subtract', 'divide', 'add', 'multiply')
      or a callable that takes two arrays and returns an array (e.g. `jnp.add`).
    operand_transform: A transform that returns the second operand for `op`.
    inputs_transform: Optional transform that returns the first operand.
    include_remaining: If True, fields in `inputs` whose keys are not in
      `operand_transform(inputs)` are passed through to the output.
  """

  op: Callable[[Any, Any], Any] | str
  operand_transform: Transform = nnx.data()
  inputs_transform: Transform = nnx.data(default_factory=Identity)
  include_remaining: bool = False

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    op = (
        _SUPPORTED_BINARY_OPS.get(self.op, self.op)
        if isinstance(self.op, str)
        else self.op
    )
    second_operand = self.operand_transform(inputs)
    keys = second_operand.keys()
    first_operand = {
        k: v for k, v in self.inputs_transform(inputs).items() if k in keys
    }
    mapped_op = cx.cmap(op)
    result = {k: mapped_op(first_operand[k], second_operand[k]) for k in keys}
    if self.include_remaining:
      result.update({k: v for k, v in inputs.items() if k not in keys})
    return result

  @classmethod
  def with_prescribed_fields(
      cls,
      op,
      fields: dict[str, cx.Field],
      inputs_transform: Transform = Identity(),
      include_remaining: bool = False,
  ):
    return cls(
        op=op,
        operand_transform=PrescribedFields(fields),
        inputs_transform=inputs_transform,
        include_remaining=include_remaining,
    )


@nnx.dataclass
class ProjectOntoBasis(TransformABC):
  """Projects inputs onto basis vectors and computes dot product.

  Applies `inputs_transform` and `basis_transform` to inputs and computes their
  dot product over `dims_to_contract`. Outputs of both transforms must have the
  same keys. The dot product is interpreted as the sum of entry-wise products of
  the fields. The dot product is output under the key `out_key`.

  Args:
    to_basis_transform: Transform that returns a basis fields.
    inputs_transform: Optional transform on inputs that will be projected.
    dims_to_contract: Sequence of dimensions to take the dot product over.
    out_key: The key for the output field containing the dot product.
    allow_missing: If True, ignores dimensions in `dims_to_contract` that are
      missing from individual fields.
  """

  to_basis_transform: Transform = nnx.data()
  inputs_transform: Transform = nnx.data(default_factory=Identity)
  dims_to_contract: Sequence[str | cx.Coordinate] = ()
  out_key: str = 'projection'
  allow_missing: bool = True

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    basis_dict = self.to_basis_transform(inputs)
    inputs_dict = self.inputs_transform(inputs)
    tree_dot = cx.cmap(
        lambda x, y: tree_math.numpy.dot(
            tree_math.Vector(x), tree_math.Vector(y)
        )
    )
    dims = self.dims_to_contract
    untag_fn = lambda f: cx.untag(f, *dims, allow_missing=self.allow_missing)
    result = tree_dot(untag_fn(inputs_dict), untag_fn(basis_dict))
    return {self.out_key: result}


TransformWithOptionalKey = typing.Transform | tuple[str, typing.Transform]


class NestedTransform(nnx.Module, pytree=False):
  """Wrapper that applies transforms to values of a nested dict[dict[Field]].

  Simplifies applying transforms to a two-level nested dictionary by assigning
  a specific or default transform to each top-level key.

  Args:
    transform: A single Transform (applied to all) or a dict associating data
      keys to Transforms. Ellipsis (...) can be used to set a default transform.

  Examples:
    >>> import coordax as cx
    >>> import jax.numpy as jnp
    >>> from neuralgcm.experimental.core import transforms
    >>> inputs = {'level1': {'a': cx.field(jnp.zeros(2))}}
    >>> transforms.NestedTransform(transforms.Identity())(inputs)
    {'level1': {'a': <Field dims=(None,) shape=(2,) axes={} >}}
  """

  def __init__(
      self,
      transform: (
          typing.Transform | dict[str | type(...), TransformWithOptionalKey]
      ),
      default_transform: typing.Transform | None = None,
  ):
    self.transforms: dict[str, tuple[str, typing.Transform]] = {}
    if isinstance(transform, dict):
      transforms_map = transform.copy()
      if ... in transforms_map:
        if default_transform is not None:
          raise ValueError(
              'Cannot specify default_transform both via `...` in '
              '`transform` dict and as a separate argument.'
          )
        self.default_transform = transforms_map.pop(...)
      else:
        self.default_transform = default_transform

      for k, v in transforms_map.items():
        if not isinstance(k, str):
          raise ValueError(f'Transform key {k} must be a string.')
        if isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], str):
          self.transforms[k] = v
        else:
          self.transforms[k] = (k, v)
    else:
      if default_transform is not None:
        raise ValueError(
            'Cannot specify default_transform when transform is not a dict.'
        )
      self.default_transform = transform

  def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
    outputs = {}
    explicit_in_keys = {in_key for (in_key, _) in self.transforms.values()}

    for out_key, (in_key, transform) in self.transforms.items():
      if in_key in inputs:
        outputs[out_key] = transform(inputs[in_key])

    for in_key, v in inputs.items():
      if in_key not in explicit_in_keys:
        if self.default_transform is None:
          raise ValueError(
              f'No default or key-specific transform for {in_key=}.'
          )
        outputs[in_key] = self.default_transform(v)
    return outputs

  def output_shapes(self, input_shapes: dict[str, Any]) -> dict[str, Any]:
    call_dispatch = lambda nested_transform, inputs: nested_transform(inputs)
    return nnx.eval_shape(call_dispatch, self, input_shapes)
