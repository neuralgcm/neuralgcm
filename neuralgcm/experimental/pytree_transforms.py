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

"""Modules that implement mappings between pytrees.

Pytree transforms describe a broad collection of mappings between pytrees. Most
of these can be crudely separated into two categories:
  1. Operations that transform individual arrays (subset of) within a pytree.
     [e.g. rescaling, reshaping, broadcasting, or other transforms]
  2. Operations that return elements (static) or a result of the computation
     dependent on inputs to the output pytree.
     [e.g. feature injection, input featurization, variable transformation]
"""

from __future__ import annotations

import abc
import dataclasses
import functools
import re
from typing import Callable, Protocol, Sequence

from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
from flax import nnx
import jax
import jax.numpy as jnp
import jax_datetime as jdt
from neuralgcm.experimental import coordax as cx
from neuralgcm.experimental import jax_solar
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import dynamic_io
from neuralgcm.experimental.core import nnx_compat
from neuralgcm.experimental.core import normalizations
from neuralgcm.experimental.core import orographies
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import pytree_utils
from neuralgcm.experimental.core import random_processes
from neuralgcm.experimental.core import spatial_filters
from neuralgcm.experimental.core import spherical_transforms
from neuralgcm.experimental.core import typing
from neuralgcm.experimental.core import units
from neuralgcm.experimental.core import xarray_utils
import numpy as np
import xarray


ShapeFloatStruct = typing.ShapeFloatStruct


class TransformParams(nnx.Variable):
  """Custom variable class for transform parameters."""


class Transform(Protocol):
  """Protocol for pytree transforms."""

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    ...

  def output_shapes(
      self, input_shapes: typing.Pytree | None = None
  ) -> typing.Pytree:
    ...


TransformFactory = Callable[..., Transform]


class TransformABC(nnx.Module, abc.ABC):
  """Abstract base class for pytree transforms."""

  @abc.abstractmethod
  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    raise NotImplementedError()

  def output_shapes(
      self, input_shapes: typing.Pytree | None = None
  ) -> typing.Pytree:
    return nnx.eval_shape(self.__call__, input_shapes)


class Broadcast(TransformABC):
  """Features transform that broadcasts all features."""

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    leaves, tree_def = jax.tree.flatten(inputs)
    leaves = jnp.broadcast_arrays(*leaves)
    return jax.tree.unflatten(tree_def, leaves)


class Identity(TransformABC):
  """Transform does not modify inputs."""

  def __call__(self, inputs: typing.PyTreeState) -> typing.PyTreeState:
    return inputs


class Empty(TransformABC):
  """Transform that returns an empty dict."""

  def __call__(self, inputs: ...) -> typing.Pytree:
    return {}


@nnx_compat.dataclass
class NestDict(TransformABC):
  """Transform that nests elements of a flat dictionary into a nested dict."""

  keys_to_nest: tuple[str, ...]
  nested_key_name: str

  def __call__(self, inputs: dict[str, typing.Array]) -> typing.Pytree:
    nested = {k: v for k, v in inputs.items() if k in self.keys_to_nest}
    non_nested = {k: v for k, v in inputs.items() if k not in self.keys_to_nest}
    return non_nested | {self.nested_key_name: nested}


# TODO(dkochkov): Consider supporting use_bias use_scale selectively.
# could use dict[str, bool] variation, or even default_dict if fiddle is happy.
class BatchShiftAndNormalize(TransformABC):
  """Shift & norm over feature axis with support for batch stat updates."""

  def __init__(
      self,
      feature_sizes: dict[str, int],
      feature_axis: int = -3,
      use_running_average: bool = True,
      use_bias: bool = False,
      use_scale: bool = False,
      momentum: float = 0.99,
      epsilon: float = 1e-5,
      bias_init: nnx.initializers.Initializer = nnx.initializers.zeros_init(),
      scale_init: nnx.initializers.Initializer = nnx.initializers.ones_init(),
      batch_axis_name: str | None = None,
      skip_unspecified: bool = False,
      rngs: nnx.Rngs | None = None,
  ):
    self.batch_transforms = {
        k: nnx.BatchNorm(
            num_features=v,
            use_running_average=False,  # controlled by class attribute.
            axis=feature_axis,
            momentum=momentum,
            epsilon=epsilon,
            use_scale=use_scale,
            use_bias=use_bias,
            bias_init=bias_init,
            scale_init=scale_init,
            axis_name=batch_axis_name,
            rngs=rngs,
        )
        for k, v in feature_sizes.items()
    }
    self.skip_unspecified = skip_unspecified
    self.use_running_average = use_running_average

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    inputs, from_dict_fn = pytree_utils.as_dict(inputs)
    transforms = pytree_utils.replace_with_matching_or_default(
        inputs,
        self.batch_transforms,
        default=lambda x, _: x if self.skip_unspecified else None,
        check_used_all_replace_keys=False,
    )
    use_avg = self.use_running_average
    outputs = jax.tree.map(lambda x, y: y(x, use_avg), inputs, transforms)
    return from_dict_fn(outputs)

  @classmethod
  def for_input_shapes(
      cls,
      input_shapes: typing.Pytree,
      feature_axis: int = -3,
      use_running_average: bool = True,
      use_bias: bool = False,
      use_scale: bool = False,
      momentum: float = 0.99,
      epsilon: float = 1e-5,
      bias_init: nnx.initializers.Initializer = nnx.initializers.zeros_init(),
      scale_init: nnx.initializers.Initializer = nnx.initializers.ones_init(),
      batch_axis_name: str | None = None,
      skip_unspecified: bool = False,
      rngs: nnx.Rngs | None = None,
  ):
    """Custom constructor based on input shapes that should be normalized."""
    feature_sizes = jax.tree.map(lambda x: x.shape[feature_axis], input_shapes)
    return cls(
        feature_sizes=feature_sizes,
        feature_axis=feature_axis,
        use_running_average=use_running_average,
        use_bias=use_bias,
        use_scale=use_scale,
        momentum=momentum,
        epsilon=epsilon,
        bias_init=bias_init,
        scale_init=scale_init,
        batch_axis_name=batch_axis_name,
        skip_unspecified=skip_unspecified,
        rngs=rngs,
    )


class StreamingStatsNormalization(TransformABC):
  """Transform that uses streaming mean and variance to normalize inputs."""

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
    self.stream_norm_transforms = {
        k: normalizations.StreamNorm(v, feature_axes, epsilon=epsilon)
        for k, v in feature_shapes.items()
    }
    self.skip_unspecified = skip_unspecified
    self.update_stats = update_stats

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    inputs, from_dict_fn = pytree_utils.as_dict(inputs)
    transforms = pytree_utils.replace_with_matching_or_default(
        inputs,
        self.stream_norm_transforms,
        default=lambda x, _: x if self.skip_unspecified else None,
        check_used_all_replace_keys=False,
    )
    normalize = lambda x, y: y(x, self.update_stats)
    outputs = jax.tree.map(normalize, inputs, transforms)
    return from_dict_fn(outputs)

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
        k: tuple(v.shape[i] for i in feature_axes)
        for k, v in input_shapes.items()
    }
    if exclude_regex is not None:
      feature_shapes = {
          k: v for k, v in feature_shapes.items()
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
class FeatureSelector(TransformABC):
  """Features transform that retains items whose keys match against regex.

  Attributes:
    regex_patterns: regular expression pattern that specifies the set of keys
      from `inputs` that will be returned by __call__ method.
  """

  regex_patterns: str

  def __call__(
      self,
      inputs: dict[str, typing.Array],
  ) -> dict[str, typing.Array]:
    outputs = {}
    for k, v in inputs.items():
      if re.fullmatch(self.regex_patterns, k):
        outputs[k] = v
    return outputs


class ComposedTransform(nnx.Module):
  """Transform that composes a sequence of transforms."""

  def __init__(
      self,
      transforms: Sequence[Transform],
  ):
    self.transforms = transforms

  def __call__(self, inputs):
    for transform in self.transforms:
      inputs = transform(inputs)
    return inputs

  def output_shapes(
      self, input_shapes: typing.Pytree | None = None
  ) -> typing.Pytree:
    # We use explicit implementation here to allow subcomponents to call custom
    # output_shapes methods.
    for transform in self.transforms:
      input_shapes = transform.output_shapes(input_shapes)
    return input_shapes


# TODO(dkochkov): Think whether these can have default init + init from file.
class ShiftAndNormalize(TransformABC):
  """Transforms inputs by shifting and normalizing values by `shifts/scales`."""

  def __init__(
      self,
      shifts: typing.Pytree,
      scales: typing.Pytree,
      global_scale: float | None = None,
  ):
    if global_scale is not None:
      scales = jax.tree.map(lambda x: x * global_scale, scales)
    self.shifts = TransformParams(shifts)
    self.scales = TransformParams(scales)

  def update_shifts_and_scales(self, shifts, scales):
    """Updates values of `shifts` and `scales`."""
    self.shifts = TransformParams(shifts)
    self.scales = TransformParams(scales)

  def __call__(self, inputs):
    inputs, from_dict_fn = pytree_utils.as_dict(inputs)
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
    # if shifts/scales have missing values present in `inputs`, we insert `None`
    # for the default. If corresponding `inputs` is not `None`, this will raise
    # an error, as expected. This works because tree_map skips `None` values in
    # the first argument, as long as all dictionary keys match.
    result = jax.tree.map(
        lambda x, y, z: (x - y) / z, inputs, shifts, scales
    )
    return from_dict_fn(result)

  @classmethod
  def identity_init(cls, input_shapes: typing.Pytree):
    """Initializes the module with identity transform."""
    shifts = jax.tree.map(jnp.zeros, input_shapes)
    scales = jax.tree.map(jnp.ones, input_shapes)
    return cls(shifts=shifts, scales=scales)


class InverseShiftAndNormalize(TransformABC):
  """Inverse of the `ShiftAndNormalize` for the same `shifts/scales`."""

  def __init__(
      self,
      shifts: typing.Pytree,
      scales: typing.Pytree,
      global_scale: float | None = None,
  ):
    if global_scale is not None:
      scales = jax.tree.map(lambda x: x * global_scale, scales)
    self.shifts = TransformParams(shifts)
    self.scales = TransformParams(scales)

  def update_shifts_and_scales(self, shifts, scales):
    """Updates values of `shifts` and `scales`."""
    self.shifts = TransformParams(shifts)
    self.scales = TransformParams(scales)

  def __call__(self, inputs):
    inputs, from_dict_fn = pytree_utils.as_dict(inputs)
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
    # if shifts/scales have missing values present in `inputs`, we insert `None`
    # for the default. If corresponding `inputs` is not `None`, this will raise
    # an error, as expected. This works because tree_map skips `None` values in
    # the first argument, as long as all dictionary keys match.
    result = jax.tree.map(
        lambda x, y, z: x * z + y, inputs, shifts, scales
    )
    return from_dict_fn(result)

  @classmethod
  def identity_init(cls, input_shapes: typing.Pytree):
    """Initializes the module with identity transform."""
    shifts = jax.tree.map(jnp.zeros, input_shapes)
    scales = jax.tree.map(jnp.ones, input_shapes)
    return cls(shifts=shifts, scales=scales)


class TakeSurfaceAdjacentSigmaLevel(TransformABC):
  """Takes surface adjacent sigma level from the input."""

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    """Returns `inputs` where only last sigma level is retained."""

    def _slice_fn(x):
      return x[slice(-1, None), ...]

    inputs, from_dict_fn = pytree_utils.as_dict(inputs)
    outputs = jax.tree.map(_slice_fn, inputs)
    return from_dict_fn(outputs)


@nnx_compat.dataclass
class ClipWavenumbers(TransformABC):
  """Sets top `wavenumbers_to_clip` total wavenumbers to zero in the input."""

  grid: coordinates.SphericalHarmonicGrid
  wavenumbers_to_clip: int = 1
  mesh: parallelism.Mesh = dataclasses.field(kw_only=True)

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    """Returns `inputs` where only last sigma level is retained."""
    # TODO(dkochkov): add clip_wavenumbers method to SphericalHarmonicGrid.
    ylm_grid = self.grid.ylm_grid
    ylm_grid = dataclasses.replace(ylm_grid, spmd_mesh=self.mesh.spmd_mesh)
    return ylm_grid.clip_wavenumbers(inputs, self.wavenumbers_to_clip)


class MaskTransform(TransformABC):
  """Masks arrays in the input dictionary with keys in `fields_to_mask`."""

  def __init__(
      self,
      fields_to_mask: Sequence[str],
      mask_shape: tuple[int, ...],
      fill_value_true: float = 0.0,
      fill_value_false: float = 1.0,
      *,
      fill_threshold: float = jnp.nan,
  ):
    """Fills `fill_value` in `fields_to_mask` entries of an input pytree."""
    self.mask_shape = mask_shape
    self.mask = TransformParams(jnp.zeros(mask_shape))
    self.fill_value_true = fill_value_true
    self.fill_value_false = fill_value_false
    self.fill_threshold = fill_threshold
    self.fields_to_mask = fields_to_mask

  def update_from_xarray(
      self, dataset: xarray.Dataset, threshold_variable: str
  ):
    dataset = dataset[threshold_variable]
    if threshold_variable == 'sea_ice_cover':
      # TODO(asubel): This is a hack, fix this.
      dataset = dataset.max('time').squeeze()
    if 'time' in dataset.dims:
      dataset = dataset.isel(time=0).squeeze()
    if dataset.shape != self.mask_shape:
      raise ValueError(
          f'dataset.shape={dataset.shape=} does not match'
          f' self.mask.shape={self.mask_shape=}'
      )
    if self.fill_threshold is jnp.nan:
      mask_value = xarray.where(
          np.isnan(dataset), self.fill_value_true, self.fill_value_false
      )
    else:
      mask_value = xarray.where(
          dataset > self.fill_threshold,
          self.fill_value_true,
          self.fill_value_false,
      )
    self.mask = TransformParams(jnp.array(mask_value.values))

  def update_from_pytree(self, data):
    assert set(data.keys()) == set(self.mask.keys())
    self.mask = TransformParams(data)

  def update_from_array(self, mask: typing.Array):
    if mask.shape != self.mask_shape:
      raise ValueError(
          f'mask.shape={mask.shape=} does not match'
          f' self.mask.shape={self.mask.shape=}'
      )
    self.mask = TransformParams(mask)

  def __call__(self, inputs):
    inputs, from_dict_fn = pytree_utils.as_dict(inputs)
    for k, v in inputs.items():
      if k in self.fields_to_mask:
        inputs[k] = v * self.mask.value
    return from_dict_fn(inputs)


@nnx_compat.dataclass
class ToModalWithDivCurl(TransformABC):
  """Module that converts inputs to modal replacing velocity with div/curl."""

  ylm_transform: spherical_transforms.SphericalHarmonicsTransform
  u_key: str = 'u_component_of_wind'
  v_key: str = 'v_component_of_wind'

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    dinosaur_grid = self.ylm_transform.dinosaur_grid
    mesh = self.ylm_transform.mesh
    if self.u_key not in inputs or self.v_key not in inputs:
      raise ValueError(
          f'{(self.u_key, self.v_key)=} not found in {inputs.keys()=}'
      )
    sec_lat = 1 / dinosaur_grid.cos_lat
    u, v = inputs.pop(self.u_key), inputs.pop(self.v_key)
    u, v = parallelism.with_physics_to_dycore_sharding(mesh, (u, v))
    # here u,v stand for velocity / cos(lat), but the cos(lat) is cancelled in
    # divergence and curl operators below.
    inputs[self.u_key] = u * sec_lat
    inputs[self.v_key] = v * sec_lat
    to_modal_fn = lambda x: dinosaur_grid.to_modal(x) if x is not None else None
    inputs = parallelism.with_dycore_sharding(mesh, inputs)
    modal_outputs = jax.tree.map(to_modal_fn, inputs)
    modal_outputs = parallelism.with_dycore_sharding(mesh, modal_outputs)
    u, v = modal_outputs.pop(self.u_key), modal_outputs.pop(self.v_key)
    modal_outputs['divergence'] = parallelism.with_dycore_sharding(
        mesh, dinosaur_grid.div_cos_lat((u, v))
    )
    modal_outputs['vorticity'] = parallelism.with_dycore_sharding(
        mesh, dinosaur_grid.curl_cos_lat((u, v))
    )
    return modal_outputs


@nnx_compat.dataclass
class ToNodalWithVelocity(TransformABC):
  """Module that converts inputs to nodal replacing div/curl with velocity."""

  ylm_transform: spherical_transforms.SphericalHarmonicsTransform
  u_key: str = 'u_component_of_wind'
  v_key: str = 'v_component_of_wind'

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    dinosaur_grid = self.ylm_transform.dinosaur_grid
    if 'divergence' not in inputs or 'vorticity' not in inputs:
      raise ValueError(f'required `u, v` not found in {inputs.keys()=}')
    divergence, vorticity = inputs.pop('divergence'), inputs.pop('vorticity')
    u, v = spherical_harmonic.vor_div_to_uv_nodal(
        dinosaur_grid, vorticity=vorticity, divergence=divergence
    )
    to_nodal_fn = lambda x: dinosaur_grid.to_nodal(x) if x is not None else None
    nodal_outputs = jax.tree.map(to_nodal_fn, inputs)
    nodal_outputs[self.u_key] = u
    nodal_outputs[self.v_key] = v
    return nodal_outputs


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

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    inputs, from_dict_fn = pytree_utils.as_dict(inputs)
    # here we handle different DataObservation types separately for now.
    result = {}
    for k, v in inputs.items():
      if cx.is_field(v):
        result[k] = self._nondim_field(v, k)
      elif isinstance(v, typing.Numeric):
        result[k] = self._nondim_numeric(v, k)
      else:
        raise ValueError(f'Unsupported type {type(v)} for key {k}.')
    return from_dict_fn(result)


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

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    inputs, from_dict_fn = pytree_utils.as_dict(inputs)
    # here we handle different DataObservation types separately for now.
    result = {}
    for k, v in inputs.items():
      if cx.is_field(v):
        result[k] = self._redim_field(v, k)
      elif isinstance(v, typing.Numeric):
        result[k] = self._redim_numeric(v, k)
      else:
        raise ValueError(f'Unsupported type {type(v)} for key {k}.')
    return from_dict_fn(result)


@nnx_compat.dataclass
class ToModal(TransformABC):
  """Module that returns inputs that are converted from nodal to modal space."""

  ylm_transform: spherical_transforms.SphericalHarmonicsTransform

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    modal_outputs = {}
    for k, v in inputs.items():
      modal_outputs[k] = self.ylm_transform.to_modal_array(v)
    return modal_outputs


class ToModalWithFilteredGradients(TransformABC):
  """Module that returns filtered grad and laplacian features of inputs fields.

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
            ylm_transform.ylm_grid, attenuation=a, order=1,
            mesh=self.ylm_transform.mesh
        )
        for a in filter_attenuations
    ]
    self.modal_filters = modal_filters

  def __call__(
      self,
      inputs: dict[typing.KeyWithCosLatFactor, typing.Array],
  ) -> dict[typing.KeyWithCosLatFactor, typing.Array]:
    dinosaur_grid = self.ylm_transform.dinosaur_grid
    features = {}
    for k, value in inputs.items():
      name, cos_lat_order = k.name, k.factor_order
      for filter_module, att in zip(self.modal_filters, self.attenuations):
        d_value_dlon, d_value_dlat = dinosaur_grid.cos_lat_grad(value)
        laplacian_value = dinosaur_grid.laplacian(value)
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
        features[dlon_key] = filter_module.filter_modal(d_value_dlon)
        features[dlat_key] = filter_module.filter_modal(d_value_dlat)
        features[del2_key] = filter_module.filter_modal(laplacian_value)
    return features


def _scale_levels_for_matching_keys(
    inputs: typing.Pytree,
    scales: typing.Array,
    keys_to_scale: Sequence[str] = tuple(),
) -> typing.Pytree:
  """Transforms `inputs` by scaling levels for keys that are in `keys_to_scale.

  Args:
    inputs: pytree of values that will be selectively scaled along levels.
    scales: scaling weights that will be applied.
    keys_to_scale: keys for which scaling operation is applied.

  Returns:
    pytree of the same structure of `inputs` with keys matching `keys_to_scale`
    scaled by `scales` along the level axis.
  """
  if scales.ndim != 1:
    raise ValueError(
        'scales must be 1d array of weights per level, got '
        f'array with shape {scales.shape}'
    )
  scales = scales[:, np.newaxis, np.newaxis]  # broadcasting shape.
  inputs, from_dict_fn = pytree_utils.as_dict(inputs)
  scale_fn = lambda x: x * scales
  inputs = pytree_utils.map_over_matching_keys(inputs, scale_fn, keys_to_scale)
  return from_dict_fn(inputs)


class LevelScale(TransformABC):
  """Transform inputs by scaling different vertical levels."""

  def __init__(
      self,
      scales: Sequence[float],
      keys_to_scale: Sequence[str] = tuple(),
  ):
    self.scale_fn = functools.partial(
        _scale_levels_for_matching_keys,
        scales=np.asarray(scales),
        keys_to_scale=keys_to_scale,
    )

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    return self.scale_fn(inputs)


class InverseLevelScale(TransformABC):
  """Transform inputs by inverse scaling different vertical levels."""

  def __init__(
      self,
      scales: Sequence[float],
      keys_to_scale: Sequence[str] = tuple(),
  ):
    self.scale_fn = functools.partial(
        _scale_levels_for_matching_keys,
        scales=1 / np.asarray(scales),
        keys_to_scale=keys_to_scale,
    )

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    return self.scale_fn(inputs)


#
# Transforms that are return pytrees with injected features.
#


@nnx_compat.dataclass
class RadiationFeatures(TransformABC):
  """Feature module that computes incident radiation flux."""

  grid: coordinates.LonLatGrid
  transform: Transform = Identity()

  @property
  def lon(self) -> typing.Array:
    # TODO(dkochkov): support field broadcasting to a coordinate.
    dummy = cx.wrap(jnp.zeros(self.grid.shape), self.grid)
    return self.grid.fields['longitude'].broadcast_like(dummy).data

  @property
  def lat(self) -> typing.Array:
    # TODO(dkochkov): support field broadcasting to a coordinate.
    dummy = cx.wrap(jnp.zeros(self.grid.shape), self.grid)
    return self.grid.fields['latitude'].broadcast_like(dummy).data

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    features = {}
    features['radiation'] = jax_solar.normalized_radiation_flux(
        time=inputs['time'], longitude=self.lon, latitude=self.lat
    )
    return self.transform(features)


@nnx_compat.dataclass
class LatitudeFeatures(TransformABC):
  """Feature module that creates cos and sin of latitude features."""

  grid: coordinates.LonLatGrid
  transform: Transform = Identity()

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    del inputs  # unused.
    # TODO(dkochkov): support field broadcasting to a coordinate.
    dummy = cx.wrap(jnp.zeros(self.grid.shape), self.grid)
    latitudes = jnp.deg2rad(
        self.grid.fields['latitude'].broadcast_like(dummy).data
    )
    features = {
        'cos_latitude': jnp.cos(latitudes),
        'sin_latitude': jnp.sin(latitudes),
    }
    return self.transform(features)


@nnx_compat.dataclass
class OrographyFeatures(TransformABC):
  """Feature module that computes orographic features."""

  orography_module: orographies.ModalOrography | nnx.Module
  transform: Transform = Identity()

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    del inputs  # unused.
    return self.transform({'orography': self.orography_module.nodal_orography})


@nnx_compat.dataclass
class OrographyWithGradsFeatures(TransformABC):
  """Feature module that computes orographic features and their gradients."""

  orography_module: orographies.ModalOrography
  compute_gradients_transform: ToModalWithFilteredGradients | None = None
  include_raw_orography: bool = True
  transform: Transform = Identity()

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    del inputs  # unused.
    modal_features = {'orography': self.orography_module.modal_orography}
    modal_features = {  # jit should eliminate to_modal if it is not used.
        typing.KeyWithCosLatFactor(k, 0): v for k, v in modal_features.items()
    }
    modal_gradient_features = self.compute_gradients_transform(modal_features)
    sh_grid = self.orography_module.ylm_transform.modal_grid.ylm_grid
    sec_lat = 1 / sh_grid.cos_lat
    sec2_lat = sh_grid.sec2_lat
    sec_lat_scales = {0: 1, 1: sec_lat, 2: sec2_lat}
    features = {}
    if self.include_raw_orography:
      all_modal_features = modal_gradient_features | modal_features
    else:
      all_modal_features = modal_gradient_features
    for k, v in all_modal_features.items():
      sec_lat_scale = sec_lat_scales[k.factor_order]
      features[k.name] = sh_grid.to_nodal(v) * sec_lat_scale
    return self.transform(features)


@nnx_compat.dataclass
class PressureFeatures(TransformABC):
  """Feature module that computes pressure."""

  ylm_transform: spherical_transforms.SphericalHarmonicsTransform
  sigma: coordinates.SigmaLevels
  transform: Transform = Identity()

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    to_nodal_fn = self.ylm_transform.to_nodal_array
    sigma = self.sigma.fields['sigma'].data
    surface_pressure = jnp.exp(to_nodal_fn(inputs['log_surface_pressure']))
    pressure = surface_pressure * sigma[:, jnp.newaxis, jnp.newaxis]
    return self.transform({'pressure': pressure})


@nnx_compat.dataclass
class RandomnessFeatures(nnx.Module):
  """Feature module that returns values from a random process."""

  random_process: random_processes.RandomProcessModule
  grid: cx.Coordinate
  feature_name: str = 'randomness'
  transform: Transform = Identity()

  def output_shapes(
      self, input_shapes: typing.Pytree | None = None
  ) -> typing.Pytree:
    del input_shapes  # unused.
    leading_shape = self.random_process.event_shape
    return {
        self.feature_name: ShapeFloatStruct(leading_shape + self.grid.shape)
    }

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    del inputs  # unused.
    return self.transform({
        self.feature_name: self.random_process.state_values(self.grid).data,
    })


@nnx_compat.dataclass
class DynamicInputFeatures(TransformABC):
  """Feature module for computes dynamic input features."""

  # TODO(dkochkov) Generalize this to work beyond surface features.

  keys: Sequence[str]
  dynamic_input_module: dynamic_io.DynamicInputSlice
  transform: Transform = Identity()

  def output_shapes(
      self, input_shapes: typing.Pytree | None = None
  ) -> typing.Pytree:
    # Note: we override this method because at init
    del input_shapes  # unused.
    data_shapes = self.dynamic_input_module.output_shapes()
    return {k: v for k, v in data_shapes.items() if k in self.keys}

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    time = inputs['time']
    data_features = self.dynamic_input_module(time)
    features = {k: data_features[k].data for k in self.keys}
    return self.transform(features)


# TODO(dkochkov) Generalize this to work on coords with different grids.
class SpatialSurfaceFeatures(TransformABC):
  """Features for atmospheric columns that are static in space and time."""

  def __init__(
      self,
      feature_sizes: dict[str, int],
      *,
      grid: cx.Coordinate,
      transform: Transform = Identity(),
      param_type: TransformParams | nnx.Param = TransformParams,
      initializer: nnx.Initializer = nnx.initializers.truncated_normal(),
      rngs: nnx.Rngs,
  ):
    self.transform = transform
    self.features = {
        k: param_type(initializer(rngs.params(), shape=((v,) + grid.shape)))
        for k, v in feature_sizes.items()
    }

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    del inputs  # unused.
    return self.transform({k: v.value for k, v in self.features.items()})

  def update_features_from_data(
      self,
      dataset: xarray.Dataset,
      sim_units: units.SimUnits,
      spatial_filter=None,
  ):
    """Updates `self.features` with filtered data from dataset."""
    # TODO(dkochkov) use units attr on dataset with default to `meter` here.
    if spatial_filter is None:
      spatial_filter = lambda x: x

    lon_name, lat_name = xarray_utils.get_longitude_latitude_names(dataset)
    # TODO(dkochkov) Add consistency verification for lon/lat.
    # lon, lat = (dataset[lon_name], dataset[lat_name])
    # xarray_utils.verify_grid_consistency(lon, lat, self.coords)
    for key, feature_value in self.features.items():
      if key in dataset:
        data = dataset[key].transpose(lon_name, lat_name)
        data_units = units.parse_units(data.attrs['units'])
        data = sim_units.nondimensionalize(data.values * data_units)
        data = spatial_filter(data)
        self.features[key] = type(feature_value)(data)


class VelocityAndPrognosticsWithModalGradients(TransformABC):
  """Features module that returns prognostics + u,v and optionally gradients."""

  def __init__(
      self,
      ylm_transform: spherical_transforms.SphericalHarmonicsTransform,
      surface_field_names: tuple[str, ...] = tuple(),
      volume_field_names: tuple[str, ...] = tuple(),
      compute_gradients_transform: ToModalWithFilteredGradients | None = None,
      transform: Transform = Identity(),
      inputs_are_modal: bool = True,
      u_key: str = 'u_component_of_wind',
      v_key: str = 'v_component_of_wind',
  ):
    if compute_gradients_transform is None:
      compute_gradients_transform = Empty()
    self.ylm_transform = ylm_transform
    self.surface_field_names = surface_field_names
    self.volume_field_names = volume_field_names
    self.fields_to_include = surface_field_names + volume_field_names
    self.compute_gradients_transform = compute_gradients_transform
    self.transform = transform
    self.u_key = u_key
    self.v_key = v_key
    if inputs_are_modal:
      self.pre_process = lambda x: x
    else:
      self.pre_process = ToModal(ylm_transform)

  def _extract_features(
      self,
      inputs: typing.Pytree,
      prefix: str = '',
  ) -> typing.Pytree:
    """Returns a nodal velocity and prognostic features."""
    # Note: all intermediate features have an explicit cos-lat factors in key.
    # These factors are removed in the `__call__` method before returning.

    dinosaur_grid = self.ylm_transform.dinosaur_grid
    # compute `u, v` if div/curl is available and `u, v` not in prognosics.
    if set(['vorticity', 'divergence']).issubset(inputs.keys()) and (
        not set([self.u_key, self.v_key]).intersection(inputs.keys())
    ):
      cos_lat_u, cos_lat_v = spherical_harmonic.get_cos_lat_vector(
          inputs['vorticity'], inputs['divergence'], dinosaur_grid
      )
      modal_features = {}
      if self.u_key in self.fields_to_include:
        modal_features[typing.KeyWithCosLatFactor(prefix + self.u_key, 1)] = (
            cos_lat_u
        )
      if self.v_key in self.fields_to_include:
        modal_features[typing.KeyWithCosLatFactor(prefix + self.v_key, 1)] = (
            cos_lat_v
        )
    elif self.u_key in inputs and self.v_key in inputs:
      modal_features = {
          typing.KeyWithCosLatFactor(prefix + self.u_key, 0): inputs[
              self.u_key
          ],
          typing.KeyWithCosLatFactor(prefix + self.v_key, 0): inputs[
              self.v_key
          ],
      }
    else:
      modal_features = {}
    prognostics_keys = list(inputs.keys())
    if 'tracers' in prognostics_keys:
      prognostics_keys.remove('tracers')
    if 'sim_time' in prognostics_keys:
      prognostics_keys.remove('sim_time')
    if 'time' in prognostics_keys:
      prognostics_keys.remove('time')
    for k in set(self.fields_to_include) - set([self.u_key, self.v_key]):
      if k in prognostics_keys:
        modal_features[typing.KeyWithCosLatFactor(prefix + k, 0)] = inputs[k]
      elif k in inputs.get('tracers', {}):
        value = inputs['tracers'][k]
        modal_features[typing.KeyWithCosLatFactor(prefix + k, 0)] = value
      else:
        raise ValueError(f'Prognostic {k} not found in inputs.')

    # Computing gradient features and adjusting cos_lat factors.
    modal_features = parallelism.with_dycore_sharding(
        self.ylm_transform.mesh, modal_features
    )
    diff_operator_features = self.compute_gradients_transform(modal_features)
    sec_lat = 1 / dinosaur_grid.cos_lat
    sec2_lat = dinosaur_grid.sec2_lat
    sec_lat_scales = {0: 1, 1: sec_lat, 2: sec2_lat}
    # Computing all features in nodal space.
    features = {}
    for k, v in (diff_operator_features | modal_features).items():
      sec_lat_scale = sec_lat_scales[k.factor_order]
      features[k.name] = dinosaur_grid.to_nodal(v) * sec_lat_scale
    features = parallelism.with_dycore_sharding(
        self.ylm_transform.mesh, features
    )
    return features

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    inputs = self.pre_process(inputs)
    nodal_features = self._extract_features(inputs)
    return self.transform(nodal_features)


@nnx_compat.dataclass
class PrognosticFeatures(TransformABC):
  """Features module that returns a subset of prognostic variables."""

  prognostic_keys: tuple[str, ...] = tuple()
  transform: Transform = Identity()

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    return self.transform({k: inputs[k] for k in self.prognostic_keys})


class EmbeddedFeatures(TransformABC):
  """Features module that returns feature generated from an embedding module."""

  def __init__(
      self,
      fields_to_embed: Sequence[str],
      *,
      embedding_module: nnx.Module,
      transform: Transform = Identity(),
  ):
    self.fields_to_embed = fields_to_embed
    self.embedding_module = embedding_module
    self.transform = transform

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    output_features = self.embedding_module(inputs)
    return self.transform(output_features)

  def output_shapes(
      self, input_shapes: typing.Pytree | None = None
  ) -> typing.Pytree:
    return self.transform.output_shapes(
        self.embedding_module.output_shapes(input_shapes)
    )


class SurfaceEmbeddingFeatures(TransformABC):
  """Features module that returns embeddings for each surface point."""

  def __init__(
      self,
      grid: coordinates.LonLatGrid,
      embedding_sizes: dict[str, int],
      embedding_factory: nnx.Module,
      transform: Transform = Identity(),
  ):
    embedding_shapes = {
        k: typing.ShapeFloatStruct((v,) + grid.shape)
        for k, v in embedding_sizes.items()
    }
    self.embedding = embedding_factory(output_shapes=embedding_shapes)
    self.transform = transform

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    output_features = self.embedding(inputs)
    return self.transform(output_features)

  def output_shapes(
      self, input_shapes: typing.Pytree | None = None
  ) -> typing.Pytree:
    return self.transform.output_shapes(self.embedding.output_shapes)


class VolumeEmbeddingFeatures(TransformABC):
  """Features module that returns embeddings for each point on the coords."""

  def __init__(
      self,
      coords: cx.Coordinate,
      embedding_names: tuple[str, ...],
      embedding_factory: nnx.Module,
      transform: Transform = Identity(),
  ):
    embedding_shapes = {
        k: typing.ShapeFloatStruct(coords.shape) for k in embedding_names
    }
    self.embedding = embedding_factory(output_shapes=embedding_shapes)
    self.transform = transform

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    output_features = self.embedding(inputs)
    return self.transform(output_features)

  def output_shapes(
      self, input_shapes: typing.Pytree | None = None
  ) -> typing.Pytree:
    return self.transform.output_shapes(self.embedding.output_shapes)


@nnx_compat.dataclass
class CombinedFeatures(TransformABC):
  """Feature module that combines multiple feature modules together.

  Feature modules that will be combined are specified as dictionary values where
  keys indicate optional feature prefix. This helps with: (1) disambiguating
  multiple differently configured features; (2) accessing feature modules of a
  configured model. By default, prefix is only added if `always_add_prefix` is
  set to True or if there's a conflict in feature names.
  """

  feature_modules: dict[str, Transform]
  transform: Transform = Identity()
  always_add_prefix: bool = False

  def __call__(self, inputs: typing.Pytree) -> dict[str, typing.Array]:
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
    return self.transform(all_features)

  def output_shapes(
      self, input_shapes: typing.Pytree | None = None
  ) -> typing.Pytree:
    out_shapes = {}
    for m in self.feature_modules.values():
      out_shapes |= m.output_shapes(input_shapes)
    return self.transform.output_shapes(out_shapes)
