# Copyright 2026 Google LLC
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

"""Parameterization transforms for hybrid atmospheric models."""

from __future__ import annotations

from collections.abc import Sequence

import coordax as cx
from flax import nnx
import jax.numpy as jnp
import jax_datetime as jdt
from terrax.atmosphere import transforms as atm_transforms
from terrax.core import coordinates
from terrax.core import parallelism
from terrax.core import pytree_utils
from terrax.core import spatial_filters
from terrax.core import spherical_harmonics
from terrax.core import transforms
from terrax.core import typing

# pylint: disable=g-classes-have-attributes


def div_curl_target_split_axes(
    levels: coordinates.SigmaLevels | coordinates.HybridLevels,
    volume_field_names: Sequence[str] = (
        'divergence',
        'vorticity',
        'temperature',
        'specific_humidity',
    ),
    surface_field_names: Sequence[str] = ('log_surface_pressure',),
    *,
    div_curl_from_uv: bool = True,
    u_key: str = 'u_component_of_wind',
    v_key: str = 'v_component_of_wind',
) -> dict[str, cx.Coordinate]:
  """Returns target split axes for nodal tendency predictions.

  When `div_curl_from_uv=True`, replaces `'divergence'` and `'vorticity'` in
  `volume_field_names` with `u_key` and `v_key` (or accepts `u_key` and `v_key`
  directly). When `div_curl_from_uv=False`, keeps `'divergence'` and
  `'vorticity'` directly and disallows `u_key` and `v_key`. Maps all volume
  fields to `levels` and all surface fields in `surface_field_names` to
  `cx.Scalar()`.

  Args:
    levels: Vertical levels for 3D volume tendencies.
    volume_field_names: Names of 3D fields whose tendencies are predicted.
    surface_field_names: Names of 2D surface fields whose tendencies are
      predicted.
    div_curl_from_uv: Whether divergence and vorticity tendencies are predicted
      via `(u_key, v_key)` or directly as `('divergence', 'vorticity')`.
    u_key: Key for zonal wind tendency predicted in nodal space when
      `div_curl_from_uv=True`.
    v_key: Key for meridional wind tendency predicted in nodal space when
      `div_curl_from_uv=True`.
  """
  div_curl_keys = {'divergence', 'vorticity'}
  uv_keys = {u_key, v_key}
  present_div_curl = div_curl_keys.intersection(volume_field_names)
  present_uv = uv_keys.intersection(volume_field_names)
  if len(present_div_curl) == 1:
    raise ValueError(
        "`volume_field_names` must contain both 'divergence' and 'vorticity' "
        f'or neither, got {list(volume_field_names)}.'
    )
  if div_curl_from_uv:
    if present_div_curl and present_uv:
      raise ValueError(
          '`volume_field_names` cannot contain both divergence/vorticity and '
          f'{(u_key, v_key)}, got {list(volume_field_names)}.'
      )
    if len(present_uv) == 1:
      raise ValueError(
          f'`volume_field_names` must contain both {u_key!r} and {v_key!r} '
          f'or neither, got {list(volume_field_names)}.'
      )
    renames = {'divergence': u_key, 'vorticity': v_key}
    mapped_volume = [renames.get(k, k) for k in volume_field_names]
  else:
    if present_uv:
      raise ValueError(
          f'`volume_field_names` cannot contain {sorted(present_uv)} when '
          f'`div_curl_from_uv=False`, got {list(volume_field_names)}.'
      )
    mapped_volume = list(volume_field_names)
  if overlapping := set(mapped_volume).intersection(surface_field_names):
    raise ValueError(
        f'Overlapping volume and surface field names: {sorted(overlapping)}.'
    )
  split_axes: dict[str, cx.Coordinate] = {
      name: levels for name in mapped_volume
  }
  for name in surface_field_names:
    split_axes[name] = cx.Scalar()
  return split_axes


def primitive_equation_state_shapes(
    ylm_map: spherical_harmonics.FixedYlmMapping,
    levels: (
        coordinates.PressureLevels
        | coordinates.SigmaLevels
        | coordinates.HybridLevels
    ),
    tracer_names: Sequence[str] = ('specific_humidity',),
    nodal_tracers: Sequence[str] = (),
    surface_field_names: Sequence[str] = ('log_surface_pressure',),
    include_time: bool = True,
) -> dict[str, cx.Field]:
  """Returns abstract shape fields for primitive equation prognostics.

  Args:
    ylm_map: Spherical harmonic mapping defining modal `ylm_grid` and nodal
      `lon_lat_grid`.
    levels: Vertical levels of 3D prognostic variables.
    tracer_names: Names of 3D tracer variables.
    nodal_tracers: Subset of `tracer_names` carried on `ylm_map.lon_lat_grid`
      rather than `ylm_map.ylm_grid`.
    surface_field_names: Names of 2D modal surface variables on `ylm_grid`.
    include_time: Whether to include a scalar `'time'` `Datetime` shape field.
  """
  if unknown_nodal := set(nodal_tracers) - set(tracer_names):
    raise ValueError(
        f'`nodal_tracers` {sorted(unknown_nodal)} must be a subset of '
        f'`tracer_names` {list(tracer_names)}.'
    )
  modal_volume = cx.coords.compose(levels, ylm_map.ylm_grid)
  nodal_volume = cx.coords.compose(levels, ylm_map.lon_lat_grid)
  volume_keys = ('divergence', 'vorticity', 'temperature')
  state_shapes: dict[str, cx.Field] = {
      k: cx.shape_struct_field(modal_volume) for k in volume_keys
  }
  for k in tracer_names:
    coord = nodal_volume if k in nodal_tracers else modal_volume
    state_shapes[k] = cx.shape_struct_field(coord)
  for k in surface_field_names:
    state_shapes[k] = cx.shape_struct_field(ylm_map.ylm_grid)
  if include_time:
    state_shapes['time'] = pytree_utils.shape_structure(
        cx.field(jdt.to_datetime('1970-01-01T00'))  # pyrefly: ignore[bad-argument-type]
    )
  return state_shapes


def _validate_tendency_keys(
    tendencies: dict[str, cx.Field],
    prognostic_targets: dict[str, cx.Field],
    *,
    div_curl_from_uv: bool,
    u_key: str,
    v_key: str,
) -> None:
  """Validates `tendencies` keys against `prognostic_targets` and wind mode."""
  has_u = u_key in tendencies
  has_v = v_key in tendencies
  if div_curl_from_uv:
    if has_u != has_v:
      raise ValueError(
          f'Both {u_key!r} and {v_key!r} must be present in tendencies to '
          f'compute divergence and vorticity, got {sorted(tendencies)}.'
      )
    if 'divergence' in tendencies or 'vorticity' in tendencies:
      raise ValueError(
          "Tendencies cannot contain 'divergence' or 'vorticity' when "
          f'`div_curl_from_uv=True`, got {sorted(tendencies)}.'
      )
    if has_u:
      missing_div_curl = {'divergence', 'vorticity'} - set(prognostic_targets)
      if missing_div_curl:
        raise ValueError(
            f'Tendencies contain ({u_key!r}, {v_key!r}), but '
            f'{sorted(missing_div_curl)} are missing from inputs '
            f'{sorted(prognostic_targets)}.'
        )
    allowed_keys = set(prognostic_targets) | {u_key, v_key}
  else:
    if has_u or has_v:
      raise ValueError(
          f'Tendencies cannot contain ({u_key!r}, {v_key!r}) when '
          f'`div_curl_from_uv=False`, got {sorted(tendencies)}.'
      )
    allowed_keys = set(prognostic_targets)

  if unknown_keys := set(tendencies) - allowed_keys:
    raise ValueError(
        f'Unexpected tendency keys {sorted(unknown_keys)} not present in '
        f'prognostic inputs {sorted(prognostic_targets)}.'
    )


@nnx.dataclass
class ModalParameterization(transforms.TransformABC):
  """Computes modal and nodal prognostic tendencies from nodal predictions.

  `ModalParameterization` is a `Transform` designed for
  `LearnedParameterizationSpectralDycore.learned_physics`. Given `prognostics`
  (containing modal variables on `ylm_map.ylm_grid`, optional nodal tracers on
  `ylm_map.lon_lat_grid`, and `'time'`), it:
    1. Optionally extracts features via `features_transform`.
    2. Evaluates `nodal_tendency_transform` to predict nodal tendencies on
      `ylm_map.lon_lat_grid` (predicting horizontal wind tendencies
      `u_key, v_key` in place of `'divergence'` and `'vorticity'` when
      `div_curl_from_uv=True`, or `'divergence'` and `'vorticity'` directly when
      `div_curl_from_uv=False`), followed by optional `tendency_transform` (e.g.
      output rescaling).
    3. Converts tendencies of modal prognostics on `ylm_map.ylm_grid` to modal
      space (converting `(u_key, v_key)` to modal `('divergence', 'vorticity')`
      when `div_curl_from_uv=True`, or transforming all modal variables directly
      via `ylm_map.to_modal` when `div_curl_from_uv=False`), while keeping
      tendencies of `nodal_tracers` in nodal space.
    4. Applies `modal_filter.filter_modal` (when `modal_filter` is set) to the
      modal tendencies only, leaving nodal tracer tendencies unfiltered.
    5. Fills any prognostic variable in `inputs` (excluding `'time'`) that was
      not predicted by `nodal_tendency_transform` with a zero `cx.Field` on the
      same coordinate as the input variable.

  Note that `features_transform` and `tendency_transform` can be left as `None`
  when feature extraction and output rescaling are bundled inside
  `nodal_tendency_transform`; these optional transforms are provided to
  modularize the computational flow.

  Attributes:
    ylm_map: Spherical harmonics mapping between the model's nodal and modal
      grids.
    nodal_tendency_transform: Transform mapping features (or `inputs` directly)
      to nodal tendencies on `ylm_map.lon_lat_grid`.
    modal_filter: Optional modal spatial filter applied to modal tendencies.
    features_transform: Optional transform mapping `inputs` to nodal features.
    tendency_transform: Optional transform applied to nodal tendencies before
      modal conversion.
    div_curl_from_uv: If `True` (default), `nodal_tendency_transform` predicts
      horizontal velocity tendencies `(u_key, v_key)` which are transformed to
      modal `('divergence', 'vorticity')` via `ToModalWithDivCurl`. If `False`,
      `nodal_tendency_transform` predicts nodal `'divergence'` and `'vorticity'`
      directly, which are converted to modal space via `ylm_map.to_modal`.
    u_key: Key for zonal wind tendency predicted by `nodal_tendency_transform`
      when `div_curl_from_uv=True`.
    v_key: Key for meridional wind tendency predicted by
      `nodal_tendency_transform` when `div_curl_from_uv=True`.
    mesh: Parallelism mesh. Defaults to `ylm_map.mesh` if `None`.
  """

  ylm_map: spherical_harmonics.FixedYlmMapping
  nodal_tendency_transform: typing.Transform = nnx.data()
  modal_filter: spatial_filters.ModalSpatialFilter | None = nnx.data(
      default=None
  )
  features_transform: typing.Transform | None = nnx.data(default=None)
  tendency_transform: typing.Transform | None = nnx.data(default=None)
  div_curl_from_uv: bool = nnx.static(kw_only=True, default=True)
  u_key: str = nnx.static(kw_only=True, default='u_component_of_wind')
  v_key: str = nnx.static(kw_only=True, default='v_component_of_wind')
  mesh: parallelism.Mesh | None = nnx.static(kw_only=True, default=None)

  def __post_init__(self):
    if self.mesh is None:
      self.mesh = self.ylm_map.mesh
    if self.div_curl_from_uv:
      self.to_div_curl = atm_transforms.ToModalWithDivCurl(
          self.ylm_map, u_key=self.u_key, v_key=self.v_key
      )
    else:
      self.to_div_curl = None

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    inputs = parallelism.with_dycore_sharding(self.mesh, inputs)
    if self.features_transform is not None:
      features = self.features_transform(inputs)
      features = parallelism.with_dycore_to_physics_sharding(
          self.mesh, features
      )
    else:
      features = inputs
    tendencies = self.nodal_tendency_transform(features)
    tendencies = parallelism.with_physics_to_dycore_sharding(
        self.mesh, tendencies
    )
    if self.tendency_transform is not None:
      tendencies = self.tendency_transform(tendencies)
    tendencies = dict(tendencies)

    prognostic_targets = {k: v for k, v in inputs.items() if k != 'time'}
    _validate_tendency_keys(
        tendencies,
        prognostic_targets,
        div_curl_from_uv=self.div_curl_from_uv,
        u_key=self.u_key,
        v_key=self.v_key,
    )

    to_modal: dict[str, cx.Field] = {}
    nodal_tendencies: dict[str, cx.Field] = {}
    has_u = self.u_key in tendencies
    if self.div_curl_from_uv and has_u:
      to_modal[self.u_key] = tendencies.pop(self.u_key)
      to_modal[self.v_key] = tendencies.pop(self.v_key)

    ylm_grid = self.ylm_map.ylm_grid
    lon_lat_grid = self.ylm_map.lon_lat_grid
    for k, v in tendencies.items():
      target = prognostic_targets[k]
      if cx.contains_dims(target, ylm_grid):
        to_modal[k] = v
      elif cx.contains_dims(target, lon_lat_grid):
        nodal_tendencies[k] = v
      else:
        raise ValueError(
            f'Prognostic input {k!r} with coordinate {target.coordinate} '
            f'does not contain modal grid {ylm_grid} or nodal grid '
            f'{lon_lat_grid}.'
        )

    if self.div_curl_from_uv and has_u:
      assert self.to_div_curl is not None
      modal_tendencies = self.to_div_curl(to_modal)
    elif to_modal:
      to_modal = parallelism.with_dycore_sharding(self.mesh, to_modal)
      modal_tendencies = self.ylm_map.to_modal(to_modal)
      modal_tendencies = parallelism.with_dycore_sharding(
          self.mesh, modal_tendencies
      )
    else:
      modal_tendencies = {}

    if self.modal_filter is not None and modal_tendencies:
      modal_tendencies = self.modal_filter.filter_modal(modal_tendencies)

    combined = modal_tendencies | nodal_tendencies
    outputs = {
        k: (
            combined[k]
            if k in combined
            else cx.cmap(jnp.zeros_like)(target_field)
        )
        for k, target_field in prognostic_targets.items()
    }
    return parallelism.with_dycore_sharding(self.mesh, outputs)

  @classmethod
  def build_using_factories(
      cls,
      *,
      ylm_map: spherical_harmonics.FixedYlmMapping,
      levels: coordinates.SigmaLevels | coordinates.HybridLevels,
      transform_factory: typing.TransformFactory,
      modal_filter: spatial_filters.ModalSpatialFilter | None = None,
      features_transform: typing.Transform | None = None,
      tendency_transform: typing.Transform | None = None,
      volume_field_names: Sequence[str] = (
          'divergence',
          'vorticity',
          'temperature',
          'specific_humidity',
      ),
      surface_field_names: Sequence[str] = ('log_surface_pressure',),
      nodal_tracers: Sequence[str] = (),
      input_state_shapes: dict[str, cx.Field] | None = None,
      div_curl_from_uv: bool = True,
      u_key: str = 'u_component_of_wind',
      v_key: str = 'v_component_of_wind',
      mesh: parallelism.Mesh | None = None,
      rngs: nnx.Rngs,
  ) -> ModalParameterization:
    """Builds a `ModalParameterization` using `transform_factory`.

    Computes `target_split_axes` from `levels`, `volume_field_names` (mapping
    `'divergence'` and `'vorticity'` to `u_key` and `v_key` when
    `div_curl_from_uv=True`), and `surface_field_names`, derives
    `input_state_shapes` if not provided, and instantiates
    `nodal_tendency_transform` via
    `transform_factory(input_shapes=..., target_split_axes=..., rngs=rngs)`.

    Args:
      ylm_map: Spherical harmonics mapping between the model's nodal and modal
        grids.
      levels: Vertical levels for 3D prognostic variables.
      transform_factory: Factory callable taking `(input_shapes=...,
        target_split_axes=..., rngs=rngs)` and returning
        `nodal_tendency_transform`.
      modal_filter: Optional modal spatial filter applied to modal tendencies.
      features_transform: Optional transform mapping `inputs` to nodal features
        prior to `nodal_tendency_transform`. When provided, `transform_factory`
        receives `features_transform.output_shapes(input_state_shapes)` as
        `input_shapes`.
      tendency_transform: Optional transform applied to nodal tendencies
        returned by `nodal_tendency_transform` before modal conversion.
      volume_field_names: Names of 3D fields whose tendencies are predicted.
      surface_field_names: Names of 2D surface fields whose tendencies are
        predicted.
      nodal_tracers: Names of tracers carried on `ylm_map.lon_lat_grid` rather
        than `ylm_map.ylm_grid` when deriving default `input_state_shapes`.
      input_state_shapes: Optional abstract shape fields of the input
        prognostics. If `None`, derived automatically via
        `primitive_equation_state_shapes`.
      div_curl_from_uv: Whether divergence and vorticity tendencies are
        predicted via `(u_key, v_key)` or directly as `('divergence',
        'vorticity')`.
      u_key: Key for zonal wind tendency predicted by `nodal_tendency_transform`
        when `div_curl_from_uv=True`.
      v_key: Key for meridional wind tendency predicted by
        `nodal_tendency_transform` when `div_curl_from_uv=True`.
      mesh: Parallelism mesh. Defaults to `ylm_map.mesh` if `None`.
      rngs: Random number generators passed to `transform_factory`.
    """
    target_split_axes = div_curl_target_split_axes(
        levels=levels,
        volume_field_names=volume_field_names,
        surface_field_names=surface_field_names,
        div_curl_from_uv=div_curl_from_uv,
        u_key=u_key,
        v_key=v_key,
    )
    if input_state_shapes is None:
      core_3d = {'divergence', 'vorticity', 'temperature', u_key, v_key}
      tracer_names = [k for k in volume_field_names if k not in core_3d]
      for k in nodal_tracers:
        if k not in tracer_names:
          tracer_names.append(k)
      surface_names = list(surface_field_names) or ['log_surface_pressure']
      input_state_shapes = primitive_equation_state_shapes(
          ylm_map=ylm_map,
          levels=levels,
          tracer_names=tracer_names,
          nodal_tracers=nodal_tracers,
          surface_field_names=surface_names,
      )
    else:
      for k in nodal_tracers:
        if k in input_state_shapes and not cx.contains_dims(
            input_state_shapes[k], ylm_map.lon_lat_grid
        ):
          raise ValueError(
              f'Nodal tracer {k!r} in `input_state_shapes` has coordinate '
              f'{input_state_shapes[k].coordinate}, expected nodal grid '
              f'{ylm_map.lon_lat_grid}.'
          )
    if features_transform is not None:
      input_shapes = features_transform.output_shapes(input_state_shapes)
    else:
      input_shapes = input_state_shapes
    nodal_tendency_transform = transform_factory(
        input_shapes=input_shapes,
        target_split_axes=target_split_axes,
        rngs=rngs,
    )
    return cls(
        ylm_map=ylm_map,
        nodal_tendency_transform=nodal_tendency_transform,
        modal_filter=modal_filter,
        features_transform=features_transform,
        tendency_transform=tendency_transform,
        div_curl_from_uv=div_curl_from_uv,
        u_key=u_key,
        v_key=v_key,
        mesh=mesh,
    )
