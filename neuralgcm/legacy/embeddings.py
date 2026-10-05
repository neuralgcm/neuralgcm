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
"""Modules that predict an embedding from the model state."""
from typing import Any, Optional
from dinosaur import coordinate_systems
from dinosaur import pytree_utils
from dinosaur import scales
from dinosaur import typing
from dinosaur import xarray_utils
import gin
import haiku as hk
import jax
import jax.numpy as jnp
from neuralgcm.legacy import features
from neuralgcm.legacy import mappings
from neuralgcm.legacy import transforms

EmbeddingFn = typing.EmbeddingFn
EmbeddingModule = typing.EmbeddingModule
Forcing = typing.Forcing
TransformModule = typing.TransformModule

units = scales.units


@gin.register
class ModalToNodalEmbedding(hk.Module):
  """Embedding that expects modal state input and returns nodal output."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      output_shapes: typing.Pytree,
      modal_to_nodal_features_module: features.FeaturesModule,
      nodal_mapping_module: mappings.MappingModule,
      output_transform_module: TransformModule = transforms.IdentityTransform,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.coords = coords
    self.output_shapes = output_shapes
    self.modal_to_nodal_features_fn = modal_to_nodal_features_module(
        coords, dt, physics_specs, aux_features
    )
    self.nodal_mapping_module = nodal_mapping_module
    self.output_transform_fn = output_transform_module(
        coords, dt, physics_specs, aux_features
    )

  def __call__(
      self,
      state: typing.Pytree,
      memory: Optional[typing.Pytree] = None,
      diagnostics: Optional[typing.Pytree] = None,
      randomness: Optional[typing.Pytree] = None,
      forcing: Optional[typing.Forcing] = None,
  ) -> typing.Pytree:
    """Returns the embedding output on nodal locations."""
    net = self.nodal_mapping_module(self.output_shapes)
    # Need to check if dict when embedding is not within the parameterization
    # (e.g., for diagnostic NN)
    state, _ = pytree_utils.as_dict(state)
    nodal_inputs = self.modal_to_nodal_features_fn(
        state, memory, diagnostics, randomness, forcing
    )
    nodal_outputs = self.output_transform_fn(net(nodal_inputs))
    return nodal_outputs


# TODO(pnorgaard) Refactor default embeddings to separate object
@gin.register
class NodalSurfaceModelEmbedding(hk.Module):
  """Embedding to represent a nodal space surface model."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      output_shapes: typing.Pytree,
      static_vars_ds_path: str,
      land_embedding: Optional[EmbeddingModule] = None,
      sea_embedding: Optional[EmbeddingModule] = None,
      sea_ice_embedding: Optional[EmbeddingModule] = None,
      snow_embedding: Optional[EmbeddingModule] = None,
      output_transform_module: TransformModule = transforms.IdentityTransform,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.coords = coords
    self.output_shapes = output_shapes

    # Basic surface embedding settings
    self.feature_axis = -3
    param_init = hk.initializers.TruncatedNormal()
    output_size = sum([x[self.feature_axis]
                       for x in jax.tree_util.tree_leaves(output_shapes)])
    param_shape = (output_size, 1, 1)  # uniform across lon, lat
    surface_nodal_shape = self.coords.surface_nodal_shape

    if land_embedding is not None:
      self.land_embedding_fn = land_embedding(
          coords=coords,
          dt=dt,
          physics_specs=physics_specs,
          aux_features=aux_features,
          output_shapes=output_shapes,
      )
    else:
      self.land_parameters = hk.get_parameter(
          'land_params', param_shape,
          jnp.float32, init=param_init)
      def land_embedding_fn(state, memory, randomness, forcing):
        del state, memory, randomness, forcing  # unused
        outputs = self.land_parameters * jnp.ones(surface_nodal_shape)
        return pytree_utils.unpack_to_pytree(
            outputs, self.output_shapes, self.feature_axis
        )
      self.land_embedding_fn = land_embedding_fn

    if sea_embedding is not None:
      self.sea_embedding_fn = sea_embedding(
          coords=coords,
          dt=dt,
          physics_specs=physics_specs,
          aux_features=aux_features,
          output_shapes=output_shapes,
      )
    else:
      self.sea_parameters = hk.get_parameter(
          'sea_params', param_shape,
          jnp.float32, init=param_init)
      def sea_embedding_fn(state, memory, randomness, forcing):
        del state, memory, randomness, forcing  # unused
        outputs = self.sea_parameters * jnp.ones(surface_nodal_shape)
        return pytree_utils.unpack_to_pytree(
            outputs, self.output_shapes, self.feature_axis
        )
      self.sea_embedding_fn = sea_embedding_fn

    if sea_ice_embedding is not None:
      self.sea_ice_embedding_fn = sea_ice_embedding(
          coords=coords,
          dt=dt,
          physics_specs=physics_specs,
          aux_features=aux_features,
          output_shapes=output_shapes,
      )
    else:
      self.sea_ice_parameters = hk.get_parameter(
          'sea_ice_params', param_shape,
          jnp.float32, init=param_init)
      def sea_ice_embedding_fn(state, memory, randomness, forcing):
        del state, memory, randomness, forcing  # unused
        outputs = self.sea_ice_parameters * jnp.ones(surface_nodal_shape)
        return pytree_utils.unpack_to_pytree(
            outputs, self.output_shapes, self.feature_axis
        )
      self.sea_ice_embedding_fn = sea_ice_embedding_fn

    if snow_embedding is not None:
      self.snow_embedding_fn = snow_embedding(
          coords=coords,
          dt=dt,
          physics_specs=physics_specs,
          aux_features=aux_features,
          output_shapes=output_shapes,
      )
    else:
      self.snow_parameters = hk.get_parameter(
          'snow_params', param_shape,
          jnp.float32, init=param_init)
      def snow_embedding_fn(state, memory, randomness, forcing):
        del state, memory, randomness, forcing  # unused
        outputs = self.snow_parameters * jnp.ones(surface_nodal_shape)
        return pytree_utils.unpack_to_pytree(
            outputs, self.output_shapes, self.feature_axis
        )
      self.snow_embedding_fn = snow_embedding_fn

    self.output_transform_fn = output_transform_module(
        coords, dt, physics_specs, aux_features
    )

    ds = xarray_utils.ds_from_path_or_aux(static_vars_ds_path, aux_features)
    self.land_sea_mask = xarray_utils.nodal_land_sea_mask_from_ds(ds)

    # snow data is provided as depth (in meters). It is converted to snow_cover
    # by choosing a threshold such that snow_cover = 0 below that value and
    # snow cover = 1 above that value.
    self.snow_cover_threshold = physics_specs.nondimensionalize(1 * units.meter)

  def __call__(
      self,
      state: typing.Pytree,
      memory: Optional[typing.Pytree] = None,
      diagnostics: Optional[typing.Pytree] = None,
      randomness: Optional[typing.Pytree] = None,
      forcing: Optional[typing.Forcing] = None,
  ) -> typing.Pytree:
    """Returns the embedding output on nodal locations."""
    land_outputs = self.land_embedding_fn(
        state, memory, diagnostics, randomness, forcing)  # pyrefly: ignore[bad-argument-count]
    sea_outputs = self.sea_embedding_fn(
        state, memory, diagnostics, randomness, forcing)  # pyrefly: ignore[bad-argument-count]
    sea_ice_outputs = self.sea_ice_embedding_fn(
        state, memory, diagnostics, randomness, forcing  # pyrefly: ignore[bad-argument-count]
    )
    snow_outputs = self.snow_embedding_fn(
        state, memory, diagnostics, randomness, forcing)  # pyrefly: ignore[bad-argument-count]

    # prepare masks with fractional values in [0, 1]
    land_fraction = self.land_sea_mask
    sea_fraction = 1 - land_fraction
    sea_ice_fraction = forcing[xarray_utils.SEA_ICE_COVER]  # pyrefly: ignore[unsupported-operation]
    snow_fraction = forcing[xarray_utils.SNOW_DEPTH] > self.snow_cover_threshold  # pyrefly: ignore[unsupported-operation]

    # weight and combine outputs
    snow_weight = snow_fraction * land_fraction  # snow covered land
    land_weight = (1 - snow_fraction) * land_fraction  # land without snow
    sea_ice_weight = sea_ice_fraction * sea_fraction  # ice covered sea
    sea_weight = (1 - sea_ice_fraction) * sea_fraction  # sea without ice

    def tree_scale(a, x):
      # Multiply leaves of `x` by `a`.
      return jax.tree_util.tree_map(lambda y: a * y, x)

    surface_outputs = jax.tree_util.tree_map(
        lambda a, b, c, d: a + b + c + d,
        tree_scale(land_weight, land_outputs),
        tree_scale(sea_weight, sea_outputs),
        tree_scale(sea_ice_weight, sea_ice_outputs),
        tree_scale(snow_weight, snow_outputs),
    )

    return self.output_transform_fn(surface_outputs)


@gin.register
class NodalLandSeaIceEmbedding(hk.Module):
  """Embedding to represent a nodal land/sea/sea-ice surface."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      output_shapes: typing.Pytree,
      static_vars_ds_path: str,
      land_embedding: Optional[EmbeddingModule] = None,
      sea_embedding: Optional[EmbeddingModule] = None,
      sea_ice_embedding: Optional[EmbeddingModule] = None,
      output_transform_module: TransformModule = transforms.IdentityTransform,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.coords = coords
    self.output_shapes = output_shapes

    # Basic surface embedding settings
    self.feature_axis = -3
    surface_nodal_shape = self.coords.surface_nodal_shape
    param_init = hk.initializers.TruncatedNormal()
    output_size = sum([x[self.feature_axis]
                       for x in jax.tree_util.tree_leaves(output_shapes)])
    uniform_param_shape = (output_size, 1, 1)  # uniform across lon, lat
    # Alternative for lon,lat dependent parameters, e.g. for land model
    # spatial_params_shape = (output_size, surface_nodal_shape[-2:])

    def get_parameters_fn(
        shape: tuple[int, int, int],
        name: str = ''):
      parameters = hk.get_parameter(
          name + '_params', shape, jnp.float32, init=param_init
      )
      def parameters_fn(state, memory, diagnostics, randomness, forcing):
        del state, memory, diagnostics, randomness, forcing  # unused
        outputs = parameters * jnp.ones(surface_nodal_shape)
        return pytree_utils.unpack_to_pytree(
            outputs, output_shapes, self.feature_axis,
        )
      return parameters_fn

    if land_embedding is not None:
      self.land_embedding_fn = land_embedding(
          coords=coords,
          dt=dt,
          physics_specs=physics_specs,
          aux_features=aux_features,
          output_shapes=output_shapes,
      )
    else:
      self.land_embedding_fn = get_parameters_fn(uniform_param_shape, 'land')

    if sea_embedding is not None:
      self.sea_embedding_fn = sea_embedding(
          coords=coords,
          dt=dt,
          physics_specs=physics_specs,
          aux_features=aux_features,
          output_shapes=output_shapes,
      )
    else:
      self.sea_embedding_fn = get_parameters_fn(uniform_param_shape, 'sea')

    if sea_ice_embedding is not None:
      self.sea_ice_embedding_fn = sea_ice_embedding(
          coords=coords,
          dt=dt,
          physics_specs=physics_specs,
          aux_features=aux_features,
          output_shapes=output_shapes,
      )
    else:
      self.sea_ice_embedding_fn = get_parameters_fn(
          uniform_param_shape, 'sea_ice'
      )

    self.output_transform_fn = output_transform_module(
        coords, dt, physics_specs, aux_features
    )
    ds = xarray_utils.ds_from_path_or_aux(static_vars_ds_path, aux_features)
    self.land_sea_mask = xarray_utils.nodal_land_sea_mask_from_ds(ds)

  def __call__(
      self,
      state: typing.Pytree,
      memory: Optional[typing.Pytree] = None,
      diagnostics: Optional[typing.Pytree] = None,
      randomness: Optional[typing.Pytree] = None,
      forcing: Optional[typing.Forcing] = None,
  ) -> typing.Pytree:
    """Returns the embedding output on nodal locations."""
    # get outputs from each model
    land_outputs = self.land_embedding_fn(
        state, memory, diagnostics, randomness, forcing)
    sea_outputs = self.sea_embedding_fn(
        state, memory, diagnostics, randomness, forcing)
    sea_ice_outputs = self.sea_ice_embedding_fn(
        state, memory, diagnostics, randomness, forcing
    )

    # prepare masks with fractional values in [0, 1]
    land_fraction = self.land_sea_mask
    sea_fraction = 1 - land_fraction
    sea_ice_fraction = forcing[xarray_utils.SEA_ICE_COVER]  # pyrefly: ignore[unsupported-operation]

    # weight and combine outputs
    land_weight = land_fraction
    sea_ice_weight = sea_ice_fraction * sea_fraction  # ice covered sea
    sea_weight = (1 - sea_ice_fraction) * sea_fraction  # sea without ice

    def tree_scale(a, x):
      # Multiply leaves of `x` by `a`.
      return jax.tree_util.tree_map(lambda y: a * y, x)

    surface_outputs = jax.tree_util.tree_map(
        lambda a, b, c: a + b + c,
        tree_scale(land_weight, land_outputs),
        tree_scale(sea_weight, sea_outputs),
        tree_scale(sea_ice_weight, sea_ice_outputs),
    )

    return self.output_transform_fn(surface_outputs)
