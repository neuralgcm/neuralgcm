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
"""Modules that computes relevant state features to be used by ML components."""

from typing import Any, Callable, Mapping, Optional, Protocol, Sequence
from dinosaur import coordinate_systems
from dinosaur import primitive_equations
from dinosaur import pytree_utils
from dinosaur import radiation
from dinosaur import scales
from dinosaur import spherical_harmonic
from dinosaur import typing
from dinosaur import xarray_utils
import gin
import haiku as hk
import jax
import jax.numpy as jnp
from neuralgcm.legacy import transforms
import numpy as np


Array = typing.Array
Pytree = typing.Pytree
TransformModule = typing.TransformModule
KeyWithCosLatFactor = typing.KeyWithCosLatFactor


class FeaturesFn(Protocol):

  def __call__(
      self,
      inputs: typing.Pytree,
      memory: Optional[typing.PyTreeState] = None,
      diagnostics: Optional[typing.Pytree] = None,
      randomness: Optional[typing.PyTreeState] = None,
      forcing: Optional[typing.Pytree] = None,
  ) -> typing.Pytree:
    ...


FeaturesModule = Callable[..., FeaturesFn]


@gin.register(denylist=['coords', 'dt', 'physics_specs', 'aux_features'])
class PrimitiveEquationsDiagnosticState(hk.Module):
  """Features modules that returns processed DiagnosticState for PE."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      features_transform_module: TransformModule = transforms.IdentityTransform,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.features_transform_fn = features_transform_module(
        coords, dt, physics_specs, aux_features
    )
    self.coords = coords

  def __call__(
      self,
      inputs: typing.Pytree,
      memory: Optional[typing.PyTreeState] = None,
      diagnostics: Optional[typing.Pytree] = None,
      randomness: Optional[typing.PyTreeState] = None,
      forcing: Optional[typing.Pytree] = None,
  ) -> primitive_equations.DiagnosticState:
    del memory, diagnostics, randomness, forcing  # unused
    if not isinstance(inputs, primitive_equations.State):
      inputs = primitive_equations.State(**inputs)
    d_state = primitive_equations.compute_diagnostic_state(inputs, self.coords)
    return self.features_transform_fn(d_state.asdict())  # pyrefly: ignore[missing-attribute]


@gin.register(denylist=['coords', 'dt', 'physics_specs', 'aux_features'])
class VelocityAndPrognostics(hk.Module):
  """Features module that returns prognostics + u,v and optionally gradients."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      fields_to_include: Optional[Sequence[str]] = None,
      features_transform_module: TransformModule = transforms.IdentityTransform,
      compute_gradients_module: TransformModule = transforms.EmptyTransform,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.features_transform_fn = features_transform_module(
        coords, dt, physics_specs, aux_features
    )
    self.coords = coords
    self.fields_to_include = fields_to_include
    self.compute_gradients_fn = compute_gradients_module(
        coords, dt, physics_specs, aux_features
    )

  def _extract_features(
      self,
      inputs: typing.Pytree,
      prefix: str = '',
  ) -> typing.Pytree:
    """Returns a nodal velocity and prognostic features."""
    # Note: all intermediate features have an explicit cos-lat factors in key.
    # These factors are removed in the `__call__` method before returning.

    # compute `u, v` if div/curl is available and `u, v` not in prognosics.
    if set(['vorticity', 'divergence']).issubset(inputs.keys()) and not set(
        ['u', 'v']
    ).intersection(inputs.keys()):
      cos_lat_u, cos_lat_v = spherical_harmonic.get_cos_lat_vector(
          inputs['vorticity'], inputs['divergence'], self.coords.horizontal
      )
      modal_features = {
          KeyWithCosLatFactor(prefix + 'u', 1): cos_lat_u,
          KeyWithCosLatFactor(prefix + 'v', 1): cos_lat_v,
      }
    else:
      modal_features = {}
    prognostics_keys = list(inputs.keys())
    prognostics_keys.remove('tracers')
    prognostics_keys.remove('sim_time')
    for k in prognostics_keys:
      if self.fields_to_include is None or k in self.fields_to_include:
        modal_features[KeyWithCosLatFactor(prefix + k, 0)] = inputs[k]

    for k, v in inputs['tracers'].items():
      if self.fields_to_include is None or k in self.fields_to_include:
        modal_features[KeyWithCosLatFactor(prefix + k, 0)] = v
    # Computing gradient features and adjusting cos_lat factors.
    modal_features = self.coords.with_dycore_sharding(modal_features)
    diff_operator_features = self.compute_gradients_fn(modal_features)
    sec_lat = 1 / self.coords.horizontal.cos_lat
    sec2_lat = self.coords.horizontal.sec2_lat
    sec_lat_scales = {0: 1, 1: sec_lat, 2: sec2_lat}
    # Computing all features in nodal space.
    features = {}
    for k, v in (diff_operator_features | modal_features).items():
      sec_lat_scale = sec_lat_scales[k.factor_order]
      features[k.name] = self.coords.horizontal.to_nodal(v) * sec_lat_scale
    features = self.coords.with_dycore_sharding(features)
    return features

  def __call__(
      self,
      inputs: typing.Pytree,
      memory: Optional[typing.PyTreeState] = None,
      diagnostics: Optional[typing.Pytree] = None,
      randomness: Optional[typing.PyTreeState] = None,
      forcing: Optional[typing.Pytree] = None,
  ) -> typing.Pytree:
    del memory, diagnostics, randomness, forcing  # unused.
    nodal_features = self._extract_features(inputs)
    return self.features_transform_fn(nodal_features)


@gin.register(denylist=['coords', 'dt', 'physics_specs', 'aux_features'])
class MemoryVelocityAndValues(VelocityAndPrognostics):
  """Similar to `VelocityAndPrognostics`, but operates on memory."""

  def __call__(
      self,
      inputs: typing.Pytree,
      memory: Optional[typing.PyTreeState] = None,
      diagnostics: Optional[typing.Pytree] = None,
      randomness: Optional[typing.PyTreeState] = None,
      forcing: Optional[typing.Pytree] = None,
  ) -> typing.Pytree:
    del inputs, diagnostics, randomness, forcing  # unused.
    nodal_features = self._extract_features(memory, 'memory_')
    return self.features_transform_fn(nodal_features)


@gin.register(denylist=['coords', 'dt', 'physics_specs', 'aux_features'])
class NodalInputVelocityAndPrognostics(VelocityAndPrognostics):
  """Features modules that returns velocities, temperature, and optionally gradients."""

  def __call__(
      self,
      inputs: typing.Pytree,
      memory: Optional[typing.PyTreeState] = None,
      diagnostics: Optional[typing.Pytree] = None,
      randomness: Optional[typing.PyTreeState] = None,
      forcing: Optional[typing.Pytree] = None,
  ) -> typing.Pytree:
    to_modal_fn = self.coords.horizontal.to_modal
    inputs = to_modal_fn(inputs)
    memory = to_modal_fn(memory)
    return super().__call__(inputs, memory, randomness, forcing)


@gin.register(denylist=['coords', 'dt', 'physics_specs', 'aux_features'])
class RadiationFeatures(hk.Module):
  """Feature module that computes incident radiation flux."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      features_transform_module: TransformModule = transforms.IdentityTransform,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.features_transform_fn = features_transform_module(
        coords, dt, physics_specs, aux_features
    )
    ref_datetime_str = str(aux_features[xarray_utils.REFERENCE_DATETIME_KEY])
    self.solar_radiation = radiation.SolarRadiation.normalized(
        coords=coords,
        physics_specs=physics_specs,
        reference_datetime=np.datetime64(ref_datetime_str),
    )

  def __call__(
      self,
      inputs: typing.Pytree,
      memory: Optional[typing.PyTreeState] = None,
      diagnostics: Optional[typing.Pytree] = None,
      randomness: Optional[typing.PyTreeState] = None,
      forcing: Optional[typing.Pytree] = None,
  ) -> typing.Pytree:
    del memory, diagnostics, randomness, forcing  # unused.
    features = {}
    features['radiation'] = self.solar_radiation.radiation_flux(
        inputs['sim_time']
    )
    # TODO(janniyuval) add a flag that allow to get radiation of next time step
    # insert a feature axis.
    features = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), features)
    return self.features_transform_fn(features)


@gin.register(denylist=['coords', 'dt', 'physics_specs', 'aux_features'])
class OrbitalTimeFeatures(hk.Module):
  """Feature module that computes orbital time features."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      features_transform_module: TransformModule = transforms.IdentityTransform,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.features_transform_fn = features_transform_module(
        coords, dt, physics_specs, aux_features
    )
    ref_datetime_str = str(aux_features[xarray_utils.REFERENCE_DATETIME_KEY])
    self.solar_radiation = radiation.SolarRadiation.normalized(
        coords=coords,
        physics_specs=physics_specs,
        reference_datetime=np.datetime64(ref_datetime_str),
    )

  def __call__(
      self,
      inputs: typing.Pytree,
      memory: Optional[typing.PyTreeState] = None,
      diagnostics: Optional[typing.Pytree] = None,
      randomness: Optional[typing.PyTreeState] = None,
      forcing: Optional[typing.Pytree] = None,
  ) -> typing.Pytree:
    del memory, diagnostics, randomness, forcing  # unused.
    features = {}
    # Cosine and sine of Earth's orbital phase around the Sun
    orbital_time = self.solar_radiation.time_to_orbital_time(inputs['sim_time'])
    # Convert from orbital_phase=0 on January 1st UTC to orbital_phase=0 at the
    # approximate perihelion (when earth is closest to the sun).
    orbital_phase = orbital_time.orbital_phase - radiation.PERIHELION
    # All longitude, latitude locations share the same orbital phase
    ones = jnp.ones(self.solar_radiation.coords.surface_nodal_shape)
    features['cos_orbital_phase'] = jnp.cos(orbital_phase) * ones
    features['sin_orbital_phase'] = jnp.sin(orbital_phase) * ones
    # Cosine and sine of local hour angle (angle from solar noon)
    solar_hour_angle = self.solar_radiation.solar_hour_angle(inputs['sim_time'])
    solar_hour_angle = jnp.expand_dims(solar_hour_angle, 0)
    features['cos_solar_hour'] = jnp.cos(solar_hour_angle)
    features['sin_solar_hour'] = jnp.sin(solar_hour_angle)
    # TODO(janniyuval) add a flag that allow to get radiation of next time step
    return self.features_transform_fn(features)


@gin.register(denylist=['coords', 'dt', 'physics_specs', 'aux_features'])
class ForcingFeatures(hk.Module):
  """Feature module that provides forcing values as features."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      forcing_to_include: Sequence[str] = tuple(),
      features_transform_module: TransformModule = transforms.IdentityTransform,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.forcing_to_include = forcing_to_include
    self.features_transform_fn = features_transform_module(
        coords, dt, physics_specs, aux_features
    )

  def __call__(
      self,
      inputs: Pytree,
      memory: Optional[typing.PyTreeState] = None,
      diagnostics: Optional[typing.Pytree] = None,
      randomness: Optional[typing.PyTreeState] = None,
      forcing: Optional[typing.Forcing] = None,
  ) -> Pytree:
    del inputs, memory, diagnostics, randomness
    features = {}
    for key in self.forcing_to_include:
      value = forcing[key]  # pyrefly: ignore[unsupported-operation]
      # Expect singleton "level" dimension for surface forcings
      if value.ndim > 3:
        raise ValueError(
            f'Expected forcing "{key}" to have ndim <= 3, got {value.ndim}'
        )
      if value.ndim == 2:
        value = jnp.expand_dims(value, axis=0)
      if value.shape[0] != 1:
        raise ValueError(
            f'Expected forcing "{key}" to have leading dimension 1'
            f'for level, got {value.shape}'
        )
      features[key] = value

    return self.features_transform_fn(features)


@gin.register(denylist=['coords', 'dt', 'physics_specs', 'aux_features'])
class LatitudeFeatures(hk.Module):
  """Feature module that creates cos and sin of latitude as features."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      features_transform_module: TransformModule = transforms.IdentityTransform,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.features_transform_fn = features_transform_module(
        coords, dt, physics_specs, aux_features
    )
    self.coords = coords

  def __call__(
      self,
      inputs: typing.Pytree,
      memory: Optional[typing.PyTreeState] = None,
      diagnostics: Optional[typing.Pytree] = None,
      randomness: Optional[typing.PyTreeState] = None,
      forcing: Optional[typing.Pytree] = None,
  ) -> typing.Pytree:
    del inputs, memory, diagnostics, randomness, forcing  # unused.
    _, sin_lat = self.coords.horizontal.nodal_mesh
    sin_features = sin_lat[np.newaxis, ...]
    cos_features = jnp.cos(jnp.arcsin(sin_features))
    features = {
        'cos_latitude': cos_features,
        'sin_latitude': sin_features,
    }
    return self.features_transform_fn(features)


@gin.register(denylist=['coords', 'dt', 'physics_specs', 'aux_features'])
class RandomnessFeatures(hk.Module):
  """Feature module that returns fields from `randomness` as features."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      features_transform_module: TransformModule = transforms.IdentityTransform,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.features_transform_fn = features_transform_module(
        coords, dt, physics_specs, aux_features
    )

  def __call__(
      self,
      inputs: typing.Pytree,
      memory: Optional[typing.PyTreeState] = None,
      diagnostics: Optional[typing.Pytree] = None,
      randomness: Optional[typing.PyTreeState] = None,
      forcing: Optional[typing.Pytree] = None,
  ) -> typing.Pytree:
    del inputs, memory, diagnostics, forcing  # unused.
    if randomness is None:
      random_features = {}
    elif isinstance(randomness, dict):
      random_features, _ = pytree_utils.flatten_dict(randomness)
    elif isinstance(randomness, jax.Array):
      random_features = {'randomness': randomness}
    else:
      raise ValueError(f'randomness has unsupported {type(randomness)=}.')
    # random fields are 2D by construction, adding a feature/level dimension.
    if randomness is not None:
      ndims = set(x.ndim for x in jax.tree_util.tree_leaves(random_features))
      if not ndims.issubset({2, 3}):
        raise ValueError(
            f'Random fields expected to be 2D and/or 3D. Found {ndims=}'
        )

      def make_3d(x):
        if x.ndim == 3:
          return x
        if x.ndim == 2:
          return x[np.newaxis, ...]

      random_features = jax.tree_util.tree_map(make_3d, random_features)
    return self.features_transform_fn(random_features)


@gin.register(denylist=['coords', 'dt', 'physics_specs', 'aux_features'])
class OrographyFeatures(hk.Module):
  """Feature module that computes orographic features."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      features_transform_module: TransformModule = transforms.IdentityTransform,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    if xarray_utils.OROGRAPHY not in aux_features:
      raise ValueError('OrographyFeatures requires orography in aux_features.')
    self.nodal_orography = aux_features[xarray_utils.OROGRAPHY]
    self.features_transform_fn = features_transform_module(
        coords, dt, physics_specs, aux_features
    )

  def __call__(
      self,
      inputs: typing.Pytree,
      memory: Optional[typing.PyTreeState] = None,
      diagnostics: Optional[typing.Pytree] = None,
      randomness: Optional[typing.PyTreeState] = None,
      forcing: Optional[typing.Pytree] = None,
  ) -> typing.Pytree:
    del inputs, memory, diagnostics, randomness, forcing  # unused.
    features = {
        xarray_utils.OROGRAPHY: jnp.expand_dims(self.nodal_orography, 0),
    }
    return self.features_transform_fn(features)


@gin.register(denylist=['coords', 'dt', 'physics_specs', 'aux_features'])
class OneHotAuxFeatures(hk.Module):
  """Feature module that produces one-hot encodings from binary covariates."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      covariate_keys: Sequence[str] = gin.REQUIRED,  # pyrefly: ignore[bad-function-definition]
      convert_float_to_int: bool = False,
      name: Optional[str] = None,
  ):
    del coords, dt, physics_specs  # unused.
    super().__init__(name=name)
    covariates = {}
    num_classes = {}
    for key in covariate_keys:
      if key not in aux_features:
        raise ValueError(f'Covariate {key} not found in aux_features.')
      if not np.issubdtype(aux_features[key].dtype, np.integer):
        if convert_float_to_int:
          aux_features[key] = np.round(aux_features[key]).astype(int)
        else:
          raise ValueError(
              f'Covariate {key} is expected to be integer dtype, '
              f'but is: {aux_features[key].dtype}'
          )
      covariates[key] = aux_features[key]
      num_classes[key] = np.unique(aux_features[key]).size
    self.covariates = covariates
    self.num_classes = num_classes

  def __call__(
      self,
      inputs: typing.Pytree,
      memory: Optional[typing.PyTreeState] = None,
      diagnostics: Optional[typing.Pytree] = None,
      randomness: Optional[typing.PyTreeState] = None,
      forcing: Optional[typing.Pytree] = None,
  ) -> dict[str, jnp.ndarray]:
    del inputs, memory, diagnostics, randomness, forcing  # unused.
    features = {
        k: jax.nn.one_hot(v, self.num_classes[k], axis=0)
        for k, v in self.covariates.items()
    }
    return features


@gin.register(denylist=['coords', 'dt', 'physics_specs', 'aux_features'])
class LearnedPositionalFeatures(hk.Module):
  """Feature module with learned params at surface nodal locations."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      latent_size: int,
      scale: float = 1.0,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.scale = scale
    self.padding = coords.horizontal.nodal_padding
    unpadded_nodal_shape = tuple(
        x - y for x, y in zip(coords.horizontal.nodal_shape, self.padding)
    )
    self.positional_features = hk.get_parameter(
        'learned_positional_features',
        (latent_size,) + unpadded_nodal_shape,
        jnp.float32,
        init=hk.initializers.Constant(0.0),
    )

  def __call__(
      self,
      inputs: typing.Pytree,
      memory: Optional[typing.PyTreeState] = None,
      diagnostics: Optional[typing.Pytree] = None,
      randomness: Optional[typing.PyTreeState] = None,
      forcing: Optional[typing.Pytree] = None,
  ) -> dict[str, jnp.ndarray]:
    """Returns scaled parameter values at surface nodal locations."""
    del inputs, memory, diagnostics, randomness, forcing  # unused.
    pad_x, pad_y = self.padding
    positional_features = self.scale * jnp.pad(
        self.positional_features, [(0, 0), (0, pad_x), (0, pad_y)]
    )
    return {'learned_positional_features': positional_features}


@gin.register(denylist=['coords', 'dt', 'physics_specs', 'aux_features'])
class EmbeddingSurfaceFeatures(hk.Module):
  """Feature module that specifies embedding surface outputs as features.

  Returns {feature_name: nn_output}
    where nn_output.shape = (output_size, lon, lat).
  """

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      feature_name: str,
      output_size: int,
      embedding_module: typing.EmbeddingModule,
      features_transform_module: TransformModule = transforms.IdentityTransform,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    # output shapes are arrays to be pytree leaves for tree_map
    output_shapes = {
        feature_name: np.asarray((output_size,) + coords.horizontal.nodal_shape)
    }
    self.embedding_fn = embedding_module(
        coords, dt, physics_specs, aux_features, output_shapes=output_shapes
    )
    self.features_transform_fn = features_transform_module(
        coords, dt, physics_specs, aux_features
    )

  def __call__(
      self,
      inputs: typing.Pytree,
      memory: Optional[typing.PyTreeState] = None,
      diagnostics: Optional[typing.Pytree] = None,
      randomness: Optional[typing.PyTreeState] = None,
      forcing: Optional[typing.Pytree] = None,
  ) -> typing.Pytree:
    features = self.embedding_fn(
        inputs, memory, diagnostics, randomness, forcing  # pyrefly: ignore[bad-argument-type]
    )
    return self.features_transform_fn(features)


@gin.register(denylist=['coords', 'dt', 'physics_specs', 'aux_features'])
class EmbeddingVolumeFeatures(hk.Module):
  """Feature module that specifies embedding volume outputs as features.

  Returns {feature_name_0: nn_output_0,
           feature_name_1: nn_output_1,
           ...
          }
    where the NN output array has shape (output_size, level, lon, lat), which is
    unpacked over output_size such that nn_output_{i}.shape = (level, lon, lat)
    for each i in range(output_size).
  """

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      feature_name: str,
      output_size: int,
      embedding_module: typing.EmbeddingModule,
      features_transform_module: TransformModule = transforms.IdentityTransform,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    # output shapes are arrays to be pytree leaves for tree_map
    output_shapes = {
        f'{feature_name}_{i}': np.asarray(coords.nodal_shape)
        for i in range(output_size)
    }
    self.embedding_fn = embedding_module(
        coords, dt, physics_specs, aux_features, output_shapes=output_shapes
    )
    self.features_transform_fn = features_transform_module(
        coords, dt, physics_specs, aux_features
    )

  def __call__(
      self,
      inputs: typing.Pytree,
      memory: Optional[typing.PyTreeState] = None,
      diagnostics: Optional[typing.Pytree] = None,
      randomness: Optional[typing.PyTreeState] = None,
      forcing: Optional[typing.Pytree] = None,
  ) -> typing.Pytree:
    features = self.embedding_fn(
        inputs, memory, diagnostics, randomness, forcing  # pyrefly: ignore[bad-argument-type]
    )
    return self.features_transform_fn(features)


@gin.register(denylist=['coords', 'dt', 'physics_specs', 'aux_features'])
class FloatDataFeatures(hk.Module):
  """Feature module that supplies floating point covariates from data."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      covariate_data_path: str = gin.REQUIRED,  # pyrefly: ignore[bad-function-definition]
      covariate_keys: Sequence[str] = gin.REQUIRED,  # pyrefly: ignore[bad-function-definition]
      renaming_dict: Optional[Mapping[str, str]] = None,
      compute_gradients_module: TransformModule = transforms.EmptyTransform,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.covariates = {}
    self.compute_gradients_fn = compute_gradients_module(
        coords, dt, physics_specs, aux_features
    )
    self.coords = coords
    ds = xarray_utils.ds_from_path_or_aux(covariate_data_path, aux_features)
    if renaming_dict is not None:
      ds = ds.rename(renaming_dict)
    lon, lat = (ds[xarray_utils.XR_LON_NAME], ds[xarray_utils.XR_LAT_NAME])
    xarray_utils.verify_grid_consistency(lon, lat, coords.horizontal)
    lon_lat_order = (xarray_utils.XR_LON_NAME, xarray_utils.XR_LAT_NAME)
    for key in covariate_keys:
      data = ds[key].transpose(*lon_lat_order)
      data_units = scales.parse_units(data.attrs['units'])
      data = physics_specs.nondimensionalize(data.values * data_units)
      if data.ndim != 3:
        data = data[np.newaxis, ...]
      self.covariates[key] = data

  def __call__(
      self,
      inputs: typing.Pytree,
      memory: Optional[typing.PyTreeState] = None,
      diagnostics: Optional[typing.Pytree] = None,
      randomness: Optional[typing.PyTreeState] = None,
      forcing: Optional[typing.Pytree] = None,
  ) -> dict[str, jnp.ndarray]:
    del inputs, memory, diagnostics, forcing, randomness  # unused.
    features = {k: v for k, v in self.covariates.items()}
    modal_features = self.coords.horizontal.to_modal(features)
    modal_features = {  # jit should eliminate to_modal if it is not used.
        KeyWithCosLatFactor(k, 0): v for k, v in modal_features.items()
    }
    modal_gradient_features = self.compute_gradients_fn(modal_features)
    sec_lat = 1 / self.coords.horizontal.cos_lat
    sec2_lat = self.coords.horizontal.sec2_lat
    sec_lat_scales = {0: 1, 1: sec_lat, 2: sec2_lat}
    for k, v in modal_gradient_features.items():
      sec_lat_scale = sec_lat_scales[k.factor_order]
      features[k.name] = self.coords.horizontal.to_nodal(v) * sec_lat_scale
    return features


@gin.register(denylist=['coords', 'dt', 'physics_specs', 'aux_features'])
class CombinedFeatures(hk.Module):
  """Feature module that combines multiple feature modules together."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      feature_modules: Sequence[FeaturesModule] = gin.REQUIRED,  # pyrefly: ignore[bad-function-definition]
      feature_module_names_to_exclude: Sequence[str] = tuple(),
      features_to_exclude: Sequence[str] = tuple(),
      features_transform_module: TransformModule = transforms.IdentityTransform,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.feature_fns = [
        module(coords, dt, physics_specs, aux_features)
        for module in feature_modules
    ]
    self.feature_module_names_to_exclude = feature_module_names_to_exclude
    self.features_to_exclude = features_to_exclude
    self.features_transform_fn = features_transform_module(
        coords, dt, physics_specs, aux_features
    )

  def __call__(
      self,
      inputs: typing.Pytree,
      memory: Optional[typing.PyTreeState] = None,
      diagnostics: Optional[typing.Pytree] = None,
      randomness: Optional[typing.PyTreeState] = None,
      forcing: Optional[typing.Forcing] = None,
  ) -> dict[str, jnp.ndarray]:
    all_features = {}
    for feature_fn in self.feature_fns:
      if type(feature_fn).__name__ not in self.feature_module_names_to_exclude:
        features = feature_fn(inputs, memory, diagnostics, randomness, forcing)
        for k, v in features.items():
          if k in all_features:
            raise ValueError(f'Encountered duplicate feature {k}')
          all_features[k] = v
    all_features = self.features_transform_fn(all_features)
    for k in self.features_to_exclude:
      all_features.pop(k, None)
    return all_features


@gin.register(denylist=['coords', 'dt', 'physics_specs', 'aux_features'])
class NullFeatures(hk.Module):
  """Placeholder features module that returns an empty dict."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      name: Optional[str] = None,
  ):
    del coords, dt, physics_specs, aux_features  # unused
    super().__init__(name=name)

  def __call__(
      self,
      inputs: typing.Pytree,
      memory: Optional[typing.PyTreeState] = None,
      diagnostics: Optional[typing.Pytree] = None,
      randomness: Optional[typing.PyTreeState] = None,
      forcing: Optional[typing.Pytree] = None,
  ) -> dict[str, jnp.ndarray]:
    del inputs, memory, diagnostics, randomness, forcing  # unused
    return {}


@gin.register(denylist=['coords', 'dt', 'physics_specs', 'aux_features'])
class PressureFeatures(hk.Module):
  """Feature module that computes pressure."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      features_transform_module: TransformModule = transforms.IdentityTransform,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.coords = coords
    self.features_transform_fn = features_transform_module(
        coords, dt, physics_specs, aux_features
    )

  def _nodal_pressure(
      self,
      inputs: typing.Pytree,
      prefix: str = '',
  ) -> Mapping[str, Array]:
    """Computes nodal pressure from model inputs."""
    # Compute nodal, dimensionalized quantities
    to_nodal_fn = self.coords.horizontal.to_nodal
    sigma = self.coords.vertical.centers
    surface_pressure = jnp.exp(to_nodal_fn(inputs['log_surface_pressure']))
    pressure = surface_pressure * sigma[:, jnp.newaxis, jnp.newaxis]
    nodal_features = {prefix + 'pressure': pressure}
    return nodal_features

  def __call__(
      self,
      inputs: typing.Pytree,
      memory: Optional[typing.PyTreeState] = None,
      diagnostics: Optional[typing.Pytree] = None,
      randomness: Optional[typing.PyTreeState] = None,
      forcing: Optional[typing.Pytree] = None,
  ) -> typing.Pytree:
    del memory, diagnostics, randomness, forcing  # unused.
    nodal_features = self._nodal_pressure(inputs)
    return self.features_transform_fn(nodal_features)


@gin.register(denylist=['coords', 'dt', 'physics_specs', 'aux_features'])
class MemoryPressureFeatures(PressureFeatures):
  """Feature module that computes pressure from memory values."""

  def __call__(
      self,
      inputs: typing.Pytree,
      memory: Optional[typing.PyTreeState] = None,
      diagnostics: Optional[typing.Pytree] = None,
      randomness: Optional[typing.PyTreeState] = None,
      forcing: Optional[typing.Pytree] = None,
  ) -> typing.Pytree:
    del inputs, diagnostics, randomness, forcing  # unused.
    nodal_features = self._nodal_pressure(memory, 'memory_')
    return self.features_transform_fn(nodal_features)
