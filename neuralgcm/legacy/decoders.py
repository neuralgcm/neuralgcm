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
"""Defines `decoder` modules that map model state to output data format."""

import functools
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar
import zlib

from dinosaur import coordinate_systems
from dinosaur import primitive_equations
from dinosaur import pytree_utils
from dinosaur import scales
from dinosaur import spherical_harmonic
from dinosaur import typing
from dinosaur import vertical_interpolation
from dinosaur import weatherbench_utils
from dinosaur import xarray_utils
import gin
import haiku as hk
import jax
import jax.numpy as jnp
from neuralgcm.legacy import diagnostics
from neuralgcm.legacy import features
from neuralgcm.legacy import filters
from neuralgcm.legacy import mappings
from neuralgcm.legacy import orographies
from neuralgcm.legacy import perturbations
from neuralgcm.legacy import stochastic
from neuralgcm.legacy import transforms
import numpy as np


# long lines are better than splitting argument definitions onto two lines
# pylint: disable=line-too-long

# We ♥ λ's
# pylint: disable=g-long-lambda

DataState = typing.DataState
DiagnosticModule = diagnostics.DiagnosticModule
FeaturesModule = features.FeaturesModule
FilterModule = Callable[..., typing.PyTreeFilterFn]
Forcing = typing.Forcing
MappingModule = mappings.MappingModule
PyTreeState = typing.PyTreeState
ModelState = typing.ModelState
TransformModule = typing.TransformModule
OrographyModule = orographies.OrographyModule
PerturbationModule = perturbations.PerturbationModule
RandomnessModule = stochastic.RandomnessModule


@gin.register
class DecoderIdentityTransform(hk.Module):
  """Transformation that returns inputs without modification."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      output_coords: coordinate_systems.CoordinateSystem,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    del coords, dt, physics_specs, aux_features, output_coords

  def __call__(self, inputs: PyTreeState) -> PyTreeState:
    return inputs


@gin.register
class DecoderFilterTransform(hk.Module):
  """Transformation that returns truncated and filtered modal inputs."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      output_coords: coordinate_systems.CoordinateSystem,
      filter_module: FilterModule = filters.DataNoFilter,
      return_nodal: bool = True,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.output_coords = output_coords
    self.filter_fn = filter_module(coords, dt, physics_specs, aux_features)
    self.return_nodal = return_nodal

  def __call__(self, inputs: PyTreeState) -> PyTreeState:
    modal_inputs = coordinate_systems.maybe_to_modal(inputs, self.output_coords)
    filtered_inputs = self.filter_fn(modal_inputs)
    if self.return_nodal:
      return self.output_coords.horizontal.to_nodal(filtered_inputs)
    return filtered_inputs


@gin.register
class OutputModalToModalTransform(hk.Module):
  """Transformation that truncates modal state to output coords."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      output_coords: coordinate_systems.CoordinateSystem,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.coords = coords
    self.output_coords = output_coords

  def __call__(self, inputs: PyTreeState) -> PyTreeState:
    downsample_fn = coordinate_systems.get_spectral_downsample_fn(
        self.coords, self.output_coords
    )
    return downsample_fn(inputs)


@gin.register
class OutputModalToNodalTransform(hk.Module):
  """Transformation that converts modal state to nodal representation."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      output_coords: coordinate_systems.CoordinateSystem,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.coords = coords
    self.output_coords = output_coords

  def __call__(self, inputs: PyTreeState) -> PyTreeState:
    to_nodal_fn = self.output_coords.horizontal.to_nodal
    downsample_fn = coordinate_systems.get_spectral_downsample_fn(
        self.coords, self.output_coords
    )
    return jax.tree_util.tree_map(
        lambda x: to_nodal_fn(downsample_fn(x)), inputs
    )


@gin.register
class OutputNodalToModalTransform(hk.Module):
  """Transformation that converts nodal state to modal representation."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      output_coords: coordinate_systems.CoordinateSystem,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.output_coords = output_coords

  def __call__(self, inputs: PyTreeState) -> PyTreeState:
    return self.output_coords.horizontal.to_modal(inputs)


@gin.register
class ModalOutputLearnedAdaptorTransform(hk.Module):
  """Transformation using a tower to adapt modal outputs to the data domain."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      output_coords: coordinate_systems.CoordinateSystem,
      modal_to_nodal_features_module: FeaturesModule,
      nodal_mapping_module: MappingModule,
      output_transform_module: TransformModule,
      name: Optional[str] = None,
  ):
    del output_coords  # unused.
    super().__init__(name=name)
    self.coords = coords
    self.modal_to_nodal_features_fn = modal_to_nodal_features_module(
        coords, dt, physics_specs, aux_features
    )
    self.nodal_mapping_module = nodal_mapping_module
    self.output_transform_fn = output_transform_module(
        coords, dt, physics_specs, aux_features
    )
    self.get_nodal_shape_fn = lambda x: coordinate_systems.get_nodal_shapes(
        x, coords
    )

  def __call__(self, inputs: PyTreeState) -> PyTreeState:
    """Applies transform to modal inputs, returns modal outputs."""
    inputs, from_dict_fn = pytree_utils.as_dict(inputs)
    prediction_shapes = jax.tree_util.tree_map(self.get_nodal_shape_fn, inputs)
    # if `inputs` contain `sim_time` - remove it from corrections.
    sim_time_shape = prediction_shapes.pop('sim_time', None)
    net = self.nodal_mapping_module(prediction_shapes)
    nodal_input_features = self.modal_to_nodal_features_fn(inputs, None)
    nodal_corrections = self.output_transform_fn(net(nodal_input_features))
    corrections = self.coords.horizontal.to_modal(nodal_corrections)
    if sim_time_shape is not None:
      corrections['sim_time'] = 0.0
    outputs = jax.tree_util.tree_map(lambda x, y: x + y, inputs, corrections)
    return from_dict_fn(outputs)


@gin.register
class NodalOutputLearnedAdaptorTransform(hk.Module):
  """Transformation using a tower to adapt nodal outputs to the data domain."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      output_coords: coordinate_systems.CoordinateSystem,
      nodal_to_nodal_features_module: FeaturesModule,
      nodal_mapping_module: MappingModule,
      output_transform_module: TransformModule,
      name: Optional[str] = None,
  ):
    del output_coords  # unused.
    super().__init__(name=name)
    self.coords = coords
    self.nodal_to_nodal_features_fn = nodal_to_nodal_features_module(
        coords, dt, physics_specs, aux_features
    )
    self.nodal_mapping_module = nodal_mapping_module
    self.output_transform_fn = output_transform_module(
        coords, dt, physics_specs, aux_features
    )
    self.get_nodal_shape_fn = lambda x: coordinate_systems.get_nodal_shapes(
        x, coords
    )

  def __call__(self, inputs: PyTreeState) -> PyTreeState:
    """Applies transform to nodal inputs, returns nodal outputs."""
    inputs, from_dict_fn = pytree_utils.as_dict(inputs)
    prediction_shapes = jax.tree_util.tree_map(self.get_nodal_shape_fn, inputs)
    # if `inputs` contain `sim_time` - remove it from corrections.
    sim_time_shape = prediction_shapes.pop('sim_time', None)
    net = self.nodal_mapping_module(prediction_shapes)
    input_features = self.nodal_to_nodal_features_fn(inputs, None)
    corrections = self.output_transform_fn(net(input_features))
    if sim_time_shape is not None:
      corrections['sim_time'] = 0.0
    outputs = jax.tree_util.tree_map(lambda x, y: x + y, inputs, corrections)
    return from_dict_fn(outputs)


@gin.register
class DecoderCombinedTransform(hk.Module):
  """Module that applies multiple transformations sequentially."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      output_coords: coordinate_systems.CoordinateSystem,
      transforms: Tuple[TransformModule, ...],  # pylint: disable=redefined-outer-name
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.transform_fns = [
        module(coords, dt, physics_specs, aux_features, output_coords)
        for module in transforms
    ]

  def __call__(self, inputs: PyTreeState) -> PyTreeState:
    for transform_fn in self.transform_fns:
      inputs = transform_fn(inputs)
    return inputs


@gin.register
class IdentityDecoder(hk.Module):
  """Decoder that returns model state unaltered."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      output_coords: coordinate_systems.CoordinateSystem,
      name: Optional[str] = None,
  ):
    del coords, dt, physics_specs, aux_features, output_coords
    super().__init__(name=name)

  def __call__(self, x: ModelState, forcing: Forcing) -> DataState:
    del forcing
    return x.state


@gin.register
class StateToDictDecoder(hk.Module):
  """Decoder that returns a dict representation of a model state."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      output_coords: coordinate_systems.CoordinateSystem,
      transform_module: TransformModule = DecoderIdentityTransform,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.transform_fn = transform_module(
        coords, dt, physics_specs, aux_features, output_coords
    )

  def __call__(self, x: ModelState, forcing: Forcing) -> DataState:
    del forcing
    state_dict, _ = pytree_utils.as_dict(x.state)
    return self.transform_fn(state_dict)


@gin.register
class LeapfrogSliceDecoder(hk.Module):
  """Decoder that returns one slice out of a leapfrog pair."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      output_coords: coordinate_systems.CoordinateSystem,
      slice_id: int = 0,
      transform_module: TransformModule = DecoderIdentityTransform,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.slice_id = slice_id
    self.transform_fn = transform_module(
        coords, dt, physics_specs, aux_features, output_coords
    )

  def __call__(self, x: ModelState, forcing: Forcing) -> DataState:
    del forcing
    return self.transform_fn(x.state[self.slice_id])


@gin.register
class LeapfrogSliceDictDecoder(hk.Module):
  """Decoder that returns one slice out of a leapfrog pair as dictionary."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      output_coords: coordinate_systems.CoordinateSystem,
      slice_id: int = 0,
      transform_module: TransformModule = DecoderIdentityTransform,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.slice_id = slice_id
    self.transform_fn = transform_module(
        coords, dt, physics_specs, aux_features, output_coords
    )

  def __call__(self, x: ModelState, forcing: Forcing) -> DataState:
    del forcing
    state_dict, _ = pytree_utils.as_dict(x.state[self.slice_id])
    return self.transform_fn(state_dict)


@gin.configurable
class PrimitiveToWeatherbenchDecoder(hk.Module):
  """Decoder that converts `StateWithTime` to  `weatherbench.State`."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      output_coords: coordinate_systems.CoordinateSystem,
      time_axis: int = 0,
      orography_module: OrographyModule = orographies.ClippedOrography,
      transform_module: TransformModule = DecoderIdentityTransform,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    ref_temps = aux_features[xarray_utils.REF_TEMP_KEY]
    self.ref_temps = ref_temps[..., np.newaxis, np.newaxis]
    self.output_coords = output_coords
    self.coords = coords
    self.physics_specs = physics_specs
    self.velocity_fn = functools.partial(
        spherical_harmonic.vor_div_to_uv_nodal,
        output_coords.horizontal,
    )
    modal_orography_init_fn = orography_module(
        coords, dt, physics_specs, aux_features
    )
    orography = modal_orography_init_fn()  # pytype: disable=not-callable  # jax-ndarray
    self.nodal_orography = coords.horizontal.to_nodal(orography)
    self.geopotential_fn = functools.partial(
        primitive_equations.get_geopotential_with_moisture,
        nodal_orography=self.nodal_orography,
        coordinates=coords.vertical,  # pyrefly: ignore[bad-argument-type]
        gravity_acceleration=physics_specs.gravity_acceleration,
        ideal_gas_constant=physics_specs.ideal_gas_constant,
        water_vapor_gas_constant=physics_specs.water_vapor_gas_constant,
    )
    self.transform_fn = transform_module(
        coords, dt, physics_specs, aux_features, output_coords
    )

  def primitive_to_weatherbench(
      self,
      inputs: primitive_equations.StateWithTime,
  ) -> weatherbench_utils.State:
    """Converts pe_state to weatherbench state on pressure levels."""
    # output state is computed on output_coords.
    to_nodal_fn = self.output_coords.horizontal.to_nodal
    u, v = self.velocity_fn(  # returned in nodal space.
        vorticity=inputs.vorticity, divergence=inputs.divergence
    )
    t = self.ref_temps + to_nodal_fn(inputs.temperature_variation)
    tracers = to_nodal_fn(inputs.tracers)
    z = self.geopotential_fn(t, tracers['specific_humidity'])
    surface_pressure = jnp.exp(to_nodal_fn(inputs.log_surface_pressure))
    u, v, t, z, tracers, surface_pressure = (
        self.coords.dycore_to_physics_sharding(
            (u, v, t, z, tracers, surface_pressure)
        )
    )
    interpolate_with_linear_extrap_fn = (
        vertical_interpolation.vectorize_vertical_interpolation(
            vertical_interpolation.linear_interp_with_linear_extrap
        )
    )
    interpolate_with_constant_extrap_fn = (
        vertical_interpolation.vectorize_vertical_interpolation(
            vertical_interpolation.vertical_interpolation
        )
    )
    regrid_with_linear_fn = functools.partial(
        vertical_interpolation.interp_sigma_to_pressure,
        pressure_coords=self.output_coords.vertical,
        sigma_coords=self.coords.vertical,
        surface_pressure=surface_pressure,
        interpolate_fn=interpolate_with_linear_extrap_fn,
    )
    regrid_with_constant_fn = functools.partial(
        vertical_interpolation.interp_sigma_to_pressure,
        pressure_coords=self.output_coords.vertical,
        sigma_coords=self.coords.vertical,
        surface_pressure=surface_pressure,
        interpolate_fn=interpolate_with_constant_extrap_fn,
    )
    # closes regridding options based on http://shortn/_X09ZAU1jsx.
    # use constant extrapolation for `u, v, tracers`.
    # use linear extrapolation for `z, t`.
    return weatherbench_utils.State(
        u=regrid_with_constant_fn(u),  # pyrefly: ignore[unexpected-keyword]
        v=regrid_with_constant_fn(v),  # pyrefly: ignore[unexpected-keyword]
        t=regrid_with_linear_fn(t),  # pyrefly: ignore[unexpected-keyword]
        z=regrid_with_linear_fn(z),  # pyrefly: ignore[unexpected-keyword]
        sim_time=inputs.sim_time,  # pyrefly: ignore[unexpected-keyword]
        tracers=regrid_with_constant_fn(tracers),  # pyrefly: ignore[unexpected-keyword]
    )

  def __call__(
      self, inputs: ModelState, forcing: Forcing
  ) -> DataState:
    del forcing
    wb_on_sigma = self.primitive_to_weatherbench(inputs.state)
    return self.transform_fn(wb_on_sigma.asdict())  # pyrefly: ignore[missing-attribute]


_DECODER_SALT = zlib.crc32(b'decoder')  # arbitrary uint32 value


def _decoder_prng_key(
    randomness: typing.RandomnessState,
) -> typing.PRNGKeyArray | None:
  """Get a PRNG Key suitable for decoder randomness."""
  if randomness.prng_key is None:
    return None
  salt = jnp.uint32(_DECODER_SALT) + jnp.uint32(randomness.prng_step)
  return jax.random.fold_in(randomness.prng_key, salt)


@gin.register
class LearnedPrimitiveToWeatherbenchDecoder(PrimitiveToWeatherbenchDecoder):
  """Similar to `PrimitiveToWeatherbenchDecoder` with learned interpolation."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      output_coords: coordinate_systems.CoordinateSystem,
      modal_to_nodal_model_features_module: FeaturesModule,
      modal_to_nodal_data_features_module: FeaturesModule,
      correction_transform_module: TransformModule,
      nodal_mapping_module: MappingModule,
      prediction_mask: typing.Pytree,
      time_axis: int = 0,
      orography_module: OrographyModule = orographies.ClippedOrography,
      transform_module: TransformModule = DecoderIdentityTransform,
      randomness_module: RandomnessModule = stochastic.ZerosRandomField,
      perturbation_module: PerturbationModule = perturbations.NoPerturbation,
      diagnostics_module: DiagnosticModule = diagnostics.NoDiagnostics,
      name: Optional[str] = None,
  ):
    super().__init__(
        coords=coords,
        dt=dt,
        physics_specs=physics_specs,
        aux_features=aux_features,
        output_coords=output_coords,
        time_axis=time_axis,
        orography_module=orography_module,
        name=name,
    )  # don't pass the transform, as we apply it at the end.
    self.prediction_mask = prediction_mask
    # features are computed on both coordinate systems.
    self.model_features_fn = modal_to_nodal_model_features_module(
        coords, dt, physics_specs, aux_features
    )
    self.data_features_fn = modal_to_nodal_data_features_module(
        output_coords, dt, physics_specs, aux_features
    )
    self.corrections_transform_fn = correction_transform_module(
        coords, dt, physics_specs, aux_features
    )
    # corrections are computed in real space on output coordinates.
    self.nodal_mapping_module = nodal_mapping_module
    self.get_nodal_shape_fn = lambda x: coordinate_systems.get_nodal_shapes(
        x, output_coords
    )
    self.transform_fn = transform_module(
        coords, dt, physics_specs, aux_features, output_coords
    )
    self.randomness_fn = randomness_module(
        coords, dt, physics_specs, aux_features
    )
    self.perturbation_fn = perturbation_module(
        coords, dt, physics_specs, aux_features
    )
    self.diagnostic_fn = diagnostics_module(
        coords, dt, physics_specs, aux_features
    )

  def __call__(
      self, inputs: ModelState, forcing: Forcing
  ) -> DataState:
    randomness = self.randomness_fn.unconditional_sample(
        _decoder_prng_key(inputs.randomness)
    )
    prognostics = self.perturbation_fn(
        inputs=self.coords.with_dycore_sharding(inputs.state),
        state=None,
        randomness=self.coords.with_dycore_sharding(randomness.nodal_value),
    )
    inputs.state = prognostics  # compute diagnostics from the perturbed state.

    # TODO(dkochkov) Could we pass physics_tendencies here?
    # TODO(janniyuval) Consider using evaporation diagnostics for training.
    decoder_diagnostics = self.diagnostic_fn(inputs, None)
    wb_on_pressure_dict = self.primitive_to_weatherbench(prognostics).asdict()  # pyrefly: ignore[missing-attribute]
    wb_on_pressure_modal = coordinate_systems.maybe_to_modal(
        self.coords.physics_to_dycore_sharding(wb_on_pressure_dict),
        self.output_coords,
    )
    wb_on_pressure_dict['diagnostics'] = decoder_diagnostics
    prediction_mask = pytree_utils.replace_with_matching_or_default(
        wb_on_pressure_dict, self.prediction_mask, default=False)
    prediction_shapes = jax.tree_util.tree_map(
        lambda x, y: self.get_nodal_shape_fn(x) if y else None,
        wb_on_pressure_dict,
        prediction_mask,
    )
    net = self.nodal_mapping_module(prediction_shapes)
    model_features = self.model_features_fn(
        prognostics.asdict(), forcing=forcing,
        randomness=randomness.nodal_value
    )
    data_features = self.data_features_fn(wb_on_pressure_modal, forcing=forcing)
    data_features = transforms.add_prefix(data_features, 'data_')
    model_features = transforms.add_prefix(model_features, 'model_')
    all_features = self.coords.dycore_to_physics_sharding(
        data_features | model_features
    )

    nodal_outputs = self.corrections_transform_fn(net(all_features))
    add_fn = lambda x, y: x + y if y is not None else x
    wb_on_pressure_dict = jax.tree_util.tree_map(
        add_fn, wb_on_pressure_dict, nodal_outputs
    )
    return self.transform_fn(wb_on_pressure_dict)


@gin.register
class DimensionalPrimitiveToWeatherbenchDecoder(PrimitiveToWeatherbenchDecoder):
  """Same as PrimitiveToWeatherbenchDecoder, but with dimensional output."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      output_coords: coordinate_systems.CoordinateSystem,
      inputs_to_units_mapping: Dict[str, str],
      time_axis: int = 0,
      orography_module: OrographyModule = orographies.ClippedOrography,
      transform_module: TransformModule = DecoderIdentityTransform,
      name: Optional[str] = None,
  ):
    nondim_pressure_centers = physics_specs.nondimensionalize(
        output_coords.vertical.centers * scales.units.millibar
    )
    nondim_output_coords = coordinate_systems.CoordinateSystem(
        output_coords.horizontal,
        vertical_interpolation.PressureCoordinates(nondim_pressure_centers),
        spmd_mesh=output_coords.spmd_mesh,
    )
    super().__init__(
        coords,
        dt,
        physics_specs,
        aux_features,
        output_coords=nondim_output_coords,
        time_axis=time_axis,
        orography_module=orography_module,
        transform_module=transform_module,
        name=name,
    )
    self.redimensionalize_fn = transforms.RedimensionalizeTransform(
        coords,
        dt,
        physics_specs,
        aux_features,
        output_coords=output_coords,
        inputs_to_units_mapping=inputs_to_units_mapping,
    )

  def __call__(
      self, inputs: ModelState, forcing: Forcing
  ) -> DataState:
    return self.redimensionalize_fn(super().__call__(inputs, forcing))


@gin.configurable
class DimensionalLearnedPrimitiveToWeatherbenchDecoder(
    LearnedPrimitiveToWeatherbenchDecoder
):
  """Same as LearnedPrimitiveToWeatherbenchDecoder, but with dimensional output."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      output_coords: coordinate_systems.CoordinateSystem,
      modal_to_nodal_model_features_module: FeaturesModule,
      modal_to_nodal_data_features_module: FeaturesModule,
      nodal_mapping_module: MappingModule,
      correction_transform_module: TransformModule,
      prediction_mask: typing.Pytree,
      inputs_to_units_mapping: Dict[str, str],
      time_axis: int = 0,
      orography_module: OrographyModule = orographies.ClippedOrography,
      transform_module: TransformModule = DecoderIdentityTransform,
      randomness_module: RandomnessModule = stochastic.ZerosRandomField,
      perturbation_module: PerturbationModule = perturbations.NoPerturbation,
      diagnostics_module: DiagnosticModule = diagnostics.NoDiagnostics,
      name: Optional[str] = None,
  ):
    nondim_pressure_centers = physics_specs.nondimensionalize(
        output_coords.vertical.centers * scales.units.millibar
    )
    nondim_output_coords = coordinate_systems.CoordinateSystem(
        output_coords.horizontal,
        vertical_interpolation.PressureCoordinates(nondim_pressure_centers),
        spmd_mesh=output_coords.spmd_mesh,
    )
    super().__init__(
        coords,
        dt,
        physics_specs,
        aux_features,
        output_coords=nondim_output_coords,
        modal_to_nodal_model_features_module=(
            modal_to_nodal_model_features_module
        ),
        modal_to_nodal_data_features_module=modal_to_nodal_data_features_module,
        nodal_mapping_module=nodal_mapping_module,
        correction_transform_module=correction_transform_module,
        prediction_mask=prediction_mask,
        time_axis=time_axis,
        orography_module=orography_module,
        transform_module=transform_module,
        randomness_module=randomness_module,
        perturbation_module=perturbation_module,
        diagnostics_module=diagnostics_module,
        name=name,
    )
    self.redimensionalize_fn = transforms.RedimensionalizeTransform(
        coords,
        dt,
        physics_specs,
        aux_features,
        output_coords=output_coords,
        inputs_to_units_mapping=inputs_to_units_mapping,
    )

  def __call__(
      self, inputs: ModelState, forcing: Forcing
  ) -> DataState:
    return self.redimensionalize_fn(super().__call__(inputs, forcing))
