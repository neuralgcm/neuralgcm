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
"""Defines `encoder` modules that map input trajectories to model states.

All encoder modules return the encoder-specific model state that represents the
state of the system at the latest time provided in the input trajectory.
The inputs are expected to consist of arrays with `time` as a leading axis.
"""

# TODO(dkochkov) make all encoders take in trajectories and return ModelState.

import functools
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

from dinosaur import coordinate_systems
from dinosaur import primitive_equations
from dinosaur import pytree_utils
from dinosaur import scales
from dinosaur import shallow_water
from dinosaur import spherical_harmonic
from dinosaur import typing
from dinosaur import vertical_interpolation
from dinosaur import weatherbench_utils
from dinosaur import xarray_utils
import gin
import haiku as hk
import jax
import jax.numpy as jnp
from neuralgcm.legacy import features
from neuralgcm.legacy import mappings
from neuralgcm.legacy import orographies
from neuralgcm.legacy import perturbations
from neuralgcm.legacy import stochastic
from neuralgcm.legacy import transforms
import numpy as np


Array = Union[np.ndarray, jnp.ndarray]
DataState = typing.DataState
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

# We ♥ λ's
# pylint: disable=g-long-lambda


@gin.register
class EncoderIdentityTransform(hk.Module):
  """Transformation that returns inputs without modification."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      input_coords: coordinate_systems.CoordinateSystem,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    del coords, dt, physics_specs, aux_features, input_coords

  def __call__(self, inputs: PyTreeState) -> PyTreeState:
    return inputs


@gin.register
class EncoderFilterTransform(hk.Module):
  """Transformation that returns truncated and filtered modal inputs."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      input_coords: coordinate_systems.CoordinateSystem,
      filter_modules: Sequence[FilterModule] = tuple(),
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.filter_fns = [
        module(coords, dt, physics_specs, aux_features)
        for module in filter_modules
    ]

  def __call__(self, inputs: PyTreeState) -> PyTreeState:
    for filter_fn in self.filter_fns:
      inputs = filter_fn(inputs)
    return inputs


@gin.register
class InputClipTransform(hk.Module):
  """Filter that clips highest total wavenumber the input state."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      input_coords: coordinate_systems.CoordinateSystem,
      wavenumbers_to_clip: int = 1,
      name: Optional[str] = None,
  ):
    """See `time_integration.exponential_filter` for details."""
    del coords, dt, physics_specs, aux_features  # unused.
    super().__init__(name=name)
    self.input_coords = input_coords
    self.wavenumbers_to_clip = wavenumbers_to_clip

  def __call__(self, inputs: PyTreeState) -> PyTreeState:
    return self.input_coords.horizontal.clip_wavenumbers(
        inputs, self.wavenumbers_to_clip
    )


@gin.register
class InputNodalToModalTransform(hk.Module):
  """Transformation that converts nodal inputs to modal representation."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      input_coords: coordinate_systems.CoordinateSystem,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.coords = coords
    self.input_coords = input_coords

  def __call__(self, inputs: PyTreeState) -> PyTreeState:
    to_modal_fn = self.input_coords.horizontal.to_modal
    downsample_fn = coordinate_systems.get_spectral_downsample_fn(
        self.input_coords, self.coords, expect_same_vertical=False
    )
    return jax.tree_util.tree_map(
        lambda x: downsample_fn(to_modal_fn(x)), inputs
    )


@gin.register
class ModalInputLearnedAdaptorTransform(hk.Module):
  """Transformation using a tower to adapt modal inputs to the model domain."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      input_coords: coordinate_systems.CoordinateSystem,
      modal_to_nodal_features_module: FeaturesModule,
      nodal_mapping_module: MappingModule,
      output_transform_module: TransformModule,
      name: Optional[str] = None,
  ):
    del input_coords  # unused.
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
class NodalInputLearnedAdaptorTransform(hk.Module):
  """Transformation using a tower to adapt nodal inputs to the model domain."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      input_coords: coordinate_systems.CoordinateSystem,
      nodal_to_nodal_features_module: FeaturesModule,
      nodal_mapping_module: MappingModule,
      output_transform_module: TransformModule,
      name: Optional[str] = None,
  ):
    del input_coords  # unused.
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
class EncoderCombinedTransform(hk.Module):
  """Module that applies multiple transformations sequentially."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      input_coords: coordinate_systems.CoordinateSystem,
      transforms: Tuple[TransformModule, ...] = tuple(),  # pylint: disable=redefined-outer-name
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.transform_fns = [
        module(coords, dt, physics_specs, aux_features, input_coords)
        for module in transforms
    ]

  def __call__(self, inputs: PyTreeState) -> PyTreeState:
    for transform_fn in self.transform_fns:
      inputs = transform_fn(inputs)
    return inputs


@gin.register
class ShallowWaterStateEncoder(hk.Module):
  """Encoder that extracts shallow_water.State pair from inputs."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      input_coords: coordinate_systems.CoordinateSystem,
      time_axis: int = 0,
      transform_module: TransformModule = EncoderIdentityTransform,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.slice_fn = functools.partial(
        pytree_utils.slice_along_axis, axis=time_axis, idx=-1)
    self.transform_fn = transform_module(
        coords, dt, physics_specs, aux_features, input_coords
    )

  def __call__(
      self, inputs: DataState, forcing: Forcing
  ) -> shallow_water.State:
    del forcing
    state = self.transform_fn(shallow_water.State(**self.slice_fn(inputs)))
    return ModelState(state)  # pyrefly: ignore[bad-return]


@gin.register
class ShallowWaterLeapfrogEncoder(hk.Module):
  """Encoder that extracts shallow_water.State pair from inputs."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      input_coords: coordinate_systems.CoordinateSystem,
      time_axis: int = 0,
      transform_module: TransformModule = EncoderIdentityTransform,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.slice_fn = functools.partial(
        pytree_utils.slice_along_axis, axis=time_axis, idx=slice(-2, None))
    self.time_axis = time_axis
    self.transform_fn = transform_module(
        coords, dt, physics_specs, aux_features, input_coords
    )

  def __call__(
      self, inputs: DataState, forcing: Forcing
  ) -> Tuple[shallow_water.State, ...]:
    del forcing
    last_two_frames = pytree_utils.split_axis(
        self.slice_fn(inputs), self.time_axis
    )
    state = self.transform_fn(
        tuple(shallow_water.State(**items) for items in last_two_frames)
    )
    return ModelState(state)  # pyrefly: ignore[bad-return]


@gin.register
class PrimitiveEquationStateEncoder(hk.Module):
  """Encoder that extracts primitive_equations.State from inputs."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      input_coords: coordinate_systems.CoordinateSystem,
      time_axis: int = 0,
      transform_module: TransformModule = EncoderIdentityTransform,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.slice_fn = functools.partial(
        pytree_utils.slice_along_axis, axis=time_axis, idx=-1)
    self.transform_fn = transform_module(
        coords, dt, physics_specs, aux_features, input_coords
    )

  def __call__(
      self, inputs: DataState, forcing: Forcing
  ) -> primitive_equations.State:
    del forcing
    state = self.transform_fn(
        primitive_equations.State(**self.slice_fn(inputs)))
    return ModelState(state)  # pyrefly: ignore[bad-return]


@gin.register
class PrimitiveEquationLeapfrogEncoder(hk.Module):
  """Encoder that extracts primitive_equations.State pair from inputs."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      input_coords: coordinate_systems.CoordinateSystem,
      time_axis: int = 0,
      transform_module: TransformModule = EncoderIdentityTransform,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.slice_fn = functools.partial(
        pytree_utils.slice_along_axis, axis=time_axis, idx=slice(-2, None))
    self.time_axis = time_axis
    self.transform_fn = transform_module(
        coords, dt, physics_specs, aux_features, input_coords
    )

  def __call__(
      self, inputs: DataState, forcing: Forcing
  ) -> Tuple[primitive_equations.State, ...]:
    del forcing
    last_two_frames = pytree_utils.split_axis(
        self.slice_fn(inputs), self.time_axis
    )
    state = self.transform_fn(
        tuple(primitive_equations.State(**items) for items in last_two_frames)
    )
    return ModelState(state)  # pyrefly: ignore[bad-return]


@gin.register
class PrimitiveEquationStateWithTimeEncoder(hk.Module):
  """Encoder that extracts primitive_equations.StateWithTime from inputs."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      input_coords: coordinate_systems.CoordinateSystem,
      time_axis: int = 0,
      transform_module: TransformModule = EncoderIdentityTransform,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.slice_fn = functools.partial(
        pytree_utils.slice_along_axis, axis=time_axis, idx=-1)
    self.transform_fn = transform_module(
        coords, dt, physics_specs, aux_features, input_coords
    )

  def __call__(
      self, inputs: DataState, forcing: Forcing
  ) -> ModelState:
    del forcing
    sliced_inputs = self.slice_fn(inputs)
    state = self.transform_fn(
        primitive_equations.StateWithTime(**sliced_inputs))
    return ModelState(state)


@gin.register
class WeatherbenchToPrimitiveEncoder(hk.Module):
  """Encoder that extracts primitive_equations.StateWithTime from WB inputs."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      input_coords: coordinate_systems.CoordinateSystem,
      time_axis: int = 0,
      orography_module: OrographyModule = orographies.ClippedOrography,
      transform_module: TransformModule = EncoderIdentityTransform,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    ref_temps = aux_features[xarray_utils.REF_TEMP_KEY]
    self.ref_temps = ref_temps[..., np.newaxis, np.newaxis]
    self.coords = coords
    self.input_coords = input_coords
    self.slice_fn = functools.partial(
        pytree_utils.slice_along_axis, axis=time_axis, idx=-1)
    modal_orography_init_fn = orography_module(
        coords, dt, physics_specs, aux_features
    )
    modal_orography = modal_orography_init_fn()  # pyrefly: ignore[not-callable]
    self.surface_pressure_fn = functools.partial(
        vertical_interpolation.get_surface_pressure,
        input_coords.vertical,
        orography=input_coords.horizontal.to_nodal(modal_orography),
        gravity_acceleration=physics_specs.gravity_acceleration,
    )
    self.curl_and_div_fn = functools.partial(
        spherical_harmonic.uv_nodal_to_vor_div_modal,
        input_coords.horizontal,
    )
    self.modal_interpolate_fn = coordinate_systems.get_spectral_interpolate_fn(
        input_coords, coords, expect_same_vertical=False
    )
    self.transform_fn = transform_module(
        coords, dt, physics_specs, aux_features, input_coords
    )

  def weatherbench_to_primitive(
      self,
      wb_state_nodal: weatherbench_utils.State,
  ) -> ModelState:
    """Converts wb_state on pressure coordinates to primitive on sigma."""
    # Note: the returned values have mixed nodal/modal representations.
    surface_pressure = self.surface_pressure_fn(wb_state_nodal.z)
    interpolate_fn = vertical_interpolation.vectorize_vertical_interpolation(
        vertical_interpolation.vertical_interpolation
    )
    regrid_fn = functools.partial(
        vertical_interpolation.interp_pressure_to_sigma,
        pressure_coords=self.input_coords.vertical,
        sigma_coords=self.coords.vertical,
        surface_pressure=surface_pressure,
        interpolate_fn=interpolate_fn,
    )
    wb_state_on_sigma = regrid_fn(wb_state_nodal)
    u, v = self.coords.physics_to_dycore_sharding(
        (wb_state_on_sigma.u, wb_state_on_sigma.v)
    )
    vorticity, divergence = self.coords.dycore_to_physics_sharding(
        self.curl_and_div_fn(u, v)
    )
    pe_state_on_sigma = primitive_equations.StateWithTime(
        divergence=divergence,
        vorticity=vorticity,
        temperature_variation=(wb_state_on_sigma.t - self.ref_temps),
        log_surface_pressure=jnp.log(surface_pressure),
        sim_time=wb_state_on_sigma.sim_time,
        tracers=wb_state_on_sigma.tracers,
    )
    return pe_state_on_sigma  # pyrefly: ignore[bad-return]

  def __call__(
      self,
      inputs: DataState,
      forcing: Forcing,
  ) -> ModelState:
    del forcing
    wb_state = weatherbench_utils.State(**self.slice_fn(inputs))
    wb_state = coordinate_systems.maybe_to_nodal(wb_state, self.input_coords)
    pe_state = self.weatherbench_to_primitive(wb_state)
    pe_state = coordinate_systems.maybe_to_modal(pe_state, self.input_coords)
    pe_state = self.modal_interpolate_fn(pe_state)
    return ModelState(state=self.transform_fn(pe_state))


@gin.register
class LearnedWeatherbenchToPrimitiveEncoder(WeatherbenchToPrimitiveEncoder):
  """Same as `WeatherbenchToPrimitiveEncoder`, but with learned corrections."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      input_coords: coordinate_systems.CoordinateSystem,
      modal_to_nodal_data_features_module: FeaturesModule,
      modal_to_nodal_model_features_module: FeaturesModule,
      nodal_mapping_module: MappingModule,
      correction_transform_module: TransformModule,
      prediction_mask: typing.Pytree,
      time_axis: int = 0,
      orography_module: OrographyModule = orographies.ClippedOrography,
      transform_module: TransformModule = EncoderIdentityTransform,
      randomness_module: RandomnessModule = stochastic.ZerosRandomField,
      perturbation_module: PerturbationModule = perturbations.NoPerturbation,
      name: Optional[str] = None,
  ):
    super().__init__(
        coords,
        dt,
        physics_specs,
        aux_features,
        input_coords=input_coords,
        time_axis=time_axis,
        orography_module=orography_module,
        name=name,
    )
    self.prediction_mask = prediction_mask
    # data features are computed in real space on input coordinates.
    self.data_features_fn = modal_to_nodal_data_features_module(
        input_coords, dt, physics_specs, aux_features
    )
    self.model_features_fn = modal_to_nodal_model_features_module(
        coords, dt, physics_specs, aux_features
    )
    self.nodal_mapping_module = nodal_mapping_module
    self.output_transform_fn = correction_transform_module(
        input_coords, dt, physics_specs, aux_features
    )
    self.get_nodal_shape_fn = lambda x: coordinate_systems.get_nodal_shapes(
        x, coords
    )
    self.transform_fn = transform_module(
        coords, dt, physics_specs, aux_features, input_coords
    )
    self.randomness_fn = randomness_module(
        coords, dt, physics_specs, aux_features
    )
    self.perturbation_fn = perturbation_module(
        coords, dt, physics_specs, aux_features
    )

  def __call__(
      self,
      inputs: DataState,
      forcing: Forcing,
  ) -> ModelState:
    randomness = self.randomness_fn.unconditional_sample(
        hk.maybe_next_rng_key()
    )
    wb_state = self.coords.with_physics_sharding(
        weatherbench_utils.State(**self.slice_fn(inputs))
    )
    wb_state_nodal = self.coords.with_physics_sharding(
        coordinate_systems.maybe_to_nodal(wb_state, self.input_coords)
    )
    wb_state_modal = self.coords.with_physics_sharding(
        coordinate_systems.maybe_to_modal(wb_state, self.input_coords)
    )
    pe_state = self.coords.physics_to_dycore_sharding(
        self.weatherbench_to_primitive(wb_state_nodal)
    )
    # Computing corrections to the primitive_equations state.
    pe_state_modal = coordinate_systems.maybe_to_modal(
        pe_state, self.input_coords
    )
    # we need to interpolate `pe_state_modal` to self.coords to compute
    # features in model space. In most cases this is no-op as grids match.
    pe_state_modal = self.modal_interpolate_fn(pe_state_modal)
    pe_state_nodal = coordinate_systems.maybe_to_nodal(
        pe_state_modal, self.coords
    )
    prediction_shapes = jax.tree_util.tree_map(
        lambda x, y: self.get_nodal_shape_fn(x) if y else None,
        pe_state_nodal.asdict(),
        self.prediction_mask,
    )
    prediction_shapes = primitive_equations.StateWithTime(**prediction_shapes)
    net = self.nodal_mapping_module(prediction_shapes)
    # we need modal values to compute features for ML corrections.
    data_features = self.data_features_fn(
        wb_state_modal.asdict(), forcing=forcing,
    )
    model_features = self.model_features_fn(
        pe_state_modal.asdict(), forcing=forcing,
        randomness=randomness.nodal_value,
    )
    data_features = transforms.add_prefix(data_features, 'data_')
    model_features = transforms.add_prefix(model_features, 'model_')

    all_features = self.coords.with_physics_sharding(
        data_features | model_features
    )

    nodal_corrections = self.coords.with_physics_sharding(
        self.output_transform_fn(net(all_features))
    )

    perturbed_correction = self.perturbation_fn(
        state=None,  # Unused
        inputs=nodal_corrections,
        randomness=randomness.nodal_value,
    )

    add_fn = lambda x, y: x + y if y is not None else x
    corrected_pe_state = self.coords.physics_to_dycore_sharding(
        jax.tree_util.tree_map(
            add_fn,
            coordinate_systems.maybe_to_modal(pe_state_nodal, self.coords),
            coordinate_systems.maybe_to_modal(
                perturbed_correction, self.coords
            ),
        )
    )
    return ModelState(state=self.transform_fn(corrected_pe_state))


@gin.register
class DimensionalWeatherbenchToPrimitiveEncoder(WeatherbenchToPrimitiveEncoder):
  """Same as WeatherbenchToPrimitiveEncoder, but with dimensional inputs."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      input_coords: coordinate_systems.CoordinateSystem,
      inputs_to_units_mapping: Dict[str, str],
      time_axis: int = 0,
      orography_module: OrographyModule = orographies.ClippedOrography,
      transform_module: TransformModule = EncoderIdentityTransform,
      name: Optional[str] = None,
  ):
    nondim_pressure_centers = physics_specs.nondimensionalize(
        input_coords.vertical.centers * scales.units.millibar
    )
    nondim_input_coords = coordinate_systems.CoordinateSystem(
        input_coords.horizontal,
        vertical_interpolation.PressureCoordinates(nondim_pressure_centers),
        spmd_mesh=input_coords.spmd_mesh,
    )
    super().__init__(
        coords,
        dt,
        physics_specs,
        aux_features,
        input_coords=nondim_input_coords,
        time_axis=time_axis,
        orography_module=orography_module,
        transform_module=transform_module,
        name=name,
    )
    self.nondim_transform_fn = transforms.NondimensionalizeTransform(
        coords,
        dt,
        physics_specs,
        aux_features,
        nondim_input_coords,
        inputs_to_units_mapping=inputs_to_units_mapping,
    )

  def __call__(
      self,
      inputs: DataState,
      forcing: Forcing,
  ) -> primitive_equations.StateWithTime:
    nondim_inputs = self.nondim_transform_fn(inputs)
    return super().__call__(nondim_inputs, forcing)  # pyrefly: ignore[bad-return]


@gin.register
class DimensionalLearnedWeatherbenchToPrimitiveEncoder(
    LearnedWeatherbenchToPrimitiveEncoder
):
  """Same as LearnedWeatherbenchToPrimitiveEncoder, but with dimensional inputs."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      input_coords: coordinate_systems.CoordinateSystem,
      inputs_to_units_mapping: Dict[str, str],
      modal_to_nodal_data_features_module: FeaturesModule,
      modal_to_nodal_model_features_module: FeaturesModule,
      nodal_mapping_module: MappingModule,
      correction_transform_module: TransformModule,
      prediction_mask: typing.Pytree,
      time_axis: int = 0,
      orography_module: OrographyModule = orographies.ClippedOrography,
      transform_module: TransformModule = EncoderIdentityTransform,
      randomness_module: RandomnessModule = stochastic.ZerosRandomField,
      perturbation_module: PerturbationModule = perturbations.NoPerturbation,
      name: Optional[str] = None,
  ):
    nondim_pressure_centers = physics_specs.nondimensionalize(
        input_coords.vertical.centers * scales.units.millibar
    )
    nondim_input_coords = coordinate_systems.CoordinateSystem(
        input_coords.horizontal,
        vertical_interpolation.PressureCoordinates(nondim_pressure_centers),
        spmd_mesh=input_coords.spmd_mesh,
    )
    super().__init__(
        coords,
        dt,
        physics_specs,
        aux_features,
        input_coords=nondim_input_coords,
        modal_to_nodal_data_features_module=modal_to_nodal_data_features_module,
        modal_to_nodal_model_features_module=(
            modal_to_nodal_model_features_module
        ),
        nodal_mapping_module=nodal_mapping_module,
        correction_transform_module=correction_transform_module,
        prediction_mask=prediction_mask,
        time_axis=time_axis,
        orography_module=orography_module,
        transform_module=transform_module,
        randomness_module=randomness_module,
        perturbation_module=perturbation_module,
        name=name,
    )
    self.nondim_transform_fn = transforms.NondimensionalizeTransform(
        coords,
        dt,
        physics_specs,
        aux_features,
        nondim_input_coords,
        inputs_to_units_mapping=inputs_to_units_mapping,
    )

  def __call__(
      self,
      inputs: DataState,
      forcing: Forcing,
  ) -> ModelState:
    nondim_inputs = self.nondim_transform_fn(inputs)
    return super().__call__(nondim_inputs, forcing)


@gin.register
class DimensionalLearnedWeatherbenchToPrimitiveWithMemoryEncoder(hk.Module):
  """Same as DimensionalLearnedWeatherbenchToPrimitiveEncoder, but with memory.
  """

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      input_coords: coordinate_systems.CoordinateSystem,
      inputs_to_units_mapping: Dict[str, str],
      modal_to_nodal_data_features_module: FeaturesModule,
      modal_to_nodal_model_features_module: FeaturesModule,
      nodal_mapping_module: MappingModule,
      correction_transform_module: TransformModule,
      prediction_mask: typing.Pytree,
      time_axis: int = 0,
      orography_module: OrographyModule = orographies.ClippedOrography,
      transform_module: TransformModule = EncoderIdentityTransform,
      randomness_module: RandomnessModule = stochastic.ZerosRandomField,
      perturbation_module: PerturbationModule = perturbations.NoPerturbation,
      name: Optional[str] = None,
  ):
    nondim_pressure_centers = physics_specs.nondimensionalize(
        input_coords.vertical.centers * scales.units.millibar)
    nondim_input_coords = coordinate_systems.CoordinateSystem(
        input_coords.horizontal,
        vertical_interpolation.PressureCoordinates(nondim_pressure_centers),
        spmd_mesh=input_coords.spmd_mesh,
    )
    super().__init__(name=name)
    make_encoder_fn = functools.partial(
        LearnedWeatherbenchToPrimitiveEncoder,
        coords=coords, dt=dt,
        physics_specs=physics_specs, aux_features=aux_features,
        input_coords=nondim_input_coords,
        modal_to_nodal_data_features_module=
        modal_to_nodal_data_features_module,
        modal_to_nodal_model_features_module=
        modal_to_nodal_model_features_module,
        nodal_mapping_module=nodal_mapping_module,
        correction_transform_module=correction_transform_module,
        prediction_mask=prediction_mask, time_axis=time_axis,
        orography_module=orography_module,
        transform_module=transform_module,
        name=name
    )

    # Memory will be deterministic. State may be random.
    self.memory_encoder = make_encoder_fn(
        randomness_module=stochastic.NoRandomField,
        perturbation_module=perturbations.NoPerturbation,
    )
    self.state_encoder = make_encoder_fn(
        randomness_module=randomness_module,
        perturbation_module=perturbation_module,
    )

    self.nondim_transform_fn = transforms.NondimensionalizeTransform(
        coords, dt, physics_specs, aux_features, nondim_input_coords,
        inputs_to_units_mapping=inputs_to_units_mapping)

  def __call__(
      self,
      inputs: DataState,
      forcing: Forcing,
  ) -> ModelState:
    nondim_inputs = self.nondim_transform_fn(inputs)
    memory = self.memory_encoder(nondim_inputs, forcing=forcing)
    model_state = self.state_encoder(nondim_inputs, forcing=forcing)
    return ModelState(
        state=model_state.state,
        memory=memory.state,
        randomness=model_state.randomness,
    )
