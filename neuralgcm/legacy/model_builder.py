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
"""Defines AbstractModel API, standard implementations and helper functions."""
from __future__ import annotations

import collections
import dataclasses
import functools
from typing import Any, Callable, Dict, Optional, Sequence, Union
from dinosaur import coordinate_systems
from dinosaur import layer_coordinates
from dinosaur import scales
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
from dinosaur import time_integration
from dinosaur import typing
from dinosaur import xarray_utils
import gin
import haiku as hk
import jax.numpy as jnp

from neuralgcm.legacy import correctors  # pylint: disable=unused-import
from neuralgcm.legacy import decoders  # pylint: disable=unused-import
from neuralgcm.legacy import embeddings  # pylint: disable=unused-import
from neuralgcm.legacy import encoders  # pylint: disable=unused-import
from neuralgcm.legacy import equations  # pylint: disable=unused-import
from neuralgcm.legacy import features  # pylint: disable=unused-import
from neuralgcm.legacy import filters  # pylint: disable=unused-import
from neuralgcm.legacy import forcings  # pylint: disable=unused-import
from neuralgcm.legacy import gin_utils
from neuralgcm.legacy import layers  # pylint: disable=unused-import
from neuralgcm.legacy import mappings  # pylint: disable=unused-import
from neuralgcm.legacy import model_utils
from neuralgcm.legacy import physics_specifications
from neuralgcm.legacy import steps  # pylint: disable=unused-import
from neuralgcm.legacy import stochastic  # pylint: disable=unused-import
from neuralgcm.legacy import towers  # pylint: disable=unused-import
from neuralgcm.legacy import transforms  # pylint: disable=unused-import
import numpy as np
import xarray
# Note: many unused imports are needed to load configurable components;


DEFAULT_REFERENCE_TEMPERATURE = 288
DEFAULT_REFERENCE_DATETIME_STR = '1979-01-01T00'

Array = typing.Array
AuxFeatures = typing.AuxFeatures
DataState = typing.DataState
PyTreeState = typing.PyTreeState
ModelState = typing.ModelState
ForcingData = typing.ForcingData
Forcing = typing.Forcing
Numeric = typing.Numeric
QuantityOrStr = Union[str, scales.Quantity]
# Specifying the full signatures of Callable would get somewhat onerous
# pylint: disable=g-bare-generic

# Overzealous linter is getting confused by ABC typing.
# pylint: disable=function-missing-types
# pylint: disable=missing-arg-types

# Register data to xarray conversion methods.
data_to_xarray = gin.external_configurable(
    xarray_utils.data_to_xarray, 'data_to_xarray'
)
# TODO(dkochkov) Remove this legacy name when no best checkpoints rely on it.
primitive_eq_to_xarray = gin.external_configurable(
    xarray_utils.data_to_xarray, 'primitive_eq_to_xarray'
)
data_to_xarray_with_renaming = gin.external_configurable(
    xarray_utils.data_to_xarray_with_renaming, 'data_to_xarray_with_renaming'
)
dynamic_covariate_data_to_xarray = gin.external_configurable(
    xarray_utils.dynamic_covariate_data_to_xarray,
    'dynamic_covariate_data_to_xarray',
)

# Register xarray to data conversion methods.
xarray_to_shallow_water = gin.external_configurable(
    xarray_utils.xarray_to_shallow_water_eq_data, 'xarray_to_shallow_water'
)
xarray_to_primitive_eq = gin.external_configurable(
    xarray_utils.xarray_to_primitive_eq_data, 'xarray_to_primitive_eq'
)
xarray_to_primitive_eq_with_time = gin.external_configurable(
    xarray_utils.xarray_to_primitive_equations_with_time_data,
    'xarray_to_primitive_eq_with_time',
)
xarray_to_weatherbench_data = gin.external_configurable(
    xarray_utils.xarray_to_weatherbench_data, 'xarray_to_weatherbench_data'
)
xarray_to_data_with_renaming = gin.external_configurable(
    xarray_utils.xarray_to_data_with_renaming, 'xarray_to_data_with_renaming'
)
xarray_to_dynamic_covariate_data = gin.external_configurable(
    xarray_utils.xarray_to_dynamic_covariate_data,
    'xarray_to_dynamic_covariate_data',
)
xarray_to_state_and_dynamic_covariate_data = gin.external_configurable(
    xarray_utils.xarray_to_state_and_dynamic_covariate_data,
    'xarray_to_state_and_dynamic_covariate_data',
)
coordinate_system_from_dataset = gin.external_configurable(
    xarray_utils.coordinate_system_from_dataset,
    'coordinate_system_from_dataset',
    allowlist=['truncation', 'spherical_harmonics_impl'],
)

# Register grids and coordinates for instantiation of coordinate systems.
Grid = gin.external_configurable(
    spherical_harmonic.Grid, denylist=['spmd_mesh']
)
GridWithWavenumbers = gin.external_configurable(
    spherical_harmonic.Grid.with_wavenumbers, 'GridWithWavenumbers'
)
GridT21 = gin.external_configurable(spherical_harmonic.Grid.T21, 'GridT21')
GridT31 = gin.external_configurable(spherical_harmonic.Grid.T31, 'GridT31')
GridT42 = gin.external_configurable(spherical_harmonic.Grid.T42, 'GridT42')
GridT85 = gin.external_configurable(spherical_harmonic.Grid.T85, 'GridT85')
GridT106 = gin.external_configurable(spherical_harmonic.Grid.T106, 'GridT106')
GridT119 = gin.external_configurable(spherical_harmonic.Grid.T119, 'GridT119')
GridT170 = gin.external_configurable(spherical_harmonic.Grid.T170, 'GridT170')
GridT213 = gin.external_configurable(spherical_harmonic.Grid.T213, 'GridT213')
GridTL31 = gin.external_configurable(spherical_harmonic.Grid.TL31, 'GridTL31')
GridTL63 = gin.external_configurable(spherical_harmonic.Grid.TL63, 'GridTL63')
GridTL95 = gin.external_configurable(spherical_harmonic.Grid.TL95, 'GridTL95')
GridTL127 = gin.external_configurable(
    spherical_harmonic.Grid.TL127, 'GridTL127'
)
GridTL159 = gin.external_configurable(
    spherical_harmonic.Grid.TL159, 'GridTL159'
)
GridTL179 = gin.external_configurable(
    spherical_harmonic.Grid.TL179, 'GridTL179'
)
GridTL255 = gin.external_configurable(
    spherical_harmonic.Grid.TL255, 'GridTL255'
)
RealSphericalHarmonics = gin.external_configurable(
    spherical_harmonic.RealSphericalHarmonics,
)
RealSphericalHarmonicsWithZeroImag = gin.external_configurable(
    spherical_harmonic.RealSphericalHarmonicsWithZeroImag,
    denylist=['spmd_mesh'],
)
LayerCoordinates = gin.external_configurable(layer_coordinates.LayerCoordinates)
SigmaCoordinates = gin.external_configurable(sigma_coordinates.SigmaCoordinates)
SigmaCoordinatesEquidistant = gin.external_configurable(
    sigma_coordinates.SigmaCoordinates.equidistant,
    'SigmaCoordinatesEquidistant',
)
CoordinateSystem = gin.external_configurable(
    coordinate_systems.CoordinateSystem, denylist=['spmd_mesh']
)

# Register vertical interpolation methods
centered_vertical_advection = gin.external_configurable(
    sigma_coordinates.centered_vertical_advection
)
upwind_vertical_advection = gin.external_configurable(
    sigma_coordinates.upwind_vertical_advection
)


@dataclasses.dataclass(frozen=True)
class ModelSpecs(collections.abc.Mapping):
  """Specification of model configuration.

  Attributes:
    coords: horizontal and vertical grid data.
    dt: nondimensionalized model time step.
    physics_specs: physical constants and definition of custom units.
    aux_features: additional static data.
  """

  coords: coordinate_systems.CoordinateSystem
  dt: float
  physics_specs: Any
  aux_features: typing.AuxFeatures

  def __len__(self):
    return len(dataclasses.fields(self))

  def __iter__(self):
    return iter(f.name for f in dataclasses.fields(self))

  def __getitem__(self, key):
    return getattr(self, key)


@gin.configurable(
    allowlist=(
        'model_time_step',
        'custom_coords',
        'reference_temperature',
        'reference_datetime_str',
    )
)
def get_model_specs(
    data_coords: coordinate_systems.CoordinateSystem,
    physics_specs: Any,
    aux_features: typing.AuxFeatures,
    model_time_step: Optional[Union[float, QuantityOrStr]] = None,
    custom_coords: Optional[coordinate_systems.CoordinateSystem] = None,
    reference_temperature: Optional[float | Sequence] = None,
    reference_datetime_str: Optional[str] = None,
) -> ModelSpecs:
  """Returns specifications for a WhirlModel configuration.

  Provides gin hooks, and in some cases defaults, for model specification
  formerly encoded in aux_features.

  Args:
    data_coords: coordinate system in which states are represented in the data.
    physics_specs: physical constants and definition of custom units.
    aux_features: auxiliary features that come with the dataset.
    model_time_step: duration of the outer time-step in our model, i.e., the
      time by which the state is advanced in a single model.advance call.
    custom_coords: optional coordinate system to be used by the model instead of
      data_coords.
    reference_temperature: reference temperature to use for sigma coordinates.
      Must be None if already defined in aux_features.  Default value of 288
      used if None and also not in aux_features.
    reference_datetime_str: reference datetime for which nondimensionalized time
      is set to 0. Must be None if already defined in aux_features. Default
      value of '1979-01-01T00' used if None and also not in aux_features.

  Returns:
    Configured specification of coordinate system, time-step, physical constants
    and units, and aux_features and for our hybrid ML/physics model.
  """
  if model_time_step is None:
    raise ValueError('must provide model_time_step or outer_time_step')

  if custom_coords is None:
    coords = data_coords
  else:
    coords = dataclasses.replace(custom_coords, spmd_mesh=data_coords.spmd_mesh)

  if aux_features.get(xarray_utils.REF_TEMP_KEY) is None:
    if reference_temperature is None:
      ones = np.ones(coords.vertical.layers, np.float32)
      ref_temps = DEFAULT_REFERENCE_TEMPERATURE * ones
      aux_features[xarray_utils.REF_TEMP_KEY] = ref_temps
    else:
      ones = np.ones(coords.vertical.layers, np.float32)
      ref_temps = np.asarray(reference_temperature)
      if ref_temps.ndim == 1 and ref_temps.shape[0] != coords.vertical.layers:
        raise ValueError(
            '`ref_temps` must be a scalar or a sequence with '
            f'{coords.vertical.layers=} elements, got {ref_temps.shape=}'
        )
      ref_temps = ref_temps * ones
      aux_features[xarray_utils.REF_TEMP_KEY] = ref_temps
  else:  # cannot set ref temp if already specified in aux_data
    if reference_temperature is not None:
      raise ValueError(
          'reference temperature already specified in aux_features'
      )

  if aux_features.get(xarray_utils.REFERENCE_DATETIME_KEY) is None:
    if reference_datetime_str is None:
      reference_datetime = np.datetime64(DEFAULT_REFERENCE_DATETIME_STR)
      aux_features[xarray_utils.REFERENCE_DATETIME_KEY] = reference_datetime
    else:
      reference_datetime = np.datetime64(reference_datetime_str)
      aux_features[xarray_utils.REFERENCE_DATETIME_KEY] = reference_datetime
  else:  # cannot set ref datetime if already specified in aux_data
    if reference_datetime_str is not None:
      raise ValueError('reference datetime already specified in aux_data')

  if isinstance(model_time_step, (str, scales.Quantity)):
    dt = physics_specs.nondimensionalize(scales.Quantity(model_time_step))
  else:
    dt = model_time_step

  return ModelSpecs(
      coords=coords,
      dt=dt,
      physics_specs=physics_specs,
      aux_features=aux_features,
  )


def _identity(x):
  return x


class DynamicalSystem(hk.Module):
  """Abstract class for modeling dynamical systems."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      input_coords: coordinate_systems.CoordinateSystem,
      output_coords: coordinate_systems.CoordinateSystem,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.coords = coords
    self.dt = dt
    self.physics_specs = physics_specs
    self.aux_features = aux_features
    self.input_coords = input_coords
    self.output_coords = output_coords

  def encode(self, x: PyTreeState, forcing: Forcing) -> PyTreeState:
    """Encodes input trajectory `x` with `forcing` to the model state."""
    raise NotImplementedError('Model subclass did not define encode')

  def decode(self, x: PyTreeState, forcing: Forcing) -> PyTreeState:
    """Decodes a model state `x` with `forcing` to a data representation."""
    raise NotImplementedError('Model subclass did not define decode')

  def advance(self, x: PyTreeState, forcing: Forcing) -> PyTreeState:
    """Returns a model state `x` with `forcing` advanced by `self.dt`."""
    raise NotImplementedError('Model subclass did not define advance')

  def forcing_fn(self, forcing_data: ForcingData, sim_time: Numeric) -> Forcing:
    """Returns forcing at sim_time, possibly using `forcing_data`."""
    raise NotImplementedError('Model subclass did not define forcing_fn')

  def trajectory(
      self,
      x,
      outer_steps: int,
      inner_steps: int = 1,
      *,
      forcing_data: ForcingData,
      start_with_input: bool = False,
      post_process_fn: Callable = _identity,
  ) -> ...:
    """Returns a final model state and trajectory."""

    def step_fn(x: PyTreeState) -> PyTreeState:
      # if x does not have `sim_time`, expect forcing_fn to handle sim_time=None
      if isinstance(x, typing.ModelState):
        sim_time = getattr(x.state, 'sim_time', None)
      else:
        sim_time = getattr(x, 'sim_time', None)
      forcing = self.forcing_fn(forcing_data, sim_time)
      x, forcing = self.coords.with_dycore_sharding((x, forcing))
      y = self.advance(x, forcing)
      y = self.coords.with_dycore_sharding(y)
      return y

    return trajectory_from_step(
        step_fn,
        outer_steps,
        inner_steps,
        start_with_input=start_with_input,
        post_process_fn=post_process_fn,
    )(x)


@gin.configurable
class ModularStepModel(DynamicalSystem):
  """Dynamical model based on independent encoder/decoder/step components."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      input_coords: coordinate_systems.CoordinateSystem,
      output_coords: coordinate_systems.CoordinateSystem,
      advance_module=gin.REQUIRED,
      encoder_module=gin.REQUIRED,
      decoder_module=gin.REQUIRED,
      forcing_module=forcings.NoForcing,
      name: Optional[str] = None,
  ):
    super().__init__(
        coords,
        dt,
        physics_specs,
        aux_features,
        input_coords,
        output_coords,
        name=name,
    )
    self.advance_fn = advance_module(coords, dt, physics_specs, aux_features)
    self.encoder_fn = encoder_module(
        coords, dt, physics_specs, aux_features, input_coords
    )
    self.decoder_fn = decoder_module(
        coords, dt, physics_specs, aux_features, output_coords
    )
    self.forcing_fn = forcing_module(coords, dt, physics_specs, aux_features)

  def encode(self, x: PyTreeState, forcing: Forcing) -> PyTreeState:
    return self.encoder_fn(x, forcing)

  def decode(self, x: PyTreeState, forcing: Forcing) -> PyTreeState:
    return self.decoder_fn(x, forcing)

  def advance(self, x: PyTreeState, forcing: Forcing) -> PyTreeState:
    return self.advance_fn(x, forcing)


@gin.configurable
class StochasticModularStepModel(DynamicalSystem):
  """Dynamical model with modular components and stochasticity.

  This instance of DynamicalSystem works with ModelState
  representation of the model state. The `advance_module` initializes a
  RandomnessModule. This must be compatible with ModelState.
  Since randomness initialization might depend on the timestep at which it is
  evolved, RandomnessModule module is initialized with `num_substeps`.
  """

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      input_coords: coordinate_systems.CoordinateSystem,
      output_coords: coordinate_systems.CoordinateSystem,
      advance_module=gin.REQUIRED,
      encoder_module=gin.REQUIRED,
      decoder_module=gin.REQUIRED,
      forcing_module=forcings.NoForcing,
      name: Optional[str] = None,
  ):
    super().__init__(
        coords,
        dt,
        physics_specs,
        aux_features,
        input_coords,
        output_coords,
        name=name,
    )
    self.advance_fn = advance_module(coords, dt, physics_specs, aux_features)
    self.encoder_fn = encoder_module(
        coords, dt, physics_specs, aux_features, input_coords
    )
    self.decoder_fn = decoder_module(
        coords, dt, physics_specs, aux_features, output_coords
    )
    self.forcing_fn = forcing_module(coords, dt, physics_specs, aux_features)

  def encode(
      self,
      x: DataState,
      forcing: Forcing,
  ) -> ModelState:
    """Encodes model state and creates a new perturbation."""
    model_state = self.encoder_fn(x, forcing=forcing)
    # encoder_fn returns `ModelState` that contains prognostic state
    # and initial values for memory, diagnostics and randomness.
    return self.advance_fn.finalize_state(model_state, forcing)

  def decode(self, x: ModelState, forcing: Forcing) -> typing.Pytree:
    """Returns model state with perturbation component removed."""
    # TODO(langmore) Consider propagating decoding fields so decoder noise at
    # different lead times is correlated.
    return self.decoder_fn(x, forcing=forcing)

  def advance(
      self,
      x: ModelState,
      forcing: Forcing,
  ) -> ModelState:
    """Advances model state."""
    return self.advance_fn(x, forcing)


@gin.configurable(
    allowlist=(
        'checkpoint_step',
        'checkpoint_multistep',
        'checkpoint_post_process',
    )
)
def trajectory_from_step(
    step_fn: Callable,
    outer_steps: int,
    inner_steps: int,
    *,
    start_with_input: bool,
    post_process_fn: Callable,
    checkpoint_step: bool = True,
    checkpoint_multistep: bool = False,
    checkpoint_post_process: bool = True,
) -> Callable:
  """Returns a function that accumulates repeated applications of `step_fn`.

  Compute a trajectory by repeatedly calling `step_fn()`
  `outer_steps * inner_steps` times.

  Args:
    step_fn: function that takes a state and returns state after one time step.
    outer_steps: number of steps to save in the generated trajectory.
    inner_steps: number of repeated calls to step_fn() between saved steps.
    start_with_input: if True, output the trajectory at steps [0, ..., steps-1]
      instead of steps [1, ..., steps].
    post_process_fn: function to apply to trajectory outputs.
    checkpoint_step: whether to use `jax.checkpoint` on `step_fn`.
    checkpoint_multistep: weather to use `jax.checkpoint` on `step_fn` repeated
      steps between outputting observations used in the loss. Multi-step
      checkpointing is off by default; turn it on to trade off ~25% increased
      computed for ~25% less memory usage.
    checkpoint_post_process: whether to use `jax.checkpoint` on
      `post_process_fn`. `checkpoint_post_process` is a no-op if multi-step
      checkpointing is enabled.

  Returns:
    A function that takes an initial state and returns a tuple consisting of:
      (1) the final frame of the trajectory.
      (2) trajectory of length `outer_steps` representing time evolution.
  """
  if checkpoint_step:
    step_fn = hk.remat(step_fn)

  if checkpoint_post_process:
    post_process_fn = hk.remat(post_process_fn)

  if checkpoint_multistep:

    def outer_scan_fn(f, init, xs, length=None):
      return hk.scan(hk.remat(f), init, xs, length=length)

  else:
    outer_scan_fn = hk.scan

  return time_integration.trajectory_from_step(
      step_fn,
      outer_steps,
      inner_steps,
      start_with_input=start_with_input,
      post_process_fn=post_process_fn,
      inner_scan_fn=hk.scan,
      outer_scan_fn=outer_scan_fn,
  )


@gin.configurable(allowlist=('model_cls', 'to_xarray_fn', 'from_xarray_fn'))
class WhirlModel:
  """Class that holds a Haiku model class and xarray conversion methods."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Optional[AuxFeatures] = None,
      input_coords: Optional[coordinate_systems.CoordinateSystem] = None,
      output_coords: Optional[coordinate_systems.CoordinateSystem] = None,
      model_cls: Callable[[], DynamicalSystem] = gin.REQUIRED,
      to_xarray_fn: Optional[Callable[..., xarray.Dataset]] = None,
      from_xarray_fn: Optional[Callable[..., DataState]] = None,
  ):
    """Constructs pre-defined model functions and holds conversion functions.

    Args:
      coords: horizontal and vertical descritization.
      dt: time step of the model.
      physics_specs: object describing the scales and physical constants.
      aux_features: dictionary holding static features that the model may use.
      input_coords: horizontal and vertical descritization of the input data. if
        `None`, uses `coords`. Default `None.
      output_coords: horizontal and vertical descritization for the output data.
        if `None`, uses `coords`. Default `None.
      model_cls: model Haiku class that implements encode/advance/decode fns.
      to_xarray_fn: function that converts decoded data slices to xarray.
      from_xarray_fn: function that extracts data slices from xarray.
    """
    if aux_features is None:
      aux_features = {}
    if input_coords is None:
      input_coords = coords
    if output_coords is None:
      output_coords = coords
    self._coords = coords
    self._data_coords = input_coords  # by data coords we refer to model inputs.
    specs = ModelSpecs(coords, dt, physics_specs, aux_features)
    model_cls = functools.partial(
        model_cls,
        coords=coords,
        dt=dt,
        physics_specs=physics_specs,
        aux_features=aux_features,
        input_coords=input_coords,
        output_coords=output_coords,
    )

    def forcing_fwd(forcing_data, sim_time):
      return model_cls().forcing_fn(forcing_data, sim_time)  # pytype: disable=wrong-keyword-args

    forcing_fn = hk.transform(forcing_fwd).apply
    encode_fwd = lambda x, forcing: model_cls().encode(x, forcing)  # pytype: disable=wrong-keyword-args
    encode_fn = hk.transform(encode_fwd).apply
    decode_fwd = lambda x, forcing: model_cls().decode(x, forcing)  # pytype: disable=wrong-keyword-args
    decode_fn = hk.transform(decode_fwd).apply
    advance_fwd = lambda x, forcing: model_cls().advance(x, forcing)  # pytype: disable=wrong-keyword-args
    advance_fn = hk.transform(advance_fwd).apply
    if to_xarray_fn is not None:
      to_xarray_fn = functools.partial(to_xarray_fn, coords=output_coords)
    self.forcing_fn = forcing_fn
    self.encode_fn = encode_fn
    self.decode_fn = decode_fn
    self.advance_fn = advance_fn
    self.specs = specs
    self.model_cls = model_cls
    self.to_xarray_fn = to_xarray_fn
    self.from_xarray_fn = from_xarray_fn

  @property
  def coords(self) -> coordinate_systems.CoordinateSystem:
    return self._coords

  @property
  def data_coords(self) -> coordinate_systems.CoordinateSystem:
    return self._data_coords

  def init_params(
      self,
      rng: Array,
      input_trajectory: typing.DataState,
      forcing_data: ForcingData,
  ) -> typing.Params:
    """Returns model parameters by initializing encode/advance/decode fn."""

    def fwd(x):
      model = self.model_cls()  # pytype: disable=wrong-keyword-args
      decode = model_utils.with_forcing(
          model.decode, model.forcing_fn, forcing_data
      )
      advance = model_utils.with_forcing(
          model.advance, model.forcing_fn, forcing_data
      )
      encode = model_utils.with_forcing(
          model.encode, model.forcing_fn, forcing_data
      )
      return decode(advance(encode(x)))

    hk_model = hk.transform(fwd)
    return hk_model.init(rng, input_trajectory)


def get_whirl_model(
    data_ds: xarray.Dataset,
    model_config_str: str,
    additional_gin_bindings: Optional[list[str]] = None,
) -> WhirlModel:
  """Returns a configured WhirlModel."""
  if additional_gin_bindings is None:
    additional_gin_bindings = []

  try:
    data_aux_features = xarray_utils.aux_features_from_xarray(data_ds)
  except KeyError:
    data_aux_features = {}

  if 'physics_config_str' in data_ds.attrs:
    physics_config_str = data_ds.attrs['physics_config_str']
  else:
    physics_config_str = ''  # empty string is equivalent to skipping.

  gin.enter_interactive_mode()
  gin.clear_config()
  gin_utils.parse_gin_config(
      physics_config_str,
      model_config_str,
      override_physics_configs_from_data=True,
      gin_bindings=additional_gin_bindings,
  )

  data_coords = coordinate_system_from_dataset(data_ds)
  physics_specs = physics_specifications.get_physics_specs()
  model_specs = get_model_specs(data_coords, physics_specs, data_aux_features)
  return WhirlModel(
      coords=model_specs.coords,
      dt=model_specs.dt,
      physics_specs=model_specs.physics_specs,
      aux_features=model_specs.aux_features,
      input_coords=data_coords,
      output_coords=data_coords,
  )


_ECMWF_CUTOFFS = {
    # On Palmer 2009 (http://shortn/_56HCcQwmSS) page 4, the cutoffs for
    # perturbations are given. Here we translate them to sigma levels.
    # low_cutoffs: (100hPa, 50hPa)
    'low_cutoffs': (0.05, 0.1),  # Will not be accurate over topography.
    # high_cutoffs: (1300m, 300m)
    'high_cutoffs': (0.86, 0.965),
}


def _piecewise_squasher(
    sigma: Array,
    low_cutoffs: Sequence[float],
    high_cutoffs: Sequence[float],
) -> Array:
  """Piecewise linear values used to "squash" values by sigma level.

  See function χ definition at: http://screen/5V3jzU7ZFA4vVJP

  Args:
    sigma: 1-D array of values for sigma levels. Should be in [0, 1].
    low_cutoffs: σ=low_cutoffs[0] is when χ starts linearly increasing from 0.
      σ=low_cutoffs[1] is when χ levels out at 1
    high_cutoffs: σ=high_cutoffs[0] is when χ starts linearly decreasing from 1.
      σ=high_cutoffs[1] is when χ reaches 0.

  Returns:
    Values χ of shape `sigma.shape + (1, 1)` that should be multiplied by
      arrays of shape (n_levels, K, L) to "squash" high/low σ values.
  """
  if sigma.ndim != 1:
    raise ValueError(f'{sigma.shape=} but should have been a 1-D array')
  if len(low_cutoffs) != 2:
    raise ValueError(f'{len(low_cutoffs)=} but should have been 2.')
  if len(high_cutoffs) != 2:
    raise ValueError(f'{len(high_cutoffs)=} but should have been 2.')

  low_func = (sigma - low_cutoffs[0]) / (low_cutoffs[1] - low_cutoffs[0])
  high_func = (high_cutoffs[1] - sigma) / (high_cutoffs[1] - high_cutoffs[0])

  # lower_bound is a function equal to the squasher between
  #   low_cutoffs[0] and high_cutoffs[1].
  # It becomes negative outside that range.
  lower_bound = jnp.minimum(1.0, jnp.minimum(low_func, high_func))
  return jnp.maximum(0.0, lower_bound)[:, jnp.newaxis, jnp.newaxis]
