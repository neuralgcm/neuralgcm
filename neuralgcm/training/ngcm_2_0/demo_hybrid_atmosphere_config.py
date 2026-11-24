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

"""Minimal self-contained TL63 hybrid atmospheric NeuralGCM 2.0 training config.

Sets up a bare-bones `RolloutTrainer` configuration for a TL63 spectral
dynamical core with learned neural parameterizations, constructed directly in
place without external configuration block helpers.

Note: This in-place configuration variant is temporary and will be removed in
the future in favor of modular configuration construction.

* Fixed TL63 model grid (128x64 Gaussian) with 32 equidistant sigma levels and
  an unaliased 190x95 dynamical core grid.
* Three explicit Zarr dataset paths in `get_config`: 37-level pressure ERA5
  (`pressure_era5_path`), surface ERA5 (`surface_era5_path`), and 6-hour
  accumulated precipitation ERA5 (`precip_era5_path`), opened lazily at
  `fdl.build` time.
* Hourly rollout training stages only, optimizing nodal + spectral CRPS on
  37-level pressure, surface, and 6-hour precipitation predictions.
* Plain 200, 500, and 850 hPa pressure-level RMSE and CRPS evaluations.
* No cloud tracers, conservation fixers, multi-resolution downscaling heads,
  station heads, or daily/weekly temporal aggregation diagnostics.
"""

from __future__ import annotations

from collections.abc import Sequence
import functools
import types
from typing import Literal

import coordax as cx
from fiddle.experimental import auto_config
from flax import nnx
import jax_datetime as jdt
from neuralgcm.base import defaults
from neuralgcm.base import parameterizations
from neuralgcm.hybrid import atmosphere
import numpy as np
import optax
from terrax.atmosphere import diagnostics as atmos_diagnostics
from terrax.atmosphere import observation_operators as atmos_observations
from terrax.atmosphere import transforms as atm_tx
from terrax.core import coordinates
from terrax.core import diagnostics
from terrax.core import dynamic_io
from terrax.core import feature_transforms as f_tx
from terrax.core import field_utils
from terrax.core import learned_transforms
from terrax.core import observation_operators
from terrax.core import orographies
from terrax.core import parallelism
from terrax.core import pytree_utils
from terrax.core import random_processes
from terrax.core import scales
from terrax.core import spatial_filters
from terrax.core import spherical_harmonics
from terrax.core import standard_layers
from terrax.core import step_filters
from terrax.core import time_integrators
from terrax.core import towers
from terrax.core import transforms as tx
from terrax.core import typing
from terrax.core import units
from terrax.metrics import aggregation
from terrax.metrics import deterministic_metrics
from terrax.metrics import evaluators
from terrax.metrics import probabilistic_losses
from terrax.metrics import probabilistic_metrics
from terrax.metrics import scaling
from terrax.metrics import weighting
from terrax.training import data_loading
from terrax.training import model_calibrators
from terrax.training import trainer
import xarray


_auto_config = functools.partial(
    auto_config.auto_config,
    experimental_allow_control_flow=True,
    experimental_allow_dataclass_attribute_access=True,
)

DEFAULT_TRAIN_TIME_SLICE: tuple[str, str] = ('1979', '2023')
DEFAULT_EVAL_TIME_SLICE: tuple[str, str] = ('2024', '2025')

CORE_PRESSURE_VARS = (
    'geopotential',
    'temperature',
    'u_component_of_wind',
    'v_component_of_wind',
    'specific_humidity',
)
CORE_SURFACE_VARS = (
    '2m_temperature',
    '2m_dewpoint_temperature',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    'mean_evaporation_rate',
    'mean_sea_level_pressure',
    'top_net_thermal_radiation',
)
PRECIP_VARS = ('total_precipitation_6hr',)
OCEAN_VARS = ('sea_surface_temperature',)
ICE_VARS = ('sea_ice_cover',)
EVAL_PRESSURE_LEVELS = (200, 500, 850)

_DEFAULT_REDUCE_DIMS = (
    'ensemble',
    'batch',
    'timedelta',
    'pressure',
    'sigma',
    'longitude',
    'latitude',
    'longitude_wavenumber',
    'total_wavenumber',
)

# Network output rescaling constants (from N(0, 1) to physical/model scales).
_SIGMA_TENDENCY_SCALAR_STDS = {
    'temperature': 6.8837e-05,
    'u_component_of_wind': 1.4930e-04,
    'v_component_of_wind': 1.6585e-04,
    'log_surface_pressure': 1.0119e-07,
}
_SIGMA_TENDENCY_VERTICAL_STDS = {
    'specific_humidity': (
        (0.05, 0.2, 0.35, 0.45, 0.6, 0.8, 0.95),
        (
            2.4055e-11,
            2.5494e-09,
            1.1891e-08,
            1.6994e-08,
            3.2305e-08,
            3.4642e-08,
            4.1070e-08,
        ),
    ),
}

_SIGMA_CORRECTION_SCALAR_STDS = {
    'temperature': 5.4359e-01,
    'u_component_of_wind': 8.193e00,
    'v_component_of_wind': 3.607e00,
    'log_surface_pressure': 4.2408e-04,
}
_SIGMA_CORRECTION_VERTICAL_STDS = {
    'specific_humidity': (
        (0.05, 0.2, 0.35, 0.45, 0.6, 0.8, 0.95),
        (
            1.0098e-07,
            1.7522e-05,
            1.0776e-04,
            1.9024e-04,
            3.9444e-04,
            5.6515e-04,
            5.6691e-04,
        ),
    ),
}

_PRESSURE_CORRECTION_SCALAR_STDS = {
    'geopotential': 1.8455e03,
    'temperature': 1.1416e00,
    'u_component_of_wind': 2.1377e00,
    'v_component_of_wind': 1.4774e00,
}
_PRESSURE_CORRECTION_VERTICAL_STDS = {
    'specific_humidity': (
        (30.0, 125.0, 300.0, 500.0, 800.0, 1000.0),
        (
            1.0219e-07,
            2.7174e-06,
            7.9600e-05,
            2.5486e-04,
            5.3891e-04,
            6.0280e-04,
        ),
    ),
}

_SURFACE_STATE_MEANS = {
    '2m_temperature': 2.8728e02,
    '2m_dewpoint_temperature': 2.8226e02,
    '10m_u_component_of_wind': -2.9631e-01,
    '10m_v_component_of_wind': 2.6810e-01,
    'mean_evaporation_rate': -1.6172e-05,
    'mean_sea_level_pressure': 1.0123e05,
    'top_net_thermal_radiation': -8.7890e05,
}
_SURFACE_STATE_STDS = {
    '2m_temperature': 1.4892e01,
    '2m_dewpoint_temperature': 1.4962e01,
    '10m_u_component_of_wind': 5.4182e00,
    '10m_v_component_of_wind': 4.5014e00,
    'mean_evaporation_rate': 4.8573e-05,
    'mean_sea_level_pressure': 9.2418e02,
    'top_net_thermal_radiation': 1.5804e05,
}

# Multiplicative loss normalization scales (~inverse 6-hour rollout CRPS).
_SURFACE_LOSS_NORM_SCALES = {
    '2m_temperature': 3.1861e00,
    '2m_dewpoint_temperature': 2.5558e00,
    '10m_u_component_of_wind': 2.8527e00,
    '10m_v_component_of_wind': 2.7250e00,
    'mean_evaporation_rate': 2.6946e05,
    'mean_sea_level_pressure': 1.4007e-02,
    'top_net_thermal_radiation': 4.9186e-05,
}
_PRECIP_LOSS_NORM_SCALES = {
    'total_precipitation_6hr': 5.3027e03,
}
_PRESSURE_LOSS_NORM_PROFILES = {
    'geopotential': (
        (30.0, 125.0, 300.0, 500.0, 800.0, 1000.0),
        (
            1.5570e-02,
            4.6932e-02,
            4.9668e-02,
            5.5386e-02,
            5.6113e-02,
            4.7300e-02,
        ),
    ),
    'specific_humidity': (
        (30.0, 125.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 800.0, 1000.0),
        (
            1.7800e06,
            1.7800e06,
            7.7121e05,
            1.9237e05,
            7.9756e04,
            5.5090e04,
            2.2100e04,
            1.0537e04,
            4.0834e03,
            5.6457e03,
        ),
    ),
    'temperature': (
        (30.0, 125.0, 300.0, 500.0, 800.0, 1000.0),
        (1.6449e00, 4.4300e00, 5.6834e00, 5.9314e00, 4.3163e00, 3.3324e00),
    ),
    'u_component_of_wind': (
        (30.0, 125.0, 300.0, 500.0, 800.0, 1000.0),
        (8.3665e-01, 1.5373e00, 1.2966e00, 1.7882e00, 1.9727e00, 2.3102e00),
    ),
    'v_component_of_wind': (
        (30.0, 125.0, 300.0, 500.0, 800.0, 1000.0),
        (8.5614e-01, 1.5247e00, 1.2742e00, 1.7013e00, 1.9472e00, 2.2228e00),
    ),
}


@_auto_config
def _reconstruct_vertical_fields(
    levels: coordinates.SigmaLevels | coordinates.PressureLevels,
    profiles: dict[str, tuple[Sequence[float], Sequence[float]]],
    interpolation_space: Literal['linear', 'log'] = 'log',
    min_level: float | None = None,
) -> dict[str, cx.Field]:
  """Interpolates 1D vertical reference profiles onto `levels`."""
  fields = {}
  for name, (ref_ticks, ref_values) in profiles.items():
    field = field_utils.reconstruct_1d_field_from_ref_values(
        levels, list(ref_ticks), list(ref_values), interpolation_space
    )
    if min_level is not None:
      field = field_utils.zero_mask_axis_outliers(field, levels, min_level)
    fields[name] = field
  return fields


@_auto_config
def _build_zero_mean_standardizer(
    variables: Sequence[str],
    scalar_stds: dict[str, float],
    vertical_stds: dict[str, tuple[Sequence[float], Sequence[float]]],
    levels: coordinates.SigmaLevels | coordinates.PressureLevels,
) -> tx.StandardizeFields:
  """Builds `StandardizeFields` mapping N(0, 1) network outputs to physical."""
  shifts = {v: cx.field(0.0) for v in variables}
  scales_dict: dict[str, cx.Field] = {}
  vertical_profiles = {}
  for var in variables:
    if var in vertical_stds:
      vertical_profiles[var] = vertical_stds[var]
    else:
      scales_dict[var] = cx.field(scalar_stds[var])
  if vertical_profiles:
    scales_dict.update(
        _reconstruct_vertical_fields(
            levels, vertical_profiles, interpolation_space='log'
        )
    )
  return tx.StandardizeFields(
      shifts=shifts, scales=scales_dict, from_normalized=True
  )


@_auto_config
def _build_normalized_forward_tower(
    *,
    input_shapes: typing.Pytree,
    target_split_axes: dict[str, cx.Coordinate],
    raw_features: tx.TransformABC,
    tower_factory: towers.ForwardTowerFactory,
    concat_dims: tuple[str | cx.Coordinate, ...],
    rngs: nnx.Rngs,
    mesh: parallelism.Mesh,
    independent_axes: tuple[cx.Coordinate, ...] = (),
    post_norm_transforms: Sequence[typing.Transform] = (),
    out_transform: typing.Transform | None = None,
    exclude_regex: str | None = 'cnn|grf',
    feature_sharding_schema: str = 'horizontal',
    result_sharding_schema: str = 'vertical',
) -> learned_transforms.ForwardTowerTransform:
  """Builds a `ForwardTowerTransform` with `StreamNorm` input normalization."""
  feature_struct = tx.TransformABC.output_shapes(raw_features, input_shapes)
  in_normalize = tx.StreamNorm.for_inputs_struct(
      inputs_struct=feature_struct,
      independent_axes=independent_axes,
      epsilon=1e-11,
      exclude_regex=exclude_regex,
      skip_unspecified=True,
  )
  in_transforms = [raw_features, in_normalize, *post_norm_transforms]
  return learned_transforms.ForwardTowerTransform.build_using_factories(
      input_shapes=input_shapes,
      target_split_axes=target_split_axes,
      tower_factory=tower_factory,
      concat_dims=concat_dims,
      inputs_transform=tx.Sequential(in_transforms),
      out_transform=out_transform or tx.Identity(),
      feature_sharding_schema=feature_sharding_schema,
      result_sharding_schema=result_sharding_schema,
      mesh=mesh,
      rngs=rngs,
  )


@_auto_config
def _build_epd_tower_factory(
    hidden_size: int,
    num_process_blocks: int,
    n_hidden_layers: int = 3,
) -> towers.ForwardTowerFactory:
  """Constructs a 1D column-wise EPD MLP `ForwardTower` factory."""
  mlp_factory = functools.partial(
      standard_layers.MlpUniform,
      hidden_size=hidden_size,
      n_hidden_layers=n_hidden_layers,
  )
  net_factory = functools.partial(
      standard_layers.Epd.build_using_factories,
      encode_factory=nnx.Linear,  # pyrefly: ignore[bad-argument-type]
      decode_factory=nnx.Linear,  # pyrefly: ignore[bad-argument-type]
      process_factory=mlp_factory,
      latent_size=hidden_size,
      num_process_blocks=num_process_blocks,
  )
  return functools.partial(
      towers.ForwardTower.build_using_factories,
      inputs_in_dims=('channel',),
      out_dims=('channel',),
      apply_remat=False,
      neural_net_factory=net_factory,
  )


@_auto_config
def _build_grf_features(
    feature_name: str,
    grid: coordinates.LonLatGrid,
    ylm_map: spherical_harmonics.FixedYlmMapping,
    dt: np.timedelta64,
    sim_units: units.SimUnits,
    rngs: nnx.Rngs,
) -> f_tx.RandomnessFeatures:
  """Constructs 20-field multi-scale spherical GRF features."""
  grf = random_processes.VectorizedGaussianRandomField.with_range_of_scales(
      ylm_map=ylm_map,
      dt=dt,
      sim_units=sim_units,
      n_fields=20,
      min_time_hrs=1,
      max_time_hrs=117,
      min_length_km=85,
      max_length_km=10_000,
      power=4.0,
      correlation_time_type=nnx.Param,
      correlation_length_type=nnx.Param,
      rngs=rngs,
  )
  return f_tx.RandomnessFeatures(
      random_process=grf,
      coord=grid,
      feature_name=feature_name,
  )


@_auto_config
def _build_land_sea_ice_embedding(
    *,
    state_shapes: typing.Pytree,
    grid: coordinates.LonLatGrid,
    model_ylm_map: spherical_harmonics.FixedYlmMapping,
    land_sea_mask_ft: typing.Transform,
    embedding_tower_factory: towers.ForwardTowerFactory,
    surf_embedding_axis: cx.Coordinate,
    tanh_clip: typing.Transform,
    mesh: parallelism.Mesh,
    rngs: nnx.Rngs,
) -> learned_transforms.LandSeaIceTowersTransform:
  """Builds blended land, ocean, and sea-ice surface boundary embeddings."""
  # Two small column towers produce `surf_embedding_axis` vectors per grid cell:
  # `sea_embedding` embeds dynamic SST + sea-ice cover over open ocean (filling
  # land NaNs with the spatial mean), while `ice_embedding` embeds lowest-level
  # atmospheric temperature/humidity/winds + sea-ice cover as a surface proxy
  # over land and sea-ice-covered cells. `LandSeaIceTowersTransform` blends them
  # by `land_sea_mask` and `sea_ice_cover` fractions.
  ocean_timed_input = dynamic_io.DynamicInputSlice(
      keys_to_coords={k: grid for k in OCEAN_VARS},
      observation_key='era5:ocean_surface',
  )
  ice_timed_input = dynamic_io.DynamicInputSlice(
      keys_to_coords={k: grid for k in ICE_VARS},
      observation_key='era5:ocean_surface',
  )
  dynamic_sst_ft = f_tx.DynamicInputFeatures(OCEAN_VARS, ocean_timed_input)
  dynamic_sic_ft = f_tx.DynamicInputFeatures(ICE_VARS, ice_timed_input)
  surface_embedding_in_ft = tx.Sequential([
      tx.SelectKeys(['temperature', 'specific_humidity']),
      atm_tx.PrognosticsWithVelocityAndGrads(ylm_map=model_ylm_map),
      tx.Isel({'sigma': -1}),
  ])
  expand_grid_ft_dim = tx.ExpandDims(axis=cx.DummyAxis('feature', 1), loc=grid)
  sea_surface_ft = tx.Sequential([
      dynamic_sic_ft,
      tx.MergeFromSource(dynamic_sst_ft),
      tx.FillNaNs(value='mean'),
      expand_grid_ft_dim,
  ])
  surface_level_proxy_ft = tx.Sequential([
      dynamic_sic_ft,
      tx.MergeFromSource(surface_embedding_in_ft),
      tx.FillNaNs(value=0.0),
      expand_grid_ft_dim,
  ])
  sea_embedding = _build_normalized_forward_tower(
      input_shapes=state_shapes,
      target_split_axes={'surface_embedding': surf_embedding_axis},
      raw_features=sea_surface_ft,
      tower_factory=embedding_tower_factory,
      concat_dims=('feature',),
      rngs=rngs,
      mesh=mesh,
      post_norm_transforms=(tanh_clip,),
      result_sharding_schema='horizontal',
  )
  ice_embedding = _build_normalized_forward_tower(
      input_shapes=state_shapes,
      target_split_axes={'surface_embedding': surf_embedding_axis},
      raw_features=surface_level_proxy_ft,
      tower_factory=embedding_tower_factory,
      concat_dims=('feature',),
      rngs=rngs,
      mesh=mesh,
      post_norm_transforms=(tanh_clip,),
      exclude_regex=None,
      result_sharding_schema='horizontal',
  )
  return learned_transforms.LandSeaIceTowersTransform(
      land_transform=ice_embedding,
      sea_transform=sea_embedding,
      sea_ice_transform=ice_embedding,
      land_sea_mask_transform=land_sea_mask_ft,
      sea_ice_value_transform=dynamic_sic_ft,
      mesh=mesh,
  )


@_auto_config
def _build_timed_data_spec(
    vars_and_coords: Sequence[tuple[Sequence[str], cx.Coordinate]],
    timedelta: coordinates.TimeDelta | None,
) -> dict[str, cx.Coordinate]:
  """Builds per-variable coordinates for `inputs_spec` or `queries_spec`."""
  td_coord = cx.Scalar() if timedelta is None else timedelta
  spec: dict[str, cx.Coordinate] = {}
  for var_names, coord in vars_and_coords:
    for var_name in var_names:
      spec[var_name] = cx.CartesianProduct([td_coord, coord])
  if timedelta is not None:
    spec['time'] = cx.CartesianProduct([td_coord, cx.Scalar()])
  return spec


@_auto_config
def construct_model(
    *,
    mesh: parallelism.Mesh | None = None,
    rng_seed: int = 42,
) -> atmosphere.LearnedParameterizationSpectralDycore:
  """Constructs the TL63 hybrid atmospheric model to be trained.

  Args:
    mesh: Optional `parallelism.Mesh` for sharding model fields and towers.
      Defaults to an unsharded `parallelism.Mesh()`.
    rng_seed: Random seed for initializing parameterizations and features.

  Returns:
    Configured `atmosphere.LearnedParameterizationSpectralDycore` instance.
  """
  if mesh is None:
    mesh = parallelism.Mesh()

  # ---------------------------------------------------------------------------
  # 1. TL63 model/dycore grids, vertical levels, and spherical harmonic maps.
  # ---------------------------------------------------------------------------
  grid = coordinates.LonLatGrid.TL63()
  model_ylm_grid = coordinates.SphericalHarmonicGrid.TL63()
  dycore_grid = coordinates.LonLatGrid.construct(
      longitude_nodes=190,
      latitude_nodes=95,
      latitude_spacing='gauss',
      longitude_offset=0.0,
  )
  dycore_ylm_grid = coordinates.SphericalHarmonicGrid.with_wavenumbers(
      longitude_wavenumbers=63,
  )
  model_levels = coordinates.SigmaLevels.equidistant(32)
  pressure_levels = coordinates.PressureLevels.with_era5_levels()

  model_ylm_map = spherical_harmonics.FixedYlmMapping(
      lon_lat_grid=grid,
      ylm_grid=model_ylm_grid,
      mesh=mesh,
      radius=scales.RADIUS.m,
      basis_as_jax_arrays=True,
  )
  dycore_ylm_map = spherical_harmonics.FixedYlmMapping(
      lon_lat_grid=dycore_grid,
      ylm_grid=dycore_ylm_grid,
      mesh=mesh,
      radius=scales.RADIUS.m,
      basis_as_jax_arrays=True,
  )

  # ---------------------------------------------------------------------------
  # 2. Dynamical core stepping, orography, and spectral filters (TL63 defaults).
  # ---------------------------------------------------------------------------
  rngs = nnx.Rngs(rng_seed)
  sim_units = units.get_si_units()
  model_timestep = np.timedelta64(1, 'h')
  parameterization_dt = np.timedelta64(3600, 's')
  dycore_substeps = 3
  dycore_dt = np.timedelta64(1200, 's')

  model_orography = orographies.ModalOrographyWithCorrection(
      ylm_map=model_ylm_map,
      correction_scale=63.71,
      correction_initializer=nnx.initializers.zeros_init(),
      mem_layout='2d',
      rngs=rngs,
  )
  dycore_orography = orographies.ModalOrographyWithCorrection(
      ylm_map=dycore_ylm_map,
      correction_scale=63.71,
      correction_initializer=nnx.initializers.zeros_init(),
      mem_layout='2d',
      rngs=rngs,
  )
  nodal_orography = orographies.Orography(grid=grid, rngs=rngs)

  smooth_filter = spatial_filters.ExponentialModalFilter.from_timescale(
      ylm_map=dycore_ylm_map,
      dt=dycore_dt,
      timescale='40 minutes',
      order=3,
      cutoff=0.0,
      sim_units=sim_units,
  )
  sharp_filter = spatial_filters.ExponentialModalFilter.from_timescale(
      ylm_map=dycore_ylm_map,
      dt=dycore_dt,
      timescale='1.33333 minutes',
      order=10,
      cutoff=0.4,
      sim_units=sim_units,
  )
  solver_filter = spatial_filters.SequentialModalFilter(
      [sharp_filter, smooth_filter],
      ylm_map=dycore_ylm_map,
  )
  nn_filter = spatial_filters.ExponentialModalFilter.from_timescale(
      ylm_map=model_ylm_map,
      dt=parameterization_dt,
      timescale='4 minutes',
      order=10,
      cutoff=0.4,
      sim_units=sim_units,
  )
  step_filter = step_filters.ModalFixedGlobalMeanFilter(
      ylm_grid=dycore_ylm_grid,
      keys=('log_surface_pressure',),
  )

  # ---------------------------------------------------------------------------
  # 3. Neural network factories, state shapes, and feature transforms.
  # ---------------------------------------------------------------------------
  tracer_names = ('specific_humidity',)
  state_3d_vars = (
      'divergence',
      'vorticity',
      'temperature',
      'specific_humidity',
  )

  epd_tower_factory = _build_epd_tower_factory(
      hidden_size=384, num_process_blocks=5, n_hidden_layers=3
  )
  embedding_tower_factory = _build_epd_tower_factory(
      hidden_size=8, num_process_blocks=1, n_hidden_layers=3
  )
  cnn_tower_factory = functools.partial(
      towers.ForwardTower.build_using_factories,
      inputs_in_dims=('channel', model_levels),
      out_dims=('channel', model_levels),
      apply_remat=True,
      neural_net_factory=functools.partial(
          standard_layers.CnnLevel,
          channels=(64, 64, 64, 64),
          kernel_sizes=5,
      ),
  )

  modal_volume_coord = cx.CartesianProduct([model_levels, model_ylm_grid])
  pressure_volume_coord = cx.CartesianProduct([pressure_levels, grid])
  time_field = cx.field(
      jdt.to_datetime('1970-01-01T00')  # pyrefly: ignore[bad-argument-type]
  )
  time_shape = pytree_utils.shape_structure(time_field)
  state_shapes = {
      k: cx.shape_struct_field(modal_volume_coord) for k in state_3d_vars
  } | {
      'log_surface_pressure': cx.shape_struct_field(model_ylm_grid),
      'time': time_shape,
  }
  era5_assimilation_vars = (
      'u_component_of_wind',
      'v_component_of_wind',
      'temperature',
      'specific_humidity',
  )
  era5_data_shapes = {
      k: cx.shape_struct_field(pressure_volume_coord)
      for k in era5_assimilation_vars
  } | {'time': time_shape}

  tanh_clip = tx.TanhClip(15.0)
  latent_axis = cx.SizedAxis('surface_latent', 8)
  surf_embedding_axis = cx.SizedAxis('embedding_axis', 8)

  compute_gradients_fn = tx.ToModalWithDerivatives(
      ylm_map=model_ylm_map, attenuations=(0.0,)
  )
  land_sea_mask_ft = f_tx.ParamFeatures({'land_sea_mask': grid}, rngs=rngs)
  default_features = {
      'lon_lat': f_tx.LatitudeFeatures(grid),
      'radiation': f_tx.RadiationFeatures(grid=grid),
      'orography': f_tx.OrographyFeatures(nodal_orography),
      'orography_grad': f_tx.OrographyWithGradsFeatures(
          model_orography,
          compute_gradients_transform=compute_gradients_fn,
          include_raw_orography=False,
      ),
      'surface_covariates': land_sea_mask_ft,
      'learned_surface': f_tx.ParamFeatures(
          coords={
              'learned_surface_features': cx.CartesianProduct(
                  [latent_axis, grid]
              )
          },
          param_type=nnx.Param,
          initializer=nnx.initializers.truncated_normal(stddev=1.0),
          rngs=rngs,
      ),
  }

  state_ft = tx.Sequential([
      tx.SelectKeys(state_3d_vars + ('log_surface_pressure',)),
      atm_tx.PrognosticsWithVelocityAndGrads(
          ylm_map=model_ylm_map,
          compute_grads_transform=compute_gradients_fn,
      ),
  ])

  # Vertical 1D CNN column embeddings.
  cnn_embedding_in_ft = tx.Sequential([
      tx.Sequential([
          tx.SelectKeys(list(state_3d_vars), strict=True),
          atm_tx.PrognosticsWithVelocityAndGrads(ylm_map=model_ylm_map),
      ]),
      tx.MergeFromSource(
          atm_tx.PressureFeatures(
              ylm_map=model_ylm_map, levels=model_levels, sim_units=sim_units
          )
      ),
      tx.ExpandDims(axis=cx.DummyAxis('feature', 1), loc=model_levels),
  ])
  cnn_ft = _build_normalized_forward_tower(
      input_shapes=state_shapes,
      target_split_axes={
          f'cnn_embedding_{i}': cx.Scalar() for i in range(32)
      },
      raw_features=cnn_embedding_in_ft,
      tower_factory=cnn_tower_factory,
      concat_dims=('feature',),
      rngs=rngs,
      mesh=mesh,
      out_transform=tx.StandardizeFields(
          shifts=cx.field(0.0),
          scales=cx.field(5.0),
      ),
      exclude_regex='grf',
      feature_sharding_schema='horizontal',
      result_sharding_schema='horizontal',
  )

  land_sea_ice_embedding = _build_land_sea_ice_embedding(
      state_shapes=state_shapes,
      grid=grid,
      model_ylm_map=model_ylm_map,
      land_sea_mask_ft=land_sea_mask_ft,
      embedding_tower_factory=embedding_tower_factory,
      surf_embedding_axis=surf_embedding_axis,
      tanh_clip=tanh_clip,
      mesh=mesh,
      rngs=rngs,
  )

  # ---------------------------------------------------------------------------
  # 4. Learned physics parameterization, assimilation, and observation heads.
  # ---------------------------------------------------------------------------
  learned_physics_raw_ft = tx.Merge(
      default_features
      | {
          'randomness': _build_grf_features(
              'advance_grf',
              grid,
              model_ylm_map,
              parameterization_dt,
              sim_units,
              rngs,
          ),
          'prognostics': state_ft,
          'cnn_embeddings': cnn_ft,
          'land_sea_ice_embedding': land_sea_ice_embedding,
      }
  )
  learned_physics_ft = tx.Sequential([
      learned_physics_raw_ft,
      tx.StreamNorm.for_inputs_struct(
          inputs_struct=tx.TransformABC.output_shapes(
              learned_physics_raw_ft, state_shapes
          ),
          independent_axes=(model_levels,),
          epsilon=1e-11,
          exclude_regex='cnn|grf',
          skip_unspecified=True,
      ),
      tanh_clip,
  ])
  physics_tendency_scale_vars = (
      'u_component_of_wind',
      'v_component_of_wind',
      'temperature',
      'specific_humidity',
      'log_surface_pressure',
  )
  learned_physics = (
      parameterizations.ModalParameterization.build_using_factories(
          input_state_shapes=state_shapes,
          ylm_map=model_ylm_map,
          levels=model_levels,
          volume_field_names=state_3d_vars,
          surface_field_names=('log_surface_pressure',),
          features_transform=learned_physics_ft,
          transform_factory=functools.partial(
              learned_transforms.ForwardTowerTransform.build_using_factories,
              tower_factory=epd_tower_factory,
              concat_dims=(
                  model_levels,
                  latent_axis,
                  surf_embedding_axis,
                  'grf',
              ),
              feature_sharding_schema='horizontal',
              result_sharding_schema='vertical',
              mesh=mesh,
          ),
          tendency_transform=_build_zero_mean_standardizer(
              physics_tendency_scale_vars,
              _SIGMA_TENDENCY_SCALAR_STDS,
              _SIGMA_TENDENCY_VERTICAL_STDS,
              model_levels,
          ),
          modal_filter=nn_filter,
          rngs=rngs,
          mesh=mesh,
      )
  )

  # Assimilation state correction from 37-level pressure ERA5 inputs.
  assimilation_raw_ft = tx.Merge(
      default_features
      | {
          'randomness': _build_grf_features(
              'assimilate_grf',
              grid,
              model_ylm_map,
              parameterization_dt,
              sim_units,
              rngs,
          ),
          'input_state': tx.Sequential([
              tx.SelectKeys(
                  ['geopotential', 'time'], invert=True, strict=False
              ),
              atm_tx.PrognosticsWithVelocityAndGrads(
                  ylm_map=model_ylm_map,
                  compute_grads_transform=compute_gradients_fn,
              ),
          ]),
      }
  )
  assimilation_split_axes = {
      'log_surface_pressure': cx.Scalar(),
      'u_component_of_wind': model_levels,
      'v_component_of_wind': model_levels,
      'temperature': model_levels,
      'specific_humidity': model_levels,
  }
  assimilation_correction = _build_normalized_forward_tower(
      input_shapes=era5_data_shapes,
      target_split_axes=assimilation_split_axes,
      raw_features=assimilation_raw_ft,
      tower_factory=epd_tower_factory,
      concat_dims=(pressure_levels, latent_axis, 'grf'),
      rngs=rngs,
      mesh=mesh,
      independent_axes=(pressure_levels,),
      post_norm_transforms=(tanh_clip,),
      out_transform=_build_zero_mean_standardizer(
          tuple(assimilation_split_axes),
          _SIGMA_CORRECTION_SCALAR_STDS,
          _SIGMA_CORRECTION_VERTICAL_STDS,
          model_levels,
      ),
  )

  # Shared features for 3D pressure and 2D surface observation operators.
  observation_raw_ft = tx.Merge(
      default_features
      | {
          'randomness': _build_grf_features(
              'observe_grf',
              grid,
              model_ylm_map,
              parameterization_dt,
              sim_units,
              rngs,
          ),
          'prognostics': state_ft,
          'sea_ice_embedding': land_sea_ice_embedding,
      }
  )

  # 37-level pressure observation operator.
  pressure_obs_correction = tx.Sequential([
      _build_normalized_forward_tower(
          input_shapes=state_shapes,
          target_split_axes={k: pressure_levels for k in CORE_PRESSURE_VARS},
          raw_features=observation_raw_ft,
          tower_factory=epd_tower_factory,
          concat_dims=(model_levels, latent_axis, surf_embedding_axis, 'grf'),
          rngs=rngs,
          mesh=mesh,
          independent_axes=(model_levels,),
          post_norm_transforms=(tanh_clip,),
          result_sharding_schema='horizontal',
      ),
      _build_zero_mean_standardizer(
          CORE_PRESSURE_VARS,
          _PRESSURE_CORRECTION_SCALAR_STDS,
          _PRESSURE_CORRECTION_VERTICAL_STDS,
          pressure_levels,
      ),
  ])
  pressure_operator = atmos_observations.StandardVariablesObservationOperator(
      ylm_map=model_ylm_map,
      orography=model_orography,
      levels=pressure_levels,
      sim_units=sim_units,
      observation_correction=pressure_obs_correction,
      below_ground_extrapolation='lapse_rate',
      pressure_interpolation_space='log',
      mesh=mesh,
  )

  # Surface and column moisture-budget precipitation observation operators.
  extract_p_plus_e = atmos_diagnostics.ExtractPrecipitationPlusEvaporation(
      ylm_map=model_ylm_map,
      levels=model_levels,
      moisture_species=tracer_names,
      sim_units=sim_units,
  )
  p_plus_e_diagnostic = diagnostics.InstantDiagnostic(
      extract_p_plus_e,
      {'precipitation_plus_evaporation_rate': grid},
  )
  predict_surf_tx = tx.Sequential([
      _build_normalized_forward_tower(
          input_shapes=state_shapes,
          target_split_axes={k: cx.Scalar() for k in CORE_SURFACE_VARS},
          raw_features=observation_raw_ft,
          tower_factory=epd_tower_factory,
          concat_dims=(model_levels, latent_axis, surf_embedding_axis, 'grf'),
          rngs=rngs,
          mesh=mesh,
          independent_axes=(model_levels,),
          post_norm_transforms=(tanh_clip,),
          result_sharding_schema='horizontal',
      ),
      tx.StandardizeFields(
          shifts={
              k: cx.field(_SURFACE_STATE_MEANS[k]) for k in CORE_SURFACE_VARS
          },
          scales={
              k: cx.field(_SURFACE_STATE_STDS[k]) for k in CORE_SURFACE_VARS
          },
          from_normalized=True,
      ),
      atm_tx.ConstrainWaterBudget(
          p_plus_e_diagnostic=p_plus_e_diagnostic,
          var_to_constrain='mean_evaporation_rate',
          precipitation_key='precipitation',
          evaporation_key='mean_evaporation_rate',
      ),
  ])
  surface_operator = observation_operators.TransformObservationOperator(
      tx.Sequential([
          predict_surf_tx,
          tx.MergeFromSource(tx.SelectKeys('time')),
      ])
  )

  precip_and_evap = atmos_diagnostics.ExtractPrecipitationAndEvaporation(
      observation_operator=surface_operator,
      operator_query={'mean_evaporation_rate': grid},
      extract_p_plus_e=extract_p_plus_e,
      prognostics_arg_key='prognostics',
      dt=parameterization_dt,
      evaporation_key='mean_evaporation_rate',
      output_precipitation_key='total_precipitation',
      output_evaporation_key='total_evaporation',
      evaporation_scaling='cumulative',
      precipitation_scaling='cumulative',
      sim_units=sim_units,
  )
  interval_precip_and_evap = diagnostics.IntervalDiagnostic(
      precip_and_evap,
      {'total_precipitation': grid, 'total_evaporation': grid},
      interval={'6hr': np.timedelta64(6, 'h')},
      resolution=np.timedelta64(1, 'h'),
  )
  precip_operator = observation_operators.TransformObservationOperator(
      f_tx.DiagnosticValueFeatures(interval_precip_and_evap)
  )

  operators: dict[str, typing.ObservationOperator] = {
      'era5:pressure': pressure_operator,
      'era5:surface': surface_operator,
      'era5:precip': precip_operator,
  }
  return atmosphere.LearnedParameterizationSpectralDycore(
      model_ylm_map=model_ylm_map,
      dycore_ylm_map=dycore_ylm_map,
      model_levels=model_levels,
      assimilation_sources={
          'pressure': atmosphere.AssimilationSource(
              levels=pressure_levels,
              correction=assimilation_correction,
          ),
      },
      model_timestep=model_timestep,
      aggregate_every=model_timestep,
      parameterization_timestep=parameterization_dt,
      model_orography=model_orography,
      dycore_orography=dycore_orography,
      time_integrator_cls=time_integrators.ImexRk3Sil,
      modal_filter=solver_filter,
      step_filter=step_filter,
      dycore_substeps=dycore_substeps,
      learned_physics=learned_physics,
      tendency_diagnostics={
          'p_plus_e': p_plus_e_diagnostic,
          'precip_and_evap': interval_precip_and_evap,
      },
      remat_substep=False,
      remat_to_aggregate=False,
      remat_advance=True,
      units_mapping=defaults.DEFAULT_UNITS_MAPPING,
      operators=operators,
      sim_units=sim_units,
      tracer_names=tracer_names,
      mesh=mesh,
  )


@_auto_config
def get_config(
    *,
    pressure_era5_path: str,
    surface_era5_path: str,
    precip_era5_path: str,
    experiment_dir: str = '',
    rng_seed: int = 42,
    ensemble_size: int = 2,
    ensemble_shards: int = 2,
    train_unroll_lengths_in_hours: Sequence[int] = (6, 24, 48, 72),
    train_stage_durations: Sequence[int] = (2500, 5000, 7500, 10000),
    train_time_slice: tuple[str, str] = DEFAULT_TRAIN_TIME_SLICE,
    eval_time_slice: tuple[str, str] = DEFAULT_EVAL_TIME_SLICE,
    calibration_time_slice: tuple[str, str] = ('2005-01-01', '2005-01-30'),
    calibration_batch_count: int = 10,
    eval_unroll_length_in_hours: int = 168,
    eval_lead_times_in_hours: Sequence[int] = (12, 24, 48, 72, 168),
    eval_cadence: int = 500,
    num_eval_batches: int = 1,
    time_sample_offset_in_hours: int = 6,
    batch_size_per_device: int = 1,
    shuffle_buffer_size: int = 20 * 1024**3,  # in GiB (20 GiB).
    learning_rate: float = 5e-4,
) -> trainer.RolloutTrainer:
  """Constructs a minimal TL63 hybrid atmospheric `RolloutTrainer` config.

  Args:
    pressure_era5_path: Path to the 37-pressure-level ERA5 Zarr dataset on the
      TL63 grid.
    surface_era5_path: Path to the surface-level ERA5 Zarr dataset on the TL63
      grid (including core surface targets, SST, sea-ice cover, orography, and
      `land_sea_mask`).
    precip_era5_path: Path to the 6-hour accumulated precipitation ERA5 Zarr
      dataset on the TL63 grid (`total_precipitation_6hr`).
    experiment_dir: Output directory for checkpoints and metrics. Can be left
      empty and overridden by `rollout_training` at launch time.
    rng_seed: Random seed for model initialization.
    ensemble_size: Number of ensemble members rolled out per forecast.
    ensemble_shards: SPMD shards along the `ensemble` mesh axis.
    train_unroll_lengths_in_hours: Rollout horizons (multiples of 6 hours) for
      each hourly training curriculum stage.
    train_stage_durations: Number of optimizer steps per training stage.
    train_time_slice: `(start, end)` date range used for training batches.
    eval_time_slice: `(start, end)` date range used for validation batches.
    calibration_time_slice: `(start, end)` date window used by
      `CollectNormalizationStats` before training starts.
    calibration_batch_count: Number of batches used to initialize `StreamNorm`
      statistics.
    eval_unroll_length_in_hours: Rollout horizon in hours for evaluation.
    eval_lead_times_in_hours: Forecast lead times (in hours) at which 200, 500,
      and 850 hPa RMSE and CRPS are reported.
    eval_cadence: Training step interval between online evaluations.
    num_eval_batches: Number of forecast batches evaluated per online eval.
    time_sample_offset_in_hours: Stride in hours between candidate forecast
      initialization times.
    batch_size_per_device: Per-device batch size.
    shuffle_buffer_size: Byte capacity of the training shuffle buffer.
    learning_rate: Peak Adam learning rate.

  Returns:
    Configured `trainer.RolloutTrainer` instance (or `fdl.Config` when invoked
    via `get_config.as_buildable(...)`).
  """
  if len(train_unroll_lengths_in_hours) != len(train_stage_durations):
    raise ValueError(
        f'{train_unroll_lengths_in_hours=} and {train_stage_durations=} must '
        'have the same length.'
    )

  # ---------------------------------------------------------------------------
  # 1. Data/ensemble parallelism mesh and model construction.
  # ---------------------------------------------------------------------------
  spmd_mesh = trainer.create_spmd_mesh(
      ensemble_shards=ensemble_shards,
      z_shards=1,
      x_shards=1,
      y_shards=1,
  )
  batch_and_ensemble: parallelism.DimPartitions = {
      'batch': 'batch',
      'ensemble': 'ensemble',
  }
  field_partitions: parallelism.FieldPartitions = {
      'horizontal': batch_and_ensemble,
      'vertical': batch_and_ensemble,
  }
  mesh = parallelism.Mesh(
      spmd_mesh=spmd_mesh,
      field_partitions=field_partitions,
  )
  model = construct_model(mesh=mesh, rng_seed=rng_seed)

  grid = coordinates.LonLatGrid.TL63()
  model_ylm_grid = coordinates.SphericalHarmonicGrid.TL63()
  pressure_levels = coordinates.PressureLevels.with_era5_levels()
  pressure_volume_coord = cx.CartesianProduct([pressure_levels, grid])

  # ---------------------------------------------------------------------------
  # 2. Nodal + spectral observation processing and CRPS rollout loss.
  # ---------------------------------------------------------------------------
  ylm_mapper = spherical_harmonics.YlmMapper(
      truncation_rule='linear',
      mesh=mesh,
      radius=scales.RADIUS.m,
      basis_as_jax_arrays=True,
  )
  to_modal: typing.Transform = tx.Sequential([
      tx.ToModal(ylm_mapper),
      tx.ShardFields(mesh=mesh, schema='horizontal'),
  ])
  identity: typing.Transform = tx.Identity()
  target_keys = ('era5:pressure', 'era5:surface', 'era5:precip')
  process_map: dict[
      str | types.EllipsisType, typing.Transform | tuple[str, typing.Transform]
  ] = {}
  for k in target_keys:
    process_map[k] = (k, identity)
    process_map[f'modal_{k}'] = (k, to_modal)
  process_observations = tx.NestedTransform(
      process_map,
      default_transform=identity,
  )

  pressure_loss_scales = _reconstruct_vertical_fields(
      pressure_levels,
      _PRESSURE_LOSS_NORM_PROFILES,
      interpolation_space='linear',
      min_level=10.0,
  )
  surface_loss_scales = {
      k: cx.field(_SURFACE_LOSS_NORM_SCALES[k]) for k in CORE_SURFACE_VARS
  }
  precip_loss_scales = {
      k: cx.field(_PRECIP_LOSS_NORM_SCALES[k]) for k in PRECIP_VARS
  }

  area_weighting = weighting.GridAreaWeighting()
  wavenumber_weighting = weighting.SigmoidWavenumberWeighting(
      cutoff_wavenumber=50.0,
      inflection_factor=0.7,
      width_factor=0.1,
  )
  wavenumber_scaler = scaling.WavenumberScaler(power=0.5)
  leadtime_scaler = scaling.GeneralizedLeadTimeScaler(
      base_squared_error_in_hours=18.0,
      asymptotic_squared_error_in_hours=240.0,
      asymptotic_norm=0.04,
      norm_transition_timescale_in_hours=36.0,
      norm_transition_power=1.0,
  )

  pressure_loss_evaluator = evaluators.Evaluator(
      metrics={
          'crps': probabilistic_losses.CRPS({
              'geopotential': 0.85,
              'specific_humidity': 0.4,
          })
      },
      getters={'crps': tx.ScaleFields(pressure_loss_scales)},
      aggregators={
          'crps': aggregation.Aggregator(
              dims_to_reduce=_DEFAULT_REDUCE_DIMS,
              weight_by=[area_weighting, wavenumber_weighting],
              scale_by=[wavenumber_scaler, leadtime_scaler],
          )
      },
  )
  surface_loss_evaluator = evaluators.Evaluator(
      metrics={
          'crps': probabilistic_losses.CRPS({
              'mean_evaporation_rate': 0.5,
              '10m_u_component_of_wind': 0.35,
              '10m_v_component_of_wind': 0.35,
              '2m_dewpoint_temperature': 0.3,
              'top_net_thermal_radiation': 0.25,
          })
      },
      getters={'crps': tx.ScaleFields(surface_loss_scales)},
      aggregators={
          'crps': aggregation.Aggregator(
              dims_to_reduce=_DEFAULT_REDUCE_DIMS,
              weight_by=[area_weighting, wavenumber_weighting],
              scale_by=[wavenumber_scaler, leadtime_scaler],
          )
      },
  )
  precip_t0_mask_scaler = scaling.PerVariableScaler(
      {
          'total_precipitation_6hr': scaling.CoordinateMaskScaler(
              mask_coord=coordinates.TimeDelta(
                  [np.timedelta64(0, 'h')]  # pyrefly: ignore[bad-argument-type]
              ),
              masked_value=0.0,
              unmasked_value=1.0,
          )
      },
      default_scaler=scaling.ConstantScaler(cx.field(1.0)),
  )
  precip_loss_evaluator = evaluators.Evaluator(
      metrics={'crps_with_time_mask': probabilistic_losses.CRPS({})},
      getters={'crps_with_time_mask': tx.ScaleFields(precip_loss_scales)},
      aggregators={
          'crps_with_time_mask': aggregation.Aggregator(
              dims_to_reduce=_DEFAULT_REDUCE_DIMS,
              weight_by=[area_weighting, wavenumber_weighting],
              scale_by=[
                  wavenumber_scaler,
                  leadtime_scaler,
                  precip_t0_mask_scaler,
              ],
              skipna=True,
              keep_weights_for_nans=True,
          )
      },
  )

  loss = evaluators.NestedEvaluators(
      {
          'era5:pressure': pressure_loss_evaluator,
          'modal_era5:pressure': pressure_loss_evaluator,
          'era5:surface': surface_loss_evaluator,
          'modal_era5:surface': surface_loss_evaluator,
          'era5:precip': precip_loss_evaluator,
          'modal_era5:precip': precip_loss_evaluator,
      },
      evaluator_weights={
          'era5:pressure': 1.0,
          'modal_era5:pressure': 1.0,
          'era5:surface': 1.0,
          'modal_era5:surface': 1.0,
          'era5:precip': 2.0,
          'modal_era5:precip': 2.0,
      },
      default_evaluator=evaluators.Evaluator.empty(),
  )

  # ---------------------------------------------------------------------------
  # 3. Hourly training schedule and plain 200, 500, 850 hPa eval schedule.
  # ---------------------------------------------------------------------------
  time_sample_offset = np.timedelta64(time_sample_offset_in_hours, 'h')
  train_queries_spec = {
      'era5:pressure': _build_timed_data_spec(
          [(CORE_PRESSURE_VARS, pressure_volume_coord)], timedelta=None
      ),
      'era5:surface': _build_timed_data_spec(
          [(CORE_SURFACE_VARS, grid)], timedelta=None
      ),
      'era5:precip': _build_timed_data_spec(
          [(PRECIP_VARS, grid)], timedelta=None
      ),
  }

  train_stages = []
  for unroll_hours, duration in zip(
      train_unroll_lengths_in_hours, train_stage_durations
  ):
    if unroll_hours <= 0 or unroll_hours % 6 != 0:
      raise ValueError(
          f'Hourly unroll length must be a positive multiple of 6, got '
          f'{unroll_hours}.'
      )
    if duration <= 0:
      raise ValueError(
          f'Train stage duration must be positive, got {duration}.'
      )
    td = coordinates.TimeDelta([  # pyrefly: ignore[bad-argument-type]
        np.timedelta64(6 * step, 'h') for step in range(unroll_hours // 6 + 1)
    ])
    stage_inputs_spec = {
        'era5:pressure': _build_timed_data_spec(
            [(CORE_PRESSURE_VARS, pressure_volume_coord)], timedelta=td
        ),
        'era5:surface': _build_timed_data_spec(
            [(CORE_SURFACE_VARS, grid)], timedelta=td
        ),
        'era5:precip': _build_timed_data_spec(
            [(PRECIP_VARS, grid)], timedelta=td
        ),
    }
    stage_dynamic_inputs_spec = {
        'era5:ocean_surface': _build_timed_data_spec(
            [(OCEAN_VARS + ICE_VARS, grid)], timedelta=td
        ),
    }
    train_stages.append(
        trainer.TrainStage(
            duration=duration,
            inputs_spec=stage_inputs_spec,
            dynamic_inputs_spec=stage_dynamic_inputs_spec,
            queries_spec=train_queries_spec,
            loss=loss,
            time_sample_offset=time_sample_offset,
            train_time_slice=train_time_slice,
            eval_time_slice=eval_time_slice,
            batch_size_per_device=batch_size_per_device,
            shuffle_buffer_size=shuffle_buffer_size,
        )
    )
  train_schedule = trainer.TrainSchedule(stages=train_stages)

  # Plain 200, 500, 850 hPa pressure-level RMSE and CRPS evaluations.
  if eval_unroll_length_in_hours <= 0 or eval_unroll_length_in_hours % 6 != 0:
    raise ValueError(
        'eval_unroll_length_in_hours must be a positive multiple of 6, got '
        f'{eval_unroll_length_in_hours}.'
    )
  eval_metrics = {}
  eval_getters = {}
  eval_aggregators = {}
  rmse_metric = deterministic_metrics.RMSE()
  crps_metric = probabilistic_metrics.CRPS()
  for level in EVAL_PRESSURE_LEVELS:
    sel_level = tx.Sel({'pressure': level})
    for lead_hours in eval_lead_times_in_hours:
      if lead_hours > eval_unroll_length_in_hours:
        raise ValueError(
            f'Evaluation lead time {lead_hours}h exceeds '
            f'{eval_unroll_length_in_hours=}.'
        )
      lead_mask = weighting.CoordinateMaskWeighting(
          coordinates.TimeDelta([  # pyrefly: ignore[bad-argument-type]
              np.timedelta64(lead_hours, 'h'),
          ]),
          masked_value=1.0,
          unmasked_value=0.0,
      )
      agg = aggregation.Aggregator(
          dims_to_reduce=_DEFAULT_REDUCE_DIMS,
          weight_by=[area_weighting, lead_mask],
      )
      for metric_name, metric in (('rmse', rmse_metric), ('crps', crps_metric)):
        key = f'{metric_name}_P{level}_{lead_hours}h'
        eval_metrics[key] = metric
        eval_getters[key] = sel_level
        eval_aggregators[key] = agg

  eval_td = coordinates.TimeDelta([  # pyrefly: ignore[bad-argument-type]
      np.timedelta64(6 * step, 'h')
      for step in range(eval_unroll_length_in_hours // 6 + 1)
  ])
  eval_schema = trainer.EvalSchema(
      cadence=eval_cadence,
      inputs_spec={
          'era5:pressure': _build_timed_data_spec(
              [(CORE_PRESSURE_VARS, pressure_volume_coord)],
              timedelta=eval_td,
          ),
      },
      dynamic_inputs_spec={
          'era5:ocean_surface': _build_timed_data_spec(
              [(OCEAN_VARS + ICE_VARS, grid)], timedelta=eval_td
          ),
      },
      queries_spec={
          'era5:pressure': _build_timed_data_spec(
              [(CORE_PRESSURE_VARS, pressure_volume_coord)], timedelta=None
          ),
      },
      metrics_evaluator=evaluators.NestedEvaluators(
          {
              'era5:pressure': evaluators.Evaluator(
                  metrics=eval_metrics,
                  getters=eval_getters,
                  aggregators=eval_aggregators,
              )
          },
          default_evaluator=evaluators.Evaluator.empty(),
      ),
      batch_size_per_device=batch_size_per_device,
      num_batches=num_eval_batches,
      time_sample_offset=time_sample_offset,
      train_time_slice=train_time_slice,
      eval_time_slice=eval_time_slice,
      name='eval',
  )
  eval_schedule = trainer.EvalSchedule(stages=[eval_schema])

  # ---------------------------------------------------------------------------
  # 4. Calibrators, explicit ERA5 DataLoader, optimizer, and RolloutTrainer.
  # ---------------------------------------------------------------------------
  pressure_era5 = xarray.open_zarr(pressure_era5_path, chunks=None)
  surface_era5 = xarray.open_zarr(surface_era5_path, chunks=None)
  precip_era5 = xarray.open_zarr(precip_era5_path, chunks=None)

  sim_units = units.get_si_units()
  data_ylm_map = spherical_harmonics.FixedYlmMapping(
      lon_lat_grid=grid,
      ylm_grid=model_ylm_grid,
      mesh=mesh,
      radius=scales.RADIUS.m,
      basis_as_jax_arrays=True,
  )
  calibrators = [
      model_calibrators.UpdateSubmodulesFromXarray(
          update_specs={
              (orographies.ModalOrography, 'model_orography'): surface_era5,
              (orographies.ModalOrography, 'dycore_orography'): surface_era5,
              (f_tx.OrographyFeatures, '.orography'): surface_era5,
              (f_tx.ParamFeatures, '.surface_covariates'): surface_era5,
          },
          kwargs={
              'sim_units': sim_units,
              'data_ylm_map': data_ylm_map,
          },
      ),
      model_calibrators.CollectNormalizationStats(
          train_stage_idx=0,
          rng_seed=0,
          time_start=calibration_time_slice[0],
          time_end=calibration_time_slice[1],
          batch_count=calibration_batch_count,
      ),
  ]

  data_loader = data_loading.DataLoader(
      all_data={
          'era5:pressure': pressure_era5,
          'era5:surface': surface_era5,
          'era5:precip': precip_era5,
          'era5:ocean_surface': surface_era5,
      },
      parallelism_mesh=mesh,
      loading_partition_schema='horizontal',
  )

  lr_schedule = optax.exponential_decay(
      init_value=learning_rate,
      transition_steps=10_000,
      transition_begin=15_000,
      decay_rate=0.5,
  )
  optimizer = optax.chain(
      optax.scale_by_adam(b1=0.9, b2=0.95, eps=1e-6),
      optax.scale_by_learning_rate(lr_schedule),
  )

  return trainer.RolloutTrainer(
      experiment_dir=experiment_dir,
      model=model,
      data_loader=data_loader,
      process_observations=process_observations,
      calibration_modules=calibrators,
      train_schedule=train_schedule,
      eval_schedule=eval_schedule,
      optimization_config=trainer.OptimizationConfig(
          optimizer=optimizer,
          ema_num_steps=100,
      ),
      initial_checkpoint=trainer.InitialCheckpoint(),
      checkpoint_config=trainer.CheckpointConfig(
          save_interval_steps=500,
          keep_every_n_steps=1000,
          model_config_str='',
      ),
      auto_restart_config=trainer.AutoRestartConfig(
          max_nan_restarts=0,
          restart_lookback_steps=2000,
          error_with_nan_loss=True,
      ),
      remat_config=trainer.RematConfig(
          step_policy='remat',
          observe_policy='remat',
          process_policy='remat',
          remat_collect_statistics=False,
      ),
      ensemble_axis=cx.SizedAxis('ensemble', ensemble_size),
      online_metrics_saver=trainer.JsonMetricsSaver(metrics_dir=''),
  )
