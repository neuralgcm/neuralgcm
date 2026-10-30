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

"""Hybrid atmospheric model with a spectral dycore and learned physics."""

from collections.abc import Callable, Sequence
from typing import Any, Literal

import coordax as cx
from flax import nnx
import jax
import jax.numpy as jnp
import jax_datetime as jdt
from neuralgcm.base import defaults
from neuralgcm.base import model
from neuralgcm.base import operator_utils
from neuralgcm.base import stepping
from neuralgcm.base import time_utils
import numpy as np
from terrax.atmosphere import equations as atm_equations
from terrax.atmosphere import fixers
from terrax.atmosphere import state_conversion
from terrax.atmosphere import transforms as atm_transforms
from terrax.core import coordinates
from terrax.core import data_specs
from terrax.core import diagnostics
from terrax.core import equations
from terrax.core import interpolators
from terrax.core import module_utils
from terrax.core import observation_operators
from terrax.core import orographies
from terrax.core import parallelism
from terrax.core import random_processes
from terrax.core import spatial_filters
from terrax.core import spherical_harmonics
from terrax.core import step_filters
from terrax.core import time_integrators
from terrax.core import transforms
from terrax.core import typing

# pylint: disable=g-classes-have-attributes


def _check_coordinates(
    fields: dict[str, cx.Field],
    expected: dict[str, cx.Field],
    description: str,
) -> None:
  """Raises if `fields` and `expected` differ in coordinates of common keys."""
  for k, v in fields.items():
    if k not in expected or not cx.is_field(v):
      continue
    expected_coord = expected[k].coordinate
    if v.coordinate != expected_coord:
      raise ValueError(
          f'{description} {k!r} has coordinate {v.coordinate}, '
          f'expected {expected_coord}.'
      )


DycoreEquation = (
    time_integrators.ImplicitExplicitODE
    | time_integrators.SemiLagrangianImplicitExplicitODE
)


AssimilationLevels = (
    coordinates.PressureLevels
    | coordinates.SigmaLevels
    | coordinates.HybridLevels
)


RandomnessAdvanceStep = Literal['substep', 'aggregation', 'advance', 'never']
_RANDOMNESS_ADVANCE_STEPS = ('substep', 'aggregation', 'advance', 'never')


@nnx.dataclass
class AssimilationSource(nnx.Module):
  """Module specifying the structure of input data and optional transforms.

  Attributes:
    levels: Vertical levels (or minimum subset) on which inputs are expected.
    grid: Optional LonLatGrid of the inputs. Defaults to the model_grid if None.
    correction: Optional transform that computes corrections to the model state
      from nondimensional inputs on `(levels, grid)`. Corrections are nodal
      fields on `(model_levels, model_grid)` keyed by model state variables,
      except that wind corrections may be given as `u_component_of_wind` and
      `v_component_of_wind` (both required), which are converted to
      `divergence` and `vorticity` corrections like the prior state itself.
    correction_timedelta: Optional timedeltas of inputs passed to `correction`,
      e.g. `TimeDelta(np.array([-6, 0], 'timedelta64[h]'))` to provide context
      from 6 hours prior. If set, inputs must include these timedeltas, which
      `correction` receives along the `timedelta` axis, while the model state
      prior is still derived from timedelta=0. Must include 0. `time` is always
      passed at timedelta=0. If None, `correction` receives inputs at
      timedelta=0 without a `timedelta` axis. Defaults to None.
    regridder: Optional regridder or transform applied to `timedelta=0` inputs
      before state conversion. Required when `grid` differs from the model
      `grid`, and must produce fields on the model `grid`. `correction` receives
      inputs on original `(levels, grid)` coordinates.
    filter_state: Optional transform applied to the model state derived from
      this source, e.g. to remove grid-scale noise introduced by regridding,
      state conversion or `correction`, which can otherwise cause an
      initialization shock. Receives the nondimensional prognostics (including
      `time`) that would be set without it, i.e. after `correction` and
      wavenumber clipping, and must return the same keys on the same
      coordinates. Outputs are clipped again.
    data_key: Dataset key in `inputs` for the vertical-level variables. If None,
      defaults to `'era5:pressure'`, `'era5:sigma'` or `'era5:hybrid'` based on
      the type of `levels`.
    surface_data_key: Dataset key in `inputs` for `surface_pressure` when
      `levels` is `SigmaLevels` or `HybridLevels` (must be None for
      `PressureLevels`). If None, defaults to `'era5:surface'` for sigma and
      hybrid levels.
  """

  levels: AssimilationLevels
  grid: coordinates.LonLatGrid | None = None
  correction: typing.Transform | None = None
  correction_timedelta: coordinates.TimeDelta | None = None
  regridder: interpolators.BaseRegridder | typing.Transform | None = None
  filter_state: typing.Transform | None = None
  data_key: str | None = nnx.static(default=None)
  surface_data_key: str | None = nnx.static(default=None)

  def __post_init__(self):
    if isinstance(self.levels, coordinates.PressureLevels):
      if self.surface_data_key is not None:
        raise ValueError(
            '`surface_data_key` cannot be set for `PressureLevels`, got '
            f'{self.surface_data_key!r}.'
        )
      default_data_key, default_surface_key = 'era5:pressure', None
    elif isinstance(self.levels, coordinates.SigmaLevels):
      default_data_key, default_surface_key = 'era5:sigma', 'era5:surface'
    elif isinstance(self.levels, coordinates.HybridLevels):
      default_data_key, default_surface_key = 'era5:hybrid', 'era5:surface'
    else:
      raise ValueError(f'Unsupported assimilation levels: {self.levels}')
    if self.data_key is None:
      self.data_key = default_data_key
    if self.surface_data_key is None:
      self.surface_data_key = default_surface_key
    if self.correction_timedelta is None:
      return
    if self.correction is None:
      raise ValueError('`correction_timedelta` requires a `correction`.')
    if np.timedelta64(0, 's') not in self.correction_timedelta.deltas:
      raise ValueError(
          '`correction_timedelta` must include 0, got '
          f'{self.correction_timedelta.deltas}.'
      )


@nnx.dataclass(kw_only=True)
class LearnedParameterizationSpectralDycore(model.Model):
  """Hybrid atmospheric model with a spectral dycore and learned physics.

  This model combines a spectral dynamical core (primitive equations) with
  learned machine learning parameterizations. It operates on a model grid
  and can interface with data on a different (data) grid.

  The model state can be initialized via `assimilate` from one of:
    * `inputs[f'nondim_{prognostics_data_key}']`: nondimensional prognostic
      variables matching `self.prognostics`, which are set as is.
    * `inputs[prognostics_data_key]`: same as above, but in physical units,
      which are nondimensionalized using `nondim_transform`. The exception is
      `log_surface_pressure`, which is always expected to be the log of the
      nondimensional surface pressure.
    * Data matching one of `assimilation_sources`, which is regridded to the
      model `grid` if `source.grid` differs, converted to the model state,
      optionally corrected by `source.correction` (which receives inputs on
      `(source.levels, source.grid)`) and optionally filtered by
      `source.filter_state`. Vertical-level variables are read from
      `source.data_key` (defaulting to `'era5:pressure'`, `'era5:sigma'` or
      `'era5:hybrid'` by level type), and `surface_pressure` for sigma and
      hybrid sources is read from `source.surface_data_key` (defaulting to
      `'era5:surface'`).
  Prognostic inputs take precedence over assimilation sources. If data for
  multiple assimilation sources is present, the first one in
  `assimilation_source_preference_order` (defaulting to the key order of
  `assimilation_sources`) is used. All inputs can optionally include a
  `timedelta` axis, in which case the `timedelta=0` slice is used for
  initialization. Inputs of sources with `correction_timedelta` must include
  these timedeltas, which are passed to the source's `correction` as additional
  context. Since `inputs_spec` lists all alternatives as optional, validation
  that one of them is complete happens in `assimilate`.

  The model advances in time by `model_timestep` in `num_aggregation_steps`
  intervals of `aggregate_every`, each consisting of `num_substeps` steps of
  `parameterization_timestep`. Within each `parameterization_timestep`:
    1. `step_start_diagnostics` are evaluated on the current `prognostics`.
    2. `learned_physics(prognostics)` is called with the current `prognostics`
       (including `'time'`). Its outputs must match `prognostics` (all keys
       except `'time'`, as nondimensional fields on the same coordinates as the
       corresponding entries in `prognostics`).
    3. `tendency_diagnostics` are evaluated on the returned tendencies, and any
       `tendency_fixers` are applied in sequence (each querying current
       diagnostics via `fixer.diagnostics_query`) to adjust the tendencies prior
       to dynamical core integration.
    4. `learned_physics` tendencies are held fixed while the dynamical core
       integrates the state over `parameterization_timestep` in
       `dycore_substeps` steps (applying `step_filter` and `modal_filter` after
       each dycore step). When `dycore_ylm_map` differs from `model_ylm_map`,
       the state and tendencies are regridded to the dycore grid before
       integration and the updated state is regridded back to the model grid
       afterward; when `dycore_ylm_map == model_ylm_map`, regridding is skipped.
       The dycore can be Eulerian (e.g. `PrimitiveEquations` integrated with
       `ImexRk3Sil`) or semi-Lagrangian (e.g.
       `SemiLagrangianPrimitiveEquations` integrated with
       `SemiLagrangianCrankNicolsonRK2`), in which case the tendencies act as
       non-advective forcing applied along trajectories.
    5. Any `state_fixers` are applied in sequence (querying diagnostics via
       `fixer.diagnostics_query`), `'time'` is incremented by
       `parameterization_timestep`, and `step_end_diagnostics` are evaluated.

  Prognostic variables other than `time` are modal fields on `ylm_grid`, except
  for `nodal_tracers`. Using `nodal_tracers` requires a semi-Lagrangian dycore
  (e.g. `SemiLagrangianPrimitiveEquations`, which transports nodal tracers on
  `grid`) and matching `model_ylm_map` and `dycore_ylm_map`. Nodal tracers
  never make a spectral round trip and are excluded from `modal_filter` and
  from wavenumber clipping at assimilation. Other components that receive
  prognostics or tendencies, i.e. `learned_physics`, `step_filter`,
  `source.filter_state`, fixers, diagnostics and `operators`, must process
  them accordingly.

  Diagnostics are evaluated at different stages of the time stepping, which
  determines both the values they see and their alignment with `prognostics`
  when observed after `advance`:
    * `step_start_diagnostics`: evaluated on the prognostics at the start of
      every `parameterization_timestep`, before `learned_physics`, hence lagging
      from time in `prognostics` by one `parameterization_timestep`.
    * `tendency_diagnostics`: evaluated on `learned_physics` tendencies (with
      prognostics from the start of the step passed as `prognostics` kwarg)
      every `parameterization_timestep`. Typically used to capture rates,
      increments or other tendency-dependent quantities. When observed, values
      correspond to the interval ending at the time of `prognostics`.
      Diagnostics are evaluated in insertion order, so later entries may observe
      values of earlier ones within the same step.
    * `step_end_diagnostics`: evaluated on the prognostics at the end of every
      `parameterization_timestep`, after `state_fixers` (when provided).
    * `aggregation_diagnostics`: evaluated on the prognostics every
      `aggregate_every`. Aligned with `prognostics` when observed. Used for
      averages, accumulations and histories.
  All diagnostics are called as `diagnostic(inputs, prognostics=prognostics)`,
  where `inputs` are the tendencies for `tendency_diagnostics` and the
  prognostics otherwise. Clocks of temporal diagnostics are advanced by the
  model with an explicit `timedelta`, which is `parameterization_timestep` for
  the first three collections and `aggregate_every` for the last, hence
  diagnostics should not specify `default_timedelta`.

  External forcings and time-dependent covariates (e.g. sea surface temperature
  or sea ice cover) used by `learned_physics` or observation operators are
  represented as `DynamicInputModule` submodules anywhere in the model tree and
  are updated via `update_dynamic_inputs`. Random processes
  (`RandomProcessModule` submodules on the model, initialized via
  `initialize_random_processes`) are advanced at the stage specified by
  `advance_randomness_every` (defaulting to `'substep'`, i.e. at the end of
  every `parameterization_timestep`), or at per-module stages specified in
  `randomness_advance_overrides` (where `'never'` excludes a module from
  advancing during `advance`).

  Observations are computed via `observe(queries)`, which dispatches each query
  key to `operators` (or to the built-in `'diagnostics'` operator exposing all
  diagnostic collections) and returns the resulting fields in physical units
  per `units_mapping` (unless the query key is prefixed with `'nondim_'`), see
  `model.Model`.

  Arguments:
    model_ylm_map: Spherical harmonics mapping for the model grid.
    model_levels: Vertical levels for the model simulation.
    model_orography: Orography used on the model grid.
    model_timestep: Time step for the model outer loop.
    modal_filter: Spatial filter applied in spectral space after each dycore
      step, except to nodal tracers.
    learned_physics: The ML parameterization that maps prognostics to
      tendencies, see the contract on its outputs above.
    dycore_ylm_map: Optional spherical harmonics mapping for the dycore grid.
      Defaults to `model_ylm_map` if None.
    dycore_orography: Optional orography used on the dycore grid. Defaults to
      `model_orography` when `dycore_ylm_map == model_ylm_map`; required when
      `dycore_ylm_map` differs from `model_ylm_map`.
    regridder_cls: Factory for regridders between model and dycore spherical
      harmonic grids, called with the target grid when `dycore_ylm_map` differs
      from `model_ylm_map` (regridding is skipped when model and dycore grids
      match).
    tracer_names: Names of tracers included in the model. Defaults to
      `('specific_humidity',)`.
    nodal_tracers: Subset of `tracer_names` carried on the nodal `grid` instead
      of `ylm_grid`. Requires a semi-Lagrangian dycore and matching model and
      dycore grids.
    parameterization_timestep: Time interval between `learned_physics`
      evaluations, over which the state is advanced with fixed tendencies. Must
      divide `aggregate_every`. Defaults to `aggregate_every` if None.
    aggregate_every: Interval at which aggregate diagnostics are collected. Must
      divide `model_timestep`. Defaults to `model_timestep` if None.
    dycore_substeps: Number of dycore steps per `parameterization_timestep`.
      Defaults to 1.
    dycore_equation_cls: Factory for the dycore equation, called with `ylm_map`,
      `levels`, `sim_units`, `reference_temperatures`, `orography_module`,
      `tracer_names`, `nodal_tracers` (when non-empty) and
      `**extra_dycore_kwargs`. Can be Eulerian (e.g. `PrimitiveEquations`) or
      semi-Lagrangian (e.g. `SemiLagrangianPrimitiveEquations`); must be
      semi-Lagrangian when `nodal_tracers` is non-empty.
    time_integrator_cls: Factory for the dycore time integrator, called with the
      equation and the dycore time step. Defaults to `ImexRk3Sil`. Must match
      the type of the dycore equation, e.g. `SemiLagrangianCrankNicolsonRK2` for
      semi-Lagrangian ones.
    step_filter: Filter applied after each dycore step. Defaults to
      `step_filters.NoFilter()` if None.
    reference_temperatures: Reference temperatures for the semi-implicit scheme.
      Defaults to `atm_equations.get_reference_temperature(model_levels)` if
      None.
    extra_dycore_kwargs: Additional keyword arguments passed to
      `dycore_equation_cls`.
    sim_units: Simulation units helper. Defaults to `units.SI_UNITS`.
    units_mapping: Physical units of variables crossing the model boundary.
      Defaults to `defaults.DEFAULT_UNITS_MAPPING`; `tracer_names` are always
      added as dimensionless.
    assimilation_sources: Dict mapping source names to `AssimilationSource`
      instances specifying input coordinates, dataset keys and optional learned
      corrections and state filters from which the model state can be derived.
      Each source must have a distinct `data_key`. May be empty if the model is
      initialized only from prognostic inputs.
    assimilation_source_preference_order: Optional sequence of keys in
      `assimilation_sources` specifying the priority order when data for
      multiple assimilation sources is present. If None, the key order of
      `assimilation_sources` is used.
    prognostics_data_key: Dataset key in `inputs` from which the model state is
      set directly. Nondimensional values are read from `nondim_` prefixed key.
    operators: Dict of observation operators for querying model state. Defaults
      to `primitive_equations_state_operator(model_ylm_map)` keyed by
      `prognostics_data_key` when empty.
    step_start_diagnostics: Diagnostics evaluated on prognostics at the start of
      every `parameterization_timestep`, see above.
    tendency_diagnostics: Diagnostics evaluated on `learned_physics` tendencies
      every `parameterization_timestep`, see above.
    step_end_diagnostics: Diagnostics evaluated on prognostics at the end of
      every `parameterization_timestep`, see above.
    aggregation_diagnostics: Diagnostics evaluated on prognostics every
      `aggregate_every`, see above.
    tendency_fixers: Modules to fix tendencies (e.g. conservation), using values
      of `tendency_diagnostics` from the same step.
    state_fixers: Modules to fix the state after the dycore step, using values
      of `step_start_diagnostics` from the start of the step.
    advance_randomness_every: Stage at which `RandomProcessModule` submodules
      not listed in `randomness_advance_overrides` are advanced: `'substep'`
      (every `parameterization_timestep`), `'aggregation'` (every
      `aggregate_every`), `'advance'` (every `model_timestep`) or `'never'`.
    randomness_advance_overrides: Dict mapping `RandomnessAdvanceStep` stages to
      sequences of `RandomProcessModule` submodules of the model to advance at
      that stage instead of `advance_randomness_every` (with `'never'` excluding
      them from advancing).
    remat_substep: Whether to use rematerialization for substeps.
    remat_to_aggregate: Whether to use rematerialization for the evolution to
      the next aggregation time, including the aggregation diagnostics.
      Preferred over `remat_advance` when `model_timestep > aggregate_every`,
      since it keeps only the scan carry live across aggregation steps.
    remat_advance: Whether to use rematerialization for advance.
  """

  # Core required configuration.
  model_ylm_map: spherical_harmonics.FixedYlmMapping
  model_levels: coordinates.SigmaLevels | coordinates.HybridLevels
  model_orography: orographies.ModalOrography
  model_timestep: np.timedelta64
  modal_filter: spatial_filters.ModalSpatialFilter
  learned_physics: typing.Transform

  # Dual-grid configuration, regridding, and tracers.
  dycore_ylm_map: spherical_harmonics.FixedYlmMapping | None = None
  dycore_orography: orographies.ModalOrography | None = None
  regridder_cls: Callable[..., interpolators.BaseRegridder] = (
      interpolators.SpectralRegridder
  )
  tracer_names: Sequence[str] = nnx.static(
      default_factory=lambda: ('specific_humidity',)
  )
  nodal_tracers: Sequence[str] = nnx.static(default=())

  # Time integration and dynamical core configuration.
  parameterization_timestep: np.timedelta64 | None = None
  aggregate_every: np.timedelta64 | None = None
  dycore_substeps: int = 1
  dycore_equation_cls: Callable[..., DycoreEquation] = (
      atm_equations.PrimitiveEquations
  )
  time_integrator_cls: Callable[..., time_integrators.DinosaurIntegrator] = (
      time_integrators.ImexRk3Sil
  )
  step_filter: step_filters.StepFilter | None = None
  reference_temperatures: Sequence[float] | None = None
  extra_dycore_kwargs: dict[str, Any] = nnx.static(default_factory=dict)

  # `sim_units`, `units_mapping` and `operators` are inherited from
  # `model.Model`.

  # Initialization and observation.
  assimilation_sources: dict[str, AssimilationSource] = nnx.data(
      default_factory=dict
  )
  assimilation_source_preference_order: Sequence[str] | None = nnx.static(
      default=None
  )
  prognostics_data_key: str = nnx.static(default='atmosphere_prognostics')

  # Diagnostics and conservation fixers.
  step_start_diagnostics: dict[str, diagnostics.DiagnosticModule] = nnx.data(
      default_factory=dict
  )
  tendency_diagnostics: dict[str, diagnostics.DiagnosticModule] = nnx.data(
      default_factory=dict
  )
  step_end_diagnostics: dict[str, diagnostics.DiagnosticModule] = nnx.data(
      default_factory=dict
  )
  aggregation_diagnostics: dict[str, diagnostics.DiagnosticModule] = nnx.data(
      default_factory=dict
  )
  tendency_fixers: Sequence[fixers.FixerModule] = ()
  state_fixers: Sequence[fixers.FixerModule] = ()

  # Stochasticity and rematerialization.
  advance_randomness_every: RandomnessAdvanceStep = nnx.static(
      default='substep'
  )
  randomness_advance_overrides: dict[
      RandomnessAdvanceStep, Sequence[random_processes.RandomProcessModule]
  ] = nnx.data(default_factory=dict)
  remat_substep: bool = False
  remat_to_aggregate: bool = False
  remat_advance: bool = False

  def __post_init__(self):
    if self.dycore_ylm_map is None:
      self.dycore_ylm_map = self.model_ylm_map
    if self.dycore_orography is None:
      if self.dycore_ylm_map != self.model_ylm_map:
        raise ValueError(
            '`dycore_orography` must be specified when `dycore_ylm_map` '
            'differs from `model_ylm_map`.'
        )
      self.dycore_orography = self.model_orography
    if self.step_filter is None:
      self.step_filter = step_filters.NoFilter()
    if self.units_mapping is None:
      self.units_mapping = defaults.DEFAULT_UNITS_MAPPING
    self.units_mapping = (
        dict(self.units_mapping)
        | {k: 'dimensionless' for k in self.tracer_names}
        | {f'{k}_ylm': 'dimensionless' for k in self.tracer_names}
    )
    if not self.operators:
      self.operators = {
          self.prognostics_data_key: (
              operator_utils.primitive_equations_state_operator(
                  self.model_ylm_map
              )
          ),
      }
    super().__post_init__()
    if self.aggregate_every is None:
      self.aggregate_every = self.model_timestep
    if self.parameterization_timestep is None:
      self.parameterization_timestep = self.aggregate_every

    for name, source in self.assimilation_sources.items():
      if source.grid is None:
        source.grid = self.grid
      if source.grid != self.grid and source.regridder is None:
        raise ValueError(
            f'Assimilation source {name!r} has grid {source.grid} different '
            f'from the model grid {self.grid}, which requires a `regridder`.'
        )
    data_keys = [s.data_key for s in self.assimilation_sources.values()]
    if len(set(data_keys)) != len(data_keys):
      raise ValueError(
          'Assimilation sources must have distinct `data_key`s, got '
          f'{data_keys}.'
      )
    operator_utils.merge_inputs_specs(*(
        self._source_inputs_spec(s) for s in self.assimilation_sources.values()
    ))
    if self.assimilation_source_preference_order is None:
      self.assimilation_source_preference_order = tuple(
          self.assimilation_sources
      )
    else:
      self.assimilation_source_preference_order = tuple(
          self.assimilation_source_preference_order
      )
      if sorted(self.assimilation_source_preference_order) != sorted(
          self.assimilation_sources
      ):
        raise ValueError(
            '`assimilation_source_preference_order` must be a permutation of '
            f'`assimilation_sources` keys {list(self.assimilation_sources)}, '
            f'got {list(self.assimilation_source_preference_order)}.'
        )
    # Number of aggregation intervals per `model_timestep`.
    self.num_aggregation_steps = time_utils.exact_timedelta_ratio(
        self.model_timestep,
        self.aggregate_every,
        f'model_timestep ({self.model_timestep}) must be a multiple of '
        f'aggregate_every ({self.aggregate_every}) to ensure that '
        'aggregation can happen at regular intervals.',
    )
    # Number of `learned_physics` evaluations per aggregation interval.
    self.num_substeps = time_utils.exact_timedelta_ratio(
        self.aggregate_every,
        self.parameterization_timestep,
        f'parameterization_timestep ({self.parameterization_timestep}) must'
        f' divide aggregate_every ({self.aggregate_every}).',
    )
    sim_units = self.sim_units
    assert sim_units is not None
    parameterization_dt = sim_units.nondimensionalize_timedelta64(
        self.parameterization_timestep
    )
    self.dycore_dt = parameterization_dt / self.dycore_substeps
    # Interpolators for regridding between model and dycore grids.
    if self.dycore_ylm_map == self.model_ylm_map:
      self.to_dycore_regrid = None
      self.to_model_regrid = None
    else:
      self.to_dycore_regrid = self.regridder_cls(self.dycore_ylm_map.ylm_grid)
      self.to_model_regrid = self.regridder_cls(self.model_ylm_map.ylm_grid)
    if self.reference_temperatures is None:
      reference_temperatures = atm_equations.get_reference_temperature(
          self.model_levels
      )
    else:
      reference_temperatures = self.reference_temperatures
    self.nodal_tracers = tuple(self.nodal_tracers)
    if 'nodal_tracers' in self.extra_dycore_kwargs:
      raise ValueError(
          'Specify `nodal_tracers` on the model rather than in '
          '`extra_dycore_kwargs`.'
      )
    dycore_kwargs = dict(self.extra_dycore_kwargs)
    if self.nodal_tracers:
      if self.dycore_ylm_map != self.model_ylm_map:
        raise ValueError(
            'Model and dycore grids must match when using nodal tracers '
            f'{self.nodal_tracers}, got model ({self.grid}, {self.ylm_grid}) '
            f'and dycore ({self.dycore_ylm_map.lon_lat_grid}, '
            f'{self.dycore_ylm_map.ylm_grid}).'
        )
      dycore_kwargs['nodal_tracers'] = self.nodal_tracers
    try:
      self.dycore_equation = self.dycore_equation_cls(
          ylm_map=self.dycore_ylm_map,
          levels=self.model_levels,
          sim_units=sim_units,
          reference_temperatures=reference_temperatures,
          orography_module=self.dycore_orography,
          tracer_names=self.tracer_names,
          **dycore_kwargs,
      )
    except TypeError as e:
      if 'nodal_tracers' in str(e):
        raise ValueError(
            '`nodal_tracers` requires a semi-Lagrangian dycore, got '
            f'{self.dycore_equation_cls}.'
        ) from e
      raise
    if self.nodal_tracers and not isinstance(
        self.dycore_equation,
        time_integrators.SemiLagrangianImplicitExplicitODE,
    ):
      raise ValueError(
          '`nodal_tracers` requires a semi-Lagrangian dycore, got '
          f'{self.dycore_equation}.'
      )

    self.to_modal_with_div_curl = atm_transforms.ToModalWithDivCurl(
        self.model_ylm_map,
    )
    self.clip_transform = transforms.ClipWavenumbers.for_grids(
        self.ylm_grid,
        wavenumbers_to_clip=1,
        skip_missing=True,
    )
    if self.advance_randomness_every not in _RANDOMNESS_ADVANCE_STEPS:
      raise ValueError(
          'Expected `advance_randomness_every` to be one of '
          f'{_RANDOMNESS_ADVANCE_STEPS}, got {self.advance_randomness_every!r}.'
      )
    overrides = self.randomness_advance_overrides
    if invalid_steps := set(overrides) - set(_RANDOMNESS_ADVANCE_STEPS):
      raise ValueError(
          f'Invalid keys {invalid_steps} in `randomness_advance_overrides`, '
          f'expected a subset of {_RANDOMNESS_ADVANCE_STEPS}.'
      )
    self.randomness_advance_overrides = {}
    model_randomness_ids = {
        id(m)
        for m in module_utils.retrieve_subclass_modules(
            self, random_processes.RandomProcessModule
        )
    }
    seen_override_ids: set[int] = set()
    normalized_overrides: dict[
        RandomnessAdvanceStep,
        Sequence[random_processes.RandomProcessModule],
    ] = {}
    for step, modules in overrides.items():
      modules = tuple(modules)
      for m in modules:
        if id(m) not in model_randomness_ids:
          raise ValueError(
              f'Module {m} in `randomness_advance_overrides[{step!r}]` is not '
              'a `RandomProcessModule` submodule of the model.'
          )
        if id(m) in seen_override_ids:
          raise ValueError(
              f'Module {m} appears multiple times in '
              '`randomness_advance_overrides`.'
          )
        seen_override_ids.add(id(m))
      normalized_overrides[step] = modules
    self.randomness_advance_overrides = normalized_overrides

    modal_volume = cx.coords.compose(self.model_levels, self.ylm_grid)
    nodal_volume = cx.coords.compose(self.model_levels, self.grid)
    nans_like = lambda c: cx.field(jnp.full(c.shape, jnp.nan), c)
    volume_keys = ('divergence', 'vorticity', 'temperature')
    tracers = {
        k: nans_like(nodal_volume if k in self.nodal_tracers else modal_volume)
        for k in self.tracer_names
    }
    # At construction time, prognostics are set to NaNs to ensure that
    # assimilate does not change the graph structure of the model.
    self.prognostics = typing.Prognostic(
        {
            'log_surface_pressure': nans_like(self.ylm_grid),
            'time': cx.field(jdt.to_datetime('1900-01-01T00:00')),  # pyrefly: ignore[bad-argument-type]
        }
        | {k: nans_like(modal_volume) for k in volume_keys}
        | tracers
    )

  @property
  def grid(self) -> coordinates.LonLatGrid:
    return self.model_ylm_map.lon_lat_grid

  @property
  def ylm_grid(self) -> coordinates.SphericalHarmonicGrid:
    return self.model_ylm_map.ylm_grid

  @property
  def nondim_prognostics_data_key(self) -> str:
    """Dataset key in `inputs` with nondimensional prognostic variables."""
    return f'nondim_{self.prognostics_data_key}'

  @property
  def inputs_spec(  # pyrefly: ignore[bad-override]
      self,
  ) -> dict[str, dict[str, data_specs.OptionalSpec[data_specs.CoordSpec]]]:
    """Returns specs of all inputs supported by `assimilate`."""
    specs = operator_utils.merge_inputs_specs(*(
        self._source_inputs_spec(s) for s in self.assimilation_sources.values()
    ))
    specs |= {
        self.prognostics_data_key: self._prognostics_inputs_spec,
        self.nondim_prognostics_data_key: self._prognostics_inputs_spec,
    }
    return jax.tree.map(
        data_specs.OptionalSpec,
        specs,
        is_leaf=lambda x: isinstance(x, data_specs.CoordSpec),
    )

  @property
  def _prognostics_inputs_spec(self) -> dict[str, data_specs.CoordSpec]:
    """Returns specs of prognostic inputs from which the state is set."""
    return {
        k: data_specs.CoordSpec.with_any_timedelta(
            v.coordinate, optional_timedelta=True
        )
        for k, v in self.prognostics.get_value().items()
    }

  def _source_inputs_spec(
      self, source: AssimilationSource
  ) -> dict[str, dict[str, data_specs.CoordSpec]]:
    """Returns specs of inputs for `source` from which state can be derived."""
    all_input_variables = [
        'geopotential',
        'temperature',
        'u_component_of_wind',
        'v_component_of_wind',
    ] + list(self.tracer_names)
    lvl_dim = source.levels.dims[0]
    if isinstance(source.levels, coordinates.HybridLevels):
      # Hybrid level ticks are indices, hence levels must match exactly.
      level_rules = {}
    else:
      level_rules = {lvl_dim: data_specs.AxisMatchRules.SUPERSET}

    def to_coord_spec(coord: cx.Coordinate) -> data_specs.CoordSpec:
      # Copy rules, since `with_*_timedelta` methods update them in place.
      rules = dict(level_rules) if lvl_dim in coord.dims else {}
      if source.correction_timedelta is None:
        return data_specs.CoordSpec.with_any_timedelta(
            coord, rules, optional_timedelta=True
        )
      # Inputs passed to `correction` must include all of its timedeltas.
      return data_specs.CoordSpec.with_given_timedelta(
          coord, source.correction_timedelta.deltas, rules
      )

    source_grid = source.grid if source.grid is not None else self.grid
    volume = cx.coords.compose(source.levels, source_grid)
    levels_specs = {k: to_coord_spec(volume) for k in all_input_variables}
    levels_specs['time'] = data_specs.CoordSpec.with_any_timedelta(
        cx.Scalar(), optional_timedelta=True
    )
    assert source.data_key is not None
    specs = {source.data_key: levels_specs}
    if source.surface_data_key is not None:
      surface_spec = {'surface_pressure': to_coord_spec(source_grid)}
      specs[source.surface_data_key] = (
          specs.get(source.surface_data_key, {}) | surface_spec
      )
    return specs

  def _diagnostics_operator(
      self,
  ) -> observation_operators.DataObservationOperator:
    """Returns an operator observing current (nondimensional) diagnostics."""
    return observation_operators.DataObservationOperator(
        self.diagnostic_values()
    )

  def _observe_diagnostics(
      self, query: dict[str, cx.Coordinate]
  ) -> dict[str, cx.Field]:
    """Helper method to observe diagnostic values without redimensionalizing."""
    diagnostics_query: typing.Query = {**query}
    return self._diagnostics_operator().observe({}, diagnostics_query)

  def _advance_randomness(self, step: RandomnessAdvanceStep) -> None:
    """Advances random processes scheduled for `step`."""
    if step == self.advance_randomness_every:
      excluded_ids = {  # pylint: disable=g-complex-comprehension
          id(m)
          for k, modules in self.randomness_advance_overrides.items()
          if k != step
          for m in modules
      }
      all_rng_modules = module_utils.retrieve_subclass_modules(
          self, random_processes.RandomProcessModule
      )
      modules = [m for m in all_rng_modules if id(m) not in excluded_ids]
    else:
      modules = self.randomness_advance_overrides.get(step, ())
    for randomness in modules:
      randomness.advance()

  ##############################################################################
  # Helper methods for model advance methods.
  ##############################################################################
  def _substep_with_fixed_tendencies(
      self, state: dict[str, cx.Field], tendencies: dict[str, cx.Field]
  ) -> dict[str, cx.Field]:
    """Evolves dycore `state` with fixed `tendencies` by one substep."""
    parameterization_eq = time_integrators.ExplicitODE.from_functions(
        lambda state: tendencies
    )
    all_equations = (self.dycore_equation, parameterization_eq)
    # Works for Eulerian and semi-Lagrangian dycores. For the latter, fixed
    # tendencies are applied as non-advective forcing.
    equation = equations.compose_equations(all_equations)  # pyrefly: ignore[bad-argument-type]
    integrator = self.time_integrator_cls(equation, self.dycore_dt)

    def dycore_step(integrator, x):
      assert self.step_filter is not None
      x_next = integrator(x)
      x_next = self.step_filter(x, x_next)
      if self.nodal_tracers:
        modal = {k: v for k, v in x_next.items() if k not in self.nodal_tracers}
        nodal = {k: v for k, v in x_next.items() if k in self.nodal_tracers}
        return self.modal_filter.filter_modal(modal) | nodal
      return self.modal_filter.filter_modal(x_next)

    return stepping.repeat_step(dycore_step, self.dycore_substeps)(
        integrator, state
    )

  def _substep(self, prognostics: dict[str, cx.Field]) -> dict[str, cx.Field]:
    """Evolves `prognostics` by `parameterization_timestep`."""
    prognostics = prognostics.copy()  # avoid mutating prognostics dict.
    stepping.update_diagnostics(
        self.step_start_diagnostics, inputs=prognostics, prognostics=prognostics
    )

    tendencies = self.learned_physics(prognostics)
    expected_tendency_keys = set(prognostics) - {'time'}
    if set(tendencies) != expected_tendency_keys:
      raise ValueError(
          '`learned_physics` must return tendencies for all prognostic '
          f'variables {sorted(expected_tendency_keys)}, got '
          f'{sorted(tendencies)}.'
      )
    _check_coordinates(tendencies, prognostics, '`learned_physics` tendency of')
    stepping.update_diagnostics(
        self.tendency_diagnostics, tendencies, prognostics
    )

    # Tendency fixers run right after tendency diagnostics.
    for tendency_fixer in self.tendency_fixers:
      imbalance = self._observe_diagnostics(tendency_fixer.diagnostics_query)
      tendencies = tendency_fixer(
          imbalance, tendencies, prognostics=prognostics
      )

    time = prognostics.pop('time')
    if self.to_dycore_regrid is not None and self.to_model_regrid is not None:
      tendencies = self.to_dycore_regrid(tendencies)
      prognostics = self.to_dycore_regrid(prognostics)
      prognostics = self._substep_with_fixed_tendencies(prognostics, tendencies)
      prognostics = self.to_model_regrid(prognostics)
    else:
      prognostics = self._substep_with_fixed_tendencies(prognostics, tendencies)

    for state_fixer in self.state_fixers:
      fixer_diagnostic = self._observe_diagnostics(
          state_fixer.diagnostics_query
      )
      prognostics = state_fixer(fixer_diagnostic, prognostics)

    assert self.parameterization_timestep is not None
    time = time + self.parameterization_timestep
    prognostics = prognostics | {'time': time}
    stepping.update_diagnostics(
        self.step_end_diagnostics, inputs=prognostics, prognostics=prognostics
    )
    stepping.advance_diagnostic_clocks(
        self.parameterization_timestep,
        self.step_start_diagnostics,
        self.tendency_diagnostics,
        self.step_end_diagnostics,
    )
    self._advance_randomness('substep')
    return prognostics

  def _to_next_aggregation_time(
      self, prognostics: dict[str, cx.Field]
  ) -> dict[str, cx.Field]:
    """Evolves `prognostics` to the next aggregation time."""
    assert self.aggregate_every is not None
    substep_fn = lambda m, x: m._substep(x)  # pylint: disable=protected-access
    if self.remat_substep:
      substep_fn = nnx.remat(substep_fn)
    prognostics = stepping.repeat_step(substep_fn, self.num_substeps)(
        self, prognostics
    )
    stepping.update_diagnostics(
        self.aggregation_diagnostics,
        inputs=prognostics,
        prognostics=prognostics,
    )
    stepping.advance_diagnostic_clocks(
        self.aggregate_every, self.aggregation_diagnostics
    )
    self._advance_randomness('aggregation')
    return prognostics

  def advance(self) -> None:
    """Computes `prognostics` evolved in time by `self.model_timestep`."""
    agg_step_fn = lambda m, x: m._to_next_aggregation_time(x)  # pylint: disable=protected-access
    # Rematerialize the whole aggregation chunk, rather than just the substeps
    # inside `_to_next_aggregation_time` can be helpful so that aggregate
    # diagnostics are recomputed as well. Otherwise their activations are
    # retained for each of the `num_aggregation_steps` iterations.
    if self.remat_to_aggregate:
      agg_step_fn = nnx.remat(agg_step_fn)
    agg_steps_fn = stepping.repeat_step(agg_step_fn, self.num_aggregation_steps)

    def step_fn(m, x):
      x = agg_steps_fn(m, x)
      m._advance_randomness('advance')  # pylint: disable=protected-access
      return x

    if self.remat_advance:
      step_fn = nnx.remat(step_fn)
    self.prognostics.set_value(step_fn(self, self.prognostics.get_value()))

  @property
  def timestep(self) -> np.timedelta64:
    """Returns the timestep of the model."""
    return self.model_timestep

  def assimilate(self, inputs: dict[str, dict[str, cx.Field]]) -> None:
    """Sets the model state from `inputs`, see class docstring for details."""
    prognostics_key = self.prognostics_data_key
    nondim_prognostics_key = self.nondim_prognostics_data_key
    prognostic_keys = [
        k for k in (prognostics_key, nondim_prognostics_key) if k in inputs
    ]
    if len(prognostic_keys) > 1:
      raise ValueError(
          f'Got ambiguous initialization from both {prognostic_keys}, expected'
          ' at most one.'
      )
    if prognostic_keys:
      [key] = prognostic_keys
      self._assimilate_prognostics(
          inputs[key], nondimensionalize=(key == prognostics_key)
      )
      return
    assert self.assimilation_source_preference_order is not None
    for name in self.assimilation_source_preference_order:
      source = self.assimilation_sources[name]
      if source.data_key in inputs:
        self._assimilate_source(inputs, source)
        return
    source_keys = {
        name: [k for k in (s.data_key, s.surface_data_key) if k is not None]
        for name, s in self.assimilation_sources.items()
    }
    raise ValueError(
        f'No initialization source found in inputs with keys {list(inputs)}. '
        f'Expected {self.prognostics_data_key!r}, '
        f'{self.nondim_prognostics_data_key!r} or datasets of one of '
        f'the assimilation sources: {source_keys}.'
    )

  def _assimilate_prognostics(
      self,
      inputs: dict[str, cx.Field],
      nondimensionalize: bool,
  ) -> None:
    """Sets the model state to `inputs` with optional nondimensionalization."""
    expected = self.prognostics.get_value()
    missing = [k for k in expected if k not in inputs]
    if missing:
      raise ValueError(
          f'Prognostic variables {missing} not found in inputs. '
          f'Available variables: {list(inputs.keys())}'
      )
    prognostics = {
        k: time_utils.select_zero_timedelta(inputs[k]) for k in expected
    }
    time = prognostics.pop('time')
    if nondimensionalize:
      prognostics = self.nondim_transform(prognostics)
    _check_coordinates(prognostics, expected, 'Prognostic variable')
    prognostics = parallelism.with_dycore_sharding(self.mesh, prognostics)
    self.prognostics.set_value(prognostics | {'time': time})

  def _assimilate_source(
      self,
      inputs: dict[str, dict[str, cx.Field]],
      source: AssimilationSource,
  ) -> None:
    """Sets the model state from inputs for `source`."""
    fields = {}
    missing_errors = []
    for data_name, spec in self._source_inputs_spec(source).items():
      if data_name not in inputs:
        missing_errors.append(
            f'dataset {data_name!r} not found in inputs '
            f'(available: {list(inputs)})'
        )
        continue
      dataset = inputs[data_name]
      if missing_vars := [k for k in spec if k not in dataset]:
        missing_errors.append(
            f'variables {missing_vars} not found in {data_name!r} '
            f'(available: {list(dataset)})'
        )
      else:
        for k in spec:
          fields[k] = dataset[k]
    if missing_errors:
      raise ValueError(
          'Missing required assimilation inputs: ' + '; '.join(missing_errors)
      )
    time = time_utils.select_zero_timedelta(fields.pop('time'))
    if source.correction_timedelta is None:
      fields = {
          k: time_utils.select_zero_timedelta(v) for k, v in fields.items()
      }
    else:
      fields = {
          k: time_utils.select_timedeltas(v, source.correction_timedelta)
          for k, v in fields.items()
      }
    # HybridLevels do not support selection and are required to match exactly.
    if not isinstance(source.levels, coordinates.HybridLevels):
      lvl_dim = source.levels.dims[0]
      sel_levels = transforms.Sel({lvl_dim: source.levels})
      fields = sel_levels(fields)
    fields = self.nondim_transform(fields)
    # The model state is derived from inputs at timedelta=0.
    input_fields = {
        k: time_utils.select_zero_timedelta(v) for k, v in fields.items()
    }
    input_fields = parallelism.with_physics_sharding(self.mesh, input_fields)
    if source.regridder is not None:
      regridded_inputs = parallelism.with_physics_sharding(
          self.mesh, source.regridder(input_fields)
      )
    else:
      regridded_inputs = input_fields
    assert self.sim_units is not None
    pe_state_nodal = state_conversion.uvtz_to_primitive_equations(
        inputs=regridded_inputs,
        levels=self.model_levels,
        orography=self.model_orography,  # pyrefly: ignore[bad-argument-type]
        sim_units=self.sim_units,
    )
    pe_state_nodal = parallelism.with_physics_sharding(
        self.mesh, pe_state_nodal
    )
    nodal_tracers = {k: pe_state_nodal.pop(k) for k in self.nodal_tracers}
    pe_state = self.to_modal_with_div_curl(pe_state_nodal) | nodal_tracers
    if source.correction is not None:
      if source.correction_timedelta is None:
        correction_inputs = input_fields
      else:
        # TODO(dkochkov): Use field sharding here, `with_physics_sharding`
        # doesn't support inputs with a `timedelta` axis.
        correction_inputs = fields
      correction = source.correction(correction_inputs | {'time': time})
      correction = parallelism.with_dycore_sharding(self.mesh, correction)
      nodal_tracer_corrections = {
          k: correction.pop(k) for k in self.nodal_tracers if k in correction
      }
      uv_keys = {
          self.to_modal_with_div_curl.u_key,
          self.to_modal_with_div_curl.v_key,
      }
      if uv_keys & set(correction):
        # Wind corrections are converted to divergence and vorticity like the
        # prior state, so that they are the exact div/curl of a vector field
        # (in particular with vanishing global means).
        if not uv_keys <= set(correction):
          raise ValueError(
              'Corrections of wind components must include both '
              f'{sorted(uv_keys)}, got {sorted(correction)}.'
          )
        correction = self.to_modal_with_div_curl(correction)
      else:
        to_modal = self.model_ylm_map.to_modal
        correction = {k: to_modal(v) for k, v in correction.items()}
      correction = parallelism.with_dycore_sharding(
          self.mesh, correction | nodal_tracer_corrections
      )
      if unknown_keys := set(correction) - set(pe_state):
        raise ValueError(
            f'Correction keys {unknown_keys} are not in the model state with '
            f'keys {list(pe_state)}.'
        )
      _check_coordinates(correction, pe_state, 'Correction of')
      pe_state |= {k: pe_state[k] + v for k, v in correction.items()}
      pe_state = parallelism.with_dycore_sharding(self.mesh, pe_state)

    pe_state = self.clip_transform(pe_state)
    pe_state = parallelism.with_dycore_sharding(self.mesh, pe_state)
    if source.filter_state is not None:
      state = pe_state | {'time': time}
      filtered = dict(source.filter_state(state))
      if set(filtered) != set(state):
        raise ValueError(
            '`filter_state` must preserve keys of the model state, got outputs '
            f'{sorted(filtered)} for inputs {sorted(state)}.'
        )
      _check_coordinates(filtered, state, '`filter_state` output')
      time = filtered.pop('time')
      # Filters that are not diagonal in spectral space, e.g. nodal operations,
      # can reintroduce the clipped wavenumber.
      pe_state = self.clip_transform(filtered)
      pe_state = parallelism.with_dycore_sharding(self.mesh, pe_state)
    self.prognostics.set_value(pe_state | {'time': time})

  def prognostic_values(self) -> dict[str, cx.Field]:
    """Returns current nondimensional prognostic values of the model."""
    return parallelism.with_dycore_sharding(
        self.mesh, self.prognostics.get_value()
    )
