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

"""Classical atmospheric model solving spectral primitive equations."""

from collections.abc import Callable, Mapping, Sequence

import coordax as cx
from flax import nnx
import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
from terrax.atmosphere import equations as atmos_equations
from terrax.core import api
from terrax.core import coordinates
from terrax.core import data_specs
from terrax.core import equations
from terrax.core import observation_operators
from terrax.core import orographies
from terrax.core import parallelism
from terrax.core import spatial_filters
from terrax.core import spherical_harmonics
from terrax.core import step_filters
from terrax.core import time_integrators
from terrax.core import transforms
from terrax.core import typing
from terrax.core import units


# pylint: disable=g-classes-have-attributes

DEFAULT_UNITS_MAPPING = {
    'u_component_of_wind': 'meter / second',
    'v_component_of_wind': 'meter / second',
    'temperature': 'kelvin',
    'temperature_variation': 'kelvin',
    'geopotential': 'm**2 s**-2',
    'sim_time': 'dimensionless',
    'specific_humidity': 'dimensionless',
    'specific_cloud_ice_water_content': 'dimensionless',
    'specific_cloud_liquid_water_content': 'dimensionless',
    'divergence': '1 / second',
    'vorticity': '1 / second',
    'surface_pressure': 'pascal',
    'log_surface_pressure': 'dimensionless',
    'tracer': 'dimensionless',
    'orography': 'meter',
}

DycoreEquation = (
    time_integrators.ImplicitExplicitODE
    | time_integrators.SemiLagrangianImplicitExplicitODE
)


def _repeat(step_fn: Callable[..., typing.Pytree], length: int):
  """Returns `step_fn(module, carry)` applied `length` times via `nnx.scan`."""
  if length == 1:
    return step_fn
  return nnx.scan(
      step_fn,
      length=length,
      in_axes=(api.DEFAULT_MODEL_STATE_AXES, nnx.Carry),
      out_axes=nnx.Carry,
  )


def _select_zero_timedelta(field: cx.Field) -> cx.Field:
  """Returns `field` at timedelta=0, or `field` as is if it has no timedelta."""
  if 'timedelta' not in field.dims:
    return field
  timedelta = field.axes.get('timedelta')
  if isinstance(timedelta, parallelism.CoordinateShard):
    timedelta = timedelta.coordinate
  if not isinstance(timedelta, coordinates.TimeDelta):
    raise ValueError(
        'Expected `timedelta` dimension to be associated with a TimeDelta '
        f'coordinate, got {timedelta=} in {field=}'
    )
  [indices] = np.nonzero(timedelta.deltas == np.timedelta64(0, 's'))
  if indices.size == 0:
    raise ValueError(
        f'Inputs must include timedelta=0, got {timedelta.deltas=} in {field=}'
    )
  idx = int(indices[0])
  return cx.cmap(lambda x: x[idx])(field.untag('timedelta'))


def nodal_prognostics_operator(
    ylm_map: spherical_harmonics.FixedYlmMapping,
) -> observation_operators.TransformObservationOperator:
  """Returns an operator observing nodal prognostics, winds, and surface pressure."""
  to_nodal = transforms.ModalToNodal(ylm_map, include_remaining=True)
  surface_pressure = transforms.Sequential([
      transforms.ApplyFnToKeys(cx.cpmap(jnp.exp), ['log_surface_pressure']),
      transforms.RenameKeys({'log_surface_pressure': 'surface_pressure'}),
  ])
  transform = transforms.Sequential([  # pyrefly: ignore[bad-argument-type]
      transforms.SelectKeys('time', invert=True),
      transforms.Merge({
          'uv': transforms.VelocityFromDivCurl(ylm_map),
          'nodal': to_nodal,
      }),
      transforms.Merge({
          'nodal': transforms.Identity(),  # pyrefly: ignore[bad-assignment]
          'sp': surface_pressure,
      }),
  ])
  return observation_operators.TransformObservationOperator(transform)


class SpectralPrimitiveEquationsModel(api.Model):
  """Atmospheric model solving primitive equations on a spectral grid.

  This model evolves the primitive equations (Eulerian or semi-Lagrangian) on
  `ylm_map` and `levels`, optionally composed with explicit forcing or
  relaxation equations (such as `HeldSuarezForcing`) via `explicit_equations`.

  The model state is initialized via `assimilate` from either:
    * `inputs[f'nondim_{prognostics_data_key}']`: nondimensional prognostic
      variables matching `self.prognostics`, which are set as is.
    * `inputs[prognostics_data_key]`: prognostic variables in physical units,
      which are nondimensionalized using `nondim_transform` (except
      `log_surface_pressure`, which is always the log of nondimensional surface
      pressure).
  All inputs may optionally include a `timedelta` axis, in which case the
  `timedelta=0` slice is used for initialization.

  Attributes:
    ylm_map: Spherical harmonics mapping for the horizontal grid.
    levels: Vertical levels coordinate (`SigmaLevels` or `HybridLevels`).
    model_timestep: Time step advanced by a single call to `advance`.
    orography: Modal orography module on `ylm_map`.
    time_integrator_cls: Factory for the time integrator, called with
      `(equation, dycore_dt)`.
    modal_filter: Spatial filter applied in spectral space after each substep.
    sim_units: Simulation units helper.
    nondim_transform: Transform to non-dimensionalize inputs.
    redim_transform: Transform to re-dimensionalize observations.
    num_substeps: Number of dycore steps per `model_timestep`.
    reference_temperatures: Reference temperatures for linearization.
    dycore_equation_cls: Factory for the dynamical core equation (e.g.
      `atmos_equations.PrimitiveEquations` or
      `atmos_equations.SemiLagrangianPrimitiveEquations`).
    explicit_equations: Tuple of explicit forcing or relaxation equations
      (e.g. `HeldSuarezForcing`) composed with `dycore_equation`. Tendencies
      returned by `explicit_equations` must match the coordinate representation
      of `prognostics` (modal for spectral variables and nodal for any
      `nodal_tracers`).
    step_filter: Step filter applied after each substep; must be compatible with
      the coordinate representation of the prognostic variables it filters.
    tracer_names: Tuple of tracer names evolved by the dynamical core.
    operators: Observation operators for querying the model state.
    prognostics_data_key: Dataset key in `inputs` from which the model state is
      initialized. Nondimensional values are read from `nondim_` prefixed key.
    extra_dycore_kwargs: Additional keyword arguments forwarded to
      `dycore_equation_cls`.
  """

  def __init__(
      self,
      ylm_map: spherical_harmonics.FixedYlmMapping,
      levels: coordinates.SigmaLevels | coordinates.HybridLevels,
      model_timestep: np.timedelta64,
      orography: orographies.ModalOrography,
      time_integrator_cls: Callable[..., time_integrators.DinosaurIntegrator],
      modal_filter: spatial_filters.ModalSpatialFilter,
      sim_units: units.SimUnits,
      nondim_transform: transforms.Nondimensionalize | None = None,
      redim_transform: transforms.Redimensionalize | None = None,
      num_substeps: int = 1,
      reference_temperatures: Sequence[float] | cx.Field | None = None,
      dycore_equation_cls: Callable[..., DycoreEquation] = (
          atmos_equations.PrimitiveEquations
      ),
      explicit_equations: Sequence[time_integrators.ExplicitODE] = (),
      step_filter: step_filters.StepFilter | None = None,
      tracer_names: Sequence[str] = (),
      operators: (
          Mapping[str, observation_operators.ObservationOperatorABC] | None
      ) = None,
      prognostics_data_key: str = 'prognostics',
      *,
      mesh: parallelism.Mesh | None = None,
      **extra_dycore_kwargs,
  ):
    if mesh is None:
      super().__init__()
    else:
      super().__init__(mesh=mesh)  # pylint: disable=unexpected-keyword-arg
    self.ylm_map = ylm_map
    self.levels = levels
    self.model_timestep = model_timestep
    self.orography = orography
    self.time_integrator_cls = time_integrator_cls
    self.modal_filter = modal_filter
    self.sim_units = sim_units
    self.num_substeps = num_substeps
    self.dycore_equation_cls = dycore_equation_cls
    self.explicit_equations = nnx.data(tuple(explicit_equations))
    self.step_filter = (
        step_filters.NoFilter() if step_filter is None else step_filter
    )
    self.tracer_names = tuple(tracer_names)
    self.prognostics_data_key = prognostics_data_key
    self.extra_dycore_kwargs = extra_dycore_kwargs

    units_mapping = DEFAULT_UNITS_MAPPING | {
        k: 'dimensionless' for k in self.tracer_names
    }
    if nondim_transform is None:
      nondim_transform = transforms.Nondimensionalize(
          self.sim_units, inputs_to_units_mapping=units_mapping
      )
    self.nondim_transform = nondim_transform
    if redim_transform is None:
      redim_transform = transforms.Redimensionalize(
          self.sim_units, inputs_to_units_mapping=units_mapping
      )
    self.redim_transform = redim_transform
    if reference_temperatures is None:
      reference_temperatures = atmos_equations.get_reference_temperature(
          self.levels
      )
    self.reference_temperatures = reference_temperatures
    self.dycore_equation = self.dycore_equation_cls(
        ylm_map=self.ylm_map,
        levels=self.levels,
        sim_units=self.sim_units,
        reference_temperatures=self.reference_temperatures,
        orography_module=self.orography,
        tracer_names=self.tracer_names,
        **extra_dycore_kwargs,
    )
    if self.explicit_equations:
      self.equation = equations.compose_equations(
          [self.dycore_equation, *self.explicit_equations]
      )
    else:
      self.equation = self.dycore_equation

    model_dt = self.sim_units.nondimensionalize_timedelta64(self.model_timestep)
    self.dycore_dt = model_dt / self.num_substeps
    self.integrator = self.time_integrator_cls(self.equation, self.dycore_dt)
    self.clip_transform = transforms.ClipWavenumbers.for_grids(
        self.ylm_grid,  # pyrefly: ignore[bad-argument-type]
        wavenumbers_to_clip=1,
        skip_missing=True,
    )
    if not operators:
      operators = {
          'prognostics': nodal_prognostics_operator(self.ylm_map),
      }
    self.operators = nnx.data(dict(operators))
    self.nodal_tracers = tuple(
        getattr(self.dycore_equation, 'nodal_tracers', ())
    )
    modal_volume = cx.coords.compose(self.levels, self.ylm_grid)
    nodal_volume = cx.coords.compose(self.levels, self.grid)
    nans_like = lambda c: cx.field(jnp.full(c.shape, jnp.nan), c)
    volume_keys = ('divergence', 'vorticity', 'temperature')
    tracers = {
        k: nans_like(nodal_volume if k in self.nodal_tracers else modal_volume)
        for k in self.tracer_names
    }
    self.prognostics = typing.Prognostic(
        {
            'log_surface_pressure': nans_like(self.ylm_grid),
            'time': cx.field(jdt.to_datetime('1900-01-01T00:00')),  # pyrefly: ignore[bad-argument-type]
        }
        | {k: nans_like(modal_volume) for k in volume_keys}
        | tracers
    )

  @property
  def grid(self) -> cx.Coordinate:
    return self.ylm_map.nodal_grid

  @property
  def ylm_grid(self) -> cx.Coordinate:
    return self.ylm_map.modal_grid

  @property
  def timestep(self) -> np.timedelta64:
    """Returns the timestep of the model."""
    return self.model_timestep

  @property
  def nondim_prognostics_data_key(self) -> str:
    """Dataset key in `inputs` with nondimensional prognostic variables."""
    return f'nondim_{self.prognostics_data_key}'

  @property
  def _prognostics_inputs_spec(self) -> dict[str, data_specs.CoordSpec]:
    """Returns specs of prognostic inputs from which the state is set."""
    return {
        k: data_specs.CoordSpec.with_any_timedelta(
            parallelism.get_unsharded(v.coordinate), optional_timedelta=True
        )
        for k, v in self.prognostics.get_value().items()
    }

  @property
  def inputs_spec(  # pyrefly: ignore[bad-override]
      self,
  ) -> dict[str, dict[str, data_specs.OptionalSpec[data_specs.CoordSpec]]]:
    """Returns specs of all inputs supported by `assimilate`."""
    specs = {
        self.prognostics_data_key: self._prognostics_inputs_spec,
        self.nondim_prognostics_data_key: self._prognostics_inputs_spec,
    }
    return jax.tree.map(
        data_specs.OptionalSpec,
        specs,
        is_leaf=lambda x: isinstance(x, data_specs.CoordSpec),
    )

  def assimilate(self, inputs: dict[str, dict[str, cx.Field]]) -> None:
    """Sets the model state from `inputs`, see class docstring for details."""
    prognostics_key = self.prognostics_data_key
    nondim_prognostics_key = self.nondim_prognostics_data_key
    prognostic_keys = [
        k for k in (prognostics_key, nondim_prognostics_key) if k in inputs
    ]
    if len(prognostic_keys) != 1:
      raise ValueError(
          f'Expected exactly one of {prognostics_key!r} or '
          f'{nondim_prognostics_key!r} in inputs, got {list(inputs.keys())}.'
      )
    [key] = prognostic_keys
    self._assimilate_prognostics(
        inputs[key], nondimensionalize=(key == prognostics_key)
    )

  def _assimilate_prognostics(
      self,
      inputs: Mapping[str, cx.Field],
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
    prognostics = {k: _select_zero_timedelta(inputs[k]) for k in expected}
    time = prognostics.pop('time')
    if nondimensionalize:
      assert self.nondim_transform is not None
      # log_surface_pressure is always the log of nondimensional pressure.
      log_sp = prognostics.pop('log_surface_pressure')
      prognostics = self.nondim_transform(prognostics)
      prognostics['log_surface_pressure'] = log_sp
    for k, v in prognostics.items():
      expected_coord = parallelism.get_unsharded(expected[k].coordinate)
      if parallelism.get_unsharded(v.coordinate) != expected_coord:
        raise ValueError(
            f'Prognostic variable {k!r} has coordinate {v.coordinate}, '
            f'expected {expected_coord}.'
        )
    prognostics = self.clip_transform(prognostics)
    prognostics = parallelism.with_dycore_sharding(self.mesh, prognostics)
    self.prognostics.set_value(prognostics | {'time': time})

  def substep(self, state: dict[str, cx.Field]) -> dict[str, cx.Field]:
    """Advances `state` by one dycore substep."""
    next_state = self.integrator(state)
    next_state = self.step_filter(state, next_state)
    if self.nodal_tracers:
      modal = {
          k: v for k, v in next_state.items() if k not in self.nodal_tracers
      }
      nodal = {
          k: v for k, v in next_state.items() if k in self.nodal_tracers
      }
      return self.modal_filter.filter_modal(modal) | nodal
    return self.modal_filter.filter_modal(next_state)

  def advance(self) -> None:
    """Advances `prognostics` in time by `self.model_timestep`."""
    prognostics = self.prognostics.get_value().copy()
    time = prognostics.pop('time')
    step_fn = _repeat(lambda model, x: model.substep(x), self.num_substeps)
    next_state = step_fn(self, prognostics)
    time = time + self.model_timestep
    self.prognostics.set_value(next_state | {'time': time})

  def observe(self, queries: typing.Queries) -> typing.Observation:
    """Computes observations specified in `queries` from the model state."""
    assert self.redim_transform is not None
    prognostics = self.prognostics.get_value()
    prognostics = parallelism.with_dycore_sharding(self.mesh, prognostics)
    final_outputs = {}
    for k, query in queries.items():
      if k not in self.operators:
        raise ValueError(f'No observation operator for {k=}')
      observation = self.operators[k].observe(prognostics, query)
      final_outputs[k] = self.redim_transform(observation)
    return final_outputs
