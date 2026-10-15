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

"""Model that advances prognostic variables with a learned update."""

from collections.abc import Mapping

import coordax as cx
from flax import nnx
import jax
import jax.numpy as jnp
import jax_datetime as jdt
from neuralgcm.utils import defaults
from neuralgcm.utils import stepping
from neuralgcm.utils import time_utils
import numpy as np
from terrax.core import api
from terrax.core import coordinates
from terrax.core import data_specs
from terrax.core import diagnostics
from terrax.core import dynamic_io  # pylint: disable=unused-import
from terrax.core import feature_transforms  # pylint: disable=unused-import
from terrax.core import learned_transforms  # pylint: disable=unused-import
from terrax.core import module_utils
from terrax.core import observation_operators
from terrax.core import parallelism
from terrax.core import pytree_utils  # pylint: disable=unused-import
from terrax.core import random_processes
from terrax.core import standard_layers  # pylint: disable=unused-import
from terrax.core import transformer_layers  # pylint: disable=unused-import
from terrax.core import transforms
from terrax.core import typing
from terrax.core import units


# pylint: disable=g-classes-have-attributes


def _shift_in(history: cx.Field, latest: cx.Field) -> cx.Field:
  """Returns `history` shifted back along `timedelta` to end with `latest`."""
  timedelta = history.axes['timedelta']
  h = history.untag('timedelta')
  shift = lambda h, x: jnp.concatenate([h[1:], x[np.newaxis]])
  return cx.cmap(shift, out_axes=h.dims)(h, latest).tag(timedelta)


def _get_required(
    mapping: Mapping[str, typing.Pytree], key: str, description: str
) -> typing.Pytree:
  """Returns `mapping[key]`, raising an informative error if it is missing."""
  if key not in mapping:
    raise ValueError(
        f'Required {description} {key!r} not found, available: {list(mapping)}.'
    )
  return mapping[key]


def _check_coordinates(
    fields: Mapping[str, cx.Field],
    coords: Mapping[str, cx.Coordinate],
    description: str,
) -> None:
  """Raises if `fields` miss entries of `coords` or have other coordinates."""
  for k, coord in coords.items():
    field = _get_required(fields, k, description)
    expected = parallelism.get_unsharded(coord)
    if parallelism.get_unsharded(field.coordinate) != expected:
      raise ValueError(
          f'Expected {description} {k!r} to have coordinate {expected}, got '
          f'{field.coordinate}.'
      )


@nnx.dataclass
class LearnedUpdateModel(api.Model):
  """Model that advances prognostic variables with a learned update.

  The model state consists of nondimensional `prognostics` on
  `prognostics_coords` and the current `time`. Every `update_timestep`,
  `update_transform` is evaluated on the prognostics and `time` and must return
  fields for all prognostic variables on `prognostics_coords`. If
  `predict_state`, these are the new prognostics, otherwise they are rates of
  change that are multiplied by the nondimensional `update_timestep` and added
  to the prognostics. The result is passed through `censor_transform`, e.g. to
  enforce physical bounds. Additional outputs of `update_transform` are ignored
  by the model, but can be recorded by `update_diagnostics`.

  If `num_history_steps` is positive, `prognostics` also include values from
  that many previous updates, stacked along `history_axis`, a `timedelta` axis
  with spacing `update_timestep` that ends at 0. `update_transform` is then
  evaluated on the full history, but still returns fields on
  `prognostics_coords`, which are appended to the history while its oldest
  values are dropped. `update_diagnostics` receive the full history as
  `prognostics`, while observation operators and `step_end_diagnostics` see
  only the current values.

  Each `advance` evolves the model by `model_timestep`, which must divide
  `update_timestep`. Between updates, the prognostics are held fixed and only
  `time` advances. Updates are applied on the last `advance` of every
  `update_timestep`, with `time` set to the start of the update interval. This
  allows `update_transform` to use inputs accumulated over the interval, e.g.
  by a coupler from a model with a shorter timestep. Random processes are
  advanced once per update.

  Diagnostics are evaluated at two stages of the time stepping, which
  determines both the values they see and their alignment with `prognostics`
  when observed after `advance`:
    * `update_diagnostics`: evaluated on the outputs of `update_transform`
      (with its inputs passed as `prognostics` kwarg) on every update. Describe
      the update, e.g. rates of change or additional outputs of
      `update_transform`. When observed, values correspond to the update that
      produced the current `prognostics`.
    * `step_end_diagnostics`: evaluated on the prognostics and `time` at the
      end of every `advance`. Aligned with `prognostics` when observed. Used
      for averages, accumulations and histories at the output cadence.
  All diagnostics are called as `diagnostic(inputs, prognostics=prognostics)`,
  where `inputs` are the outputs of `update_transform` for
  `update_diagnostics` and the prognostics otherwise. Clocks of temporal
  diagnostics are advanced by the model with an explicit `timedelta`, which is
  `update_timestep` for `update_diagnostics` and `model_timestep` for
  `step_end_diagnostics`, hence diagnostics should not specify
  `default_timedelta`. Diagnostic values are observed with the `'diagnostics'`
  query key.

  The model state is set by `assimilate` from one of:
    * `inputs[f'nondim_{prognostics_data_key}']`: nondimensional prognostic
      variables on the coordinates of `prognostics` and `time`, which are set
      as is.
    * `inputs[prognostics_data_key]`: same as above, but in physical units,
      which are nondimensionalized using `nondim_transform`.
    * `inputs[var_to_dataset_keys[k]]` for every prognostic variable `k` and
      `time`, with prognostic variables on `input_coords[k]` (defaulting to
      `prognostics_coords[k]`). Inputs are nondimensionalized and passed through
      `assimilate_transform`, which must return the same keys (including `time`)
      with prognostic variables on the coordinates of `prognostics`.
  Prognostic inputs take precedence over `var_to_dataset_keys`. All inputs can
  optionally include a `timedelta` axis, in which case the `timedelta=0` slice
  is used. If `num_history_steps` is positive, inputs of prognostic variables
  must instead include a `timedelta` axis with all values of `history_axis`.
  Since `inputs_spec` lists all alternatives as optional, validation that one of
  them is complete happens in `assimilate`.

  Attributes:
    prognostics_coords: Coordinates of prognostic variables, keyed by name.
    update_transform: Transform that maps prognostics and `time` to rates of
      change or new values of all prognostic variables, see above.
    var_to_dataset_keys: Keys of the datasets in `inputs` that contain each of
      the prognostic variables and `time`.
    model_timestep: Time interval by which `advance` evolves the model.
    update_timestep: Time interval between updates of the prognostics, over
      which they are held fixed. Must be a multiple of `model_timestep`.
    predict_state: Whether outputs of `update_transform` are new values of the
      prognostic variables or their rates of change.
    num_history_steps: Number of previous updates stored in `prognostics`, see
      above.
    input_coords: Coordinates of input variables that differ from
      `prognostics_coords`, which `assimilate_transform` must map to
      `prognostics_coords`.
    assimilate_transform: Transform applied to nondimensionalized inputs in
      `assimilate`, which must preserve keys.
    censor_transform: Transform applied to updated prognostics, e.g. to enforce
      physical constraints.
    sim_units: Simulation units helper. Defaults to `units.SI_UNITS`.
    nondim_transform: Transform that nondimensionalizes inputs. Defaults to
      `transforms.Nondimensionalize` using `sim_units` and
      `defaults.DEFAULT_UNITS_MAPPING` if None.
    redim_transform: Transform that redimensionalizes observations. Defaults to
      `transforms.Redimensionalize` using `sim_units` and
      `defaults.DEFAULT_UNITS_MAPPING` if None.
    update_diagnostics: Diagnostics evaluated on outputs of `update_transform`
      on every update, see above.
    step_end_diagnostics: Diagnostics evaluated on prognostics at the end of
      every `advance`, see above.
    remat_advance: Whether to rematerialize `advance` to reduce memory usage
      during backpropagation.
    prognostics_data_key: Dataset key in `inputs` from which the model state is
      set directly. Nondimensional values are read from `nondim_` prefixed key.
    operators: Observation operators for querying the prognostics.
  """

  prognostics_coords: dict[str, cx.Coordinate]
  update_transform: typing.Transform
  var_to_dataset_keys: dict[str, str]
  model_timestep: np.timedelta64 = nnx.static(kw_only=True)
  update_timestep: np.timedelta64 = nnx.static(kw_only=True)
  predict_state: bool = nnx.static(kw_only=True, default=False)
  num_history_steps: int = nnx.static(kw_only=True, default=0)
  input_coords: dict[str, cx.Coordinate] = nnx.static(
      kw_only=True, default_factory=dict
  )
  assimilate_transform: typing.Transform = nnx.data(
      kw_only=True, default_factory=transforms.Identity
  )
  censor_transform: typing.Transform = nnx.data(
      kw_only=True, default_factory=transforms.Identity
  )
  sim_units: units.SimUnits = nnx.static(kw_only=True, default=units.SI_UNITS)
  nondim_transform: transforms.Nondimensionalize | None = nnx.data(
      kw_only=True, default=None
  )
  redim_transform: transforms.Redimensionalize | None = nnx.data(
      kw_only=True, default=None
  )
  update_diagnostics: dict[str, diagnostics.DiagnosticModule] = nnx.data(
      kw_only=True, default_factory=dict
  )
  step_end_diagnostics: dict[str, diagnostics.DiagnosticModule] = nnx.data(
      kw_only=True, default_factory=dict
  )
  remat_advance: bool = False
  prognostics_data_key: str = nnx.static(kw_only=True, default='prognostics')
  operators: dict[str, observation_operators.ObservationOperatorABC] = nnx.data(
      default_factory=dict
  )

  def __post_init__(self):
    if self.nondim_transform is None:
      self.nondim_transform = transforms.Nondimensionalize(
          self.sim_units,
          inputs_to_units_mapping=defaults.DEFAULT_UNITS_MAPPING,
      )
    if self.redim_transform is None:
      self.redim_transform = transforms.Redimensionalize(
          self.sim_units,
          inputs_to_units_mapping=defaults.DEFAULT_UNITS_MAPPING,
      )
    expected_keys = {*self.prognostics_coords, 'time'}
    if set(self.var_to_dataset_keys) != expected_keys:
      raise ValueError(
          f'var_to_dataset_keys must have entries for {sorted(expected_keys)},'
          f' got {sorted(self.var_to_dataset_keys)}.'
      )
    prognostic_data_keys = {
        self.prognostics_data_key,
        self.nondim_prognostics_data_key,
    }
    if overlapping := prognostic_data_keys & set(
        self.var_to_dataset_keys.values()
    ):
      raise ValueError(
          f'var_to_dataset_keys cannot use prognostic dataset keys '
          f'{sorted(overlapping)}.'
      )
    if unknown := set(self.input_coords) - set(self.prognostics_coords):
      raise ValueError(
          f'input_coords has entries for {sorted(unknown)} that are not '
          f'prognostic variables {list(self.prognostics_coords)}.'
      )
    # Number of `advance` steps per update of the prognostics.
    self.steps_per_update = time_utils.exact_timedelta_ratio(
        self.update_timestep,
        self.model_timestep,
        f'update_timestep ({self.update_timestep}) must be a multiple of '
        f'model_timestep ({self.model_timestep}).',
    )
    if self.num_history_steps < 0:
      raise ValueError(
          'num_history_steps must be non-negative, got '
          f'{self.num_history_steps}.'
      )
    nans_like = lambda c: cx.field(jnp.full(c.shape, jnp.nan), c)
    self.prognostics = typing.Prognostic(
        {k: nans_like(c) for k, c in self._stored_coords().items()}
    )
    self.time = typing.Prognostic(cx.field(jdt.to_datetime('1900-01-01T00:00')))
    zero = jnp.array(0, dtype=jnp.int32)
    self.steps_since_update = typing.Prognostic(cx.field(zero))

  @property
  def timestep(self) -> np.timedelta64:
    """Returns the time interval by which `advance` evolves the model."""
    return self.model_timestep

  @property
  def nondim_prognostics_data_key(self) -> str:
    """Dataset key in `inputs` with nondimensional prognostic variables."""
    return f'nondim_{self.prognostics_data_key}'

  @property
  def history_axis(self) -> coordinates.TimeDelta:
    """Returns timedeltas of values stored in `prognostics`, ending at 0."""
    deltas = np.arange(-self.num_history_steps, 1) * self.update_timestep
    return coordinates.TimeDelta(deltas)

  def _stored_coords(self) -> dict[str, cx.Coordinate]:
    """Returns coordinates of `prognostics` stored in the model state."""
    if not self.num_history_steps:
      return self.prognostics_coords
    history = self.history_axis
    return {
        k: cx.coords.compose(history, c)
        for k, c in self.prognostics_coords.items()
    }

  def current_prognostics(self) -> dict[str, cx.Field]:
    """Returns `prognostics` at the current time, i.e. without history."""
    prognostics = self.prognostics.get_value()
    if not self.num_history_steps:
      return prognostics
    return {k: v.isel(timedelta=-1) for k, v in prognostics.items()}

  @property
  def _prognostics_inputs_spec(self) -> dict[str, data_specs.CoordSpec]:
    """Returns specs of prognostic inputs from which the state is set."""
    with_timedelta = data_specs.CoordSpec.with_given_timedelta
    specs = {
        k: with_timedelta(
            coord,
            self.history_axis.deltas,
            optional_timedelta=not self.num_history_steps,
        )
        for k, coord in self.prognostics_coords.items()
    }
    specs['time'] = with_timedelta(cx.Scalar(), optional_timedelta=True)
    return specs

  @property
  def inputs_spec(  # pyrefly: ignore[bad-override]
      self,
  ) -> dict[str, dict[str, data_specs.OptionalSpec[data_specs.CoordSpec]]]:
    """Returns specs of all inputs supported by `assimilate`."""
    with_timedelta = data_specs.CoordSpec.with_given_timedelta
    specs = {}
    for k, coord in (self.prognostics_coords | self.input_coords).items():
      spec = with_timedelta(
          coord,
          self.history_axis.deltas,
          optional_timedelta=not self.num_history_steps,
      )
      specs.setdefault(self.var_to_dataset_keys[k], {})[k] = spec
    time_spec = with_timedelta(cx.Scalar(), optional_timedelta=True)
    specs.setdefault(self.var_to_dataset_keys['time'], {})['time'] = time_spec
    specs |= {
        self.prognostics_data_key: self._prognostics_inputs_spec,
        self.nondim_prognostics_data_key: self._prognostics_inputs_spec,
    }
    return jax.tree.map(
        data_specs.OptionalSpec,
        specs,
        is_leaf=lambda x: isinstance(x, data_specs.CoordSpec),
    )

  def _apply_update(self) -> None:
    """Updates the prognostics using `update_transform`."""
    prognostics = self.prognostics.get_value()
    # Updates are applied at the end of the update interval, but are evaluated
    # with `time` at its start, see class docstring.
    dt = jdt.to_timedelta(self.model_timestep)
    steps_since_update = self.steps_since_update.get_value()
    start_time = self.time.get_value() - dt * steps_since_update
    inputs = prognostics | {'time': start_time}
    updates = self.update_transform(inputs)
    _check_coordinates(
        updates, self.prognostics_coords, 'output of update_transform'
    )
    stepping.update_diagnostics(self.update_diagnostics, updates, inputs)
    stepping.advance_diagnostic_clocks(
        self.update_timestep, self.update_diagnostics
    )
    if self.predict_state:
      next_prognostics_slice = {k: updates[k] for k in self.prognostics_coords}
    else:
      assert self.nondim_transform is not None
      sim_units = self.nondim_transform.sim_units
      update_dt = sim_units.nondimensionalize_timedelta64(self.update_timestep)
      next_prognostics_slice = {
          k: v + updates[k] * update_dt
          for k, v in self.current_prognostics().items()
      }
    next_prognostics_slice = self.censor_transform(next_prognostics_slice)
    if self.num_history_steps:
      next_prognostics = {
          k: _shift_in(prognostics[k], v)
          for k, v in next_prognostics_slice.items()
      }
    else:
      next_prognostics = next_prognostics_slice
    self.prognostics.set_value(next_prognostics)
    for randomness in module_utils.retrieve_subclass_modules(
        self, random_processes.RandomProcessModule
    ):
      randomness.advance()

  def advance(self) -> None:
    """Evolves the model state by `model_timestep`."""

    def step_fn(model: LearnedUpdateModel):
      steps_since_update = model.steps_since_update.get_value()
      if model.steps_per_update == 1:
        model._apply_update()  # pylint: disable=protected-access
      else:
        nnx.cond(
            steps_since_update.data == model.steps_per_update - 1,
            lambda m: m._apply_update(),  # pylint: disable=protected-access
            lambda m: None,
            model,
        )
      time = model.time.get_value() + model.model_timestep
      model.time.set_value(time)
      next_step = (steps_since_update + 1) % model.steps_per_update
      model.steps_since_update.set_value(next_step)
      state = model.current_prognostics() | {'time': time}
      stepping.update_diagnostics(model.step_end_diagnostics, state, state)
      stepping.advance_diagnostic_clocks(
          model.model_timestep, model.step_end_diagnostics
      )

    if self.remat_advance:
      step_fn = nnx.remat(step_fn)
    step_fn(self)

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
    fields = {}
    for k, data_key in self.var_to_dataset_keys.items():
      dataset = _get_required(inputs, data_key, 'dataset')
      field = _get_required(dataset, k, f'variable in {data_key!r}')
      if self.num_history_steps and k != 'time':
        fields[k] = time_utils.select_timedeltas(field, self.history_axis)
      else:
        fields[k] = time_utils.select_zero_timedelta(field)
    assert self.nondim_transform is not None
    fields = self.nondim_transform(fields)
    state = self.assimilate_transform(fields)
    if set(state) != set(fields):
      raise ValueError(
          'assimilate_transform must preserve keys, got outputs '
          f'{sorted(state)} for inputs {sorted(fields)}.'
      )
    time = state.pop('time')
    _check_coordinates(state, self._stored_coords(), 'assimilated variable')
    self.prognostics.set_value(state)
    self.time.set_value(time)
    self.steps_since_update.set_value(cx.field(jnp.array(0, dtype=jnp.int32)))

  def _assimilate_prognostics(
      self,
      inputs: Mapping[str, cx.Field],
      nondimensionalize: bool,
  ) -> None:
    """Sets the model state to `inputs` with optional nondimensionalization."""
    expected_keys = [*self.prognostics_coords, 'time']
    if missing := [k for k in expected_keys if k not in inputs]:
      raise ValueError(
          f'Prognostic variables {missing} not found in inputs. '
          f'Available variables: {list(inputs)}'
      )
    state = {}
    for k in self.prognostics_coords:
      if self.num_history_steps:
        state[k] = time_utils.select_timedeltas(inputs[k], self.history_axis)
      else:
        state[k] = time_utils.select_zero_timedelta(inputs[k])
    time = time_utils.select_zero_timedelta(inputs['time'])
    if nondimensionalize:
      assert self.nondim_transform is not None
      state = self.nondim_transform(state)
    _check_coordinates(state, self._stored_coords(), 'Prognostic variable')
    self.prognostics.set_value(state)
    self.time.set_value(time)
    self.steps_since_update.set_value(cx.field(jnp.array(0, dtype=jnp.int32)))

  def observe(self, queries: typing.Queries) -> typing.Observation:
    assert self.redim_transform is not None
    prognostics = self.current_prognostics()
    operators = self.operators | {
        'diagnostics': observation_operators.DataObservationOperator(
            self.diagnostic_values()
        )
    }
    outputs = {}
    for k, query in queries.items():
      operator = _get_required(operators, k, 'observation operator')
      outputs[k] = self.redim_transform(operator.observe(prognostics, query))
    return outputs
