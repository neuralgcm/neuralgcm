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

"""Coupled ocean-atmosphere and ocean-land-atmosphere models."""

import coordax as cx
from flax import nnx
from neuralgcm.base import model
from neuralgcm.base import operator_utils
from neuralgcm.base import units_utils
import numpy as np
from terrax.core import data_specs
from terrax.core import typing

# pylint: disable=g-classes-have-attributes

DataCoupler = operator_utils.DataCoupler


def _validate_coupler_units(
    coupler: DataCoupler | None,
    source: model.Model,
    target: model.Model,
    coupler_name: str,
) -> None:
  """Validates unit compatibility between `source` and `target` for `coupler`."""
  if coupler is None:
    return
  # Includes `time_data_key` of `MultiTimeDataCoupler`.
  data_keys = operator_utils.build_coupler_query(coupler).keys()
  nondim_keys = sorted(k for k in data_keys if k.startswith('nondim_'))
  if nondim_keys and source.sim_units != target.sim_units:
    raise ValueError(
        f'{coupler_name} uses nondimensional query keys {nondim_keys}, which '
        'requires matching `sim_units` between source '
        f'({source.sim_units}) and target ({target.sim_units}).'
    )
  dim_keys = sorted(k for k in data_keys if not k.startswith('nondim_'))
  assert source.sim_units is not None and target.sim_units is not None
  if dim_keys and source.sim_units.scale != target.sim_units.scale:
    assert source.units_mapping is not None
    coupler_query = operator_utils.build_coupler_query(coupler)
    coupler_vars: set[str] = set()
    for data_key in dim_keys:
      coupler_vars.update(coupler_query[data_key])
    coupler_vars.discard('time')
    missing = sorted(coupler_vars - set(source.units_mapping))
    if missing:
      raise ValueError(
          f'{coupler_name} converts {missing} between distinct `sim_units` '
          'scales, which requires their units in `units_mapping` of source '
          f'model {type(source).__name__}.'
      )


def _nondim_query_key(data_key: str) -> str:
  return data_key if data_key.startswith('nondim_') else f'nondim_{data_key}'


def _nondim_coupler_query(*couplers: DataCoupler | None) -> typing.Queries:
  """Builds the coupler query requesting nondimensional source observations."""
  query: typing.Queries = {}
  coupler_query = operator_utils.build_coupler_query(*couplers)
  for data_key, dataset_query in coupler_query.items():
    nondim_key = _nondim_query_key(data_key)
    query[nondim_key] = {**query.get(nondim_key, {}), **dataset_query}
  return query


def _nondim_for_coupler(
    observations: typing.Observation,
    coupler: DataCoupler,
    source: model.Model,
    target: model.Model,
) -> typing.Observation:
  """Selects nondimensional source `observations` for `coupler` on `target`.

  `observations` are the response of `source.observe` to
  `_nondim_coupler_query`. They are re-expressed in `target.sim_units` via
  `units_utils.units_passage`, which is a no-op when both components share
  the same scale.

  Args:
    observations: response of `source.observe` to `_nondim_coupler_query`.
    coupler: data coupler whose query selects the fields to pass to `target`.
    source: model component that produced `observations`.
    target: model component that consumes the selected fields.

  Returns:
    Observations keyed by `coupler` data keys, expressed in `target.sim_units`.
  """
  coupler_query = operator_utils.build_coupler_query(coupler)
  assert source.units_mapping is not None
  outputs: typing.Observation = {}
  for data_key, query in coupler_query.items():
    nondim_obs = observations[_nondim_query_key(data_key)]
    fields = {var: nondim_obs[var] for var in query}
    outputs[data_key] = units_utils.units_passage(
        fields, source.sim_units, target.sim_units, source.units_mapping
    )
  return outputs


@nnx.dataclass(kw_only=True)
class OceanAtmosphere(model.Model):
  """Model that couples ocean and atmosphere components.

  `OceanAtmosphere` couples an `ocean` component (which may also evolve sea ice)
  and an `atmosphere` component at a shared outer `model_timestep` via data
  couplers (`InstantDataCoupler` or `MultiTimeDataCoupler`). Either coupler may
  be `None` to configure one-way coupling.

  Coupler queries observe nondimensional fields from the source component and
  re-express them in the receiving component's `sim_units` via
  `units_utils.units_passage` (a no-op when both components share the same
  scale, and otherwise a static per-variable rescaling defined by the source
  component's units mapping). Coupler queries prefixed with `'nondim_'` require
  matching `sim_units` between the components.

  `assimilate(inputs)` initializes `ocean` and `atmosphere` from `inputs` and
  populates any active couplers with nondimensionalized initial observations.

  `advance()` evolves the coupled system by `model_timestep` in two phases:
    1. `atmosphere.advance()` steps the atmosphere using the ocean state at the
       start of the step, after which `atmosphere_to_ocean_coupler` (if set) is
       updated with nondimensionalized observations from `atmosphere`.
    2. `ocean.advance()` steps the ocean using the updated atmospheric fields
       (which may be step-ahead aggregated fields on an `InstantDataCoupler` or
       buffered snapshots on a `MultiTimeDataCoupler`), after which
       `ocean_to_atmosphere_coupler` (if set) is updated with
       nondimensionalized observations from `ocean`.
  Since `atmosphere_to_ocean_coupler` is updated before `ocean.advance()`, care
  is needed to ensure that the coupler accumulates all the necessary data
  expected by the ocean (e.g. endpoints of the time interval).

  `observe(queries)` dispatches each query key via
  `operator_utils.dispatch_observe_nondim` to top-level `operators` (evaluated
  on `prognostic_values()` merged across `ocean` and `atmosphere`), to the
  built-in `'diagnostics'` operator exposing all `DiagnosticModule` submodules
  across `ocean` and `atmosphere`, or to the unique component model whose
  `operators` contain the key. Observations are returned in the units of this
  model (physical units per `units_mapping`, or nondimensional `sim_units` for
  query keys prefixed with `'nondim_'`), see `model.Model`.

  Attributes:
    ocean: Ocean model component (including sea ice when applicable).
    atmosphere: Atmosphere model component.
    ocean_to_atmosphere_coupler: Optional coupler passing nondimensionalized
      fields observed from `ocean` to `atmosphere`.
    atmosphere_to_ocean_coupler: Optional coupler passing nondimensionalized
      fields observed from `atmosphere` to `ocean`.
    model_timestep: Outer time step of the coupled model. Defaults to
      `atmosphere.timestep` if `None`. Must match both `ocean.timestep` and
      `atmosphere.timestep`.
    sim_units: Simulation units of the coupled model. Defaults to
      `atmosphere.sim_units` if `None`.
    units_mapping: Physical units of variables observed from the coupled model.
      Defaults to the merged `units_mapping` of `ocean` and `atmosphere` if
      `None`.
    operators: Optional top-level observation operators on the coupled model.
    remat_advance: Whether to wrap `advance` with `nnx.remat`.
  """

  ocean: model.Model
  atmosphere: model.Model
  ocean_to_atmosphere_coupler: DataCoupler | None = nnx.data(default=None)
  atmosphere_to_ocean_coupler: DataCoupler | None = nnx.data(default=None)
  model_timestep: np.timedelta64 | None = nnx.static(default=None)
  remat_advance: bool = nnx.static(default=False)

  def __post_init__(self):
    if self.model_timestep is None:
      self.model_timestep = self.atmosphere.timestep
    if (
        self.model_timestep != self.ocean.timestep
        or self.model_timestep != self.atmosphere.timestep
    ):
      raise ValueError(
          'All model components must have the same timestep as '
          f'model_timestep ({self.model_timestep}), got '
          f'ocean.timestep={self.ocean.timestep} and '
          f'atmosphere.timestep={self.atmosphere.timestep}.'
      )
    operator_utils.setup_composite_units(self, [self.atmosphere, self.ocean])
    super().__post_init__()
    _validate_coupler_units(
        self.ocean_to_atmosphere_coupler,
        self.ocean,
        self.atmosphere,
        'ocean_to_atmosphere_coupler',
    )
    _validate_coupler_units(
        self.atmosphere_to_ocean_coupler,
        self.atmosphere,
        self.ocean,
        'atmosphere_to_ocean_coupler',
    )
    self.ocean_query = _nondim_coupler_query(self.ocean_to_atmosphere_coupler)
    self.atm_query = _nondim_coupler_query(self.atmosphere_to_ocean_coupler)
    _ = self.inputs_spec

  @property
  def timestep(self) -> np.timedelta64:
    assert self.model_timestep is not None
    return self.model_timestep

  def prognostic_values(self) -> dict[str, cx.Field]:
    return operator_utils.merge_prognostics(
        self.ocean, self.atmosphere, sim_units=self.sim_units
    )

  @nnx.remat
  def _observe_ocean(self) -> typing.Observation:
    assert self.ocean_to_atmosphere_coupler is not None
    return _nondim_for_coupler(
        self.ocean.observe(self.ocean_query),
        self.ocean_to_atmosphere_coupler,
        self.ocean,
        self.atmosphere,
    )

  @nnx.remat
  def _observe_atmosphere(self) -> typing.Observation:
    assert self.atmosphere_to_ocean_coupler is not None
    return _nondim_for_coupler(
        self.atmosphere.observe(self.atm_query),
        self.atmosphere_to_ocean_coupler,
        self.atmosphere,
        self.ocean,
    )

  def assimilate(self, inputs: dict[str, dict[str, cx.Field]]) -> None:
    self.ocean.assimilate(inputs)
    self.atmosphere.assimilate(inputs)
    if self.ocean_to_atmosphere_coupler is not None:
      self.ocean_to_atmosphere_coupler.update_fields(self._observe_ocean())
    if self.atmosphere_to_ocean_coupler is not None:
      self.atmosphere_to_ocean_coupler.update_fields(self._observe_atmosphere())

  def advance(self) -> None:
    # pylint: disable=protected-access
    def step_fn(coupled: OceanAtmosphere):
      # The atmosphere is advanced first and atm -> ocean coupler is updated.
      coupled.atmosphere.advance()
      if coupled.atmosphere_to_ocean_coupler is not None:
        coupled.atmosphere_to_ocean_coupler.update_fields(
            coupled._observe_atmosphere()
        )
      # Ocean is advanced with coupler updated to step-ahead values.
      coupled.ocean.advance()
      if coupled.ocean_to_atmosphere_coupler is not None:
        coupled.ocean_to_atmosphere_coupler.update_fields(
            coupled._observe_ocean()
        )

    # pylint: enable=protected-access
    if self.remat_advance:
      step_fn = nnx.remat(step_fn)
    step_fn(self)

  def observe_nondim(self, queries: typing.Queries) -> typing.Observation:
    return operator_utils.dispatch_observe_nondim(
        queries,
        composite=self,
        components={'ocean': self.ocean, 'atmosphere': self.atmosphere},
    )

  @property
  def inputs_spec(
      self,
  ) -> dict[str, dict[str, cx.Coordinate | data_specs.CoordLikeSpec]]:
    return operator_utils.merge_inputs_specs(
        self.ocean.inputs_spec, self.atmosphere.inputs_spec
    )


@nnx.dataclass(kw_only=True)
class OceanLandAtmosphere(model.Model):
  """Model that couples ocean, land, and atmosphere components.

  `OceanLandAtmosphere` couples `ocean` (including sea ice when applicable),
  `land`, and `atmosphere` components at a shared outer `model_timestep` via
  data couplers (`InstantDataCoupler` or `MultiTimeDataCoupler`). Any subset of
  couplers may be `None` to configure one-way or partial coupling.

  Coupler queries observe nondimensional fields from the source component and
  re-express them in the receiving component's `sim_units` via
  `units_utils.units_passage` (a no-op when both components share the same
  scale, and otherwise a static per-variable rescaling defined by the source
  component's units mapping). Coupler queries prefixed with `'nondim_'` require
  matching `sim_units` between the components.

  `assimilate(inputs)` initializes `ocean`, `land`, and `atmosphere` from
  `inputs` and populates any active couplers with nondimensionalized initial
  observations (merging `atmosphere_to_ocean_coupler` and
  `atmosphere_to_land_coupler` queries into a single `atmosphere.observe` call).

  `advance()` evolves the coupled system by `model_timestep` in two phases:
    1. `atmosphere.advance()` steps the atmosphere using surface states at the
       start of the step, after which `atmosphere` is observed once for
       `atm_query` and both `atmosphere_to_ocean_coupler` and
       `atmosphere_to_land_coupler` (when set) are updated.
    2. `ocean.advance()` and `land.advance()` step the surface components using
       the updated atmospheric fields, after which
       `ocean_to_atmosphere_coupler` and `land_to_atmosphere_coupler` (when
       set) are updated with nondimensionalized surface observations.
  Since `atmosphere-to-surface` couplers are updated before the surface
  components advance, care is needed to ensure that the coupler accumulates
  all the necessary data expected by the surface component (e.g. endpoints of
  the time interval).

  `observe(queries)` dispatches each query key via
  `operator_utils.dispatch_observe_nondim` to top-level `operators` (evaluated
  on `prognostic_values()` merged across `ocean`, `land`, and `atmosphere`), to
  the built-in `'diagnostics'` operator exposing all `DiagnosticModule`
  submodules across `ocean`, `land`, and `atmosphere`, or to the unique
  component model whose `operators` contain the key. Observations are returned
  in the units of this model (physical units per `units_mapping`, or
  nondimensional `sim_units` for query keys prefixed with `'nondim_'`), see
  `model.Model`.

  Attributes:
    ocean: Ocean model component (including sea ice when applicable).
    land: Land model component.
    atmosphere: Atmosphere model component.
    ocean_to_atmosphere_coupler: Optional coupler passing nondimensionalized
      fields observed from `ocean` to `atmosphere`.
    land_to_atmosphere_coupler: Optional coupler passing nondimensionalized
      fields observed from `land` to `atmosphere`.
    atmosphere_to_ocean_coupler: Optional coupler passing nondimensionalized
      fields observed from `atmosphere` to `ocean`.
    atmosphere_to_land_coupler: Optional coupler passing nondimensionalized
      fields observed from `atmosphere` to `land`.
    model_timestep: Outer time step of the coupled model. Defaults to
      `atmosphere.timestep` if `None`. Must match `ocean.timestep`,
      `land.timestep`, and `atmosphere.timestep`.
    sim_units: Simulation units of the coupled model. Defaults to
      `atmosphere.sim_units` if `None`.
    units_mapping: Physical units of variables observed from the coupled model.
      Defaults to the merged `units_mapping` of `ocean`, `land`, and
      `atmosphere` if `None`.
    operators: Optional top-level observation operators on the coupled model.
    remat_advance: Whether to wrap `advance` with `nnx.remat`.
  """

  ocean: model.Model
  land: model.Model
  atmosphere: model.Model
  ocean_to_atmosphere_coupler: DataCoupler | None = nnx.data(default=None)
  land_to_atmosphere_coupler: DataCoupler | None = nnx.data(default=None)
  atmosphere_to_ocean_coupler: DataCoupler | None = nnx.data(default=None)
  atmosphere_to_land_coupler: DataCoupler | None = nnx.data(default=None)
  model_timestep: np.timedelta64 | None = nnx.static(default=None)
  remat_advance: bool = nnx.static(default=False)

  def __post_init__(self):
    if self.model_timestep is None:
      self.model_timestep = self.atmosphere.timestep
    if (
        self.model_timestep != self.ocean.timestep
        or self.model_timestep != self.land.timestep
        or self.model_timestep != self.atmosphere.timestep
    ):
      raise ValueError(
          'All model components must have the same timestep as '
          f'model_timestep ({self.model_timestep}), got '
          f'ocean.timestep={self.ocean.timestep}, '
          f'land.timestep={self.land.timestep}, and '
          f'atmosphere.timestep={self.atmosphere.timestep}.'
      )
    operator_utils.setup_composite_units(
        self, [self.atmosphere, self.ocean, self.land]
    )
    super().__post_init__()
    _validate_coupler_units(
        self.ocean_to_atmosphere_coupler,
        self.ocean,
        self.atmosphere,
        'ocean_to_atmosphere_coupler',
    )
    _validate_coupler_units(
        self.land_to_atmosphere_coupler,
        self.land,
        self.atmosphere,
        'land_to_atmosphere_coupler',
    )
    _validate_coupler_units(
        self.atmosphere_to_ocean_coupler,
        self.atmosphere,
        self.ocean,
        'atmosphere_to_ocean_coupler',
    )
    _validate_coupler_units(
        self.atmosphere_to_land_coupler,
        self.atmosphere,
        self.land,
        'atmosphere_to_land_coupler',
    )
    self.ocean_query = _nondim_coupler_query(self.ocean_to_atmosphere_coupler)
    self.land_query = _nondim_coupler_query(self.land_to_atmosphere_coupler)
    self.atm_query = _nondim_coupler_query(
        self.atmosphere_to_ocean_coupler,
        self.atmosphere_to_land_coupler,
    )
    _ = self.inputs_spec

  @property
  def timestep(self) -> np.timedelta64:
    assert self.model_timestep is not None
    return self.model_timestep

  def prognostic_values(self) -> dict[str, cx.Field]:
    return operator_utils.merge_prognostics(
        self.ocean, self.land, self.atmosphere, sim_units=self.sim_units
    )

  @nnx.remat
  def _observe_ocean(self) -> typing.Observation:
    assert self.ocean_to_atmosphere_coupler is not None
    return _nondim_for_coupler(
        self.ocean.observe(self.ocean_query),
        self.ocean_to_atmosphere_coupler,
        self.ocean,
        self.atmosphere,
    )

  @nnx.remat
  def _observe_land(self) -> typing.Observation:
    assert self.land_to_atmosphere_coupler is not None
    return _nondim_for_coupler(
        self.land.observe(self.land_query),
        self.land_to_atmosphere_coupler,
        self.land,
        self.atmosphere,
    )

  @nnx.remat
  def _observe_atmosphere(
      self,
  ) -> tuple[typing.Observation | None, typing.Observation | None]:
    atmosphere_values = self.atmosphere.observe(self.atm_query)
    to_ocean = None
    if self.atmosphere_to_ocean_coupler is not None:
      to_ocean = _nondim_for_coupler(
          atmosphere_values,
          self.atmosphere_to_ocean_coupler,
          self.atmosphere,
          self.ocean,
      )
    to_land = None
    if self.atmosphere_to_land_coupler is not None:
      to_land = _nondim_for_coupler(
          atmosphere_values,
          self.atmosphere_to_land_coupler,
          self.atmosphere,
          self.land,
      )
    return to_ocean, to_land

  def assimilate(self, inputs: dict[str, dict[str, cx.Field]]) -> None:
    self.ocean.assimilate(inputs)
    self.land.assimilate(inputs)
    self.atmosphere.assimilate(inputs)
    if self.ocean_to_atmosphere_coupler is not None:
      self.ocean_to_atmosphere_coupler.update_fields(self._observe_ocean())
    if self.land_to_atmosphere_coupler is not None:
      self.land_to_atmosphere_coupler.update_fields(self._observe_land())
    if self.atm_query:
      to_ocean, to_land = self._observe_atmosphere()
      if self.atmosphere_to_ocean_coupler is not None:
        assert to_ocean is not None
        self.atmosphere_to_ocean_coupler.update_fields(to_ocean)
      if self.atmosphere_to_land_coupler is not None:
        assert to_land is not None
        self.atmosphere_to_land_coupler.update_fields(to_land)

  def advance(self) -> None:
    # pylint: disable=protected-access
    def step_fn(coupled: OceanLandAtmosphere):
      # The atmosphere is advanced first and atm -> ocean/land couplers updated.
      coupled.atmosphere.advance()
      if coupled.atm_query:
        to_ocean, to_land = coupled._observe_atmosphere()
        if coupled.atmosphere_to_ocean_coupler is not None:
          assert to_ocean is not None
          coupled.atmosphere_to_ocean_coupler.update_fields(to_ocean)
        if coupled.atmosphere_to_land_coupler is not None:
          assert to_land is not None
          coupled.atmosphere_to_land_coupler.update_fields(to_land)
      # Ocean and land are advanced with couplers updated to step-ahead values.
      coupled.ocean.advance()
      coupled.land.advance()
      if coupled.ocean_to_atmosphere_coupler is not None:
        coupled.ocean_to_atmosphere_coupler.update_fields(
            coupled._observe_ocean()
        )
      if coupled.land_to_atmosphere_coupler is not None:
        coupled.land_to_atmosphere_coupler.update_fields(
            coupled._observe_land()
        )

    # pylint: enable=protected-access
    if self.remat_advance:
      step_fn = nnx.remat(step_fn)
    step_fn(self)

  def observe_nondim(self, queries: typing.Queries) -> typing.Observation:
    return operator_utils.dispatch_observe_nondim(
        queries,
        composite=self,
        components={
            'ocean': self.ocean,
            'land': self.land,
            'atmosphere': self.atmosphere,
        },
    )

  @property
  def inputs_spec(
      self,
  ) -> dict[str, dict[str, cx.Coordinate | data_specs.CoordLikeSpec]]:
    return operator_utils.merge_inputs_specs(
        self.ocean.inputs_spec,
        self.land.inputs_spec,
        self.atmosphere.inputs_spec,
    )
