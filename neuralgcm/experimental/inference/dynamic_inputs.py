# Copyright 2025 Google LLC
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
"""API for defining dynamic inputs (forcings) used for inference."""

from __future__ import annotations

import abc
import concurrent.futures
import dataclasses
import functools
import operator
from typing import Generic, TypeVar

import jax
import numpy as np
import numpy.typing as npt
import xarray


# TODO(shoyer): remove dict[str, xarray.Dataset] in favor of DataTree once it is
# supported by all APIs
XarrayData = TypeVar(
    'XarrayData', xarray.Dataset, xarray.DataTree, dict[str, xarray.Dataset]
)


def _map_over_datasets(func, *data: XarrayData) -> XarrayData:
  """Map a function that acts on xarray.Dataset objects."""
  if isinstance(data[0], xarray.Dataset):
    return func(*data)
  elif isinstance(data[0], xarray.DataTree):
    return xarray.map_over_datasets(func, *data)
  else:
    return jax.tree.map(
        func, *data, is_leaf=lambda x: isinstance(x, xarray.Dataset)
    )


def _get_dataset_or_datatree(
    data: XarrayData,
) -> xarray.Dataset | xarray.DataTree:
  """Get a dataset or datatree from a dataset, datatree, or dict[str, dataset]."""
  if isinstance(data, xarray.Dataset | xarray.DataTree):
    return data
  else:
    return next(iter(data.values()))


def _get_climatology(
    climatology: XarrayData, times: np.datetime64 | npt.NDArray[np.datetime64]
) -> XarrayData:
  """Index along a 'time' dimension from a dataset in terms of 'dayofyear'."""
  times = np.asarray(times)
  match times.ndim:
    case 0:
      times = xarray.DataArray(times)
    case 1:
      times = xarray.DataArray(times, dims=['time'], coords={'time': times})
    case _:
      raise ValueError(f'Times must have 0 or 1 dimensions. Got {times.ndim=}')

  # This uses Xarray's vectorized indexing to produce an output dataset with a
  # 'time' dimension.
  indexers = {'dayofyear': times.dt.dayofyear}
  if 'hour' in _get_dataset_or_datatree(climatology).dims:
    indexers['hour'] = times.dt.hour
  return _map_over_datasets(
      lambda x: x.sel(indexers).reset_coords('dayofyear', drop=True),
      climatology,
  )


@dataclasses.dataclass(frozen=True)
class DynamicInputs(abc.ABC, Generic[XarrayData]):
  """Base class for describing how to calculate dynamic inputs (i.e., forcings).

  Subclasses of DynamicInputs can be reused to generate multiple forecast
  objects, each of which can be repeated queried to sample data at different
  times, e.g., to forecast 100 days in 10 day blocks:

      dynamic_inputs = DynamicInputs(...)
      period = np.timedelta64(10, 'D')
      forecast = dynamic_inputs.get_forecast(init_time)
      for i in range(10):
        start = init_time + i * period
        stop = init_time + (i + 1) * period
        data = forecast.get_data(start, stop)

  Attributes:
    full_data: dataset defining dynamic inputs at explicitly available times,
      which must include at `init_time`.
    climatology: optional dataset defining climatological values for dynamic
      inputs. If provided, must have a 'dayofyear' coordinate that spans 1 to
      366.
    update_freq: optional time delta between consecutive dynamic inputs. By
      default, dynamic inputs are sampled at the same frequency as `full_data`.
  """

  full_data: XarrayData | None
  climatology: XarrayData | None
  update_freq: np.timedelta64 | None

  def __post_init__(self):
    if (climatology := self.climatology) is not None:
      coords = _get_dataset_or_datatree(climatology).coords
      expected_dayofyear = np.arange(1, 367)
      if 'dayofyear' not in coords:
        raise ValueError(
            "climatology does not have a 'dayofyear' coordinates:"
            f' {climatology}'
        )
      actual_dayofyear = coords['dayofyear'].data
      if not np.array_equal(actual_dayofyear, expected_dayofyear):
        raise ValueError(
            'dayofyear coordinate must include 1 through 366, got:'
            f' {actual_dayofyear}'
        )

  @abc.abstractmethod
  def get_forecast(self, init_time: np.datetime64) -> _Forecast[XarrayData]:
    """Returns a forecast of dynamic inputs for a fixed initial time."""
    return NotImplementedError()


@dataclasses.dataclass(frozen=True)
class _Forecast(abc.ABC, Generic[XarrayData]):
  """Forecast of dynamic inputs for a fixed initial time."""

  full_data: XarrayData | None
  climatology: XarrayData | None
  update_freq: np.timedelta64
  init_time: np.datetime64

  def _get_lead_times(
      self, lead_start: np.timedelta64, lead_stop: np.timedelta64
  ) -> npt.NDArray[np.timedelta64]:
    """Returns lead times from lead_start to lead_stop with update_freq."""
    return np.arange(lead_start, lead_stop, self.update_freq)

  @functools.cached_property
  def _executor(self) -> concurrent.futures.ThreadPoolExecutor:
    # Use a large thread pool for reading data variables in parallel.
    # This should suffice for maximum concurrency when reading from Zarr, as
    # long as the Zarr store uses internal concurrency for loading each
    # variable.
    return concurrent.futures.ThreadPoolExecutor(max_workers=64)

  def _read_in_parallel(self, inputs: XarrayData) -> XarrayData:
    """Read data in parallel, using a large thread pool for concurrency."""
    get_values = lambda x: x.values

    def submit_get_values(ds):
      return {k: self._executor.submit(get_values, v) for k, v in ds.items()}

    def copy_with_results(ds, futures):
      return ds.copy(data={k: f.result() for k, f in futures.items()})

    futures = _map_over_datasets(submit_get_values, inputs)
    outputs = _map_over_datasets(copy_with_results, inputs, futures)
    return outputs

  @abc.abstractmethod
  def get_data(
      self, lead_start: np.timedelta64, lead_stop: np.timedelta64
  ) -> XarrayData:
    """Get forecasts of dynamics inputs for the specified lead times.

    Args:
      lead_start: start of lead times to forecast, inclusive.
      lead_stop: end of lead times to forecast, exclusive.

    Returns:
      A dataset with a 'time' dimension containing specified forecasts.
    """
    raise NotImplementedError()


def _get_forecast(
    forecast_cls: type[_Forecast[XarrayData]],
    model: DynamicInputs[XarrayData],
    init_time: np.datetime64,
) -> _Forecast[XarrayData]:
  return forecast_cls(
      full_data=model.full_data,
      climatology=model.climatology,
      update_freq=model.update_freq,
      init_time=init_time,
  )


@dataclasses.dataclass(frozen=True)
class _PersistenceForecast(_Forecast[XarrayData]):  # pylint: disable=missing-class-docstring

  @functools.cached_property
  def init_value(self) -> XarrayData:
    assert (full_data := self.full_data) is not None
    return self._read_in_parallel(
        _map_over_datasets(lambda x: x.sel(time=self.init_time), full_data)
    )

  def get_data(
      self, lead_start: np.timedelta64, lead_stop: np.timedelta64
  ) -> XarrayData:
    lead_times = self._get_lead_times(lead_start, lead_stop)
    new_times = self.init_time + lead_times
    return _map_over_datasets(
        lambda x: x.expand_dims(time=new_times), self.init_value
    )


@dataclasses.dataclass(frozen=True)
class Persistence(DynamicInputs[XarrayData]):
  """Persistent the initial state for dynamic inputs."""

  def __post_init__(self):
    if self.full_data is None:
      raise TypeError('full_data is required for Persistence')
    super().__post_init__()

  def get_forecast(self, init_time: np.datetime64) -> _Forecast[XarrayData]:
    return _get_forecast(_PersistenceForecast, self, init_time)


@dataclasses.dataclass(frozen=True)
class _PrescribedForecast(_Forecast[XarrayData]):

  def get_data(
      self, lead_start: np.timedelta64, lead_stop: np.timedelta64
  ) -> XarrayData:
    lead_times = self._get_lead_times(lead_start, lead_stop)
    times = self.init_time + lead_times
    assert (full_data := self.full_data) is not None
    return self._read_in_parallel(
        _map_over_datasets(lambda x: x.sel(time=times), full_data)
    )


@dataclasses.dataclass(frozen=True)
class Prescribed(DynamicInputs[XarrayData]):
  """Prescribed time-varying dynamic inputs."""

  def __post_init__(self):
    if self.full_data is None:
      raise TypeError('full_data is required for Prescribed')
    super().__post_init__()

  def get_forecast(self, init_time: np.datetime64) -> _Forecast[XarrayData]:
    return _get_forecast(_PrescribedForecast, self, init_time)


@dataclasses.dataclass(frozen=True)
class _ClimatologyForecast(_Forecast[XarrayData]):

  def get_data(
      self, lead_start: np.timedelta64, lead_stop: np.timedelta64
  ) -> XarrayData:
    lead_times = self._get_lead_times(lead_start, lead_stop)
    times = self.init_time + lead_times
    return self._read_in_parallel(_get_climatology(self.climatology, times))


@dataclasses.dataclass(frozen=True)
class Climatology(DynamicInputs[XarrayData]):
  """Climatological dynamic inputs."""

  def __post_init__(self):
    if self.climatology is None:
      raise TypeError('climatology is required for Climatology')
    super().__post_init__()

  def get_forecast(self, init_time: np.datetime64) -> _Forecast[XarrayData]:
    return _get_forecast(_ClimatologyForecast, self, init_time)


@dataclasses.dataclass(frozen=True)
class _AnomalyPersistenceForecast(_Forecast[XarrayData]):  # pylint: disable=missing-class-docstring

  @functools.cached_property
  def init_anomaly(self) -> XarrayData:
    assert (full_data := self.full_data) is not None
    init_climatology = self._read_in_parallel(
        _get_climatology(self.climatology, self.init_time)
    )
    init_data = self._read_in_parallel(
        _map_over_datasets(lambda x: x.sel(time=self.init_time), full_data)
    )
    return _map_over_datasets(operator.sub, init_data, init_climatology)

  def get_data(
      self, lead_start: np.timedelta64, lead_stop: np.timedelta64
  ) -> XarrayData:
    lead_times = self._get_lead_times(lead_start, lead_stop)
    valid_clim = self._read_in_parallel(
        _get_climatology(self.climatology, self.init_time + lead_times)
    )
    forecast = _map_over_datasets(operator.add, self.init_anomaly, valid_clim)
    # TODO(shoyer): Make a more generic solution for clipping anomaly forecasts.
    if 'sea_ice_cover' in forecast:
      forecast['sea_ice_cover'] = forecast['sea_ice_cover'].clip(0, 1)
    return forecast


@dataclasses.dataclass(frozen=True)
class AnomalyPersistence(DynamicInputs[XarrayData]):
  """Persistent the initial difference from climatology for dynamic inputs."""

  def __post_init__(self):
    if self.full_data is None:
      raise TypeError('full_data is required for AnomalyPersistence')
    if self.climatology is None:
      raise TypeError('climatology is required for AnomalyPersistence')
    super().__post_init__()

  def get_forecast(self, init_time: np.datetime64) -> _Forecast[XarrayData]:
    return _get_forecast(_AnomalyPersistenceForecast, self, init_time)
