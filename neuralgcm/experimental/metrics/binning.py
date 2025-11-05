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

"""Defines classes that implement binning schemes for metric aggregation."""


from __future__ import annotations
import abc
import dataclasses
import coordax as cx
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass
class Binning(abc.ABC):
  """Abstract base class for binning."""

  bin_dim_name: str

  @abc.abstractmethod
  def create_bin_mask(
      self,
      field: cx.Field,
      field_name: str | None = None,
      context: dict[str, cx.Field] | None = None,
  ) -> cx.Field:
    """Creates a bin mask for a statistic.

    Args:
      field: The field to generate a binning mask for.
      field_name: The name of the field to generate a binning mask for.
      context: Optional context for coordinate-dependent binning.

    Returns:
      A boolean cx.Field that can be broadcast against the input field. It
      should contain a new dimension with the name `bin_dim_name`.
    """


@dataclasses.dataclass
class Regions(Binning):
  """Class for rectangular region binning."""

  regions: dict[str, tuple[tuple[int, int], tuple[int, int]]]

  def create_bin_mask(
      self,
      field: cx.Field,
      field_name: str | None = None,
      context: dict[str, cx.Field] | None = None,
  ) -> cx.Field:
    """Creates a bin mask for a set of named regions."""
    del context  # unused.
    lon_lat_dims = ('longitude', 'latitude')
    if not all(d in field.axes for d in lon_lat_dims):
      raise ValueError(f'{field.dims=} must have {lon_lat_dims} axes to bin.')

    lat = field.coord_fields['latitude']
    lon = field.coord_fields['longitude']
    masks = []
    region_names = []
    for region_name, (lat_lims, lon_lims) in self.regions.items():
      lat_mask = (lat >= lat_lims[0]) & (lat <= lat_lims[1])
      lon_in_deg = lon % 360
      lon_lims_in_deg = (lon_lims[0] % 360, lon_lims[1] % 360)
      if lon_lims_in_deg[1] > lon_lims_in_deg[0]:
        lon_mask = (lon_in_deg >= lon_lims_in_deg[0]) & (
            lon_in_deg <= lon_lims_in_deg[1]
        )
      else:  # wraps around the periodic boundary
        lon_mask = (lon_in_deg >= lon_lims_in_deg[0]) | (
            lon_in_deg <= lon_lims_in_deg[1]
        )
      masks.append(lat_mask & lon_mask)
      region_names.append(region_name)

    region_ax = cx.LabeledAxis(self.bin_dim_name, np.array(region_names))
    return cx.cmap(jnp.stack)(masks).tag(region_ax)
