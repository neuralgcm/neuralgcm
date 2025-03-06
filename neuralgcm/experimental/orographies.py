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

"""Modules that hold orographic data."""

import dataclasses

from dinosaur import xarray_utils as dino_xarray_utils
from flax import nnx
import jax.numpy as jnp
from neuralgcm.experimental import coordax as cx
from neuralgcm.experimental import coordinates
from neuralgcm.experimental import interpolators
from neuralgcm.experimental import parallelism
from neuralgcm.experimental import spatial_filters
from neuralgcm.experimental import typing
from neuralgcm.experimental import units
from neuralgcm.experimental import xarray_utils
import xarray


class OrographyVariable(nnx.Variable):
  """Variable class for orography data."""


class ModalOrography(nnx.Module):
  """Orogrphay module that provoides elevation in modal representation."""

  def __init__(
      self,
      *,
      grid: coordinates.SphericalHarmonicGrid,
      initializer: nnx.initializers.Initializer = nnx.initializers.zeros_init(),
      mesh: parallelism.Mesh,
      rngs: nnx.Rngs,
  ):
    self.grid = grid
    self.opt_grid = dataclasses.replace(grid, spmd_mesh=mesh.spmd_mesh)
    modal_shape_1d = (grid.ylm_grid.mask.sum(),)
    self.orography = OrographyVariable(initializer(rngs, modal_shape_1d))

  @property
  def nodal_orography(self) -> typing.Array:
    padded_modal_orography = self.modal_orography
    return self.opt_grid.ylm_grid.to_nodal(padded_modal_orography)

  @property
  def modal_orography(self) -> typing.Array:
    """Returns orography converted to modal representation with filtering."""
    ylm_grid = self.opt_grid.ylm_grid
    mask = ylm_grid.mask
    modal_orography_2d = jnp.zeros(ylm_grid.modal_shape)
    return modal_orography_2d.at[mask].set(self.orography.value)

  def update_orography_from_data(
      self,
      dataset: xarray.Dataset,
      sim_units: units.SimUnits,
      spatial_filter=None,
  ):
    """Updates ``self.orography`` with filtered orography from dataset."""
    # TODO(dkochkov) use units attr on dataset with default to `meter` here.
    if spatial_filter is None:
      spatial_filter = lambda x: x
    nodal_orography = xarray_utils.nodal_orography_from_ds(dataset)
    nodal_orography = xarray_utils.xarray_nondimensionalize(
        nodal_orography, sim_units
    )
    nodal_orography = coordinates.field_from_xarray(nodal_orography)
    data_grid = cx.get_coordinate(nodal_orography)
    # TODO(dkochkov) Introduce objects for specifying nodal-modal conversions.
    ylm_grid = dino_xarray_utils.LINEAR_SHAPE_TO_GRID_DICT[data_grid.shape](
        spherical_harmonics_impl=self.grid.spherical_harmonics_impl
    )
    data_grid = coordinates.LonLatGrid.from_dinosaur_grid(ylm_grid)
    nodal_orography = nodal_orography.data
    if not isinstance(spatial_filter, spatial_filters.ModalSpatialFilter):
      nodal_orography = spatial_filter(nodal_orography)

    data_modal_grid = data_grid.to_spherical_harmonic_grid()
    modal_orography = data_modal_grid.ylm_grid.to_modal(nodal_orography)
    interpolator = interpolators.SpectralRegridder(self.grid)
    modal_orography = interpolator(cx.wrap(modal_orography, data_modal_grid))
    modal_orography = modal_orography.unwrap(self.grid)
    if isinstance(spatial_filter, spatial_filters.ModalSpatialFilter):
      modal_orography = spatial_filter.filter_modal(modal_orography)
    self.orography.value = modal_orography[self.grid.ylm_grid.mask]


class ModalOrographyWithCorrection(ModalOrography):
  """ModalOrography module with learned correction in modal representation."""

  def __init__(
      self,
      *,
      grid: coordinates.SphericalHarmonicGrid,
      initializer: nnx.initializers.Initializer = nnx.initializers.zeros_init(),
      correction_scale: float,
      correction_param_type: nnx.Param = nnx.Param,
      correction_initializer: nnx.initializers.Initializer = (
          nnx.initializers.truncated_normal()
      ),
      mesh: parallelism.Mesh,
      rngs: nnx.Rngs,
  ):
    super().__init__(grid=grid, initializer=initializer, mesh=mesh, rngs=rngs)
    self.correction_scale = correction_scale
    self.correction = correction_param_type(
        correction_initializer(rngs.params(), self.orography.shape)
    )

  @property
  def modal_orography(self) -> typing.Array:
    """Returns orography converted to modal representation with filtering."""
    ylm_grid = self.opt_grid.ylm_grid
    mask = ylm_grid.mask
    modal_orography_2d = jnp.zeros(ylm_grid.modal_shape)
    modal_orography_1d = (
        self.orography.value + self.correction.value * self.correction_scale
    )
    return modal_orography_2d.at[mask].set(modal_orography_1d)


class Orography(nnx.Module):
  """Orography module that provides elevation in real space."""

  def __init__(
      self,
      *,
      grid: coordinates.LonLatGrid,
      initializer: nnx.initializers.Initializer = nnx.initializers.zeros_init(),
      rngs: nnx.Rngs,
  ):
    self.grid = grid
    self.orography = OrographyVariable(
        initializer(rngs, grid.ylm_grid.nodal_shape)
    )

  @property
  def nodal_orography(self) -> typing.Array:
    return self.orography.value

  def update_orography_from_data(
      self,
      dataset: xarray.Dataset,
      sim_units: units.SimUnits,
      spatial_filter=None,
  ):
    """Updates ``self.orography`` with filtered orography from dataset."""
    # TODO(dkochkov) use units attr on dataset with default to `meter` here.
    if spatial_filter is None:
      spatial_filter = lambda x: x
    nodal_orography = xarray_utils.nodal_orography_from_ds(dataset)
    nodal_orography = xarray_utils.xarray_nondimensionalize(
        nodal_orography, sim_units
    )
    nodal_orography = coordinates.field_from_xarray(nodal_orography)
    data_grid = cx.get_coordinate(nodal_orography)
    # TODO(dkochkov) Introduce objects for specifying nodal-modal conversions.
    ylm_grid = dino_xarray_utils.LINEAR_SHAPE_TO_GRID_DICT[data_grid.shape](
        spherical_harmonics_impl=self.grid.spherical_harmonics_impl
    )
    data_grid = coordinates.LonLatGrid.from_dinosaur_grid(ylm_grid)
    nodal_orography = nodal_orography.data
    nodal_orography = spatial_filter(nodal_orography)
    if data_grid != self.grid:
      raise ValueError(f'{data_grid=} does not match {self.grid=}.')
    self.orography.value = nodal_orography
