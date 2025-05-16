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

from dinosaur import xarray_utils as dino_xarray_utils
from flax import nnx
import jax.numpy as jnp
from neuralgcm.experimental import coordax as cx
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import interpolators
from neuralgcm.experimental.core import spatial_filters
from neuralgcm.experimental.core import spherical_transforms
from neuralgcm.experimental.core import typing
from neuralgcm.experimental.core import units
from neuralgcm.experimental.core import xarray_utils
import xarray


class OrographyVariable(nnx.Variable):
  """Variable class for orography data."""


class ModalOrography(nnx.Module):
  """Orogrphay module that provoides elevation in modal representation."""

  def __init__(
      self,
      *,
      ylm_transform: spherical_transforms.SphericalHarmonicsTransform,
      initializer: nnx.initializers.Initializer = nnx.initializers.zeros_init(),
      rngs: nnx.Rngs,
  ):
    self.ylm_transform = ylm_transform
    # TODO(dkochkov): add mask field to SphericalHarmonicGrid.
    modal_shape_1d = (ylm_transform.dinosaur_grid.mask.sum(),)
    self.orography = OrographyVariable(initializer(rngs, modal_shape_1d))

  @property
  def nodal_orography(self) -> typing.Array:
    return self.ylm_transform.to_nodal_array(self.modal_orography)

  @property
  def modal_orography(self) -> typing.Array:
    """Returns orography converted to modal representation with filtering."""
    # TODO(dkochkov): add mask field to SphericalHarmonicGrid.
    mask = self.ylm_transform.dinosaur_grid.mask
    modal_orography_2d = jnp.zeros(self.ylm_transform.modal_grid.shape)
    return modal_orography_2d.at[mask].set(self.orography.value)

  def update_orography_from_data(
      self,
      dataset: xarray.Dataset,
      data_ylm_transform: spherical_transforms.SphericalHarmonicsTransform,
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
    nodal_orography = nodal_orography.unwrap(data_ylm_transform.nodal_grid)
    if not isinstance(spatial_filter, spatial_filters.ModalSpatialFilter):
      nodal_orography = spatial_filter(nodal_orography)
    modal_orography = data_ylm_transform.to_modal_array(nodal_orography)
    interpolator = interpolators.SpectralRegridder(
        self.ylm_transform.modal_grid
    )
    modal_orography = interpolator(
        cx.wrap(modal_orography, data_ylm_transform.modal_grid)
    )
    modal_orography = modal_orography.unwrap(self.ylm_transform.modal_grid)
    if isinstance(spatial_filter, spatial_filters.ModalSpatialFilter):
      modal_orography = spatial_filter.filter_modal(modal_orography)
    # TODO(dkochkov): add mask field to SphericalHarmonicGrid.
    self.orography.value = modal_orography[
        self.ylm_transform.dinosaur_grid.mask
    ]


class ModalOrographyWithCorrection(ModalOrography):
  """ModalOrography module with learned correction in modal representation."""

  def __init__(
      self,
      *,
      ylm_transform: spherical_transforms.SphericalHarmonicsTransform,
      initializer: nnx.initializers.Initializer = nnx.initializers.zeros_init(),
      correction_scale: float,
      correction_param_type: nnx.Param = nnx.Param,
      correction_initializer: nnx.initializers.Initializer = (
          nnx.initializers.truncated_normal()
      ),
      rngs: nnx.Rngs,
  ):
    super().__init__(
        ylm_transform=ylm_transform, initializer=initializer, rngs=rngs
    )
    self.correction_scale = correction_scale
    self.correction = correction_param_type(
        correction_initializer(rngs.params(), self.orography.shape)
    )

  @property
  def modal_orography(self) -> typing.Array:
    """Returns orography converted to modal representation with filtering."""
    # TODO(dkochkov): add mask field to SphericalHarmonicGrid.
    mask = self.ylm_transform.dinosaur_grid.mask
    modal_orography_2d = jnp.zeros(mask.shape)
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
    self.orography = OrographyVariable(initializer(rngs, grid.shape))

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
