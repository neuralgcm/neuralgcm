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
"""Geometry and interpolation utilities for spherical tasks."""

import coordax as cx
from flax import nnx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import random_processes
from neuralgcm.experimental.core import typing
import numpy as np


# pylint: disable=protected-access
advance_prng_key = random_processes._advance_prng_key
# pylint: enable=protected-access


def apply_anisotropy(ylm_coeffs: cx.Field, alpha: float = 0.3) -> cx.Field:
  """Applies zonal anisotropy filter to spherical harmonic coefficients.

  Args:
    ylm_coeffs: The spectral coefficients (organized by l, m).
    alpha: Anisotropy factor (lower = more jet-like/zonal).

  Returns:
    Filtered coefficients.
  """
  ylm_grid = cx.coords.extract(
      ylm_coeffs.coordinate, coordinates.SphericalHarmonicGrid
  )
  l = ylm_grid.fields['total_wavenumber'].broadcast_like(ylm_coeffs)
  m = ylm_grid.fields['longitude_wavenumber'].broadcast_like(ylm_coeffs)
  sigma_m = cx.cmap(jnp.maximum)(1.0, alpha * l)
  weight = cx.cmap(jnp.exp)(-(m**2) / (2 * sigma_m**2))
  return ylm_coeffs * weight


class UncorrelatedGaussiansOnASphere(random_processes.RandomProcessModule):
  """Generates a field of uncorrelated Gaussian blobs on a sphere."""

  def __init__(
      self,
      grid: coordinates.LonLatGrid,
      n_blobs: int,
      width_deg: float,
      amplitude_range: tuple[float, float],
      *,
      rngs: nnx.Rngs,
  ):
    """Inits a random process that generates Gaussian blobs on a sphere.

    Args:
      grid: The LonLatGrid on which to generate the blobs.
      n_blobs: The number of blobs to generate in each sample.
      width_deg: The characteristic width of each blob in degrees.
      amplitude_range: A tuple (min, max) for the amplitude of each blob.
      rngs: The random number generators for the process.
    """
    self.grid = grid
    self.n_blobs = n_blobs
    self.width_deg = width_deg
    self.amplitude_range = amplitude_range
    k, rng = cx.cmap(lambda k: tuple(jax.random.split(k, 2)))(rngs.params())
    self.state_rng = typing.Randomness(rng)
    self.rng_step = typing.Randomness(cx.field(0))
    self.core = typing.Randomness(k)

  def _sample_blobs_on_grid(
      self,
      key: jax.Array,
  ) -> cx.Field:
    """Samples blobs on the grid based on a single PRNG key."""
    k1, k2, k3 = jax.random.split(key, 3)
    blob_lons_deg = jax.random.uniform(
        k1, (self.n_blobs,), minval=0.0, maxval=360.0
    )
    # Sample latitude uniformly on sphere: sin(lat) ~ U[-1, 1]
    blob_lats_deg = jnp.rad2deg(
        jnp.arcsin(
            jax.random.uniform(k2, (self.n_blobs,), minval=-1.0, maxval=1.0)
        )
    )
    blob_amps = jax.random.uniform(
        k3,
        (self.n_blobs,),
        minval=self.amplitude_range[0],
        maxval=self.amplitude_range[1],
    )
    d2r = jnp.deg2rad
    lons = cx.cpmap(d2r)(
        self.grid.fields['longitude'].broadcast_like(self.grid)
    )
    lats = cx.cpmap(d2r)(
        self.grid.fields['latitude'].broadcast_like(self.grid)
    )
    width_rad = d2r(self.width_deg)

    total_field = cx.cpmap(jnp.zeros_like)(lons)
    for i in range(self.n_blobs):
      lon_center, lat_center = d2r(blob_lons_deg[i]), d2r(blob_lats_deg[i])
      dlat = lats - lat_center
      dlon = lons - lon_center
      amplitude = blob_amps[i]
      # Haversine formula for angular distance
      a = (
          cx.cpmap(lambda x: jnp.sin(x / 2) ** 2)(dlat)
          + jnp.cos(lat_center)
          * cx.cpmap(jnp.cos)(lats)
          * cx.cpmap(lambda x: jnp.sin(x / 2) ** 2)(dlon)
      )
      # clip to avoid sqrt of small negative numbers due to precision.
      angle = cx.cpmap(lambda x: 2 * jnp.arcsin(jnp.sqrt(jnp.clip(x, 0, 1))))(a)
      # pylint: disable=cell-var-from-loop
      blob_i = cx.cpmap(
          lambda ang: amplitude * jnp.exp(-((ang / width_rad) ** 2))
      )(angle)
      # pylint: enable=cell-var-from-loop
      total_field += blob_i
    return total_field

  def unconditional_sample(self, rng: cx.Field) -> None:
    k, rng = cx.cmap(lambda k: tuple(jax.random.split(k)))(rng)
    self.state_rng.set_value(rng)
    self.rng_step.set_value(
        cx.field(jnp.zeros(k.shape, jnp.uint32), k.coordinate)
    )
    self.core.set_value(k)

  def advance(self) -> None:
    k = cx.cmap(advance_prng_key)(
        self.state_rng.get_value(), self.rng_step.get_value()
    )
    self.rng_step.set_value(self.rng_step.get_value() + 1)
    self.core.set_value(k)

  def state_values(self, coord: cx.Coordinate | None = None) -> cx.Field:
    if coord is not None and coord != self.grid:
      raise ValueError(f'coord must be None or {self.grid}, got {coord}')
    sample_fn = self._sample_blobs_on_grid
    key = self.core.get_value()
    # If key is batched, map over batch dimension.
    if key.ndim > 0:
      out_axes = {c: i for i, c in enumerate(key.axes + self.grid.axes)}
      sample_fn = cx.cmap(sample_fn, out_axes=out_axes)
    return sample_fn(key)


def evaluate_modal_field(
    modal_field: cx.Field,
    lon: cx.Field,
    lat: cx.Field,
) -> cx.Field:
  """Evaluates a `modal_field` at arbitrary (lon, lat) points."""
  if lon.coordinate != lat.coordinate:
    raise ValueError(
        'Requested locations lon/lat are not on the same coordinates:'
        f' {lon.coordinate=}, {lat.coordinate=}'
    )
  ylm_grid = cx.coords.extract(
      modal_field.coordinate, coordinates.SphericalHarmonicGrid
  )
  to_theta = lambda x: jnp.clip(-jnp.deg2rad(x) + jnp.pi / 2, min=0, max=jnp.pi)
  to_phi = lambda x: jnp.clip(jnp.deg2rad(x), min=0, max=2 * jnp.pi)
  thetas = cx.cpmap(to_theta)(lat)
  phis = cx.cpmap(to_phi)(lon)
  l_max = ylm_grid.total_wavenumbers

  ls = ylm_grid.fields['total_wavenumber']
  ms = ylm_grid.fields['longitude_wavenumber']
  l_mesh, m_mesh = np.meshgrid(ls.data, ms.data, indexing='xy')
  l_flat = l_mesh[ylm_grid.fields['mask'].data]
  m_flat = m_mesh[ylm_grid.fields['mask'].data]

  def eval_at_point(ylm_coefficients, theta, phi):
    cs = ylm_coefficients[ylm_grid.fields['mask'].data]
    abs_m = jnp.abs(m_flat)
    ylm_cplx = jsp.special.sph_harm_y(l_flat, abs_m, theta, phi, n_max=l_max)
    ylm_real = jnp.where(m_flat >= 0, jnp.real(ylm_cplx), jnp.imag(ylm_cplx))
    factor = jnp.where(abs_m > 0, jnp.sqrt(2), 1.0)
    return jnp.sum(cs * factor * ylm_real)

  in_coord = modal_field.coordinate
  real_space_coord = cx.coords.compose(
      *[ax for ax in lon.coordinate.axes if ax not in in_coord.axes]
  )
  out_coord = cx.coords.replace_axes(in_coord, ylm_grid, real_space_coord)
  out_axes = {d: i for i, d in enumerate(out_coord.dims)}
  eval_at_point_fn = cx.cmap(eval_at_point, out_axes=out_axes)
  return eval_at_point_fn(modal_field.untag(ylm_grid), thetas, phis)


def compute_lon_lat_departure_points(
    lon: cx.Field,
    lat: cx.Field,
    u: cx.Field,
    v: cx.Field,
    dt: float,
    radius: float = 1.0,
) -> tuple[cx.Field, cx.Field]:
  """Computes departure points of lon, lat locations under u, v.

  Approximates departure point by mapping to Cartesian coordinates,
  transporting by velocities, and projecting back to the sphere.

  Args:
    lon: longitude values for which to compute the departure points.
    lat: latitude values for which to compute the departure points.
    u: velocity along the longitudinal direction.
    v: velocity along the latitudinal direction.
    dt: time step.
    radius: radius of the sphere.

  Returns:
    lon_dep, lat_dep: departure points.
  """
  to_rad = cx.cpmap(jnp.deg2rad)
  to_deg = cx.cpmap(jnp.rad2deg)
  lon, lat = to_rad(lon), to_rad(lat)

  cos_lon = cx.cpmap(jnp.cos)(lon)
  sin_lon = cx.cpmap(jnp.sin)(lon)
  cos_lat = cx.cpmap(jnp.cos)(lat)
  sin_lat = cx.cpmap(jnp.sin)(lat)

  x = cos_lat * cos_lon * radius
  y = cos_lat * sin_lon * radius
  z = sin_lat * radius

  dx_lon = -sin_lon
  dy_lon = cos_lon
  dz_lon = cx.cpmap(jnp.zeros_like)(lon)

  dx_lat = -(sin_lat * cos_lon)
  dy_lat = -(sin_lat * sin_lon)
  dz_lat = cos_lat

  dx = u * dt * dx_lon + v * dt * dx_lat
  dy = u * dt * dy_lon + v * dt * dy_lat
  dz = u * dt * dz_lon + v * dt * dz_lat

  x_dep = x - dx
  y_dep = y - dy
  z_dep = z - dz

  def _norm(x_val, y_val, z_val):
    n = jnp.sqrt(x_val**2 + y_val**2 + z_val**2)
    return jnp.where(n == 0.0, 1e-10, n)

  norm_dep = cx.cpmap(_norm)(x_dep, y_dep, z_dep)
  lat_dep = cx.cpmap(
      lambda z_v, n_v: jnp.arcsin(jnp.clip(z_v / n_v, -1.0, 1.0))
  )(z_dep, norm_dep)
  lon_dep = cx.cpmap(jnp.arctan2)(y_dep, x_dep)
  lon_dep = cx.cpmap(lambda l: l % (2 * jnp.pi))(lon_dep)

  return to_deg(lon_dep), to_deg(lat_dep)
