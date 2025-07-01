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
"""Modules that parameterize random processes."""

import abc
from collections.abc import Sequence
import zlib

import coordax as cx
from flax import nnx
import jax
import jax.numpy as jnp
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import spherical_transforms
from neuralgcm.experimental.core import typing
from neuralgcm.experimental.core import units
import numpy as np


Quantity = typing.Quantity
_ADVANCE_SALT = zlib.crc32(b'advance')  # arbitrary uint32 value


class RandomnessValue(nnx.Intermediate):
  """Variable type in which randomness values are stored."""


class RandomnessParam(nnx.Variable):
  """Variable type for parameters of the random processes."""


def _advance_prng_key(prng_key: jax.Array, prng_step: int):
  """Get a PRNG Key suitable for advancing randomness from key and step."""
  salt = jnp.uint32(_ADVANCE_SALT) + jnp.uint32(prng_step)
  return jax.random.fold_in(prng_key, salt)


def _get_advance_prng_key(randomness: typing.Randomness) -> jax.Array | None:
  """Extracts advance_prng_key from Randomness."""
  return _advance_prng_key(randomness.prng_key, randomness.prng_step)


class RandomProcessModule(nnx.Module, abc.ABC):
  """Base class for random processes."""

  @abc.abstractmethod
  def unconditional_sample(
      self,
      rng: jax.Array,
  ) -> typing.Randomness:
    """Samples the random field unconditionally."""

  @abc.abstractmethod
  def advance(
      self,
      state: typing.Randomness | None = None,
  ) -> typing.Randomness:
    """Updates the state of a random field."""

  @abc.abstractmethod
  def state_values(
      self,
      coords: cx.Coordinate | None = None,
      state: typing.Randomness | None = None,
  ) -> cx.Field:
    ...

  @property
  def event_shape(self) -> tuple[int, ...]:
    """Shape of the random process event."""
    return tuple()


class UniformUncorrelated(RandomProcessModule):
  """Scalar time-independent uniform random process."""

  def __init__(
      self,
      coords: cx.Coordinate,
      minval: float,
      maxval: float,
      rngs: nnx.Rngs,
  ):
    self.coords = coords
    self.minval = minval
    self.maxval = maxval
    self.state = RandomnessValue(self.unconditional_sample(rngs.params()))

  def unconditional_sample(self, rng: jax.Array) -> typing.Randomness:
    key, state_rng = jax.random.split(rng)
    value = jax.random.uniform(
        key,
        minval=self.minval,
        maxval=self.maxval,
        shape=self.coords.shape,
    )
    self.state = RandomnessValue(typing.Randomness(state_rng, 0, value))
    return self.state.value

  def advance(
      self,
      state: typing.Randomness | None = None,
  ) -> typing.Randomness:
    if state is None:
      state = self.state.value
    key = _get_advance_prng_key(state)
    new_value = jax.random.uniform(
        key,
        minval=self.minval,
        maxval=self.maxval,
        shape=self.coords.shape,
    )
    self.state.value = typing.Randomness(
        state.prng_key, state.prng_step + 1, new_value
    )
    return self.state.value

  def state_values(
      self,
      coords: cx.Coordinate | None = None,
      state: typing.Randomness | None = None,
  ) -> cx.Field:
    if state is None:
      state = self.state.value
    if coords is None:
      coords = self.coords
    if coords != self.coords:
      raise ValueError(
          f'Interpolation is not supported yet: {coords=} {self.coords=}'
      )
    return cx.wrap(state.core, coords)


class NormalUncorrelated(RandomProcessModule):
  """Time-independent normal random process."""

  def __init__(
      self,
      coords: cx.Coordinate,
      mean: float,
      std: float,
      rngs: nnx.Rngs,
  ):
    self.coords = coords
    self.mean = mean
    self.std = std
    self.state = RandomnessValue(self.unconditional_sample(rngs.params()))

  def unconditional_sample(self, rng: typing.PRNGKeyArray) -> typing.Randomness:
    key, state_rng = jax.random.split(rng)
    value = jax.random.normal(key, shape=self.coords.shape)
    value = self.mean + self.std * value
    self.state = RandomnessValue(typing.Randomness(state_rng, 0, value))
    return self.state.value

  def advance(
      self,
      state: typing.Randomness | None = None,
  ) -> typing.Randomness:
    if state is None:
      state = self.state.value
    key = _get_advance_prng_key(state)
    new_value = self.mean + self.std * jax.random.normal(key, self.coords.shape)
    self.state.value = typing.Randomness(
        state.prng_key, state.prng_step + 1, new_value
    )
    return self.state.value

  def state_values(
      self,
      coords: cx.Coordinate | None = None,
      state: typing.Randomness | None = None,
  ) -> cx.Field:
    if state is None:
      state = self.state.value
    if coords is None:
      coords = self.coords
    if coords != self.coords:
      raise ValueError(
          f'Interpolation is not supported yet: {coords=} {self.coords=}'
      )
    return cx.wrap(state.core, coords)


class GaussianRandomFieldCore(nnx.Module):
  """Core functionality of a spatio-temporal gaussian random process.

  This is a core class that is used to define multiple RandomProcessModules, but
  is not a RandomProcessModule itself. The rationale for parameterizing the core
  logic separately is to avoid nesting in RandomProcessModule classes, which
  could interfere with global initialization of the randomness.

  For implementation details see Appendix 8 in [1].
    [1] Stochastic Parametrization and Model Uncertainty, Palmer Et al. 2009.

  With x ∈ EarthSurface, this field U is initialized at t=0 with
    U(0, x) = Σₖ Ψₖ(x) (1 - ϕ²)^(-0.5) σₖ γₖ σₖ ηₖ₀,
  where Ψₖ is the kth spherical harmonic basis function, ϕ² is the one timestep
  correlation, σₖ > 0 is a scaling factor, and ηₖ₀ are iid 1D unit Gaussians.

  With `variance` an init kwarg,
    E[U(0, x)] ≡ 0,
    1 / (4πR²) ∫ Var(U(0, x))dx = variance,
  regardless of coords (and the radius).

  Further states are generated with the recursion
    U(t + δ) = ϕ U(t) + σₖ ηₖₜ
  This ensures that U is stationary.

  In general, the j timestep correlation is
    Cov(U(t, x), U(t + jδ, y)) = ϕ²ʲ Σₖ Ψₖ(x) Ψₖ(y) (γₖ)²
  """

  def __init__(
      self,
      ylm_transform: spherical_transforms.SphericalHarmonicsTransform,
      dt: float,
      sim_units: units.SimUnits,
      correlation_time: typing.Numeric | typing.Quantity,
      correlation_length: typing.Numeric | typing.Quantity,
      variance: typing.Numeric,
      correlation_time_type: nnx.Param | RandomnessParam = RandomnessParam,
      correlation_length_type: nnx.Param | RandomnessParam = RandomnessParam,
      variance_type: nnx.Param | RandomnessParam = RandomnessParam,
      clip: float = 6.0,
  ):
    """Constructs a core of a Gaussian Random Field.

    Args:
      ylm_transform: spherical harmonic transform defining the default basis for
        the random process.
      dt: time step of the random process.
      sim_units: object defining nondimensionalization and physical constants.
      correlation_time: correlation time of the random process.
      correlation_length: correlation length of the random process.
      variance: variance of the random process.
      correlation_time_type: parameter type for correlation time that allows for
        granular selection of the subsets of model parameters.
      correlation_length_type: parameter type for correlation length that allows
        for granular selection of the subsets of model parameters.
      variance_type: parameter type for variance  that allows for granular
        selection of the subsets of model parameters.
      clip: number of standard deviations at which to clip randomness to ensure
        numerical stability.
    """
    nondimensionalize = lambda x: units.maybe_nondimensionalize(x, sim_units)
    correlation_time = nondimensionalize(correlation_time)
    correlation_length = nondimensionalize(correlation_length)
    variance = nondimensionalize(variance)
    # we make parameters 1d to streamline broadcasting when code is vmapped.
    as_1d_param = lambda x, t: t(jnp.array([x]))
    self.ylm_transform = ylm_transform
    self.dt = dt
    self.corr_time = as_1d_param(correlation_time, correlation_time_type)
    self.corr_length = as_1d_param(correlation_length, correlation_length_type)
    self._variance = as_1d_param(variance, variance_type)
    self.clip = clip

  @property
  def _surf_area(self) -> jax.Array | float:
    """Surface area of the sphere used by self.grid."""
    return 4 * jnp.pi * self.ylm_transform.radius**2

  @property
  def variance(self):
    """An estimate of pointwise (in nodal space) variance of this random field.

    This random field is defined in spectral space, and has no precise
    pointwise variance quantity. However, it does have a precise integrated
    variance, which is used to define the field.

    If we assume the field is stationary (with higher spectral
    precision it is near stationary), then the average of this quantity is a
    good pointwise estimate. So define
      σ² := (1 / (4πR²)) ∫ Var(U(0, x))dx
          = (1 / (4πR²)) integrated_grf_variance

    Therefore the init parameter `variance` can be used to define
      `_integrated_grf_variance := variance * surf_area`
    and then `_integrated_grf_variance` is used to define this field. The result
    is a field with pointwise variance close to the init kwarg `variance`.

    Returns:
      Numeric estimate of pointwise variance.
    """
    return self._variance.value

  @property
  def phi(self) -> jax.Array:
    """Correlation coefficient between two timesteps."""
    return jnp.exp(-self.dt / self.corr_time.value)

  @property
  def relative_corr_len(self):
    """Correlation length of the random process relative to the radius."""
    return self.corr_length.value / self.ylm_transform.radius

  def _integrated_grf_variance(self):
    """Integral of the GRF's variance over the earth's surface."""
    return self.variance * self._surf_area

  def _sigma_array(self) -> jax.Array:
    """Array of σₙ from Appendix 8 in [Palmer] http://shortn/_56HCcQwmSS."""
    dinosaur_grid = self.ylm_transform.dinosaur_grid
    # n = [0, 1, ..., N]
    n = dinosaur_grid.modal_axes[1]  # total wavenumbers.
    # Number of longitudinal wavenumbers at each total wavenumber n.
    #  L = 2n + 1, except for the last entry.
    n_longitudian_wavenumbers = dinosaur_grid.mask.sum(axis=0)
    # [Palmer] states correlation_length = sqrt(2κT) / R, therefore
    kt = 0.5 * self.relative_corr_len**2
    # sigmas_unnormed[n] is proportional to the standard deviation for each
    # longitudinal wavenumbers at each total wavenumber n.
    sigmas_unnormed = jnp.exp(-0.5 * kt * n * (n + 1))
    # The sum of unnormalized variance for all longitudinal wavenumbers at each
    # total wavenumber.
    sum_unnormed_vars = jnp.sum(n_longitudian_wavenumbers * sigmas_unnormed**2)
    # This is analogous to F₀ from [Palmer].
    # (normalization * sigmas_unnormed)² would sum to 1. The leading factor
    #   self._integrated_grf_variance * (1 - self.phi ** 2)
    # ensures that the AR(1) process has variance self._integrated_grf_variance.
    # We do not include the extra fator of 2 in the denominator. I do not know
    # why [Palmer] has this factor.
    # In sampling, phi appears as 1 - phi**2 = 1 - exp(-2 dt / tau)
    one_minus_phi2 = -jnp.expm1(-2 * self.dt / self.corr_time.value)
    normalization = jnp.sqrt(
        self._integrated_grf_variance() * one_minus_phi2 / sum_unnormed_vars
    )
    # The factor of coords.horizontal.radius appears because our basis vectors
    # have L2 norm = radius.
    return normalization * sigmas_unnormed / dinosaur_grid.radius

  def sample_core(self, rng: typing.PRNGKeyArray) -> jax.Array:
    """Helper method for sampling the core of the gaussian random field."""
    dinosaur_grid = self.ylm_transform.dinosaur_grid
    modal_shape = dinosaur_grid.modal_shape
    sigmas = self._sigma_array()
    weights = jnp.where(
        dinosaur_grid.mask,
        jax.random.truncated_normal(rng, -self.clip, self.clip, modal_shape),
        jnp.zeros(modal_shape),
    )
    one_minus_phi2 = -jnp.expm1(-2 * self.dt / self.corr_time.value)
    return one_minus_phi2 ** (-0.5) * sigmas * weights

  def advance_core(
      self, state_core: jax.Array, state_key: jax.Array, state_step: int
  ) -> jax.Array:
    """Helper method for advancing the core of the gaussian random field."""
    dinosaur_grid = self.ylm_transform.dinosaur_grid
    modal_shape = dinosaur_grid.modal_shape
    rng = _advance_prng_key(state_key, state_step)
    eta = jax.random.truncated_normal(rng, -self.clip, self.clip, modal_shape)
    return state_core * self.phi + self._sigma_array() * jnp.where(
        dinosaur_grid.mask, eta, jnp.zeros(modal_shape)
    )


class GaussianRandomField(RandomProcessModule):
  """Spatially and temporally correlated gaussian process on the sphere."""

  def __init__(
      self,
      ylm_transform: spherical_transforms.SphericalHarmonicsTransform,
      dt: float,
      sim_units: units.SimUnits,
      correlation_time: typing.Numeric | typing.Quantity,
      correlation_length: typing.Numeric | typing.Quantity,
      variance: typing.Numeric,
      correlation_time_type: nnx.Param | RandomnessParam = RandomnessParam,
      correlation_length_type: nnx.Param | RandomnessParam = RandomnessParam,
      variance_type: nnx.Param | RandomnessParam = RandomnessParam,
      clip: float = 6.0,
      *,
      rngs: nnx.Rngs,
  ):
    """Initializes a Gaussian Random Field."""
    self.grf = GaussianRandomFieldCore(
        ylm_transform=ylm_transform,
        dt=dt,
        sim_units=sim_units,
        correlation_time=correlation_time,
        correlation_length=correlation_length,
        variance=variance,
        correlation_time_type=correlation_time_type,
        correlation_length_type=correlation_length_type,
        variance_type=variance_type,
        clip=clip,
    )
    # rngs are used to create an initial state. This ensures that the structure
    # of the module is not changed by sampling or advancing the state.
    self.ylm_transform = ylm_transform
    prng_key = rngs.params()
    core_rngs = rngs.params()
    self.randomness_state = RandomnessValue(
        typing.Randomness(
            prng_key=prng_key,
            prng_step=0,
            core=self.grf.sample_core(core_rngs),
        )
    )

  def unconditional_sample(self, rng: typing.PRNGKeyArray) -> typing.Randomness:
    """Returns a randomly initialized state for the autoregressive process."""
    rng, next_rng = jax.random.split(rng)
    randomness_state = typing.Randomness(
        prng_key=next_rng,
        prng_step=0,
        core=self.grf.sample_core(rng),
    )
    self.randomness_state = RandomnessValue(randomness_state)
    return randomness_state

  def advance(
      self, state: typing.Randomness | None = None
  ) -> typing.Randomness:
    """Updates the CoreRandomState of a random gaussian field."""
    if state is None:
      state = self.randomness_state.value
    next_randomness_state = typing.Randomness(
        core=self.grf.advance_core(state.core, state.prng_key, state.prng_step),
        prng_key=state.prng_key,
        prng_step=state.prng_step + 1,
    )
    self.randomness_state = RandomnessValue(next_randomness_state)
    return next_randomness_state

  def state_values(
      self,
      coords: cx.Coordinate | None = None,
      state: typing.Randomness | None = None,
  ) -> cx.Field:
    if coords is None:
      coords = self.ylm_transform.nodal_grid
    if not isinstance(coords, coordinates.LonLatGrid):
      raise ValueError('Interpolation of randomness is not supported yet')
    if state is None:
      state = self.randomness_state.value
    return cx.wrap(self.ylm_transform.to_nodal_array(state.core), coords)


class BatchGaussianRandomField(RandomProcessModule):
  """Batched version of GaussianRandomField process."""

  def __init__(
      self,
      ylm_transform: spherical_transforms.SphericalHarmonicsTransform,
      dt: float,
      sim_units: units.SimUnits,
      correlation_times: typing.Numeric | typing.Quantity,
      correlation_lengths: typing.Numeric | typing.Quantity,
      variances: Sequence[float],
      correlation_time_type: nnx.Param | RandomnessParam = RandomnessParam,
      correlation_length_type: nnx.Param | RandomnessParam = RandomnessParam,
      variance_type: nnx.Param | RandomnessParam = RandomnessParam,
      clip: float = 6.0,
      *,
      rngs: nnx.Rngs,
  ):
    lengths = [
        len(correlation_times),
        len(correlation_lengths),
        len(variances),
    ]
    if len(set(lengths)) != 1:
      raise ValueError(f'Argument lengths differed: {lengths=}')
    (n_fields,) = set(lengths)

    nondimensionalize = lambda x: units.maybe_nondimensionalize(x, sim_units)
    correlation_times = np.array(
        [nondimensionalize(tau) for tau in correlation_times]
    )
    correlation_lengths = np.array(
        [nondimensionalize(length) for length in correlation_lengths]
    )
    variances = np.array(
        [nondimensionalize(variance) for variance in variances]
    )
    make_grf = lambda length, tau, variance: GaussianRandomFieldCore(
        ylm_transform=ylm_transform,
        dt=dt,
        sim_units=sim_units,
        correlation_time=tau,
        correlation_length=length,
        variance=variance,
        correlation_time_type=correlation_time_type,
        correlation_length_type=correlation_length_type,
        variance_type=variance_type,
        clip=clip,
    )
    self.n_fields = n_fields
    self.batch_grf_core = nnx.vmap(make_grf, axis_size=self.n_fields)(
        correlation_lengths, correlation_times, variances
    )
    prng_key = rngs.params()
    core_rngs = jax.random.split(rngs.params(), num=self.n_fields)
    sample_fn = lambda grf, rng: grf.sample_core(rng)
    self.randomness_state = RandomnessValue(
        typing.Randomness(
            prng_key=prng_key,
            prng_step=0,
            core=nnx.vmap(sample_fn)(self.batch_grf_core, core_rngs),
        )
    )
    self.ylm_transform = ylm_transform

  @property
  def event_shape(self) -> tuple[int, ...]:
    return tuple([self.n_fields])

  def unconditional_sample(self, rng: typing.PRNGKeyArray) -> typing.Randomness:
    all_rngs = jax.random.split(rng, num=(self.n_fields + 1))
    next_rng = all_rngs[-1]
    rngs = all_rngs[:-1]
    sample_fn = lambda grf, rng: grf.sample_core(rng)
    randomness_state = typing.Randomness(
        prng_key=next_rng,
        prng_step=0,
        core=nnx.vmap(sample_fn)(self.batch_grf_core, rngs),
    )
    self.randomness_state = RandomnessValue(randomness_state)
    return randomness_state

  def advance(
      self, state: typing.Randomness | None = None
  ) -> typing.Randomness:
    if state is None:
      state = self.randomness_state.value
    advance_fn = lambda grf, core, key, step: grf.advance_core(core, key, step)
    in_axes = (0, 0, 0, None)
    advance_fn = nnx.vmap(advance_fn, in_axes=in_axes)
    rngs = jax.random.split(state.prng_key, num=self.n_fields)
    next_randomness_state = typing.Randomness(
        core=advance_fn(self.batch_grf_core, state.core, rngs, state.prng_step),
        prng_key=state.prng_key,
        prng_step=state.prng_step + 1,
    )
    self.randomness_state = RandomnessValue(next_randomness_state)
    return next_randomness_state

  def state_values(
      self,
      coords: cx.Coordinate | None = None,
      state: typing.Randomness | None = None,
  ) -> cx.Field:
    if coords is None:
      coords = self.ylm_transform.nodal_grid
    if not isinstance(coords, coordinates.LonLatGrid):
      raise ValueError('Interpolation of randomness is not supported yet')
    if state is None:
      state = self.randomness_state.value
    # TODO(dkochkov): consider an option where we return a field with only
    # trailing axes labeled, rather than assigning names to all dimensions.
    return cx.wrap(self.ylm_transform.to_nodal_array(state.core), 'grf', coords)
