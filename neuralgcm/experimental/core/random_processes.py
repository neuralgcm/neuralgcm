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
import dataclasses
import functools
from typing import Protocol, Sequence
import zlib

import coordax as cx
from flax import nnx
import jax
import jax.numpy as jnp
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import spherical_harmonics
from neuralgcm.experimental.core import typing
from neuralgcm.experimental.core import units
import numpy as np

Quantity = typing.Quantity
_ADVANCE_SALT = zlib.crc32(b'advance')  # arbitrary uint32 value


Randomness = typing.Randomness


class RandomnessParam(nnx.Variable):
  """Variable type for parameters of random processes."""


class RandomnessTape(typing.Randomness):
  """Variable type for holding recorded/replayed samples.

  RandomnessTape is a subclass of Randomness is ensure that when modules are
  vectorized we generate copies for each copy. It is a separate type to ensure
  that we set different scan mechanics, as tape should be closed over, rather
  than being passed through as a scan carry which could result in broadcasting.
  """


class DistributionLike(Protocol):
  """Protocol that described distribution requirements to be compatible with Samplers."""

  def sample(
      self, *, seed: jax.Array, sample_shape: tuple[int, ...]
  ) -> jax.Array:
    ...

  def log_prob(self, value: jax.Array) -> jax.Array:
    ...


class Sampler(nnx.Module, abc.ABC):
  """Base class for seeding potentially correlated random processes.

  Random processes use `Sampler`s to produce independent random variables that
  may be transformed into correlated random state. A random process may rely on
  one or multiple `Sampler` instances. Changing the underlying sampler can be
  used to record or inject samples which is needed for some Data Assimilation
  algorithms.

  Attributes:
    sample_coord: coordinate describing a single draw event.
  """

  def __init__(self, sample_coord: cx.Coordinate):
    self.sample_coord = sample_coord

  @abc.abstractmethod
  def draw(
      self,
      seed: cx.Field,
  ) -> cx.Field:
    """Draws samples `sample_coord` coordinate for each entry in `seed`."""


class DistributionSampler(Sampler):
  """Sampler that draws i.i.d. variables from a distribution."""

  def __init__(
      self, distribution: DistributionLike, sample_coord: cx.Coordinate
  ):
    super().__init__(sample_coord)
    self.distribution = distribution

  def draw(
      self,
      seed: cx.Field,
  ) -> cx.Field:
    sample_fn = lambda k: self.distribution.sample(
        seed=k, sample_shape=self.sample_coord.shape
    )
    return cx.cmap(sample_fn, seed.named_axes)(seed).tag(self.sample_coord)


class RecordingSampler(Sampler):
  """Sampler wrapper that records first "N" draws to a tape."""

  def __init__(
      self,
      sampler: Sampler,
      tape: cx.Field,
      tape_axis: str | cx.Coordinate = 'tape_idx',
  ):
    super().__init__(sampler.sample_coord)
    self.sampler = sampler
    self.tape = RandomnessTape(tape)
    self.tape_position = RandomnessTape(cx.field(0))
    self.tape_axis = tape_axis

  @classmethod
  def with_fixed_tape_length(
      cls,
      sampler: Sampler,
      max_steps: int,
      tape_axis: str | cx.Coordinate = 'tape_idx',
  ):
    """Creates a RecordingSampler with a fixed tape length."""
    if isinstance(tape_axis, str):
      tape_axis = cx.DummyAxis(tape_axis, max_steps)
    tape_coord = cx.coords.compose(tape_axis, sampler.sample_coord)
    tape_data = jnp.zeros(tape_coord.shape)
    tape = cx.field(tape_data, tape_coord)
    return cls(sampler, tape, tape_axis)

  def draw(
      self,
      seed: cx.Field,
  ) -> cx.Field:
    """Draws a sampler from the wrapped sampler."""
    draw = self.sampler.draw(seed)
    # Updating the tape value if it is not full.
    tape = self.tape.get_value()
    tape_axis = cx.get_coordinate_part(tape, self.tape_axis)
    tape_length = tape_axis.shape[0]
    idx = self.tape_position.get_value()
    update_fn = lambda t, d, i: jnp.where(i < tape_length, t.at[i].set(d), t)
    update_fn = cx.cmap(update_fn, out_axes=tape.untag(tape_axis).named_axes)
    updated_tape = update_fn(tape.untag(tape_axis), draw, idx).tag(tape_axis)
    self.tape.set_value(updated_tape)
    # Update tape position.
    clip_fn = cx.cmap(functools.partial(jnp.clip, max=tape_length))
    self.tape_position.set_value(clip_fn(idx + 1))
    return draw


class TapeSampler(Sampler):
  """Sampler that replays first "N" draws from a provided tape."""

  def __init__(
      self,
      sampler: Sampler,
      tape: cx.Field,
      tape_axis: str | cx.Coordinate = 'tape_idx',
      allow_broadcasting: bool = False,
  ):
    super().__init__(sampler.sample_coord)
    self.sampler = sampler
    self.tape = RandomnessTape(tape)
    self.allow_broadcasting = allow_broadcasting
    self.tape_position = RandomnessTape(cx.field(0))
    self.tape_axis = tape_axis

  def draw(
      self,
      seed: cx.Field,
  ) -> cx.Field:
    """Draws a sample either from sampler or tape depending on call count."""
    idx = self.tape_position.get_value()
    tape = self.tape.get_value()
    tape_axis = cx.get_coordinate_part(tape, self.tape_axis)
    tape_draw = cx.cmap(lambda x, i: x[i])(tape.untag(tape_axis), idx)
    sampler_draw = self.sampler.draw(seed)

    if sampler_draw.coordinate != tape_draw.coordinate:
      if not self.allow_broadcasting:
        raise ValueError(
            f'Tape sample has {tape_draw.coordinate} that is not equal to the '
            f'coordinate from the sampler {sampler_draw.coordinate} and '
            'allow_broadcasting is False.'
        )
      tape_draw = tape_draw.broadcast_like(sampler_draw)

    tape_length = tape_axis.shape[0]
    pick_draw_fn = lambda idx, t, s: jnp.where(idx < tape_length, t, s)
    draw = cx.cmap(pick_draw_fn)(idx, tape_draw, sampler_draw)
    clip_fn = cx.cmap(functools.partial(jnp.clip, max=tape_length))
    self.tape_position.set_value(clip_fn(idx + 1))
    return draw

  def tape_log_prob(self) -> cx.Field:
    """Computes the log-likelihood of the tape under the stored distribution."""
    if not isinstance(self.sampler, DistributionSampler):
      raise ValueError(
          'Wrapped sampler must be a DistributionSampler to compute log-prob'
      )
    idx = self.tape_position.get_value()
    tape = self.tape.get_value()
    tape_axis = cx.get_coordinate_part(tape, self.tape_axis)
    tape_length = tape_axis.shape[0]
    mask = cx.field(jnp.arange(tape_length), tape_axis) < idx
    log_prob_fn = cx.cmap(self.sampler.distribution.log_prob)
    masked_log_prob = log_prob_fn(tape) * mask
    return cx.cmap(jnp.sum)(masked_log_prob.untag(tape_axis, self.sample_coord))


@dataclasses.dataclass
class UniformDistribution:
  """Uniform distribution."""

  minval: float = 0.0
  maxval: float = 1.0

  def sample(
      self, *, seed: jax.Array, sample_shape: tuple[int, ...]
  ) -> jax.Array:
    return jax.random.uniform(
        seed, shape=sample_shape, minval=self.minval, maxval=self.maxval
    )

  def log_prob(self, value: jax.Array) -> jax.Array:
    log_prob = -jnp.log(self.maxval - self.minval)
    within_bounds = (value >= self.minval) & (value <= self.maxval)
    return jnp.where(within_bounds, log_prob, -jnp.inf)


@dataclasses.dataclass
class NormalDistribution:
  """Normal distribution."""

  mean: float = 0.0
  std: float = 1.0

  def sample(
      self, *, seed: jax.Array, sample_shape: tuple[int, ...]
  ) -> jax.Array:
    return self.mean + self.std * jax.random.normal(seed, shape=sample_shape)

  def log_prob(self, value: jax.Array) -> jax.Array:
    return -0.5 * ((value - self.mean) / self.std) ** 2 - jnp.log(
        self.std * jnp.sqrt(2 * jnp.pi)
    )


@dataclasses.dataclass
class TruncatedNormalDistribution:
  """Truncated normal distribution."""

  lower: float
  upper: float
  mean: float = 0.0
  std: float = 1.0

  def sample(
      self, *, seed: jax.Array, sample_shape: tuple[int, ...]
  ) -> jax.Array:
    std_lower = (self.lower - self.mean) / self.std
    std_upper = (self.upper - self.mean) / self.std
    val = jax.random.truncated_normal(
        seed, std_lower, std_upper, shape=sample_shape
    )
    return self.mean + self.std * val

  def log_prob(self, value: jax.Array) -> jax.Array:
    std_value = (value - self.mean) / self.std
    std_lower = (self.lower - self.mean) / self.std
    std_upper = (self.upper - self.mean) / self.std
    log_phi_diff = jnp.log(
        jax.scipy.special.ndtr(std_upper) - jax.scipy.special.ndtr(std_lower)
    )
    normal_log_prob = -0.5 * std_value**2 - jnp.log(
        self.std * jnp.sqrt(2 * jnp.pi)
    )
    within_bounds = (value >= self.lower) & (value <= self.upper)
    return jnp.where(within_bounds, normal_log_prob - log_phi_diff, -jnp.inf)


def _advance_prng_key(prng_key: jax.Array, prng_step: int):
  """Get a PRNG Key suitable for advancing randomness from key and step."""
  salt = jnp.uint32(_ADVANCE_SALT) + jnp.uint32(prng_step)
  return jax.random.fold_in(prng_key, salt)


class RandomProcessModule(nnx.Module, abc.ABC):
  """Base class for random processes."""

  @abc.abstractmethod
  def unconditional_sample(
      self,
      rng: jax.Array,
  ) -> None:
    """Updates the state with an unconditional sample from the process."""

  @abc.abstractmethod
  def advance(
      self,
  ) -> None:
    """Updates the state of a random field."""

  @abc.abstractmethod
  def state_values(
      self,
      coord: cx.Coordinate | None = None,
  ) -> cx.Field:
    """Returns the values of the current random state evaluated on `coords`."""

  @property
  def event_shape(self) -> tuple[int, ...]:
    """Shape of the random process event."""
    return ()


class UniformUncorrelated(RandomProcessModule):
  """Scalar time-independent uniform random process."""

  def __init__(
      self,
      minval: float,
      maxval: float,
      coord: cx.Coordinate,
      rngs: nnx.Rngs,
  ):
    self.coord = coord
    self.minval = minval
    self.maxval = maxval
    k, rng = cx.cmap(lambda k: tuple(jax.random.split(k, 2)))(rngs.params())
    self.state_rng = Randomness(rng)
    self.rng_step = Randomness(cx.field(0))
    self.core = Randomness(k)

  def _sample_core(
      self, rng: cx.Field, coord: cx.Coordinate | None
  ) -> cx.Field:
    """Samples the core of the uniform random field."""
    if coord is None:
      coord = self.coord
    sample_fn = lambda rng: jax.random.uniform(
        rng,
        minval=self.minval,
        maxval=self.maxval,
        shape=coord.shape,
    )
    out_axes = {c: i for i, c in enumerate(rng.axes)}
    return cx.cmap(sample_fn, out_axes)(rng).tag(coord)

  def unconditional_sample(self, rng: cx.Field) -> None:
    k, rng = cx.cmap(lambda k: tuple(jax.random.split(k)))(rng)
    self.state_rng.set_value(rng)
    self.rng_step.set_value(
        cx.field(jnp.zeros(k.shape, jnp.uint32), k.coordinate)
    )
    self.core.set_value(k)

  def advance(self) -> None:
    k = cx.cmap(_advance_prng_key)(
        self.state_rng.get_value(), self.rng_step.get_value()
    )
    self.rng_step.set_value(self.rng_step.get_value() + 1)
    self.core.set_value(k)

  def state_values(
      self,
      coord: cx.Coordinate | None = None,
  ) -> cx.Field:
    return self._sample_core(self.core.get_value(), coord)


class NormalUncorrelated(RandomProcessModule):
  """Time-independent normal random process."""

  def __init__(
      self,
      mean: float,
      std: float,
      coord: cx.Coordinate,
      rngs: nnx.Rngs,
  ):
    self.coord = coord
    self.mean = mean
    self.std = std
    k, rng = cx.cmap(lambda k: tuple(jax.random.split(k, 2)))(rngs.params())
    self.state_rng = Randomness(rng)
    self.rng_step = Randomness(cx.field(0))
    self.core = Randomness(k)  # core of uncorrelated is just a key.

  def _sample_core(
      self, rng: cx.Field, coord: cx.Coordinate | None
  ) -> cx.Field:
    """Samples the core of the normal random field."""
    if coord is None:
      coord = self.coord
    sample_fn = lambda rng: self.mean + self.std * jax.random.normal(
        rng, shape=coord.shape
    )
    out_axes = {c: i for i, c in enumerate(rng.axes)}
    return cx.cmap(sample_fn, out_axes)(rng).tag(coord)

  def unconditional_sample(self, rng: cx.Field):
    k, rng = cx.cmap(lambda k: tuple(jax.random.split(k)))(rng)
    self.state_rng.set_value(rng)
    self.rng_step.set_value(
        cx.field(jnp.zeros(k.shape, jnp.uint32), k.coordinate)
    )
    self.core.set_value(k)

  def advance(self):
    k = cx.cmap(_advance_prng_key)(
        self.state_rng.get_value(), self.rng_step.get_value()
    )
    self.rng_step.set_value(self.rng_step.get_value() + 1)
    self.core.set_value(k)

  def state_values(
      self,
      coord: cx.Coordinate | None = None,
  ) -> cx.Field:
    return self._sample_core(self.core.get_value(), coord)


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
      ylm_map: spherical_harmonics.FixedYlmMapping,
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
      ylm_map: spherical harmonic transform defining the default basis for the
        random process.
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
    self.ylm_map = ylm_map
    self.dt = dt
    self.corr_time = as_1d_param(correlation_time, correlation_time_type)
    self.corr_length = as_1d_param(correlation_length, correlation_length_type)
    self._variance = as_1d_param(variance, variance_type)
    self.clip = clip

  @property
  def _surf_area(self) -> jax.Array | float:
    """Surface area of the sphere used by self.grid."""
    return 4 * jnp.pi * self.ylm_map.radius**2

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
    return self._variance[...]

  @property
  def phi(self) -> jax.Array:
    """Correlation coefficient between two timesteps."""
    return jnp.exp(-self.dt / self.corr_time[...])

  @property
  def relative_corr_len(self):
    """Correlation length of the random process relative to the radius."""
    return self.corr_length[...] / self.ylm_map.radius

  def _integrated_grf_variance(self):
    """Integral of the GRF's variance over the earth's surface."""
    return self.variance * self._surf_area

  def _sigma_array(self) -> jax.Array:
    """Array of σₙ from Appendix 8 in [Palmer] http://shortn/_56HCcQwmSS."""
    dinosaur_grid = self.ylm_map.dinosaur_grid
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
    one_minus_phi2 = -jnp.expm1(-2 * self.dt / self.corr_time)
    normalization = jnp.sqrt(
        self._integrated_grf_variance() * one_minus_phi2 / sum_unnormed_vars
    )
    # The factor of coords.horizontal.radius appears because our basis vectors
    # have L2 norm = radius.
    return normalization * sigmas_unnormed / dinosaur_grid.radius

  def sample_core(self, rng: typing.PRNGKeyArray) -> jax.Array:
    """Helper method for sampling the core of the gaussian random field."""
    dinosaur_grid = self.ylm_map.dinosaur_grid
    modal_shape = dinosaur_grid.modal_shape
    sigmas = self._sigma_array()
    weights = jnp.where(
        dinosaur_grid.mask,
        jax.random.truncated_normal(rng, -self.clip, self.clip, modal_shape),
        jnp.zeros(modal_shape),
    )
    one_minus_phi2 = -jnp.expm1(-2 * self.dt / self.corr_time)
    return one_minus_phi2 ** (-0.5) * sigmas * weights

  def advance_core(
      self, state_core: jax.Array, state_key: jax.Array, state_step: int
  ) -> jax.Array:
    """Helper method for advancing the core of the gaussian random field."""
    dinosaur_grid = self.ylm_map.dinosaur_grid
    modal_shape = dinosaur_grid.modal_shape
    rng = _advance_prng_key(state_key, state_step)
    eta = jax.random.truncated_normal(rng, -self.clip, self.clip, modal_shape)
    return state_core * self.phi + self._sigma_array() * jnp.where(
        dinosaur_grid.mask, eta, jnp.zeros(modal_shape)
    )

  @property
  def core_grid(self) -> coordinates.SphericalHarmonicGrid:
    return self.ylm_map.modal_grid


class GaussianRandomField(RandomProcessModule):
  """Spatially and temporally correlated gaussian process on the sphere."""

  def __init__(
      self,
      ylm_map: spherical_harmonics.FixedYlmMapping,
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
        ylm_map=ylm_map,
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
    self.ylm_map = ylm_map
    k, rng = cx.cmap(lambda k: tuple(jax.random.split(k, 2)))(rngs.params())
    self.state_rng = Randomness(rng)
    self.rng_step = Randomness(cx.field(0))
    self.core = Randomness(
        cx.cmap(self.grf.sample_core)(k).tag(self.grf.core_grid)
    )

  def unconditional_sample(self, rng: cx.Field) -> None:
    """Returns a randomly initialized state for the autoregressive process."""
    k, rng = cx.cmap(lambda k: tuple(jax.random.split(k)))(rng)
    self.state_rng.set_value(rng)
    self.rng_step.set_value(
        cx.field(jnp.zeros(k.shape, jnp.uint32), k.coordinate)
    )
    sample_core = cx.cmap(self.grf.sample_core, out_axes='leading')
    self.core.set_value(sample_core(k).tag(self.grf.core_grid))

  def advance(self) -> None:
    """Updates the CoreRandomState of a random gaussian field."""
    self.core.set_value(
        cx.cpmap(self.grf.advance_core)(
            self.core.get_value().untag(self.grf.core_grid),
            self.state_rng.get_value(),
            self.rng_step.get_value(),
        ).tag(self.grf.core_grid)
    )
    self.rng_step.set_value(self.rng_step.get_value() + 1)

  def state_values(
      self,
      coord: cx.Coordinate | None = None,
  ) -> cx.Field:
    if coord is None:
      coord = self.ylm_map.nodal_grid
    if coord == self.ylm_map.modal_grid:
      return self.core.get_value()
    elif coord == self.ylm_map.nodal_grid:
      return self.ylm_map.to_nodal(self.core.get_value())
    else:
      raise ValueError(
          f'Interpolation is not supported yet, requested {coord=} '
          f'but the process is defined on {self.ylm_map.nodal_grid=}'
      )


class VectorizedGaussianRandomField(RandomProcessModule):
  """Vectorized version of GaussianRandomField process."""

  def __init__(
      self,
      ylm_map: spherical_harmonics.FixedYlmMapping,
      dt: float,
      sim_units: units.SimUnits,
      axis: cx.Coordinate,
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
    if axis.ndim != 1:
      raise ValueError(f'collection_axis must be 1d, got {axis=}')
    [n_fields] = axis.shape
    init_specs = [correlation_times, correlation_lengths, variances]
    if any(len(p) != n_fields for p in init_specs):
      raise ValueError(
          'Argument lengths must match collection_axis size, but got: '
          f'{axis=} {init_specs=}'
      )

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
        ylm_map=ylm_map,
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
    self.axis = axis
    self.ylm_map = ylm_map
    self.batch_grf_core = nnx.vmap(make_grf, axis_size=self.n_fields)(
        correlation_lengths, correlation_times, variances
    )
    # Initializing the state of the process.
    k, rng = cx.cmap(lambda k: tuple(jax.random.split(k, 2)))(rngs.params())
    self.state_rng = Randomness(rng)
    self.rng_step = Randomness(cx.field(0))
    self.core = Randomness(self._sample_core(k))

  @property
  def event_shape(self) -> tuple[int, ...]:
    return tuple([self.n_fields])

  def _sample_core(self, rng: cx.Field) -> cx.Field:
    """Samples the core state of a collection of gaussian random fields."""
    split = lambda k: jax.random.split(k, self.n_fields)
    rngs = cx.cmap(split)(rng).tag(self.axis)
    sample_single = lambda grf, rng: grf.sample_core(rng)
    sample_collection = nnx.vmap(sample_single)
    return cx.cmap(sample_collection, vmap=nnx.vmap, out_axes='leading')(
        self.batch_grf_core, rngs.untag(self.axis)
    ).tag(self.axis, self.batch_grf_core.core_grid)

  def unconditional_sample(self, rng: cx.Field) -> None:
    k, next_rng = cx.cmap(lambda k: tuple(jax.random.split(k, 2)))(rng)
    self.state_rng.set_value(next_rng)
    self.rng_step.set_value(
        cx.field(jnp.zeros(k.shape, jnp.uint32), k.coordinate)
    )
    self.core.set_value(self._sample_core(k))

  def advance(self) -> None:
    rng, step = self.state_rng.get_value(), self.rng_step.get_value()
    split = lambda k: jax.random.split(k, self.n_fields)
    rngs = cx.cmap(split)(rng).tag(self.axis)
    advance_keys = cx.cmap(_advance_prng_key)(rngs, step)
    advance_single = lambda grf, core, k, step: grf.advance_core(core, k, step)
    advance_collection = nnx.vmap(advance_single, in_axes=(0, 0, 0, None))
    next_core = cx.cmap(advance_collection, vmap=nnx.vmap, out_axes='leading')(
        self.batch_grf_core,
        self.core.get_value().untag(self.axis, self.batch_grf_core.core_grid),
        advance_keys.untag(self.axis),
        step,
    ).tag(self.axis, self.batch_grf_core.core_grid)
    self.rng_step.set_value(self.rng_step.get_value() + 1)
    self.core.set_value(next_core)

  def state_values(
      self,
      coord: cx.Coordinate | None = None,
  ) -> cx.Field:
    if coord is None:
      coord = self.ylm_map.nodal_grid
    if coord == self.ylm_map.modal_grid:
      return self.core.get_value()
    elif coord == self.ylm_map.nodal_grid:
      return self.ylm_map.to_nodal(self.core.get_value())
    else:
      raise ValueError(
          f'Interpolation is not supported yet, requested {coord=} '
          f'but the process is defined on {self.ylm_map.nodal_grid=}'
      )

  @classmethod
  def with_range_of_scales(
      cls,
      *,
      ylm_map: spherical_harmonics.FixedYlmMapping,
      dt: float | np.timedelta64,
      sim_units: units.SimUnits,
      n_fields: int,
      min_time_hrs: int = 1,
      max_time_hrs: int = 117,
      min_length_km: int = 85,
      max_length_km: int = 10_000,
      power: float = 4.0,
      extra_n_constant_fields: int = 0,
      axis_name: str = 'grf',
      correlation_time_type: type[
          nnx.Param | RandomnessParam
      ] = RandomnessParam,
      correlation_length_type: type[
          nnx.Param | RandomnessParam
      ] = RandomnessParam,
      variance_type: type[nnx.Param | RandomnessParam] = RandomnessParam,
      clip: float = 6.0,
      rngs: nnx.Rngs,
  ):
    """Constructs GRF fields with a range of correlation times and lengths.

    This method generates correlation times and lengths for gaussian random
    fields that range between min and max values. The variance of each field is
    set to 1.0.

    Args:
      ylm_map: Spherical harmonic transform defining the default basis.
      dt: Time step of the random process. If float, must be nondimensionalized.
      sim_units: Object defining nondimensionalization and physical constants.
      n_fields: Number of fields to generate with varying correlation scales.
      min_time_hrs: Minimum time in hours for GRF correlation time.
      max_time_hrs: Maximum time in hours for GRF correlation time.
      min_length_km: Minimum length in km for GRF correlation length.
      max_length_km: Maximum length in km for GRF correlation length.
      power: Power to raise curve from min to max. Power > 1 results in a convex
        curve, and therefore more times/powers near the minimum.
      extra_n_constant_fields: Number of additional trailing fields that are
        initialized as effectively constant in space and time.
      axis_name: Name of coordinate axis for vectorized fields.
      correlation_time_type: Parameter type for correlation time that allows for
        granular selection of the subsets of model parameters.
      correlation_length_type: Parameter type for correlation length that allows
        for granular selection of the subsets of model parameters.
      variance_type: Parameter type for variance that allows for granular
        selection of the subsets of model parameters.
      clip: Number of standard deviations at which to clip randomness to ensure
        numerical stability.
      rngs: Instance for random number generation.

    Returns:
      VectorizedGaussianRandomField instance with specified fields.
    """
    if isinstance(dt, np.timedelta64):
      dt = sim_units.nondimensionalize_timedelta64(dt)
    times = list(
        int(t)
        for t in (
            np.linspace(
                min_time_hrs ** (1 / power),
                max_time_hrs ** (1 / power),
                n_fields,
            )
            ** power
        )
    )
    lengths = list(
        int(l)
        for l in (
            np.linspace(
                min_length_km ** (1 / power),
                max_length_km ** (1 / power),
                n_fields,
            )
            ** power
        )
    )
    correlation_times = [f'{t} hours' for t in times]
    correlation_lengths = [f'{l} km' for l in lengths]
    variances = [1.0] * n_fields

    if extra_n_constant_fields > 0:
      constant_time_hrs = 24 * 365 * 1000  # 1000 years in hours
      constant_length_km = 40_075 * 10  # 10x circumference of earth in km
      for _ in range(extra_n_constant_fields):
        correlation_times.append(f'{constant_time_hrs} hours')
        correlation_lengths.append(f'{constant_length_km} km')
        variances.append(1.0)

    axis = cx.SizedAxis(axis_name, n_fields + extra_n_constant_fields)
    return cls(
        ylm_map=ylm_map,
        dt=dt,
        sim_units=sim_units,
        axis=axis,
        correlation_times=correlation_times,
        correlation_lengths=correlation_lengths,
        variances=variances,
        correlation_time_type=correlation_time_type,
        correlation_length_type=correlation_length_type,
        variance_type=variance_type,
        clip=clip,
        rngs=rngs,
    )
