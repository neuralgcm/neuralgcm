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
"""Implementation of stochastic modules."""

import abc
import dataclasses
import enum
import logging
from typing import Any, Callable, Optional, Sequence, TypeVar, Union
import zlib

from dinosaur import coordinate_systems
from dinosaur import typing
import gin
import haiku as hk
import jax
import jax.numpy as jnp


tree_map = jax.tree_util.tree_map
tree_leaves = jax.tree_util.tree_leaves

Numeric = typing.Numeric
Quantity = typing.Quantity
_SOFTPLUS_INVERSE_1 = 0.5413248546129181

# CoreRandomState is advanced by a RandomField, and .to_*_values(core_state)
# produces the final (usable) random Array.
CoreRandomState = typing.Pytree
RandomnessState = typing.RandomnessState


def _validate_randomness_state(state: RandomnessState) -> None:
  """Validates that `state.core` is not `None`, raises an error otherwise."""
  if state.core is None:
    raise ValueError(
        f'Got {state.core=} when value is expected. '
        'Check how incoming randomness is initialized.'
    )


def make_positive_scalar(raw_parameter: typing.Array) -> jax.Array:
  """Positive [batch] scalar values, maps 0 --> 1 using a softplus(...)."""
  raw_parameter = jnp.asarray(raw_parameter)
  return jax.nn.softplus(raw_parameter + _SOFTPLUS_INVERSE_1)


# pylint: disable=logging-fstring-interpolation


################################################################################
# Single random fields that stand on their own.
################################################################################


class PreferredRepresentation(enum.Enum):
  """The preferred (for computational reasons) representation of a field."""

  NODAL = 'NODAL'
  MODAL = 'MODAL'


class RandomField(abc.ABC):
  """Base class for random fields."""

  def __init__(self, coords):
    self.coords = coords

  @property
  @abc.abstractmethod
  def preferred_representation(self) -> PreferredRepresentation | None:
    """The PreferredRepresentation for this field, or None if no preference."""

  @abc.abstractmethod
  def unconditional_sample(self, rng: typing.PRNGKeyArray) -> RandomnessState:
    """Sample the random field unconditionally."""

  @abc.abstractmethod
  def advance(self, state: RandomnessState) -> RandomnessState:
    """Updates the core state of a random field."""

  @abc.abstractmethod
  def to_modal_values(self, core_state: CoreRandomState) -> typing.Array | None:
    """Returns the modal rep. of the random field specified by this class."""

  @abc.abstractmethod
  def to_nodal_values(self, core_state: CoreRandomState) -> typing.Array | None:
    """Returns the nodal rep. of the random field specified by this class."""


RandomnessModule = Callable[..., RandomField]


_ADVANCE_SALT = zlib.crc32(b'advance')  # arbitrary uint32 value


T = TypeVar('T', typing.PRNGKeyArray, None)


def _prng_key_for_current_advance_step(
    randomness: typing.RandomnessState,
) -> typing.PRNGKeyArray | None:
  """Get a PRNG Key suitable for randomness in the current advance step."""
  if randomness.prng_key is None:
    return None
  salt = jnp.uint32(_ADVANCE_SALT) + jnp.uint32(randomness.prng_step)
  return jax.random.fold_in(randomness.prng_key, salt)


@gin.register
class NoRandomField(RandomField):
  """Module that disables randomness in a given module returning `None`."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Any,
      prefer_nodal: bool = True,
  ):
    """Constructs a ZerosRandomField.

    Args:
      coords: horizontal and vertical grid data.
      dt: nondimensionalized model time step.
      physics_specs: physical constants and definition of custom units.
      aux_features: additional static data.
      prefer_nodal: Whether this field should prefer a nodal representation.
    """
    super().__init__(coords)
    logging.info('[NGCM] Initializing NoRandomField')
    del dt, physics_specs, aux_features, prefer_nodal  # unused.

  @property
  def preferred_representation(self) -> PreferredRepresentation | None:
    return None

  def unconditional_sample(
      self, rng: typing.PRNGKeyArray | None
  ) -> RandomnessState:
    """Returns a zeros initialized state."""
    return RandomnessState(prng_key=rng, prng_step=0)

  def advance(self, state: RandomnessState) -> RandomnessState:
    """Updates the state of a random gaussian field."""
    return RandomnessState(
        prng_key=state.prng_key, prng_step=state.prng_step + 1
    )

  def to_nodal_values(self, core_state: CoreRandomState) -> typing.Array | None:
    del core_state  # unused.
    return None

  def to_modal_values(self, core_state: CoreRandomState) -> typing.Array | None:
    del core_state  # unused.
    return None


@gin.register
class ZerosRandomField(RandomField):
  """Implements a constant random field identically equal to zero."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Any,
      prefer_nodal: bool = True,
  ):
    """Constructs a ZerosRandomField.

    Args:
      coords: horizontal and vertical grid data.
      dt: nondimensionalized model time step.
      physics_specs: physical constants and definition of custom units.
      aux_features: additional static data.
      prefer_nodal: Whether this field should prefer a nodal representation.
    """
    super().__init__(coords)
    logging.info('[NGCM] Initializing ZerosRandomField')
    del dt  # unused
    del physics_specs  # unused.
    del aux_features  # unused.
    self._prefer_nodal = prefer_nodal

  @property
  def preferred_representation(self) -> PreferredRepresentation | None:
    if self._prefer_nodal:
      return PreferredRepresentation.NODAL
    else:
      return PreferredRepresentation.MODAL

  def unconditional_sample(
      self, rng: typing.PRNGKeyArray | None
  ) -> RandomnessState:
    """Returns a zeros initialized state."""
    if self._prefer_nodal:
      core = jnp.zeros(self.coords.horizontal.nodal_shape)
    else:
      core = jnp.zeros(self.coords.horizontal.modal_shape)
    return RandomnessState(
        core=core,
        nodal_value=jnp.zeros(self.coords.horizontal.nodal_shape),
        modal_value=jnp.zeros(self.coords.horizontal.modal_shape),
        prng_key=rng,
        prng_step=0,
    )

  def advance(self, state: RandomnessState) -> RandomnessState:
    """Updates the state of a random gaussian field."""
    _validate_randomness_state(state)
    return RandomnessState(
        core=jnp.zeros_like(state.core),
        nodal_value=jnp.zeros(self.coords.horizontal.nodal_shape),
        modal_value=jnp.zeros(self.coords.horizontal.modal_shape),
        prng_key=state.prng_key,
        prng_step=state.prng_step + 1,
    )

  def to_nodal_values(self, core_state: CoreRandomState) -> typing.Array | None:
    """Returns the ready-for-use Zeros random field."""
    return jnp.zeros(self.coords.horizontal.nodal_shape)

  def to_modal_values(self, core_state: CoreRandomState) -> typing.Array | None:
    """Returns the ready-for-use Zeros random field."""
    return jnp.zeros(self.coords.horizontal.modal_shape)


@gin.register
class GaussianRandomField(RandomField):
  """Implements gaussian random field with spatial and temporal correlations.

  This type of random fields is used in SPPT (stochastic physics
  parameterization tendencies) schemes, where each tendency due to physics
  parameterizations are multiplicatively perturbed by the value of such field.

  For implementation details see Appendix 8 in http://shortn/_56HCcQwmSS.

  With x ∈ EarthSurface, this field U is initialized at t=0 with
    U(0, x) = Σₖ Ψₖ(x) (1 - φ²)^(-0.5) σₖ γₖ σₖ ηₖ₀,
  where Ψₖ is the kth spherical harmonic basis function, φ² is the one timestep
  correlation, σₖ > 0 is a scaling factor, and ηₖ₀ are iid 1D unit Gaussians.

  With `variance` an init kwarg,
    E[U(0, x)] ≡ 0,
    1 / (4πR²) ∫ Var(U(0, x))dx = variance,
  regardless of coords (and the radius).

  Further states are generated with the recursion
    U(t + δ) = ϕ U(t) + σₖ ηₖₜ
  This ensures that U is stationary.

  In general,
    Cov(U(t, x), U(t + δ, y)) = ϕᵟ Σₖ Ψₖ(x) Ψₖ(y) (γₖ)².
  """

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Any,
      correlation_time: Union[jax.Array, Quantity, str] = gin.REQUIRED,
      correlation_length: Union[jax.Array, Quantity, str] = gin.REQUIRED,
      variance: Optional[Union[jax.Array, Quantity, str]] = gin.REQUIRED,
      clip: float = 6.0,
  ):
    """Constructs a GaussianRandomField.

    Args:
      coords: horizontal and vertical grid data.
      dt: nondimensionalized model time step.
      physics_specs: physical constants and definition of custom units.
      aux_features: additional static data.
      correlation_time: timescale with units over which autoregressive process
        decorrelates. Typical values in NWP range from hours to days.
      correlation_length: lengthscale with units over which random field is
        correlated. Typical values in NWP range from 500-2500 km.
      variance: The average (over EarthSurface) variance of the random field If
        None, this GRF always returns a zeros field and no RNGS are drawn.
      clip: number of standard deviations at which to clip randomness to ensure
        numerical stability.
    """
    del aux_features  # unused.
    super().__init__(coords)
    logging.info(
        '[NGCM] Initializing GaussianRandomField (possibly via'
        f' CenteredLognormalRandomField) with {variance=}, {correlation_time=},'
        f' {correlation_length=}'
    )

    tau = maybe_nondimensionalize(correlation_time, physics_specs)
    correlation_length = maybe_nondimensionalize(
        correlation_length, physics_specs
    )

    # In sampling, phi appears as 1 - phi**2 = 1 - exp(-2 dt / tau)
    self.one_minus_phi2 = -jnp.expm1(-2 * dt / tau)

    self.phi = jnp.exp(-dt / tau)

    self._variance = maybe_nondimensionalize(variance, physics_specs)  # σ²

    # [Palmer] states correlation_length = sqrt(2κT) / R, therefore
    self.kt = (correlation_length / self.coords.horizontal.radius) ** 2 / 2
    self.clip = clip

  @property
  def preferred_representation(self) -> PreferredRepresentation | None:
    return PreferredRepresentation.MODAL

  @property
  def _surf_area(self) -> jax.Array:
    """Surface area of sphere of radius self.coords.horizontal.radius."""
    return 4 * jnp.pi * self.coords.horizontal.radius**2  # pytype: disable=bad-return-type  # jnp-type

  def _sigma_array(self) -> jax.Array:
    """Array of σₙ from Appendix 8 in [Palmer] http://shortn/_56HCcQwmSS."""
    # n = [0, 1, ..., N]
    n = self.coords.horizontal.modal_axes[1]  # total wavenumbers.

    # Number of longitudinal wavenumbers at each total wavenumber n.
    #  L = 2n + 1, except for the last entry.
    n_longitudian_wavenumbers = self.coords.horizontal.mask.sum(axis=0)

    # sigmas_unnormed[n] is proportional to the standard deviation for each
    # longitudinal wavenumbers at each total wavenumber n.
    sigmas_unnormed = jnp.exp(-0.5 * self.kt * n * (n + 1))

    # The sum of unnormalized variance for all longitudinal wavenumbers at each
    # total wavenumber.
    sum_unnormed_vars = jnp.sum(n_longitudian_wavenumbers * sigmas_unnormed**2)

    # This is analogous to F₀ from [Palmer].
    # (normalization * sigmas_unnormed)² would sum to 1. The leading factor
    #   self._integrated_grf_variance * (1 - self.phi ** 2)
    # ensures that the AR(1) process has variance self._integrated_grf_variance.
    # We do not include the extra fator of 2 in the denominator. I do not know
    # why [Palmer] has this factor.
    normalization = jnp.sqrt(
        self._integrated_grf_variance()
        * self.one_minus_phi2
        / sum_unnormed_vars
    )

    # The factor of coords.horizontal.radius appears because our basis vectors
    # have L2 norm = radius.  See http://screen/9FYVXZ5cMHoGDZk
    return normalization * sigmas_unnormed / self.coords.horizontal.radius

  def unconditional_sample(self, rng: typing.PRNGKeyArray) -> RandomnessState:
    """Returns a randomly initialized state for the autoregressive process."""
    modal_shape = self.coords.horizontal.modal_shape
    rng, next_rng = jax.random.split(rng)
    if self.variance is None:
      return RandomnessState(
          core=jnp.zeros(modal_shape),
          nodal_value=jnp.zeros(self.coords.horizontal.nodal_shape),
          modal_value=jnp.zeros(modal_shape),
          prng_key=next_rng,
          prng_step=0,
      )
    sigmas = self._sigma_array()
    weights = jnp.where(
        self.coords.horizontal.mask,
        jax.random.truncated_normal(rng, -self.clip, self.clip, modal_shape),
        jnp.zeros(modal_shape),
    )
    core = self.one_minus_phi2 ** (-0.5) * sigmas * weights
    return RandomnessState(
        core=core,
        nodal_value=self.to_nodal_values(core),
        modal_value=self.to_modal_values(core),
        prng_key=next_rng,
        prng_step=0,
    )

  def advance(self, state: RandomnessState) -> RandomnessState:
    """Updates the CoreRandomState of a random gaussian field."""
    _validate_randomness_state(state)
    if self.variance is None:
      return RandomnessState(
          core=jnp.zeros_like(state.core),
          nodal_value=jnp.zeros(self.coords.horizontal.nodal_shape),
          modal_value=jnp.zeros(self.coords.horizontal.modal_shape),
          prng_key=state.prng_key,
          prng_step=state.prng_step + 1,
      )
    modal_shape = self.coords.horizontal.modal_shape
    rng = _prng_key_for_current_advance_step(state)
    eta = jax.random.truncated_normal(rng, -self.clip, self.clip, modal_shape)
    next_core = state.core * self.phi + self._sigma_array() * jnp.where(
        self.coords.horizontal.mask, eta, jnp.zeros(modal_shape)
    )
    return RandomnessState(
        core=next_core,
        nodal_value=self.to_nodal_values(next_core),
        modal_value=self.to_modal_values(next_core),
        prng_key=state.prng_key,
        prng_step=state.prng_step + 1,
    )

  @property
  def variance(self) -> Numeric | None:
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
    return self._variance

  def _integrated_grf_variance(self) -> Numeric | None:
    """Integral of the GRF's variance over the earth's surface."""
    if self.variance is None:
      return self.variance
    return self.variance * self._surf_area

  def to_modal_values(self, core_state: CoreRandomState) -> typing.Array | None:
    """Returns the ready-for-use Gaussian random field."""
    return core_state

  def to_nodal_values(self, core_state: CoreRandomState) -> typing.Array | None:
    """Returns the ready-for-use Gaussian random field."""
    return self.coords.horizontal.to_nodal(core_state)


@gin.register
class GaussianRandomFieldModule(GaussianRandomField, hk.Module):
  """Module wrapper of GaussianRandomField with trainable parameters."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Any,
      initial_correlation_time: Union[Quantity, str] = gin.REQUIRED,
      initial_correlation_length: Union[Quantity, str] = gin.REQUIRED,
      initial_variance: Optional[Union[Quantity, str]] = gin.REQUIRED,
      variance_bound: Optional[Union[Quantity, str]] = gin.REQUIRED,
      tune_variance: bool = True,
      clip: float = 6.0,
      name: Optional[str] = None,
  ):
    """Constructs a GaussianRandomFieldModule.

    Stochastic parameters are initialized at provided `initial_*` values.
    This hk.Module can then be used to tune values.

    Args:
      coords: horizontal and vertical grid data.
      dt: nondimensionalized model time step.
      physics_specs: physical constants and definition of custom units.
      aux_features: additional static data.
      initial_correlation_time: timescale with units over which autoregressive
        process decorrelates. Typical values in NWP range from hours to days.
      initial_correlation_length: lengthscale with units over which random field
        is correlated. Typical values in NWP range from 500-2500 km.
      initial_variance: The average (over EarthSurface) variance of the random
        field. If None, this GRF always returns a zeros field and no RNGS will
        be drawn
      variance_bound: If provided, an upper bound on tuned variance values.
      tune_variance: Whether variance should be a tunable hk.parameter, or fixed
      clip: number of standard deviations at which to clip randomness to ensure
        numerical stability.
      name: Something no one cares about and we just use None.
    """
    # You must call hk.Module.__init__ before initializing this class.
    hk.Module.__init__(self, name=name)

    correlation_time_raw = hk.get_parameter(
        'correlation_time_raw', shape=(), init=hk.initializers.Constant(0.0)
    )
    correlation_length_raw = hk.get_parameter(
        'correlation_length_raw', shape=(), init=hk.initializers.Constant(0.0)
    )

    if tune_variance:
      variance_raw = hk.get_parameter(
          'variance_raw', shape=(), init=hk.initializers.Constant(0.0)
      )
    else:
      variance_raw = 0.0

    initial_variance = maybe_nondimensionalize(initial_variance, physics_specs)
    _assert_positive_or_none(initial_variance, 'initial_variance')

    if initial_variance is None:
      variance = None
    elif variance_bound in {None, 'None'}:  # Allow strings for gin.
      variance = convert_hk_param_to_positive_scalar(
          variance_raw, initial_variance
      )
    else:
      variance_bound = maybe_nondimensionalize(variance_bound, physics_specs)
      _assert_positive_or_none(variance_bound, 'variance_bound')
      _assert_positive_or_none(
          variance_bound - initial_variance, 'variance_bound - initial_variance'
      )
      variance = convert_hk_param_to_bounded_scalar(
          variance_raw,
          initial_variance,
          low=0.0,
          high=variance_bound,
      )

    # We call GaussianRandomFieldModule.__init__ rather than super().__init__
    # since we don't want to call hk.Module.__init__ twice... although doing
    # that didn't hurt anything.
    GaussianRandomField.__init__(
        self,
        coords=coords,
        dt=dt,
        physics_specs=physics_specs,
        aux_features=aux_features,
        correlation_time=convert_hk_param_to_positive_scalar(
            correlation_time_raw,
            maybe_nondimensionalize(initial_correlation_time, physics_specs),
        ),
        correlation_length=convert_hk_param_to_positive_scalar(
            correlation_length_raw,
            maybe_nondimensionalize(initial_correlation_length, physics_specs),
        ),
        variance=variance,
        clip=clip,
    )


################################################################################
# Single random fields that are derived from "stand on their own" fields.
################################################################################


@gin.register
class CenteredLognormalRandomField(GaussianRandomField):
  """A lognormal random field shifted to have mean zero."""

  @property
  def preferred_representation(self) -> PreferredRepresentation | None:
    return PreferredRepresentation.NODAL

  def _integrated_grf_variance(self) -> jax.Array | None:
    """Integrated variance of the associated GRF (not this Lognormal field)."""
    if self.variance is None:
      return None
    # If Z ~ Normal(μ, σ²), then X ~ exp(Z) has
    #  variance = (exp(σ²) - 1) exp(2μ + σ²).
    # We have centered this field, which involved setting μ = -σ² / 2.
    # => variance = exp(σ²) - 1,
    # and thus
    #  σ² = log(1 + variance)
    return jnp.log1p(self.variance) * self._surf_area

  def to_nodal_values(self, core_state: CoreRandomState) -> jax.Array:
    """Returns the ready-for-use Lognormal random field."""
    if self.variance is None:
      grf_variance = 0.0
    else:
      grf_variance = self._integrated_grf_variance() / self._surf_area
    # If Z ~ Normal(μ, σ²), then X ~ exp(Z) has mean exp(μ + σ²/2).
    # To ensure E[X] = 1, we must set μ = -σ²/2.
    x = self.coords.horizontal.to_nodal(core_state)  # ~ Normal(0, σ²)
    return jnp.expm1(x - grf_variance / 2)  # ~ Exp(Normal(-σ²/2, σ²)) - 1

  def to_modal_values(self, core_state: CoreRandomState) -> jax.Array:
    """Returns the ready-for-use Lognormal random field."""
    return self.coords.horizontal.to_modal(self.to_nodal_values(core_state))


@gin.register
class CenteredLognormalRandomFieldModule(
    CenteredLognormalRandomField, GaussianRandomFieldModule
):
  """A lognormal random hk.Module field shifted to have mean zero."""


################################################################################
# Fields made from many different fields.
################################################################################


@gin.register
class BatchGaussianRandomFieldModule(hk.Module):
  """Batch of independent GaussianRandomFieldModules.

  These GRFs are meant to be fed into a neural network as generic "signals".

  The state arrays have leading batch dim indexing independent GRFs.
  """

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Any,
      initial_correlation_times: Sequence[Quantity | str] = gin.REQUIRED,
      initial_correlation_lengths: Sequence[Quantity | str] = gin.REQUIRED,
      variances: Sequence[Quantity | str] = gin.REQUIRED,
      field_subset: Optional[Sequence[int]] = None,
      n_fixed_fields: Optional[int] = None,
      clip: float = 6.0,
      name: Optional[str] = None,
  ):
    """Constructs a BatchGaussianRandomFieldModule.

    Correlation scales are initialized to `initial_*` args and will be tuned
    by Haiku optimizers. Variance will be fixed.

    Args:
      coords: horizontal and vertical grid data.
      dt: nondimensionalized model time step.
      physics_specs: physical constants and definition of custom units.
      aux_features: additional static data.
      initial_correlation_times: timescales with units over which autoregressive
        process decorrelates. Typical values in NWP range from hours to days.
      initial_correlation_lengths: lengthscale with units over which random
        field is correlated. Typical values in NWP range from 500-2500 km.
      variances: The average (over EarthSurface) variance of the random field.
        These are fixed arrays (not tunable hk.parameters).
      field_subset: Optional nonempty subset of indices into initial parameters.
        Specifies which fields to construct. If None, use all fields. E.g.,
        field_subset=[0, 5] means form 3 GRFs from the 0th and 5th parameter
        values.
      n_fixed_fields: Number of fields that use fixed parameters. These will
        be fixed at the trailing `n_fixed_fields` initial correlations. The
        total number of fields is unchanged, since these fixed fields replace
        learnable fields.
      clip: number of standard deviations at which to clip randomness to ensure
        numerical stability.
      name: Name to show in xprof.
    """
    ## You must call hk.Module.__init__ before initializing this class.
    hk.Module.__init__(self, name=name)

    lengths = [
        len(initial_correlation_times),
        len(initial_correlation_lengths),
        len(variances),
    ]
    if len(set(lengths)) != 1:
      raise ValueError(f'Argument lengths differed: {lengths=}')
    n_fixed_fields = n_fixed_fields or 0

    # Get subset of args using `field_subset`
    if field_subset is not None:
      if not field_subset:
        raise ValueError(
            '`field_subset` must be `None` or non-empty sequence. Found'
            f' {field_subset=}'
        )
      get_subset = lambda seq: [seq[i] for i in field_subset]
      initial_correlation_lengths = get_subset(initial_correlation_lengths)
      initial_correlation_times = get_subset(initial_correlation_times)
      variances = get_subset(variances)

    logging.info(
        '[NGCM] Initializing BatchGaussianRandomFieldModule with'
        f' {initial_correlation_times=}, and {initial_correlation_lengths=},'
        f' and {variances=}'
    )

    # Get Haiku parameters.
    self._n_fields = len(variances)
    self._variances = jnp.array(
        [nondimensionalize(v, physics_specs) for v in variances]
    )

    initial_correlation_lengths = jnp.array([
        nondimensionalize(l, physics_specs) for l in initial_correlation_lengths
    ])
    correlation_lengths_raw = hk.get_parameter(
        'correlation_lengths_raw',
        shape=(self.n_fields - n_fixed_fields,),
        init=hk.initializers.Constant(0.0),
    )
    if n_fixed_fields:
      correlation_lengths_raw = jnp.concatenate([
          correlation_lengths_raw, jnp.zeros([n_fixed_fields])])
    self._correlation_lengths = convert_hk_param_to_positive_scalar(
        correlation_lengths_raw, initial_correlation_lengths
    )

    initial_correlation_times = jnp.array(
        [nondimensionalize(t, physics_specs) for t in initial_correlation_times]
    )
    correlation_times_raw = hk.get_parameter(
        'correlation_times_raw',
        shape=(self.n_fields - n_fixed_fields,),
        init=hk.initializers.Constant(0.0),
    )
    if n_fixed_fields:
      correlation_times_raw = jnp.concatenate([
          correlation_times_raw, jnp.zeros([n_fixed_fields])])
    self._correlation_times = convert_hk_param_to_positive_scalar(
        correlation_times_raw, initial_correlation_times
    )

    def make_rf(correlation_time, correlation_length, variance):
      return GaussianRandomField(
          coords=coords,
          dt=dt,
          physics_specs=physics_specs,
          aux_features=aux_features,
          correlation_time=correlation_time,
          correlation_length=correlation_length,
          variance=variance,
          clip=clip,
      )

    self._make_rf = make_rf

  @property
  def n_fields(self) -> int:
    return self._n_fields

  def unconditional_sample(self, rng: typing.PRNGKeyArray) -> RandomnessState:
    """Sample the batch GRFs unconditionally."""
    logging.info(
        '[NGCM] Calling BatchGaussianRandomFieldModule.unconditional_sample'
    )

    def _unconditional_sample_one_rf(
        key, correlation_time, correlation_length, variance
    ):
      rf = self._make_rf(correlation_time, correlation_length, variance)
      return rf.unconditional_sample(key)

    rngs = jax.random.split(rng, self.n_fields + 1)
    rngs, next_rng = rngs[:-1], rngs[-1]
    sample = jax.vmap(_unconditional_sample_one_rf)(
        rngs,
        self._correlation_times,
        self._correlation_lengths,
        self._variances,
    )
    # We have RNG keys and steps associated with each field from vmap, but
    # RandomnessState should only have a single (scalar) RNG key/step.
    return dataclasses.replace(sample, prng_key=next_rng, prng_step=0)

  def advance(self, state: RandomnessState) -> RandomnessState:
    """Updates the state of the batch of GRFs."""
    logging.info('[NGCM] Calling BatchGaussianRandomFieldModule.advance')

    def _advance_one_rf(state, correlation_time, correlation_length, variance):
      rf = self._make_rf(correlation_time, correlation_length, variance)
      return rf.advance(state)

    rng = _prng_key_for_current_advance_step(state)
    rngs = jax.random.split(rng, self.n_fields)
    steps = jnp.ones(self.n_fields, int) * state.prng_step
    advanced = jax.vmap(_advance_one_rf)(
        dataclasses.replace(state, prng_key=rngs, prng_step=steps),
        self._correlation_times,
        self._correlation_lengths,
        self._variances,
    )
    return dataclasses.replace(
        advanced, prng_key=state.prng_key, prng_step=state.prng_step + 1
    )


@gin.register
class DictOfGaussianRandomFieldModules(hk.Module):
  """Dictionary of independent GaussianRandomFieldModules.

  These GRFs are meant to be fed into a neural network as generic "signals".
  """

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Any,
      initial_correlation_times: Sequence[Quantity | str] = gin.REQUIRED,
      initial_correlation_lengths: Sequence[Quantity | str] = gin.REQUIRED,
      variances: Sequence[Quantity | str] = gin.REQUIRED,
      field_names: Optional[Sequence[str]] = None,
      field_subset: Optional[Sequence[int]] = None,
      clip: float = 6.0,
      name: Optional[str] = None,
  ):
    """Constructs a DictOfGaussianRandomFieldModules.

    Correlation scales are initialized to `initial_*` args and will be tuned
    by Haiku optimizers. Variance will be fixed.

    Args:
      coords: horizontal and vertical grid data.
      dt: nondimensionalized model time step.
      physics_specs: physical constants and definition of custom units.
      aux_features: additional static data.
      initial_correlation_times: timescales with units over which autoregressive
        process decorrelates. Typical values in NWP range from hours to days.
      initial_correlation_lengths: lengthscale with units over which random
        field is correlated. Typical values in NWP range from 500-2500 km.
      variances: The average (over EarthSurface) variance of the random field.
        These are fixed arrays (not tunable hk.parameters).
      field_names: Optional names to give the fields. If None, the fields are
        named like "GRF0", "GRF1",...
      field_subset: Optional nonempty subset of indices into initial parameters.
        Specifies which fields to construct. If None, use all fields. E.g.,
        field_subset=[0, 5] means form 3 GRFs from the 0th and 5th parameter
        values.
      clip: number of standard deviations at which to clip randomness to ensure
        numerical stability.
      name: Name to show in xprof.
    """
    ## You must call hk.Module.__init__ before initializing this class.
    hk.Module.__init__(self, name=name)
    logging.info(
        '[NGCM] Initializing DictOfGaussianRandomFieldModules with'
        f' {initial_correlation_times=}, and {initial_correlation_lengths=},'
        f' and {variances=}'
    )

    field_names = field_names or [
        f'GRF{i}' for i in range(len(initial_correlation_times))
    ]

    lengths = [
        len(initial_correlation_times),
        len(initial_correlation_lengths),
        len(variances),
        len(field_names),
    ]
    if len(set(lengths)) != 1:
      raise ValueError(f'Argument lengths differed: {lengths=}')

    if field_subset is not None:
      if not field_subset:
        raise ValueError(
            '`field_subset` must be `None` or non-empty sequence. Found'
            f' {field_subset=}'
        )
      subset = lambda seq: [seq[i] for i in field_subset]
      field_names = subset(field_names)
      initial_correlation_lengths = subset(initial_correlation_lengths)
      initial_correlation_times = subset(initial_correlation_times)
      variances = subset(variances)

    self._field_names = tuple(field_names)

    self._random_fields = {}
    for tau, lam, var, field_name in zip(
        initial_correlation_times,
        initial_correlation_lengths,
        variances,
        self.field_names,
        strict=True,
    ):
      self._random_fields[field_name] = GaussianRandomFieldModule(
          coords,
          dt,
          physics_specs,
          aux_features,
          initial_correlation_time=tau,
          initial_correlation_length=lam,
          initial_variance=var,
          tune_variance=False,
          variance_bound=None,
          clip=clip,
          name=field_name,
      )

  @property
  def n_fields(self) -> int:
    return len(self._random_fields)

  @property
  def field_names(self) -> tuple[str, ...]:
    return self._field_names

  def unconditional_sample(self, rng: typing.PRNGKeyArray) -> RandomnessState:
    """Sample the random field unconditionally."""
    core = {}
    nodal_values = {}
    modal_values = {}
    *rngs, next_rng = jax.random.split(rng, self.n_fields + 1)
    for (name, rf), sample_key in zip(self._random_fields.items(), rngs):
      rvs = rf.unconditional_sample(sample_key)
      core[name] = rvs.core
      nodal_values[name] = rvs.nodal_value
      modal_values[name] = rvs.modal_value
    return RandomnessState(
        core=core,
        nodal_value=nodal_values,
        modal_value=modal_values,
        prng_key=next_rng,
        prng_step=0,
    )

  def advance(self, state: RandomnessState) -> RandomnessState:
    """Updates the core state of a random field."""
    core = {}
    nodal_values = {}
    modal_values = {}
    rng = _prng_key_for_current_advance_step(state)
    rngs = jax.random.split(rng, self.n_fields)
    for (name, rf), sample_key in zip(self._random_fields.items(), rngs):
      # rvs is a RandomnessState.
      rvs = rf.advance(
          RandomnessState(state.core[name], prng_key=sample_key, prng_step=0)
      )
      core[name] = rvs.core
      nodal_values[name] = rvs.nodal_value
      modal_values[name] = rvs.modal_value
    return RandomnessState(
        core=core,
        nodal_value=nodal_values,
        modal_value=modal_values,
        prng_key=state.prng_key,
        prng_step=state.prng_step + 1,
    )


class SumOfRandomFields(RandomField):
  """RandomField that is the sum of multiple fields."""

  def __init__(self, random_fields: Sequence[RandomField]):
    self._random_fields = list(random_fields)  # Shallow copy
    coords = self._random_fields[0].coords
    if any(rf.coords != coords for rf in self._random_fields):
      raise ValueError(f'All fields must have the same coords. Found {coords=}')
    super().__init__(coords)

  @property
  def preferred_representation(self) -> PreferredRepresentation | None:
    n_nodal = sum(
        rf.preferred_representation == PreferredRepresentation.NODAL
        for rf in self._random_fields
    )
    n_modal = sum(
        rf.preferred_representation == PreferredRepresentation.MODAL
        for rf in self._random_fields
    )
    if n_nodal > n_modal:
      return PreferredRepresentation.NODAL
    elif n_nodal < n_modal:
      return PreferredRepresentation.MODAL
    return None

  def unconditional_sample(self, rng: typing.PRNGKeyArray) -> RandomnessState:
    """Sample the random field unconditionally."""
    rvs = []
    *rngs, next_rng = jax.random.split(rng, len(self._random_fields) + 1)
    for rf, sample_key in zip(self._random_fields, rngs, strict=True):
      rvs.append(rf.unconditional_sample(sample_key).core)
    return RandomnessState(
        core=rvs,
        nodal_value=self.to_nodal_values(rvs),
        modal_value=self.to_modal_values(rvs),
        prng_key=next_rng,
        prng_step=0,
    )

  def advance(self, state: RandomnessState) -> RandomnessState:
    """Updates the core state of a random field."""
    rvs = []
    rng = _prng_key_for_current_advance_step(state)
    rngs = jax.random.split(rng, len(self._random_fields))
    for rf, s, k in zip(
        self._random_fields, state.core, rngs, strict=True
    ):
      rs = RandomnessState(s, prng_key=k, prng_step=state.prng_step)
      rvs.append(rf.advance(rs).core)
    return RandomnessState(
        core=rvs,
        nodal_value=self.to_nodal_values(rvs),
        modal_value=self.to_modal_values(rvs),
        prng_key=state.prng_key,
        prng_step=state.prng_step + 1,
    )

  def to_modal_values(self, core_state: CoreRandomState) -> typing.Array | None:
    """Finishes `core_state` by summing components."""
    modal_sum = 0.0
    nodal_sum = 0.0
    for rf, s in zip(self._random_fields, core_state, strict=True):
      if rf.preferred_representation == PreferredRepresentation.NODAL:
        nodal_sum += rf.to_nodal_values(s)
      elif rf.preferred_representation in [PreferredRepresentation.MODAL, None]:
        modal_sum += rf.to_modal_values(s)
    return modal_sum + self.coords.horizontal.to_modal(nodal_sum)

  def to_nodal_values(self, core_state: CoreRandomState) -> typing.Array | None:
    """Finishes `core_state` by summing components."""
    modal_sum = 0.0
    nodal_sum = 0.0
    for rf, s in zip(self._random_fields, core_state, strict=True):
      if rf.preferred_representation == PreferredRepresentation.MODAL:
        modal_sum += rf.to_modal_values(s)
      elif rf.preferred_representation in [PreferredRepresentation.NODAL, None]:
        nodal_sum += rf.to_nodal_values(s)
    return nodal_sum + self.coords.horizontal.to_nodal(modal_sum)


class SumOfGaussianLikeRandomFields(SumOfRandomFields, abc.ABC):
  """Base class for sum of independent Gaussian-like random fields."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Any,
      correlation_times: Sequence[
          Union[jax.Array, Quantity, str]
      ] = gin.REQUIRED,
      correlation_lengths: Sequence[
          Union[jax.Array, Quantity, str]
      ] = gin.REQUIRED,
      variances: Sequence[Union[jax.Array, Quantity, str]] = gin.REQUIRED,
      clip: float = 6.0,
  ):
    """Constructs a SumOfGaussianLikeRandomFields."""
    n_fields = len(correlation_times)
    variances = variances or [None] * n_fields
    random_fields = []
    logging.info(
        '[NGCM] Initializing SumOfGaussianLikeRandomFields with '
        f'{variances=}, {correlation_times=}, {correlation_lengths=}'
    )
    for tau, lam, var in zip(
        correlation_times, correlation_lengths, variances, strict=True
    ):
      random_fields.append(
          self.get_cls_constructor()(
              coords,
              dt,
              physics_specs,
              aux_features,
              correlation_time=tau,
              correlation_length=lam,
              variance=var,
              clip=clip,
          )
      )

    super().__init__(random_fields)

  @abc.abstractmethod
  def get_cls_constructor(self) -> type[GaussianRandomField]:
    """Gets class constructor that is initialized with Gaussian-like kwargs."""


class SumOfGaussianLikeRandomFieldsModule(
    SumOfRandomFields, hk.Module, abc.ABC
):
  """Base class for sums of independent Gaussian-like RandomFieldModules."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Any,
      initial_correlation_times: Sequence[Quantity | str] = gin.REQUIRED,
      initial_correlation_lengths: Sequence[Quantity | str] = gin.REQUIRED,
      initial_variances: Optional[Sequence[Quantity | str]] = gin.REQUIRED,
      variance_bounds: Optional[Sequence[Quantity | str]] = gin.REQUIRED,
      clip: float = 6.0,
      name: Optional[str] = None,
  ):
    """Constructs a SumOfGaussianLikeRandomFieldsModule."""
    # You must call hk.Module.__init__ before initializing this class.
    hk.Module.__init__(self, name=name)

    n_fields = len(initial_correlation_times)
    initial_variances = initial_variances or [None] * n_fields
    variance_bounds = variance_bounds or [None] * n_fields
    random_fields = []
    for tau, lam, var, bound in zip(
        initial_correlation_times,
        initial_correlation_lengths,
        initial_variances,
        variance_bounds,
        strict=True,
    ):
      random_fields.append(
          self.get_cls_constructor()(
              coords,
              dt,
              physics_specs,
              aux_features,
              initial_correlation_time=tau,
              initial_correlation_length=lam,
              initial_variance=var,
              variance_bound=bound,
              clip=clip,
              name=name,
          )
      )
    # We call SumOfRandomFields.__init__ rather than super().__init__
    # since we don't want to call hk.Module.__init__ twice... although doing
    # that didn't hurt anything.
    SumOfRandomFields.__init__(self, random_fields)

  @abc.abstractmethod
  def get_cls_constructor(self) -> type[GaussianRandomFieldModule]:
    """Gets class constructor that is initialized with Gaussian-like kwargs."""


@gin.register
class SumOfGaussianRandomFields(SumOfGaussianLikeRandomFields):

  def get_cls_constructor(self) -> type[GaussianRandomField]:
    return GaussianRandomField


@gin.register
class SumOfGaussianRandomFieldsModule(SumOfGaussianLikeRandomFieldsModule):
  """A sum of independent GaussianRandomFieldModules."""

  def get_cls_constructor(self) -> type[GaussianRandomFieldModule]:
    return GaussianRandomFieldModule


@gin.register
class SumOfCenteredLognormalRandomFields(SumOfGaussianLikeRandomFields):

  def get_cls_constructor(self) -> type[CenteredLognormalRandomField]:
    return CenteredLognormalRandomField


@gin.register
class SumOfCenteredLognormalRandomFieldsModule(
    SumOfGaussianLikeRandomFieldsModule
):
  """A sum of independent CenteredLognormalRandomFieldModules."""

  def get_cls_constructor(self) -> type[CenteredLognormalRandomFieldModule]:
    return CenteredLognormalRandomFieldModule


################################################################################
# Helper functions for creating fields.
################################################################################


def convert_hk_param_to_positive_scalar(
    param: jax.Array,
    initial_value: Numeric,
) -> jax.Array:
  """Converts [batch] scalar parameter to scalar value using Softplus."""
  return initial_value * make_positive_scalar(param)


def _sigmoid(low: Numeric, high: Numeric, x: jax.Array) -> jax.Array:
  """Numerically stable sigmoid, adapted from tfp.bijectors.Sigmoid."""
  diff = high - low
  left = low + diff * jax.nn.sigmoid(x)
  right = high - diff * jax.nn.sigmoid(-x)
  return jnp.where(x < 0, left, right)


def _inv_sigmoid(low: Numeric, high: Numeric, x: jax.Array) -> jax.Array:
  """Inverse sigmoid, adapted from tfp.bijectors.Sigmoid."""
  return jnp.log(x - low) - jnp.log(high - x)


def convert_hk_param_to_bounded_scalar(
    param: jax.Array,
    initial_value: Numeric,
    low: Numeric,
    high: Numeric,
) -> jax.Array:
  """Converts a [batch] scalar parameter to scalar value using Sigmoid."""
  offset = _inv_sigmoid(low, high, initial_value)
  return _sigmoid(low, high, offset + param)


def nondimensionalize(
    x: Union[typing.Numeric, Quantity, str],
    physics_specs: Any,
) -> typing.Numeric:
  if isinstance(x, (Quantity, str)):
    return physics_specs.nondimensionalize(Quantity(x))
  else:
    return x


def maybe_nondimensionalize(
    x: Optional[Union[typing.Numeric, Quantity, str]],
    physics_specs: Any,
) -> None | typing.Numeric:
  """Calls nondimensionalize on Quantity or str, otherwise passthrough."""
  if x == 'None':  # Allow strings for gin
    return None
  return nondimensionalize(x, physics_specs)


def _assert_positive_or_none(x: typing.Numeric | None, name: str) -> None:
  if x is None:
    return
  if x <= 0:
    raise ValueError(f'{name}={x} but should have been positive or None')
