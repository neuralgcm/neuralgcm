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

"""Implementation of the Lorenz96 model components for demostration purposes."""

from collections.abc import Sequence
import dataclasses
import functools
from typing import Callable, Protocol

import coordax as cx
from flax import nnx
import jax
import jax.numpy as jnp
import jax_datetime as jdt
from neuralgcm.experimental.core import api
from neuralgcm.experimental.core import data_specs
from neuralgcm.experimental.core import module_utils
from neuralgcm.experimental.core import nnx_compat
from neuralgcm.experimental.core import observation_operators
from neuralgcm.experimental.core import random_processes
from neuralgcm.experimental.core import time_integrators
from neuralgcm.experimental.core import transforms
from neuralgcm.experimental.core import typing
import numpy as np


IntegratorCls = Callable[[time_integrators.ExplicitODE, float], nnx.Module]


def lorenz96_x_term(x: jax.Array, f: float) -> jax.Array:
  """Computes uncoupled term for `X` variables in L96 system."""
  x_p1 = jnp.roll(x, -1)
  x_m1 = jnp.roll(x, 1)
  x_m2 = jnp.roll(x, 2)
  return -x_m1 * (x_m2 - x_p1) - x + f


def lorenz96_y_term(y: jax.Array, c: float, b: float) -> jax.Array:
  """Computes uncoupled term for `Y` variables in L96 system."""
  y_p1 = jnp.roll(y, -1)
  y_p2 = jnp.roll(y, -2)
  y_m1 = jnp.roll(y, 1)
  return -(c * b) * y_p1 * (y_p2 - y_m1) - c * y


@nnx_compat.dataclass
class Lorenz96Base(api.Model):
  """Base class for Lorenz-96 models providing common methods."""

  dt: float = dataclasses.field(default=0.002, kw_only=True)
  time_integrator_cls: IntegratorCls = dataclasses.field(
      default=time_integrators.RungeKutta4, kw_only=True
  )
  operators: dict[str, observation_operators.ObservationOperatorABC] = (
      dataclasses.field(default_factory=dict, kw_only=True)
  )

  def __post_init__(self):
    # Adds input-like obs-operators if not set explicitly for dmeo purposes.
    if 'slow' not in self.operators and hasattr(self, 'x'):
      sel_x_and_t = transforms.Select(['x', 'time'])
      obs_op = observation_operators.TransformObservationOperator(sel_x_and_t)
      self.operators['slow'] = obs_op
    if 'fast' not in self.operators and hasattr(self, 'y'):
      sel_y_and_t = transforms.Select(['y', 'time'])
      obs_op = observation_operators.TransformObservationOperator(sel_y_and_t)
      self.operators['fast'] = obs_op

  @property
  def prognostics(self):
    prognostic_fields = {}
    if hasattr(self, 'x'):
      prognostic_fields['x'] = self.x.get_value()
    if hasattr(self, 'y'):
      prognostic_fields['y'] = self.y.get_value()
    prognostic_fields['time'] = self.time.get_value()
    return prognostic_fields

  @module_utils.ensure_unchanged_state_structure
  def observe(
      self,
      queries: dict[str, dict[str, cx.Field | cx.Coordinate]],
  ) -> dict[str, dict[str, cx.Field]]:
    result = {}
    for k, q in queries.items():
      if k in self.operators:
        result[k] = self.operators[k].observe(self.prognostics, q)
      else:
        raise ValueError(f'No observation operator for key "{k}"')
    return result

  @property
  def timestep(self) -> np.timedelta64:
    # Lorenz96 non dimensional time roughly corresponds to 120h.
    return np.timedelta64(int(self.dt * 120 * 3600), 's')

  @property
  def inputs_spec(self):
    specs = {}
    if hasattr(self, 'x'):
      cs = {'x': self.x.get_value().coordinate, 'time': cx.Scalar()}
      specs['slow'] = {
          k: data_specs.CoordSpec.with_any_timedelta(v) for k, v in cs.items()
      }
    if hasattr(self, 'y'):
      cs = {'y': self.y.get_value().coordinate}
      if not hasattr(self, 'x'):  # fast-only
        cs['time'] = cx.Scalar()
      specs['fast'] = {
          k: data_specs.CoordSpec.with_any_timedelta(v) for k, v in cs.items()
      }
    return specs


@nnx_compat.dataclass
class Lorenz96WithTwoScales(Lorenz96Base):
  """Two timescale Lorenz-96 model."""

  k_axis: cx.Coordinate
  j_axis: cx.Coordinate
  forcing: float | nnx.Param = 10.0
  c: float | nnx.Param = 10.0
  b: float | nnx.Param = 10.0
  h: float | nnx.Param = 1.0
  # Prognostic fields initialized in __post_init__
  x: typing.Prognostic = dataclasses.field(init=False)
  y: typing.Prognostic = dataclasses.field(init=False)
  time: typing.Prognostic = dataclasses.field(init=False)

  def __post_init__(self):
    ks, js = self.k_axis, self.j_axis
    kjs = cx.coords.compose(ks, js)
    self.x = typing.Prognostic(cx.field(jnp.zeros(ks.shape), ks))
    self.y = typing.Prognostic(cx.field(jnp.zeros(kjs.shape), kjs))
    self.time = typing.Prognostic(cx.field(jdt.Datetime(jdt.Timedelta())))
    super().__post_init__()

  @module_utils.ensure_unchanged_state_structure
  def assimilate(self, inputs: dict[str, dict[str, cx.Field]]) -> None:
    slice_last_time = lambda f: cx.cmap(lambda a: a[-1])(f.untag('timedelta'))
    self.x.set_value(slice_last_time(inputs['slow']['x']))
    self.y.set_value(slice_last_time(inputs['fast']['y']))
    self.time.set_value(slice_last_time(inputs['slow']['time']))

  @module_utils.ensure_unchanged_state_structure
  def advance(self) -> None:
    k, j = self.k_axis, self.j_axis
    f, c, b, h = self.forcing, self.c, self.b, self.h
    x_dot = cx.cpmap(functools.partial(lorenz96_x_term, f=f))
    y_dot = cx.cpmap(functools.partial(lorenz96_y_term, b=b, c=c))
    sum_fn = cx.cmap(jnp.sum)
    # Equation for coupled X and Y variables.
    # fmt: off
    all_terms_fn = lambda s: {
        'x': x_dot(s['x'].untag(k)).tag(k) - (h*c/b) * sum_fn(s['y'].untag(j)),
        'y': y_dot(s['y'].untag(j)).tag(j) + (h*c/b) * s['x'],
    }
    # fmt: on
    full_ode = time_integrators.ExplicitODE.from_functions(all_terms_fn)
    x, y = self.x.get_value(), self.y.get_value()
    next_xy = self.time_integrator_cls(full_ode, self.dt)({'x': x, 'y': y})
    self.x.set_value(next_xy['x'])
    self.y.set_value(next_xy['y'])
    # treated separately to avoid round off.
    self.time.set_value(self.time.get_value() + self.timestep)


class L96Parameterization(Protocol):
  """Protocol class defining the interface for Lorenz96 parameterizations."""

  def __call__(
      self,
      prognostics: dict[str, cx.Field],
      dt: float,
  ) -> cx.Field:
    """Returns cumulative correction over `dt` for a `x` or `y` variable."""


@nnx_compat.dataclass
class StochasticLinearParameterization(nnx.Module):
  """A simple stochastic linear parameterization.

  This implements update of the form:
    update = -(b0 + b1 * X) * dt + noise_amplitude * N(0, 1) * sqrt(dt)

  # Default values for `b0, b1` are taken from M2Lines book:
  # https://m2lines.github.io/L96_demo/notebooks/gcm-analogue.html
  """

  random_process: random_processes.RandomProcessModule
  b0: float | nnx.Param = 0.75218026
  b1: float | nnx.Param = 0.85439536
  noise_amplitude: float | nnx.Param = 1.0

  def __call__(
      self,
      prognostics: dict[str, cx.Field],
      dt: float,
  ) -> cx.Field:
    x = prognostics['x']
    k = x.axes['k']
    poly_fn = cx.cpmap(lambda x_k: -(self.b0 + self.b1 * x_k))
    deterministic_term = poly_fn(x.untag(k)).tag(k)
    randomness = self.random_process.state_values(k)
    stochastic_term = self.noise_amplitude * randomness
    self.random_process.advance()
    return deterministic_term * dt + stochastic_term * jnp.sqrt(dt)


@nnx_compat.dataclass
class Lorenz96(Lorenz96Base):
  """Single timescale Lorenz-96 model with a modular parameterizations."""

  k_axis: cx.Coordinate
  parameterizations: Sequence[L96Parameterization] = ()
  forcing: float | nnx.Param = 10.0
  # Prognostic fields initialized in __post_init__
  x: typing.Prognostic = dataclasses.field(init=False)
  time: typing.Prognostic = dataclasses.field(init=False)

  def __post_init__(self):
    self.x = typing.Prognostic(
        cx.field(jnp.zeros(self.k_axis.shape), self.k_axis)
    )
    self.time = typing.Prognostic(cx.field(jdt.Datetime(jdt.Timedelta())))
    super().__post_init__()

  @module_utils.ensure_unchanged_state_structure
  def assimilate(self, inputs: dict[str, dict[str, cx.Field]]) -> None:
    slice_last_time = lambda f: cx.cmap(lambda a: a[-1])(f.untag('timedelta'))
    self.x.set_value(slice_last_time(inputs['slow']['x']))
    self.time.set_value(slice_last_time(inputs['slow']['time']))

  @module_utils.ensure_unchanged_state_structure
  def advance(self) -> None:
    k, f = self.k_axis, self.forcing
    x, time = self.x.get_value(), self.time.get_value()
    x_dot_fn = cx.cpmap(functools.partial(lorenz96_x_term, f=f))
    x_ode = time_integrators.ExplicitODE.from_functions(
        lambda s: x_dot_fn(s.untag(k)).tag(k)
    )
    next_x = self.time_integrator_cls(x_ode, self.dt)(x)
    for parameterization in self.parameterizations:
      next_x += parameterization({'x': x, 'time': time}, self.dt)
    self.x.set_value(next_x)
    self.time.set_value(self.time.get_value() + self.timestep)


@nnx_compat.dataclass
class Lorenz96FastMode(Lorenz96Base):
  """Fast timescale Lorenz-96 model."""

  k_axis: cx.Coordinate
  j_axis: cx.Coordinate
  parameterizations: Sequence[L96Parameterization] = ()
  c: float | nnx.Param = 10.0
  b: float | nnx.Param = 10.0
  # Prognostic fields initialized in __post_init__
  y: typing.Prognostic = dataclasses.field(init=False)
  time: typing.Prognostic = dataclasses.field(init=False)

  def __post_init__(self):
    kjs = cx.coords.compose(self.k_axis, self.j_axis)
    self.y = typing.Prognostic(cx.field(jnp.zeros(kjs.shape), kjs))
    self.time = typing.Prognostic(cx.field(jdt.Datetime(jdt.Timedelta())))
    super().__post_init__()

  @module_utils.ensure_unchanged_state_structure
  def assimilate(self, inputs: dict[str, dict[str, cx.Field]]) -> None:
    slice_last_time = lambda f: cx.cmap(lambda a: a[-1])(f.untag('timedelta'))
    self.y.set_value(slice_last_time(inputs['fast']['y']))
    self.time.set_value(slice_last_time(inputs['fast']['time']))

  @module_utils.ensure_unchanged_state_structure
  def advance(self) -> None:
    c, b = self.c, self.b
    y, time = self.y.get_value(), self.time.get_value()
    y_dot_fn = cx.cpmap(functools.partial(lorenz96_y_term, b=b, c=c))
    kjs = cx.coords.compose(self.k_axis, self.j_axis)
    y_ode = time_integrators.ExplicitODE.from_functions(
        lambda s: y_dot_fn(s.untag(kjs)).tag(kjs)
    )
    next_y = self.time_integrator_cls(y_ode, self.dt)(y)
    for parameterization in self.parameterizations:
      next_y += parameterization({'y': y, 'time': time}, self.dt)
    self.y.set_value(next_y)
    self.time.set_value(self.time.get_value() + self.timestep)
