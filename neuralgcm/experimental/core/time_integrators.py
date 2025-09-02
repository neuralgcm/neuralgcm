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

"""Defines equation and time integration modules."""

from typing import Callable

from dinosaur import time_integration
from flax import nnx
from neuralgcm.experimental.core import nnx_compat
from neuralgcm.experimental.core import typing
import tree_math


# nnx_compat is not recognized as a dataclass decorator raising lint errors.
# pylint: disable=unexpected-keyword-arg


class ExplicitODE(time_integration.ExplicitODE, nnx.Module):
  """Module wrapper for ExplicitODE.

  This module is wrapped as nnx.Module to ensure that any submodule that is
  a part of the equation class is included in the model's parameter tree.
  """

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(pytree=False, **kwargs)


class ImplicitExplicitODE(time_integration.ImplicitExplicitODE, nnx.Module):
  """Module wrapper for ImplicitExplicitODE.

  This module is wrapped as nnx.Module to ensure that any submodule that is
  a part of the equation class is included in the model's parameter tree.
  """

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(pytree=False, **kwargs)


def forward_euler(equation: ExplicitODE, time_step: float) -> typing.StepFn:
  """Time stepping for an explicit ODE via forward Euler method.

  This method is first order accurate.

  Args:
    equation: equation to solve.
    time_step: time step.

  Returns:
    Function that performs a time step.
  """
  # pylint: disable=invalid-name
  dt = time_step
  F = tree_math.unwrap(equation.explicit_terms)

  @tree_math.wrap
  def step_fn(u0):
    return u0 + dt * F(u0)

  return step_fn


@nnx_compat.dataclass
class DinosaurIntegrator(nnx.Module, pytree=False):
  """Module that wraps time integrators from dinosaur package."""

  equation: ExplicitODE | ImplicitExplicitODE
  time_step: float
  integrator: Callable[
      [ExplicitODE | ImplicitExplicitODE, float], typing.StepFn
  ]

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    return self.integrator(self.equation, self.time_step)(inputs)

  def __init_subclasses__(self, **kwargs):
    super().__init__(pytree=False, **kwargs)


# Note: we don't use functools.partial here because it would cause issues with
# fiddle serialization.
class ImexRk3Sil(DinosaurIntegrator):

  def __init__(
      self,
      equation: ExplicitODE | ImplicitExplicitODE,
      time_step: float,
  ):
    super().__init__(
        equation=equation,
        time_step=time_step,
        integrator=time_integration.imex_rk_sil3,
    )


class ExplicitEuler(DinosaurIntegrator):

  def __init__(
      self,
      equation: ExplicitODE,
      time_step: float,
  ):
    super().__init__(
        equation=equation,
        time_step=time_step,
        integrator=forward_euler,
    )
