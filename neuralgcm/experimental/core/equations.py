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

"""Modules that define differential equations."""

from typing import Sequence

import coordax as cx
import jax
from neuralgcm.experimental.core import nnx_compat
from neuralgcm.experimental.core import pytree_utils
from neuralgcm.experimental.core import time_integrators
from neuralgcm.experimental.core import transforms
from neuralgcm.experimental.core import typing


ImplicitExplicitODE = time_integrators.ImplicitExplicitODE
ExplicitODE = time_integrators.ExplicitODE
ShapeFloatStruct = typing.ShapeFloatStruct


class SimTimeEquation(time_integrators.ExplicitODE):
  """Equation module describing evolution of `sim_time` variable."""

  def explicit_terms(self, x: typing.Pytree) -> typing.Pytree:
    x_dict, from_dict_fn = pytree_utils.as_dict(x)
    if 'sim_time' not in x_dict:
      raise ValueError(f'sim_time not found in {x_dict.keys()}')
    terms = {k: None if k != 'sim_time' else 1.0 for k in x_dict.keys()}
    return from_dict_fn(terms)


@nnx_compat.dataclass
class ExplicitTransformEquation(time_integrators.ExplicitODE):
  """Explicit equation whose terms are parameterized by a transform."""

  explicit_terms_transform: transforms.Transform

  def explicit_terms(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    tendencies = self.explicit_terms_transform(inputs)
    tendencies = pytree_utils.replace_with_matching_or_default(
        inputs, tendencies, None
    )
    return tendencies


def _sum_non_nones(*args: jax.Array | None) -> jax.Array | None:
  terms = [x for x in args if x is not None]
  return sum(terms) if terms else None


@nnx_compat.dataclass
class ComposedODE(ImplicitExplicitODE):
  """Composed equation with exactly one ImplicitExplicitODE instance."""

  equations: Sequence[ExplicitODE | ImplicitExplicitODE]

  def __post_init__(self):
    imex_equations = [
        x for x in self.equations if isinstance(x, ImplicitExplicitODE)
    ]
    if len(imex_equations) != 1:
      raise ValueError(
          'ComposedODE only supports exactly 1 ImplicitExplicitODE, '
          f'got {imex_equations=}'
      )
    (implicit_explicit_eq,) = imex_equations
    self.implicit_explicit_equation = implicit_explicit_eq

  def explicit_terms(self, x: typing.Pytree) -> typing.Pytree:
    explicit_tendencies = [fn.explicit_terms(x) for fn in self.equations]
    return jax.tree.map(
        _sum_non_nones, *explicit_tendencies, is_leaf=lambda x: x is None
    )

  def implicit_terms(self, x: typing.Pytree) -> typing.Pytree:
    return self.implicit_explicit_equation.implicit_terms(x)

  def implicit_inverse(
      self, x: typing.Pytree, step_size: float
  ) -> typing.Pytree:
    return self.implicit_explicit_equation.implicit_inverse(x, step_size)


@nnx_compat.dataclass
class ComposedExplicitODE(ExplicitODE):
  """Composed explicit equation."""

  equations: Sequence[ExplicitODE]

  def explicit_terms(self, x: typing.Pytree) -> typing.Pytree:
    explicit_tendencies = [fn.explicit_terms(x) for fn in self.equations]
    return jax.tree.map(
        lambda *args: sum([x for x in args if x is not None]),
        *explicit_tendencies,
        is_leaf=lambda x: x is None,
    )
