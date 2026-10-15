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

"""Stepping and diagnostics update utilities for NeuralGCM models."""

from collections.abc import Callable, Mapping

import coordax as cx
from flax import nnx
import jax_datetime as jdt
import numpy as np
from terrax.core import api
from terrax.core import diagnostics
from terrax.core import typing


def repeat_step(step_fn: Callable[..., typing.Pytree], length: int):
  """Returns `step_fn(module, carry)` applied `length` times via `nnx.scan`."""
  if length == 1:
    return step_fn
  return nnx.scan(
      step_fn,
      length=length,
      in_axes=(api.DEFAULT_MODEL_STATE_AXES, nnx.Carry),
      out_axes=nnx.Carry,
  )


def update_diagnostics(
    collection: Mapping[str, diagnostics.DiagnosticModule],
    inputs: dict[str, cx.Field],
    prognostics: dict[str, cx.Field],
) -> None:
  """Updates diagnostics in `collection` in insertion order."""
  for diagnostic in collection.values():
    diagnostic(inputs, prognostics=prognostics)


def advance_diagnostic_clocks(
    timedelta: np.timedelta64,
    *collections: Mapping[str, diagnostics.DiagnosticModule],
) -> None:
  """Advances clocks of temporal diagnostics in `collections` by `timedelta`."""
  inputs = {'timedelta': jdt.to_timedelta(timedelta)}
  for collection in collections:
    for diagnostic in collection.values():
      if isinstance(diagnostic, diagnostics.TemporalDiagnosticModule):
        diagnostic.advance_clock(inputs)
