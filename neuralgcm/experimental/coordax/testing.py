"""Testing utilities for coordax."""

from typing import Mapping

import chex
import jax
from neuralgcm.experimental import coordax
import numpy as np


def assert_field_properties(
    actual: coordax.Field,
    data: np.ndarray | jax.Array | coordax.NDArray | None = None,
    dims: tuple[str, ...] | None = None,
    shape: tuple[int, ...] | None = None,
    axes: Mapping[str, coordax.Coordinate] | None = None,
    coord_field_keys: set[str] | None = None,
    named_shape: Mapping[str, int] | None = None,
    positional_shape: tuple[int, ...] | None = None,
    rtol: float | None = 1e-5,
    atol: float | None = 1e-5,
):
  """Asserts that a Field has expected properties."""
  if data is not None:
    if atol is None and rtol is None:
      chex.assert_trees_all_equal(actual.data, data)
    else:
      chex.assert_trees_all_close(actual.data, data, rtol=rtol, atol=atol)
  if dims is not None:
    chex.assert_equal(actual.dims, dims)
  if shape is not None:
    chex.assert_shape(actual, shape)
  if axes is not None:
    chex.assert_trees_all_equal(actual.axes, axes)
  if coord_field_keys is not None:
    chex.assert_trees_all_equal(
        set(actual.coord_fields.keys()), coord_field_keys
    )
  if named_shape is not None:
    chex.assert_equal(actual.named_shape, named_shape)
  if positional_shape is not None:
    chex.assert_equal(actual.positional_shape, positional_shape)


def assert_fields_allclose(
    actual: coordax.Field,
    desired: coordax.Field,
    rtol: float = 1e-5,
    atol: float = 1e-5,
):
  """Asserts that two Fields are close and have matching coordinates."""
  assert_field_properties(
      actual=actual,
      data=desired.data,
      dims=desired.dims,
      shape=desired.shape,
      axes=desired.axes,
      named_shape=desired.named_shape,
      positional_shape=desired.positional_shape,
      rtol=rtol,
      atol=atol,
  )


def assert_fields_equal(actual: coordax.Field, desired: coordax.Field):
  """Asserts that two Fields are equal and have matching coordinates."""
  assert_field_properties(
      actual=actual,
      data=desired.data,
      dims=desired.dims,
      shape=desired.shape,
      axes=desired.axes,
      named_shape=desired.named_shape,
      positional_shape=desired.positional_shape,
      rtol=None,
      atol=None,
  )
