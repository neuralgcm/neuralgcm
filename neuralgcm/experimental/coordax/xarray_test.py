import textwrap
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
from typing import Callable, Self

from absl.testing import absltest
import jax
from neuralgcm.experimental import coordax
from neuralgcm.experimental.coordax import testing
from neuralgcm.experimental.coordax import xarray as coordax_xarray
import numpy as np
import xarray


# TODO(shoyer): consider moving this into coordax's public API
@jax.tree_util.register_static
class AdhocCoordinate(coordax.Coordinate):
  """Adhoc coordinate class for testing purposes.

  For simplicity, equality and hashing of adhoc coordinates is by identity.
  """

  def __init__(
      self,
      dims: tuple[str, ...],
      shape: tuple[int, ...],
      fields: Callable[[Self], dict[str, coordax.Field]] | None = None,
  ):
    self._dims = dims
    self._shape = shape
    self._fields = fields

  @property
  def dims(self) -> tuple[str, ...]:
    return self._dims

  @property
  def shape(self) -> tuple[int, ...]:
    return self._shape

  @property
  def fields(self) -> dict[str, coordax.Field]:
    return {} if self._fields is None else self._fields(self)


class XarrayTest(absltest.TestCase):

  def test_field_to_data_array_with_named_axis(self):
    data = np.arange(2 * 3).reshape((2, 3))
    field = coordax.wrap(data, 'x', 'y')
    actual = coordax_xarray.field_to_data_array(field)
    expected = xarray.DataArray(data, dims=['x', 'y'])
    xarray.testing.assert_identical(actual, expected)

  def test_field_to_data_array_with_labeled_axis(self):
    data = np.arange(3)
    ticks = np.array([1, 2, 3])
    axis = coordax.LabeledAxis('x', ticks)
    field = coordax.wrap(data, axis)
    actual = coordax_xarray.field_to_data_array(field)
    expected = xarray.DataArray(data, dims=['x'], coords={'x': ticks})
    xarray.testing.assert_identical(actual, expected)

  def test_field_to_data_array_custom_coord(self):
    data_2d = np.arange(2 * 3).reshape((2, 3))
    custom_coord = AdhocCoordinate(
        dims=('x', 'y'),
        shape=(2, 3),
        fields=lambda c: {'custom': coordax.wrap(data_2d, c)},
    )
    field = coordax.wrap(10 * data_2d, custom_coord)
    actual = coordax_xarray.field_to_data_array(field)
    expected = xarray.DataArray(
        data=10 * data_2d,
        dims=['x', 'y'],
        coords={'custom': (('x', 'y'), data_2d)},
    )
    xarray.testing.assert_identical(actual, expected)

  def test_field_to_data_array_missing_dimension_names(self):
    data = np.arange(2 * 3).reshape((2, 3))
    field = coordax.wrap(data)
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'can only convert Field objects with fully named dimensions to Xarray'
        ' objects, got dimensions (None, None)',
    ):
      coordax_xarray.field_to_data_array(field)

  def test_field_to_data_array_inconsistent_coordinates(self):

    x = AdhocCoordinate(
        dims=('x',),
        shape=(3,),
        fields=lambda c: {'z': coordax.wrap(0)},
    )
    y = AdhocCoordinate(
        dims=('y',),
        shape=(2,),
        fields=lambda c: {'z': coordax.wrap(1)},
    )
    field = coordax.wrap(np.zeros((3, 2)), x, y)
    with self.assertRaisesRegex(
        ValueError, "inconsistent coordinate fields for 'z'"
    ):
      coordax_xarray.field_to_data_array(field)

  def test_data_array_to_field_default_types(self):
    data = np.arange(2 * 3).reshape((2, 3))
    data_array = xarray.DataArray(
        data=data, dims=['x', 'y'], coords={'y': [1, 2, 3]}
    )
    actual = coordax_xarray.data_array_to_field(data_array)
    expected = coordax.wrap(data, 'x', coordax.LabeledAxis('y', [1, 2, 3]))
    testing.assert_fields_equal(actual, expected)

  def test_data_array_to_field_no_match(self):
    data = np.arange(2 * 3).reshape((2, 3))
    data_array = xarray.DataArray(
        data=data, dims=['x', 'y'], coords={'y': [1, 2, 3]}
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        textwrap.dedent(
            """\
            failed to convert xarray.DataArray to coordax.Field, because no coordinate type matched the dimensions starting with ('x', 'y'):
            <xarray.DataArray (x: 2, y: 3)> Size: 48B
            array([[0, 1, 2],
                   [3, 4, 5]])
            Coordinates:
              * y        (y) int64 24B 1 2 3
            Dimensions without coordinates: x

            Reasons why coordinate matching failed:
            neuralgcm.experimental.coordax.core.Coordinate: from_xarray not implemented"""
        ),
    ):
      coordax_xarray.data_array_to_field(
          data_array, coord_types=(coordax.Coordinate,)
      )

  def test_data_array_to_field_custom_matchers(self):
    data = np.arange(2 * 3).reshape((2, 3))
    data_array = xarray.DataArray(data=data, dims=['x', 'y'])
    custom_coord = AdhocCoordinate(dims=('x', 'y'), shape=(2, 3))
    custom_coord.from_xarray = lambda dims, _: custom_coord
    actual = coordax_xarray.data_array_to_field(data_array, [custom_coord])
    expected = coordax.wrap(data, custom_coord)
    testing.assert_fields_equal(actual, expected)

  def test_default_matchers_roundtrip(self):
    data = np.arange(2 * 3).reshape((2, 3))
    data_array = xarray.DataArray(
        data=data, dims=['x', 'y'], coords={'y': [1, 2, 3]}
    )
    field = coordax_xarray.data_array_to_field(data_array)
    actual = coordax_xarray.field_to_data_array(field)
    xarray.testing.assert_identical(actual, data_array)


if __name__ == '__main__':
  absltest.main()
