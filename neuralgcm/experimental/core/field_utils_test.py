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

import re
from absl.testing import absltest
from absl.testing import parameterized
import chex
import coordax as cx
import jax
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import field_utils
import numpy as np


class SplitFieldAxisTest(parameterized.TestCase):
  """Tests that split_field_axis works as expected."""

  def test_split_field_axis(self):
    axis = cx.SizedAxis('x', 3)
    field = cx.field(np.arange(3), axis)

    split_axes = {
        'a': cx.Scalar(),
        'b': cx.SizedAxis('sub_x', 2),
    }
    expected = {
        'a': cx.field(np.array(0), cx.Scalar()),
        'b': cx.field(np.array([1, 2]), cx.SizedAxis('sub_x', 2)),
    }

    with self.subTest('axis_as_coordinate'):
      actual = field_utils.split_field_axis(field, axis, split_axes)
      chex.assert_trees_all_close(actual, expected)

    with self.subTest('axis_as_string'):
      actual = field_utils.split_field_axis(field, 'x', split_axes)
      chex.assert_trees_all_close(actual, expected)

  def test_split_field_axis_preserves_coords_order(self):
    b = cx.SizedAxis('b', 4)
    z = cx.SizedAxis('z', 2)
    grid = coordinates.LonLatGrid.T21()
    field = cx.field(np.zeros((4, 2,) + grid.shape), b, z, grid)

    split_axes = {
        'z1': cx.SizedAxis('z1', 1),
        'z2': cx.SizedAxis('z2', 1),
    }
    actual = field_utils.split_field_axis(field, z, split_axes)

    expected_z1 = cx.coords.compose(b, cx.SizedAxis('z1', 1), grid)
    expected_z2 = cx.coords.compose(b, cx.SizedAxis('z2', 1), grid)
    self.assertEqual(actual['z1'].coordinate, expected_z1)
    self.assertEqual(actual['z2'].coordinate, expected_z2)

  def test_split_field_axis_to_multidim_coord(self):
    flat_size = 10
    flat_axis = cx.SizedAxis('flat', flat_size)
    field = cx.field(np.zeros((flat_size,)), flat_axis)

    x_patch = cx.SizedAxis('x', 3)
    y_patch = cx.SizedAxis('y', 3)
    xy_patch = cx.coords.compose(x_patch, y_patch)

    split_axes = {'patch': xy_patch, 'scalar': cx.Scalar()}
    actual = field_utils.split_field_axis(field, flat_axis, split_axes)

    self.assertEqual(actual['patch'].coordinate, xy_patch)
    self.assertEqual(actual['scalar'].shape, ())


class SplitToFieldsTest(parameterized.TestCase):
  """Tests that split_to_fields works as expected."""

  def test_split_to_fields_flat_out(self):
    b, x = cx.SizedAxis('batch', 3), cx.LabeledAxis('x', np.array([0, 1]))
    field = cx.field(np.arange(6).reshape(3, 2), b, x)
    targets = {'a': x, 'b': x, 'c': x}
    expected = {
        'a': cx.field(np.array([0, 1]), x),
        'b': cx.field(np.array([2, 3]), x),
        'c': cx.field(np.array([4, 5]), x),
    }
    actual = field_utils.split_to_fields(field, targets)
    chex.assert_trees_all_close(actual, expected)

  def test_split_to_fields_mixed_out(self):
    b, x = cx.SizedAxis('batch', 6), cx.LabeledAxis('x', np.array([0, 1]))
    s, d = cx.SizedAxis('s', 2), cx.SizedAxis('d', 3)
    field = cx.field(np.arange(12).reshape(6, 2), b, x)
    targets = {
        'a': x,  # takes size 1.
        'b': cx.coords.compose(s, x),  # takes size 2.
        'c': cx.coords.compose(d, x),  # takes size 3.
    }
    expected = {
        'a': cx.field(np.array([0, 1]), x),
        'b': cx.field(np.array([[2, 3], [4, 5]]), s, x),
        'c': cx.field(np.array([[6, 7], [8, 9], [10, 11]]), d, x),
    }
    actual = field_utils.split_to_fields(field, targets)
    chex.assert_trees_all_close(actual, expected)

  def test_split_to_fields_multi_dims(self):
    xy = cx.coords.compose(cx.SizedAxis('x', 6), cx.SizedAxis('y', 7))
    field = cx.field(np.ones((5,) + xy.shape), None, xy)
    s = cx.SizedAxis('s', 2)
    sxy = cx.coords.compose(s, xy)
    targets = {'a': xy, 'b': sxy, 'c': sxy}
    expected = {
        'a': cx.field(np.ones(xy.shape), xy),
        'b': cx.field(np.ones(sxy.shape), sxy),
        'c': cx.field(np.ones(sxy.shape), sxy),
    }
    actual = field_utils.split_to_fields(field, targets)
    chex.assert_trees_all_close(actual, expected)

  def test_split_to_fields_aligns_outputs(self):
    x, y = cx.SizedAxis('x', 6), cx.SizedAxis('y', 7)
    field = cx.field(np.ones((3,) + x.shape + y.shape), None, x, y)
    yx = cx.coords.compose(x, y)
    targets = {'a': yx, 'b': yx, 'c': yx}  # requests transposed xy;
    expected = {
        'a': cx.field(np.ones(yx.shape), yx),
        'b': cx.field(np.ones(yx.shape), yx),
        'c': cx.field(np.ones(yx.shape), yx),
    }
    actual = field_utils.split_to_fields(field, targets)
    chex.assert_trees_all_close(actual, expected)

  def test_split_to_fields_raises_on_misaligned_coords(self):
    """Tests that split_to_fields raises on misaligned coordinates."""
    x, y = cx.LabeledAxis('x', np.arange(3)), cx.LabeledAxis('y', np.arange(2))
    xy = cx.coords.compose(x, y)
    field = cx.field(np.ones((2,) + xy.shape), None, xy)
    good_targets = {'a': xy, 'b': xy}  # should not raise
    _ = field_utils.split_to_fields(field, good_targets)
    bad_xy = cx.coords.compose(cx.SizedAxis('x', 3), cx.SizedAxis('y', 2))
    bad_targets = {'a': bad_xy, 'b': bad_xy}
    with self.assertRaisesRegex(
        ValueError,
        re.escape(
            r'does not specify a valid split element because it is not aligned '
            'with the non-split part of input field'
        ),
    ):
      field_utils.split_to_fields(field, bad_targets)

  def test_split_to_fields_raises_on_wrong_split_size(self):
    """Tests that split_to_fields raises on wrong split size."""
    b, x = cx.SizedAxis('batch', 3), cx.LabeledAxis('x', np.array([0, 1]))
    field = cx.field(np.arange(6).reshape(3, 2), b, x)
    targets = {'a': x, 'b': x}  # requests 2*2 != 3*2.
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'The total size of the dimensions defined in `targets` (2)'
        ' does not match the size of the dimension being split in the input'
        ' field (3).',
    ):
      field_utils.split_to_fields(field, targets)

  def test_split_to_fields_raises_if_too_many_new_dims(self):
    """Tests that split_to_fields raises if more than 1 new dim is detected."""
    b, x = cx.SizedAxis('batch', 2), cx.LabeledAxis('x', np.arange(7))
    field = cx.field(np.zeros((2, 7)), b, x)
    d = cx.SizedAxis('d', 1)
    targets = {
        'a': cx.coords.compose(d, x),
        'b': cx.coords.compose(d, cx.SizedAxis('second_new', 1), x),
    }
    with self.assertRaisesRegex(
        ValueError,
        re.escape(r'has more than 1 new axis compared to input field'),
    ):
      field_utils.split_to_fields(field, targets)


class CombineFieldsTest(parameterized.TestCase):
  """Tests that combine_fields works as expected."""

  def test_combine_fields_supports_mixed_concat_axes(self):
    x, y = cx.SizedAxis('x', 3), cx.SizedAxis('y', 5)
    z, level = cx.SizedAxis('z', 2), cx.SizedAxis('level', 7)
    fields = {  # concatenates z and level when aligned on (x, y).
        'a': cx.field(np.ones((2, 3, 5)), z, x, y),
        'b': cx.field(np.ones((7, 3, 5)), level, x, y),
        'c': cx.field(np.ones((7, 3, 5)), level, x, y),
    }
    actual = field_utils.combine_fields(fields, dims_to_align=(x, y))
    expected = cx.field(np.ones((7 + 7 + 2, 3, 5)), None, x, y)
    chex.assert_trees_all_close(actual, expected)

  def test_combine_fields_supports_missing_concat_axis(self):
    x, y = cx.SizedAxis('x', 3), cx.SizedAxis('y', 5)
    level = cx.SizedAxis('level', 7)
    fields = {
        'a_surf': cx.field(np.ones((3, 5)), x, y),  # expands as (1, 3, 5).
        'b': cx.field(np.ones((7, 3, 5)), level, x, y),
        'c': cx.field(np.ones((7, 3, 5)), level, x, y),
    }
    actual = field_utils.combine_fields(fields, dims_to_align=(x, y))
    expected = cx.field(np.ones((1 + 7 + 7, 3, 5)), None, x, y)
    chex.assert_trees_all_close(actual, expected)

  def test_combine_fields_works_as_stack(self):
    x, y = cx.SizedAxis('x', 3), cx.SizedAxis('y', 5)
    level = cx.SizedAxis('level', 7)
    fields = {  # all expand to (1, 7, 3, 5) since aligned on level, x, y.
        'a': cx.field(np.ones((7, 3, 5)), level, x, y),
        'b': cx.field(np.ones((7, 3, 5)), level, x, y),
        'c': cx.field(np.ones((7, 3, 5)), level, x, y),
    }
    actual = field_utils.combine_fields(fields, dims_to_align=(level, x, y))
    expected = cx.field(np.ones((3, 7, 3, 5)), None, level, x, y)
    chex.assert_trees_all_close(actual, expected)

  def test_combine_fields_out_axis_tag(self):
    x = cx.SizedAxis('x', 5)
    fields = {'a': cx.field(np.ones(5), x), 'b': cx.field(np.ones(5), x)}

    with self.subTest('coordinate_out_tag'):
      out_tag = cx.SizedAxis('out', 2)  # 2
      actual = field_utils.combine_fields(fields, (x,), out_tag)
      expected = cx.field(np.ones((2, 5)), out_tag, x)
      chex.assert_trees_all_close(actual, expected)

    with self.subTest('name_out_tag'):
      actual = field_utils.combine_fields(fields, (x,), 'out')
      expected = cx.field(np.ones((2, 5)), 'out', x)
      chex.assert_trees_all_close(actual, expected)

  def test_combine_fields_supports_dims_and_coords(self):
    x, y = cx.SizedAxis('x', 3), cx.SizedAxis('y', 5)
    z, level = cx.SizedAxis('z', 2), cx.SizedAxis('level', 7)
    fields = {
        'a': cx.field(np.ones((2, 7, 3, 5)), z, level, x, y),
        'b': cx.field(np.ones((7, 3, 5)), level, x, y),
        'c': cx.field(np.ones((7, 3, 5)), level, x, y),
    }
    xy = cx.coords.compose(x, y)  # can pass coords as dims_to_align.
    actual = field_utils.combine_fields(fields, dims_to_align=('level', xy))
    expected = cx.field(np.ones((4, 7, 3, 5)), None, level, x, y)
    chex.assert_trees_all_close(actual, expected)

  def test_combine_fields_raises_on_repeated_dims_to_align(self):
    x = cx.SizedAxis('x', 3)
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "`dims_to_align` must be unique, but got repeated_dims=['x'].",
    ):
      field_utils.combine_fields({}, dims_to_align=('x', x))

  def test_combine_fields_raises_on_too_many_new_axes(self):
    x, y = cx.SizedAxis('x', 3), cx.SizedAxis('y', 5)
    z, level = cx.SizedAxis('z', 2), cx.SizedAxis('level', 7)
    fields = {
        'two_new': cx.field(np.ones((2, 7, 3, 5)), z, level, x, y),
        'no_new': cx.field(np.ones((3, 5)), x, y),
    }
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        f"Field {fields['two_new']} has more than 1 axis other than"
        " aligned_dims_and_axes=('x', 'y').",
    ):
      field_utils.combine_fields(fields, dims_to_align=('x', 'y'))

  def test_combine_fields_raises_on_missing_alignment_dim(self):
    x, y = cx.SizedAxis('x', 3), cx.SizedAxis('y', 5)
    level = cx.SizedAxis('level', 7)
    fields = {
        'missing_x': cx.field(np.ones((7, 5)), level, y),
        'valid': cx.field(np.ones((3, 5)), x, y),
    }
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        f"Cannot combine {fields['missing_x']} because it does not align with"
        " ('x', 'y')",
    ):
      field_utils.combine_fields(fields, dims_to_align=('x', 'y'))

  def test_combine_fields_no_unique_axis_order(self):
    x, y = cx.SizedAxis('x', 3), cx.SizedAxis('y', 5)
    fields = {
        'y_then_x': cx.field(np.ones((7, 5, 3)), 'l', y, x),
        'x_then_y': cx.field(np.ones((3, 5)), x, y),
    }
    with self.assertRaisesRegex(
        ValueError,
        re.escape('No unique out_axes found in inputs'),
    ):
      field_utils.combine_fields(fields, dims_to_align=('x', 'y'))


class UtilsTest(parameterized.TestCase):

  def test_shape_struct_fields_from_coords(self):
    coords = {
        'a': cx.LabeledAxis('a', np.array([1, 2, 3])),
        'b': cx.LabeledAxis('b', np.array([4, 5, 6])),
    }
    actual = field_utils.shape_struct_fields_from_coords(coords)
    self.assertEqual(cx.get_coordinate(actual['a']), coords['a'])
    self.assertEqual(cx.get_coordinate(actual['b']), coords['b'])
    self.assertIsInstance(actual['a'].data, jax.ShapeDtypeStruct)


class ZeroMaskAxisOutliersTest(parameterized.TestCase):
  """Tests zero_mask_axis_outliers utility function."""

  def test_mask_lower_bound_only(self):
    axis = cx.LabeledAxis('x', np.array([-1, 0, 1, 2]))
    field = cx.field(np.ones(4), axis)
    actual = field_utils.zero_mask_axis_outliers(field, axis, lower=0.0)
    expected = cx.field(np.array([0, 1, 1, 1]), axis)
    chex.assert_trees_all_close(actual, expected)

  def test_mask_upper_bound_only(self):
    axis = cx.LabeledAxis('x', np.array([-1, 0, 1, 2]))
    field = cx.field(np.ones(4), axis)
    actual = field_utils.zero_mask_axis_outliers(field, axis, upper=1.0)
    expected = cx.field(np.array([1, 1, 1, 0]), axis)
    chex.assert_trees_all_close(actual, expected)

  def test_mask_lower_and_upper_bounds(self):
    axis = cx.LabeledAxis('x', np.array([-1, 0, 1, 2]))
    field = cx.field(np.ones(4), axis)
    actual = field_utils.zero_mask_axis_outliers(
        field, axis, lower=0.0, upper=1.0
    )
    expected = cx.field(np.array([0, 1, 1, 0]), axis)
    chex.assert_trees_all_close(actual, expected)

  def test_mask_multidim_field(self):
    x = cx.LabeledAxis('x', np.array([-1, 0, 1, 2]))
    y = cx.SizedAxis('y', 2)
    field = cx.field(np.ones((4, 2)), x, y)
    actual = field_utils.zero_mask_axis_outliers(field, x, lower=0.0, upper=1.0)
    expected_data = np.array([[0, 0], [1, 1], [1, 1], [0, 0]])
    expected = cx.field(expected_data, x, y)
    chex.assert_trees_all_close(actual, expected)

  def test_no_values_masked(self):
    axis = cx.LabeledAxis('x', np.array([-1, 0, 1, 2]))
    field = cx.field(np.ones(4), axis)
    actual = field_utils.zero_mask_axis_outliers(
        field, axis, lower=-1.0, upper=2.0
    )
    expected = cx.field(np.array([1, 1, 1, 1]), axis)
    chex.assert_trees_all_close(actual, expected)

  def test_all_values_masked_if_lower_gt_upper(self):
    axis = cx.LabeledAxis('x', np.array([-1, 0, 1, 2]))
    field = cx.field(np.ones(4), axis)
    actual = field_utils.zero_mask_axis_outliers(
        field, axis, lower=1.0, upper=0.0
    )
    expected = cx.field(np.zeros(4), axis)
    chex.assert_trees_all_close(actual, expected)

  def test_raises_on_no_lower_or_upper_bound(self):
    axis = cx.LabeledAxis('x', np.array([-1, 0, 1, 2]))
    field = cx.field(np.ones(4), axis)
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'Must specify at least one of `lower` or `upper`.',
    ):
      field_utils.zero_mask_axis_outliers(field, axis)

  def test_raises_on_axis_without_ticks(self):
    axis = cx.SizedAxis('x', 4)
    field = cx.field(np.ones(4), axis)
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        f'Axis must be 1d with specified tick values got {axis}',
    ):
      field_utils.zero_mask_axis_outliers(field, axis, lower=0.0)

  def test_raises_on_multidim_axis(self):
    x, y = cx.SizedAxis('x', 2), cx.SizedAxis('y', 2)
    axis = cx.coords.compose(x, y)
    field = cx.field(np.ones((2, 2)), axis)
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        f'Axis must be 1d with specified tick values got {axis}',
    ):
      field_utils.zero_mask_axis_outliers(field, axis, lower=0.0)


class InAxesUtilTest(parameterized.TestCase):
  """Tests in_axes utility functions."""

  def test_in_axes_for_coord(self):
    x, y = cx.SizedAxis('x', 3), cx.SizedAxis('y', 4)
    f1 = cx.field(np.zeros((3, 4)), x, y)
    f2 = cx.field(np.zeros((4, 3)), y, x)
    f3 = cx.field(np.zeros((4,)), y)
    inputs = {'a': f1, 'b': (f2, 123, f3)}
    with self.subTest('map_over_x'):
      actual = field_utils.in_axes_for_coord(inputs, x)
      expected = {'a': 0, 'b': (1, None, None)}
      chex.assert_trees_all_equal(actual, expected)
    with self.subTest('map_over_y'):
      actual = field_utils.in_axes_for_coord(inputs, y)
      expected = {'a': 1, 'b': (0, None, 0)}
      chex.assert_trees_all_equal(actual, expected)

  def test_in_axes_for_coord_raises_for_non_1d_coord(self):
    x, y = cx.SizedAxis('x', 3), cx.SizedAxis('y', 4)
    f1 = cx.field(np.zeros((3, 4)), x, y)
    xy = cx.coords.compose(x, y)
    with self.assertRaisesRegex(ValueError, 'idx can be computed only for 1d'):
      field_utils.in_axes_for_coord(f1, xy)

  def test_in_axes_for_coord_with_nesting(self):
    x, y = cx.SizedAxis('x', 3), cx.SizedAxis('y', 4)
    f = {
        'a': cx.field(np.zeros((3, 4)), x, y),
        'b': cx.field(np.zeros((4, 3)), y, x),
    }
    actual_outer, actual_inner = field_utils.in_axes_for_coord(f, [x, y])
    expected_outer = {'a': 0, 'b': 1}
    expected_inner = {'a': 0, 'b': 0}
    chex.assert_trees_all_equal(actual_outer, expected_outer)
    chex.assert_trees_all_equal(actual_inner, expected_inner)

  @parameterized.named_parameters(
      dict(
          testcase_name='one_level',
          in_axes=((0, 1),),
          expected=((0, 1),),
      ),
      dict(
          testcase_name='two_levels_no_shift',
          in_axes=((1, 1), (0, 0)),
          expected=((1, 1), (0, 0)),
      ),
      dict(
          testcase_name='two_levels_with_shift',
          in_axes=({'a': 0, 'b': 1}, {'a': 1, 'b': 0}),
          expected=({'a': 0, 'b': 1}, {'a': 0, 'b': 0}),
      ),
      dict(
          testcase_name='three_levels_with_shifts_and_none',
          in_axes=((1, None, 2), (0, 1, 0), (2, 0, 1)),
          expected=(
              (1, None, 2),
              (0, 1, 0),
              (0, 0, 0),
          ),
      ),
      dict(
          testcase_name='two_levels_not_leading_axes',
          in_axes=((0, 2), (1, 0)),
          expected=((0, 2), (0, 0)),
      ),
      dict(
          testcase_name='nested_dict_structure',
          in_axes=({'a': {'b': 0}}, {'a': {'b': 1}}),
          expected=({'a': {'b': 0}}, {'a': {'b': 0}}),
      ),
      dict(
          testcase_name='repeated_none',
          in_axes=((None, 0), (None, 1)),
          expected=((None, 0), (None, 0)),
      ),
  )
  def test_nest_in_axes(self, in_axes, expected):
    actual = field_utils.nest_in_axes(*in_axes)
    chex.assert_trees_all_equal(actual, expected)

  def test_nest_in_axes_raises_on_negative_axis(self):
    with self.assertRaisesRegex(ValueError, 'Negative axes are not allowed'):
      field_utils.nest_in_axes((-1, 0))

  def test_nest_in_axes_raises_on_repeated_axis(self):
    with self.assertRaisesRegex(
        ValueError,
        'leaf in.*is mapped over the same axis multiple times',
    ):
      field_utils.nest_in_axes((0, 1), (0, 2))


class Reconstruct1dFieldFromRefValuesTest(parameterized.TestCase):
  """Tests reconstruct_1d_field_from_ref_values utility function."""

  @parameterized.named_parameters(
      dict(
          testcase_name='linear',
          interpolation_space='linear',
          ref_ticks=[0, 1],
          ref_values=[1, 3],
          axis_ticks=[0, 0.5, 1],
          expected_values=[1, 2, 3],
      ),
      dict(
          testcase_name='log',
          interpolation_space='log',
          ref_ticks=[0, 2],
          ref_values=[1, np.exp(2)],
          axis_ticks=[0, 1, 2],
          expected_values=[1, np.exp(1), np.exp(2)],
      ),
      dict(
          testcase_name='sqrt',
          interpolation_space='sqrt',
          ref_ticks=[0, 1],
          ref_values=[1, 4],
          axis_ticks=[0, 0.5, 1],
          expected_values=[1, 2.25, 4],
      ),
      dict(
          testcase_name='square',
          interpolation_space='square',
          ref_ticks=[0, 1],
          ref_values=[1, 2],
          axis_ticks=[0, 0.5, 1],
          expected_values=[1, np.sqrt(2.5), 2],
      ),
  )
  def test_interpolation(
      self,
      interpolation_space,
      ref_ticks,
      ref_values,
      axis_ticks,
      expected_values,
  ):
    axis = cx.LabeledAxis('x', np.array(axis_ticks))
    actual = field_utils.reconstruct_1d_field_from_ref_values(
        axis, ref_ticks, ref_values, interpolation_space
    )
    expected = cx.field(np.array(expected_values, dtype=np.float32), axis)
    chex.assert_trees_all_close(actual, expected)

  def test_raises_on_axis_without_ticks(self):
    axis = cx.SizedAxis('x', 4)
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        f'Axis must be 1d with specified tick values got {axis}',
    ):
      field_utils.reconstruct_1d_field_from_ref_values(
          axis, ref_ticks=[0, 1], ref_values=[1, 2]
      )

  def test_raises_on_multidim_axis(self):
    x, y = cx.SizedAxis('x', 2), cx.SizedAxis('y', 2)
    axis = cx.coords.compose(x, y)
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        f'Axis must be 1d with specified tick values got {axis}',
    ):
      field_utils.reconstruct_1d_field_from_ref_values(
          axis, ref_ticks=[0, 1], ref_values=[1, 2]
      )

  def test_raises_on_bad_interpolation_space(self):
    axis = cx.LabeledAxis('x', np.array([0, 1]))
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'Unsupported interpolation space: bad_space',
    ):
      field_utils.reconstruct_1d_field_from_ref_values(
          axis,
          ref_ticks=[0, 1],
          ref_values=[1, 2],
          interpolation_space='bad_space',
      )


class EnsurePositionalAxisIdxTest(parameterized.TestCase):
  """Tests ensure_positional_axis_idx utility function."""

  def test_mixed_coords_negative_idx(self):
    x = cx.SizedAxis('x', 3)
    y = cx.SizedAxis('y', 4)
    f1 = cx.field(np.zeros((3,)), x)
    f2 = cx.field(np.zeros((3, 2)), x, None)
    f3 = cx.field(np.zeros((4, 3, 1)), y, x, None)

    pytree = [f1, {'a': f2, 'b': f3}]
    actual = field_utils.ensure_positional_axis_idx(pytree, idx=-1)

    add_dummy = lambda s, *c: cx.coords.compose(*c, cx.DummyAxis(None, s))
    self.assertEqual(actual[0].coordinate, add_dummy(1, x))
    self.assertEqual(actual[1]['a'].coordinate, add_dummy(2, x))
    self.assertEqual(actual[1]['b'].coordinate, add_dummy(1, y, x))

  def test_mixed_coords_positive_idx(self):
    x = cx.SizedAxis('x', 3)
    y = cx.SizedAxis('y', 4)
    f1 = cx.field(np.zeros((3, 4)), x, y)
    f2 = cx.field(np.zeros((3, 5)), x, None)
    f3 = cx.field(np.zeros((4, 2, 3)), y, None, x)

    pytree = (f1, f2, f3)
    actual = field_utils.ensure_positional_axis_idx(pytree, idx=1)

    dummy = lambda s: cx.DummyAxis(None, s)
    self.assertEqual(actual[0].coordinate, cx.coords.compose(x, dummy(1), y))
    self.assertEqual(actual[1].coordinate, cx.coords.compose(x, dummy(5)))
    self.assertEqual(actual[2].coordinate, cx.coords.compose(y, dummy(2), x))

  def test_raises_on_multiple_positional_axes(self):
    f = cx.field(np.zeros((1, 1)), None, None)
    with self.assertRaisesRegex(
        ValueError, 'Field has multiple positional axes'
    ):
      field_utils.ensure_positional_axis_idx(f, idx=0)

  def test_raises_on_misaligned_positional_axes_positive_idx(self):
    x = cx.SizedAxis('x', 3)
    # Target idx=0; f1 has positional at 1: (x, None)
    f1 = cx.field(np.zeros((3, 1)), x, None)
    with self.assertRaisesRegex(ValueError, 'Expected positional axes at'):
      field_utils.ensure_positional_axis_idx({'f': f1}, idx=0)

  def test_raises_on_misaligned_positional_axes_negative_idx(self):
    x = cx.SizedAxis('x', 3)
    # Target idx=-1; f1 has positional at 0: (None, x). dims[-1] is x.
    f1 = cx.field(np.zeros((1, 3)), None, x)
    with self.assertRaisesRegex(ValueError, 'Expected positional axes at'):
      field_utils.ensure_positional_axis_idx([f1], idx=-1)


if __name__ == '__main__':
  absltest.main()
