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

from absl.testing import absltest
from absl.testing import parameterized
from neuralgcm.experimental.xreader import stencils
import numpy as np


class StencilsTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(start=0, stop=1, step=0.5, closed='both', expected=[0.0, 0.5, 1.0]),
      dict(start=0, stop=1, step=0.5, closed='left', expected=[0.0, 0.5]),
      dict(start=0, stop=1, step=0.5, closed='right', expected=[0.5, 1.0]),
      dict(start=0, stop=1, step=0.5, closed='neither', expected=[0.5]),
      dict(
          start=-1,
          stop=1,
          step=0.5,
          closed='both',
          expected=[-1.0, -0.5, 0.0, 0.5, 1.0],
      ),
  )
  def test_stencil_points(self, start, stop, step, closed, expected):
    stencil = stencils.Stencil(start, stop, step, closed=closed)
    np.testing.assert_allclose(stencil.points, expected)

  def test_stencil_default_closed(self):
    stencil = stencils.Stencil(0, 1, 0.5)
    self.assertEqual(stencil.closed, 'left')

  def test_stencil_invalid_stop(self):
    with self.assertRaisesRegex(ValueError, 'stop must be greater than start'):
      stencils.Stencil(0, 0, 1)

  def test_stencil_invalid_closed(self):
    with self.assertRaisesRegex(ValueError, 'invalid value for closed'):
      stencils.Stencil(0, 1, 0.5, closed='invalid')

  def test_stencil_non_divisible_step(self):
    stencil = stencils.Stencil(0, 1, 0.3, closed='both')
    with self.assertRaisesRegex(ValueError, 'must evenly divide'):
      stencil.points

  def test_time_stencil(self):
    stencil = stencils.TimeStencil('-3h', '+2h', '1h', closed='both')
    self.assertEqual(stencil.start, np.timedelta64(-3, 'h'))
    self.assertEqual(stencil.stop, np.timedelta64(2, 'h'))
    self.assertEqual(stencil.step, np.timedelta64(1, 'h'))

    expected = np.array([-3, -2, -1, 0, 1, 2], dtype='timedelta64[h]')
    np.testing.assert_array_equal(stencil.points, expected)

  def test_time_stencil_repr(self):
    stencil = stencils.TimeStencil('-9h', '3h', '1h', closed='both')
    self.assertEqual(
        repr(stencil),
        "TimeStencil(start='-9 hours', stop='3 hours', step='1 hours',"
        " closed='both')",
    )

  @parameterized.parameters(
      dict(value='1h', expected=np.timedelta64(1, 'h')),
      dict(value='-1h', expected=np.timedelta64(-1, 'h')),
      dict(value='+1h', expected=np.timedelta64(1, 'h')),
      dict(value='1hr', expected=np.timedelta64(1, 'h')),
      dict(value='1hour', expected=np.timedelta64(1, 'h')),
      dict(value='1hours', expected=np.timedelta64(1, 'h')),
      dict(value='1m', expected=np.timedelta64(1, 'm')),
      dict(value='1min', expected=np.timedelta64(1, 'm')),
      dict(value='1minute', expected=np.timedelta64(1, 'm')),
      dict(value='1minutes', expected=np.timedelta64(1, 'm')),
      dict(value='1s', expected=np.timedelta64(1, 's')),
      dict(value='1sec', expected=np.timedelta64(1, 's')),
      dict(value='1second', expected=np.timedelta64(1, 's')),
      dict(value='1seconds', expected=np.timedelta64(1, 's')),
      dict(value='1D', expected=np.timedelta64(1, 'D')),
      dict(value='1day', expected=np.timedelta64(1, 'D')),
      dict(value='1days', expected=np.timedelta64(1, 'D')),
      dict(value='24 hours', expected=np.timedelta64(24, 'h')),
  )
  def test_parse_timedelta_string(self, value, expected):
    self.assertEqual(stencils._parse_timedelta_string(value), expected)

  def test_parse_timedelta_string_invalid(self):
    with self.assertRaisesRegex(ValueError, 'invalid time delta string'):
      stencils._parse_timedelta_string('invalid')

  def test_parse_timedelta_string_invalid_unit(self):
    with self.assertRaisesRegex(ValueError, 'unsupported time unit'):
      stencils._parse_timedelta_string('1w')

  @parameterized.parameters(
      dict(
          source_points=[0, 1, 2, 3, 4],
          sample_origins=[1, 3],
          stencil=stencils.Stencil(start=-1, stop=1, step=1, closed='both'),
          expected=[slice(0, 3, 1), slice(2, 5, 1)],
      ),
      dict(
          source_points=[0, 1, 2, 3, 4],
          sample_origins=[1, 3],
          stencil=stencils.Stencil(start=-1, stop=1, step=1, closed='left'),
          expected=[slice(0, 2, 1), slice(2, 4, 1)],
      ),
      dict(
          source_points=[0, 1, 2, 3, 4],
          sample_origins=[1, 3],
          stencil=stencils.Stencil(start=-1, stop=1, step=1, closed='right'),
          expected=[slice(1, 3, 1), slice(3, 5, 1)],
      ),
      dict(
          source_points=[0, 1, 2, 3, 4],
          sample_origins=[1, 3],
          stencil=stencils.Stencil(start=-1, stop=1, step=1, closed='neither'),
          expected=[slice(1, 2, 1), slice(3, 4, 1)],
      ),
      dict(
          source_points=[0, 2, 4, 6, 8],
          sample_origins=[2, 6],
          stencil=stencils.Stencil(start=-2, stop=2, step=2, closed='both'),
          expected=[slice(0, 3, 1), slice(2, 5, 1)],
      ),
      dict(
          source_points=np.arange(10),
          sample_origins=[2],
          stencil=stencils.Stencil(start=-2, stop=4, step=2, closed='left'),
          expected=[slice(0, 6, 2)],
      )
  )
  def test_build_sampling_slices(
      self, source_points, sample_origins, stencil, expected
  ):
    actual = stencils.build_sampling_slices(
        source_points, sample_origins, stencil
    )
    self.assertEqual(actual, expected)

    source_values = np.asarray(source_points)[actual[0]]
    target_values = stencil.points + sample_origins[0]
    np.testing.assert_array_equal(source_values, target_values)

  def test_build_sampling_slices_invalid_source_points(self):
    with self.assertRaisesRegex(ValueError, 'source_points must be 1D'):
      stencils.build_sampling_slices(
          source_points=[[0, 1], [2, 3]],
          sample_origins=[1],
          stencil=stencils.Stencil(0, 1, 1),
      )

  def test_build_sampling_slices_invalid_sample_origins(self):
    with self.assertRaisesRegex(ValueError, 'sample_origins must be 1D'):
      stencils.build_sampling_slices(
          source_points=[0, 1],
          sample_origins=[[1], [2]],
          stencil=stencils.Stencil(0, 1, 1),
      )

  def test_build_sampling_slices_non_constant_source_step(self):
    with self.assertRaisesRegex(
        ValueError, 'source_points must have constant step'
    ):
      stencils.build_sampling_slices(
          source_points=[0, 1, 3],
          sample_origins=[1],
          stencil=stencils.Stencil(0, 1, 1),
      )

  def test_build_sampling_slices_negative_start(self):
    with self.assertRaisesRegex(
        ValueError, 'first stencil point must be non-negative'
    ):
      stencils.build_sampling_slices(
          source_points=[0, 1, 2],
          sample_origins=[-1],
          stencil=stencils.Stencil(0, 1, 1),
      )

  def test_build_sampling_slices_out_of_bounds_stop(self):
    with self.assertRaisesRegex(
        ValueError,
        'last stencil point must be less than the number of source points',
    ):
      stencils.build_sampling_slices(
          source_points=[0, 1, 2],
          sample_origins=[2],
          stencil=stencils.Stencil(0, 2, 1),
      )


if __name__ == '__main__':
  absltest.main()
