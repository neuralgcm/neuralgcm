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
import math
from unittest import mock

from absl.testing import absltest
from neuralgcm.experimental.inference import timing


class TimerTest(absltest.TestCase):

  def test_basics(self):
    counter = mock.Mock()
    counter.side_effect = [0, 0.5]
    timer = timing.Timer(counter=counter)
    self.assertTrue(math.isnan(timer.last))
    self.assertEqual(timer.total, 0.0)
    with timer:
      pass
    self.assertEqual(timer.last, 0.5)
    self.assertEqual(timer.total, 0.5)

  def test_multiple_steps(self):
    counter = mock.Mock()
    counter.side_effect = [0, 1, 10, 11]
    timer = timing.Timer(counter=counter)
    with timer:
      pass
    with timer:
      pass
    self.assertEqual(timer.last, 1.0)
    self.assertEqual(timer.total, 2.0)

  def test_reenter(self):
    timer = timing.Timer()
    timer.begin_step()
    with self.assertRaises(RuntimeError):
      timer.begin_step()


if __name__ == '__main__':
  absltest.main()
