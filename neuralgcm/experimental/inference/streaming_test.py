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

from neuralgcm.experimental.inference import streaming


class SentinelError(Exception):
  pass


def raises_error():
  raise SentinelError


class SingleThreadExecutorTest(parameterized.TestCase):

  def test_executor(self):
    add1 = streaming.SingleTaskExecutor(lambda x: x + 1)
    add1.submit(1)
    result = add1.get()
    self.assertEqual(result, 2)

    with self.assertRaises(streaming.EmptyStreamError):
      add1.get()

  def test_executor_blocking(self):
    out = []
    stream = streaming.SingleTaskExecutor(out.append)
    stream.submit(1)
    stream.wait()
    stream.submit(2)
    stream.wait()
    stream.submit(3)
    stream.wait()
    self.assertEqual(out, [1, 2, 3])

  def test_executor_errors(self):

    stream = streaming.SingleTaskExecutor(raises_error)
    stream.submit()

    with self.assertRaises(streaming.FullStreamError):
      stream.submit()

    with self.assertRaises(SentinelError):
      stream.get()
    with self.assertRaises(streaming.EmptyStreamError):
      stream.get()

    stream.submit()
    with self.assertRaises(SentinelError):
      stream.wait()


if __name__ == '__main__':
  absltest.main()
