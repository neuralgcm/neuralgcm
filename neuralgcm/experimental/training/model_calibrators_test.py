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

from absl.testing import absltest
from flax import nnx
from neuralgcm.experimental.training import model_calibrators
import xarray


class MockModule(nnx.Module, pytree=False):

  def __init__(self):
    self.updated = False
    self.kwargs = {}

  def update_from_xarray(self, dataset, **kwargs):
    self.updated = True
    self.dataset = dataset
    self.kwargs = kwargs


class Model(nnx.Module, pytree=False):

  def __init__(self):
    self.submodule_a = MockModule()
    self.sub_dict = {'b': MockModule()}
    self.submodule_c = MockModule()


class ModelCalibratorsTest(absltest.TestCase):

  def test_update_submodules_from_xarray(self):
    model = Model()
    dataset_a = xarray.Dataset({'a': 1})
    dataset_b = xarray.Dataset({'b': 2})

    specs = {
        (MockModule, 'submodule_a'): dataset_a,
        (MockModule, 'b'): dataset_b,
    }
    calibrator = model_calibrators.UpdateSubmodulesFromXarray(
        specs, kwargs={'foo': 'bar'}
    )

    calibrator(model, None, None)

    with self.subTest('updated_submodules'):
      self.assertTrue(model.submodule_a.updated)
      self.assertEqual(model.submodule_a.dataset, dataset_a)
      self.assertEqual(model.submodule_a.kwargs, {'foo': 'bar'})

      self.assertTrue(model.sub_dict['b'].updated)
      self.assertEqual(model.sub_dict['b'].dataset, dataset_b)
      self.assertEqual(model.sub_dict['b'].kwargs, {'foo': 'bar'})

    with self.subTest('untouched_submodules'):
      self.assertFalse(model.submodule_c.updated)

  def test_update_duplicates(self):
    model = Model()
    model.submodule_d = model.submodule_a  # Create a duplicate reference.
    dataset_d = xarray.Dataset({'d': 3})
    # Update using the alias 'submodule_d'
    specs = {(MockModule, 'submodule_d'): dataset_d}
    calibrator = model_calibrators.UpdateSubmodulesFromXarray(
        specs, kwargs={'baz': 'qux'}
    )

    calibrator(model, None, None)

    self.assertTrue(model.submodule_a.updated)
    self.assertEqual(model.submodule_a.dataset, dataset_d)
    self.assertEqual(model.submodule_a.kwargs, {'baz': 'qux'})
    # Ensure both point to the same updated object state
    self.assertTrue(model.submodule_d.updated)

  def test_missing_module_raises_error(self):
    model = Model()
    specs = {
        (MockModule, 'non_existent'): xarray.Dataset({'z': 9}),
    }
    calibrator = model_calibrators.UpdateSubmodulesFromXarray(specs)

    with self.assertRaisesRegex(ValueError, 'No module of type .* found'):
      calibrator(model, None, None)


if __name__ == '__main__':
  absltest.main()
