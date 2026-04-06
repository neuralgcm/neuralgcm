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

from unittest import mock

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


@nnx.dataclass
class MockModelParams(nnx.Module):
  l1: nnx.Param = nnx.data(default_factory=lambda: nnx.Param(1.0))
  l2: nnx.Param = nnx.data(default_factory=lambda: nnx.Param(2.0))


@nnx.dataclass
class MockModelSubsetParams(nnx.Module):
  l1: nnx.Param = nnx.data(default_factory=lambda: nnx.Param(1.0))


@nnx.dataclass
class MockLargeModel(nnx.Module):
  l1: nnx.Param = nnx.data(default_factory=lambda: nnx.Param(1.0))
  l2: nnx.Param = nnx.data(init=False)
  l3: nnx.Param = nnx.data(default_factory=lambda: nnx.Param(3.0))

  def __post_init__(self):
    self.l2 = self.l1  # l2 is an alias of l1


class ParentModel(nnx.Module):
  """Mock module for testing component update."""

  def __init__(self, val1=1.0, val2=2.0):
    self.my_component = MockModelParams(l1=nnx.Param(val1), l2=nnx.Param(val2))


@nnx.dataclass
class MockSmallModel(nnx.Module):
  l2: nnx.Param = nnx.data(default_factory=lambda: nnx.Param(0.0))


class ModelCalibratorsTest(absltest.TestCase):

  def test_shared_params_subset(self):
    model = MockSmallModel(l2=nnx.Param(0.0))
    loaded_model = MockLargeModel(l1=nnx.Param(42.0), l3=nnx.Param(99.0))

    with mock.patch.object(
        model_calibrators.checkpointing, 'load_model_checkpoint'
    ) as mock_load:
      mock_load.return_value = loaded_model

      calibrator = model_calibrators.LoadModelComponentParams(
          component_key=None,
          ckpt_path_or_dir='/dummy/path',
          load_subset=True,
      )
      calibrator(model, None, None)

      self.assertEqual(model.l2.get_value(), 42.0)

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

  @mock.patch.object(model_calibrators.checkpointing, 'load_model_checkpoint')
  def test_load_model_component_params(self, mock_load):
    mock_load.return_value = MockModelParams(
        l1=nnx.Param(3.0), l2=nnx.Param(4.0)
    )
    model = MockModelParams(l1=nnx.Param(1.0), l2=nnx.Param(2.0))

    calibrator = model_calibrators.LoadModelComponentParams(
        component_key=None,
        ckpt_path_or_dir='/dummy/path',
        is_training_checkpoint=False,
    )
    calibrator(model, None, None)

    self.assertEqual(model.l1.get_value(), 3.0)
    self.assertEqual(model.l2.get_value(), 4.0)
    mock_load.assert_called_once_with(path='/dummy/path')

  @mock.patch.object(model_calibrators.checkpointing, 'load_model_checkpoint')
  def test_load_model_component_params_with_key(self, mock_load):
    mock_load.return_value = MockModelParams(
        l1=nnx.Param(3.0), l2=nnx.Param(4.0)
    )
    model = ParentModel(1.0, 2.0)

    calibrator = model_calibrators.LoadModelComponentParams(
        component_key='my_component',
        ckpt_path_or_dir='/dummy/path',
        is_training_checkpoint=False,
    )
    calibrator(model, None, None)

    self.assertEqual(model.my_component.l1.get_value(), 3.0)
    self.assertEqual(model.my_component.l2.get_value(), 4.0)
    mock_load.assert_called_once_with(path='/dummy/path')

  @mock.patch.object(model_calibrators.checkpointing, 'load_model_checkpoint')
  def test_load_model_component_params_subset(self, mock_load):
    mock_load.return_value = MockModelParams(
        l1=nnx.Param(3.0), l2=nnx.Param(4.0)
    )
    model = MockModelSubsetParams(l1=nnx.Param(1.0))

    calibrator = model_calibrators.LoadModelComponentParams(
        component_key=None,
        ckpt_path_or_dir='/dummy/path',
        load_subset=True,
    )
    calibrator(model, None, None)

    self.assertEqual(model.l1.get_value(), 3.0)

  @mock.patch.object(model_calibrators.checkpointing, 'load_model_checkpoint')
  def test_load_model_component_params_subset_raises_when_false(
      self, mock_load
  ):
    mock_load.return_value = MockModelParams(
        l1=nnx.Param(3.0), l2=nnx.Param(4.0)
    )
    model = MockModelSubsetParams(l1=nnx.Param(1.0))

    calibrator = model_calibrators.LoadModelComponentParams(
        component_key=None,
        ckpt_path_or_dir='/dummy/path',
        load_subset=False,
    )

    with self.assertRaisesRegex(
        ValueError, 'Parameter structures do not match'
    ):
      calibrator(model, None, None)


if __name__ == '__main__':
  absltest.main()
