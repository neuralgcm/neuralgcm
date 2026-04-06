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
"""Model calibration utilities."""

import abc
import dataclasses
from typing import Any, TypeAlias

import coordax as cx
from flax import nnx
import jax
from neuralgcm.experimental.core import checkpointing
from neuralgcm.experimental.core import data_specs
from neuralgcm.experimental.core import module_utils
from neuralgcm.experimental.core import transforms
from neuralgcm.experimental.training import checkpointing as training_checkpointing
from neuralgcm.experimental.training import data_loading
import xarray


Model: TypeAlias = nnx.Module  # Avoid circular import with api.py
DataLoader: TypeAlias = data_loading.DataLoader
TrainSchedule: TypeAlias = Any  # Avoid circular import with trainer.py


class ModelCalibrator(abc.ABC):
  """Abstract base class for model calibrators."""

  @abc.abstractmethod
  def __call__(
      self,
      model: Model,
      data_loader: DataLoader,
      train_schedule: TrainSchedule,
  ) -> Model:
    """Calibrates the model."""
    raise NotImplementedError()


@dataclasses.dataclass
class UpdateSubmodulesFromXarray(ModelCalibrator):
  """Updates submodules from Xarray datasets."""

  # pylint: disable=g-bare-generic
  update_specs: dict[tuple[type, str], xarray.Dataset]
  kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
  raise_on_no_update: bool = True
  # pylint: enable=g-bare-generic

  def __call__(
      self,
      model: Model,
      data_loader: DataLoader,
      train_schedule: TrainSchedule,
  ) -> Model:
    del data_loader, train_schedule  # Unused.

    # Find duplicates to handle multiple paths to the same object.
    duplicate_paths_groups = nnx.graph.find_duplicates(model)
    # Maps a path string to a set of all equivalent path strings (aliases).
    path_to_aliases = {}

    def path_to_str(path):
      return '.'.join(str(p) for p in path)

    for group in duplicate_paths_groups:
      # group is a list of paths (tuples of keys).
      aliases = {path_to_str(path) for path in group}
      for path in group:
        path_to_aliases[path_to_str(path)] = aliases

    for (cls, suffix), dataset in self.update_specs.items():
      modules, paths = module_utils.retrieve_subclass_modules(
          model, cls, return_paths=True
      )

      updated = False
      for module, path in zip(modules, paths):
        full_path = path_to_str(path)
        candidates = path_to_aliases.get(full_path, {full_path})

        if any(candidate.endswith(suffix) for candidate in candidates):
          module.update_from_xarray(dataset, **self.kwargs)
          updated = True

      if not updated and self.raise_on_no_update:
        raise ValueError(
            f'No module of type {cls} with suffix "{suffix}" found for update.'
        )

    return model


@dataclasses.dataclass
class CollectNormalizationStats(ModelCalibrator):
  """Collects normalization statistics by running the model on data."""

  train_stage_idx: int
  rng_seed: int
  time_start: str
  time_end: str
  batch_count: int

  def __call__(
      self,
      model: Model,
      data_loader: DataLoader,
      train_schedule: TrainSchedule,
  ) -> Model:
    rng = jax.random.key(self.rng_seed)

    def unroll_fn(model, inputs, dynamic_data, rng, query):
      model.initialize_random_processes(cx.field(rng))
      model.update_dynamic_inputs(dynamic_data)
      model.assimilate(inputs)
      model.advance()
      observed = model.observe(query)
      return observed

    unroll_fn = nnx.jit(unroll_fn)
    stats_modules = module_utils.retrieve_subclass_modules(
        model, transforms.StreamNorm
    )
    for m in stats_modules:
      m.update_stats = True

    train_stage = train_schedule.stages[self.train_stage_idx]
    data_iter = data_loader.build_eval_inputs(
        train_stage.inputs_spec,
        train_stage.dynamic_inputs_spec,
        dataset_time_slice=(self.time_start, self.time_end),
        batch_size_per_device=None,
        time_sample_offset=train_stage.time_sample_offset,
        batch_count=self.batch_count,
    )

    # Helper to select a single last time slice in trajectory.
    sel_td_in_f = lambda f: cx.cmap(lambda x: x[-1])(f.untag('timedelta'))
    sel_timedelta = lambda t: jax.tree.map(sel_td_in_f, t, is_leaf=cx.is_field)

    for i, (inputs, dynamic_data) in enumerate(data_iter):
      init_slice = data_loading.sel_init_fields(inputs)
      # For query we need a single time slice of input data.
      query = data_specs.construct_query(
          sel_timedelta(init_slice), train_stage.queries_spec
      )
      result = unroll_fn(
          model, inputs, dynamic_data, jax.random.fold_in(rng, i), query
      )
      jax.block_until_ready(result)

    # Set update_stats to False to avoid updating stats during training.
    for m in stats_modules:
      m.update_stats = False

    return model


@dataclasses.dataclass
class LoadModelComponentParams(ModelCalibrator):
  """Loads parameters for a specific model component from a checkpoint."""

  component_key: str | None
  ckpt_path_or_dir: str
  ckpt_step: int | None = None  # only used for training checkpoints.
  param_types: tuple[Any, ...] = (nnx.Param,)
  is_training_checkpoint: bool = False
  load_subset: bool = False

  def __call__(
      self,
      model: Model,
      data_loader: DataLoader,
      train_schedule: TrainSchedule,
  ) -> Model:
    del data_loader, train_schedule  # Unused.
    # TODO(dkochkov): Consider if we need to provide spmd_mesh updates. For
    # standard model parameters this should not be necessary, since we tend to
    # store them in a non-padded format, but this might be safer or useful in
    # other cases.
    if self.is_training_checkpoint:
      component = training_checkpointing.model_from_training_checkpoint(
          checkpoints_dir=self.ckpt_path_or_dir, step=self.ckpt_step
      )
    else:
      component = checkpointing.load_model_checkpoint(
          path=self.ckpt_path_or_dir
      )

    loaded_state = nnx.state(component, *self.param_types)
    key = self.component_key
    to_update = getattr(model, key) if key else model
    target_state = nnx.state(to_update, *self.param_types)

    if self.load_subset:
      loaded_vars = {
          path: node
          for path, node in nnx.graph.iter_graph(component)
          if isinstance(node, self.param_types)
      }
      # Expanding loaded_vars to include duplicate aliases.
      for group in nnx.graph.find_duplicates(component):
        canonical = next((p for p in group if p in loaded_vars), None)
        if canonical is not None:
          val = loaded_vars[canonical]
          for p in group:
            loaded_vars[p] = val
      target_vars = {
          path: node
          for path, node in nnx.graph.iter_graph(to_update)
          if isinstance(node, self.param_types)
      }
      common_keys = set(loaded_vars.keys()).intersection(target_vars.keys())
      for path in common_keys:
        target_vars[path].set_value(loaded_vars[path].get_value())
    else:
      # Ensure there are no discrepancies between the parameter structures.
      loaded_struct = jax.tree.structure(loaded_state)
      target_struct = jax.tree.structure(target_state)
      if loaded_struct != target_struct:
        raise ValueError(
            f'Parameter structures do not match for component {key}.\n'
            f'Loaded structure: {loaded_struct}\n'
            f'Target structure: {target_struct}'
        )
      nnx.update(to_update, loaded_state)
    return model
