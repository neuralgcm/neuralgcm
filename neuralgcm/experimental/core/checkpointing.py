# Copyright 2024 Google LLC
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
"""Helper utilities for checkpointing Fiddle configs used in NeuralGCM."""

import dataclasses
import json

from etils import epath
import fiddle as fdl
from fiddle.experimental import serialization
from flax import nnx
import jax
from neuralgcm.experimental.core import api
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import typing
import orbax.checkpoint as ocp


ParamInfo = ocp.type_handlers.ParamInfo
Metadata = ocp.metadata.value.Metadata


@dataclasses.dataclass(frozen=True)
class _SplitState:
  params: nnx.GraphState
  non_params: nnx.GraphState


UNSAVED_VARIABLE_TYPES = (
    typing.Diagnostic,
    typing.DynamicInput,
    typing.Randomness,
)


def split_model_state_for_saving(model: nnx.Module) -> _SplitState:
  """Extracts model state to save from an nnx.Module."""
  # TODO(shoyer): Consider adding a base-class for temporary variables in
  # in neuralgcm.experimental. This would allow for collecting them together
  # with nnx.state() and similar utilities.
  params, *_, non_params = nnx.state(
      model, nnx.Param, *UNSAVED_VARIABLE_TYPES, ...
  )
  return _SplitState(params=params, non_params=non_params)


_STATE_KEY = 'state'
_CONFIG_KEY = 'fiddle_config'


def load_model_checkpoint(
    path: str | epath.PathLike,
    spmd_mesh_updates: (
        dict[parallelism.TagOrMeshType, jax.sharding.Mesh | None] | None
    ) = None,
    array_partitions_updates: (
        dict[parallelism.TagOrMeshType, parallelism.ArrayPartitions] | None
    ) = None,
    field_partitions_updates: (
        dict[parallelism.TagOrMeshType, parallelism.FieldPartitions] | None
    ) = None,
) -> api.ForecastSystem:
  """Loades a ForecastSystem model from a checkpoint."""
  checkpointer = ocp.Checkpointer(ocp.CompositeCheckpointHandler())

  # Create model from checkpoint metadata.
  config_args = ocp.args.Composite(**{_CONFIG_KEY: ocp.args.JsonRestore()})
  model_config_dict = checkpointer.restore(path, config_args)[_CONFIG_KEY]
  model_config = serialization.load_json(json.dumps(model_config_dict))
  model = api.ForecastSystem.from_fiddle_config(
      model_config,
      spmd_mesh_updates=spmd_mesh_updates,
      array_partitions_updates=array_partitions_updates,
      field_partitions_updates=field_partitions_updates,
  )

  # Set model parameters from checkpoint.
  state_tuple = split_model_state_for_saving(model)
  state = nnx.merge_state(state_tuple.params, state_tuple.non_params)
  state_args = ocp.args.Composite(**{_STATE_KEY: ocp.args.PyTreeRestore(state)})
  restored = checkpointer.restore(path, state_args)[_STATE_KEY]
  nnx.update(model, restored)
  return model


def save_checkpoint(
    model: api.ForecastSystem,
    path: str | epath.PathLike,
    fiddle_config: fdl.Config[api.ForecastSystem] | None = None,
):
  """Saves model to a checkpoint."""
  if fiddle_config is None:
    fiddle_config = model.fiddle_config

  if not isinstance(fiddle_config, fdl.Config):
    raise TypeError(f'must supply a fiddle.Config, got {fiddle_config=}')

  state_tuple = split_model_state_for_saving(model)
  state = nnx.merge_state(state_tuple.params, state_tuple.non_params)
  model_config_dict = json.loads(serialization.dump_json(fiddle_config))
  args = ocp.args.Composite(**{
      _STATE_KEY: ocp.args.PyTreeSave(state),
      _CONFIG_KEY: ocp.args.JsonSave(model_config_dict),
  })
  checkpointer = ocp.Checkpointer(ocp.CompositeCheckpointHandler())
  checkpointer.save(path, args)
