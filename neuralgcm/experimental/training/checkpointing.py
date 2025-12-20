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
"""Utilities for checkpointing training runs."""

from collections.abc import Sequence
import dataclasses
import json
from typing import Any

from absl import logging
import fiddle
from fiddle.experimental import serialization
from flax import nnx
from neuralgcm.experimental.core import api
from neuralgcm.experimental.core import checkpointing as model_checkpointing
import orbax.checkpoint as ocp
from orbax.checkpoint import checkpoint_managers as ocp_managers


# pylint: disable=logging-fstring-interpolation


@dataclasses.dataclass
class LastNSteps(ocp_managers.PreservationPolicy):
  """Ensures sufficient checkpoints are preserved for lookback restoration."""

  lookback_steps: int

  def should_preserve(
      self,
      checkpoints: Sequence[ocp_managers.PolicyCheckpointInfo],
      *,
      context: ocp_managers.PreservationContext,
  ) -> Sequence[bool]:
    latest_step = max(ckpt.step for ckpt in checkpoints)
    result = [
        latest_step - ckpt.step <= self.lookback_steps for ckpt in checkpoints
    ]
    return result


_MODEL_CONFIG_KEY = 'fiddle_model_config'
_STEP_FORMAT_FIXED_LENGTH = 6


def _checkpoint_manager_options_with_fixed_step_format(
    options: ocp.CheckpointManagerOptions | None = None,
) -> ocp.CheckpointManagerOptions:
  """Returns a CheckpointManagerOptions with fixed step format."""
  if options is None:
    options = ocp.CheckpointManagerOptions()
  if options.step_format_fixed_length is not None:
    raise ValueError('cannot customize step_format_fixed_length')
  return dataclasses.replace(
      options, step_format_fixed_length=_STEP_FORMAT_FIXED_LENGTH
  )


def training_manager(
    directory: str,
    options: ocp.CheckpointManagerOptions,
    model_config_str: str,
    metadata: dict[str, Any] | None = None,
) -> ocp.CheckpointManager:
  """Returns a CheckpointManager for training."""
  model_config_dict = json.loads(model_config_str)
  options = _checkpoint_manager_options_with_fixed_step_format(options)
  if metadata is None:
    metadata = {}
  if _MODEL_CONFIG_KEY in metadata:
    raise ValueError(f'{_MODEL_CONFIG_KEY!r} already set in metadata')
  metadata[_MODEL_CONFIG_KEY] = model_config_dict
  return ocp.CheckpointManager(directory, options=options, metadata=metadata)


def read_only_manager(directory: str) -> ocp.CheckpointManager:
  """Returns a CheckpointManager that is read-only."""
  options = _checkpoint_manager_options_with_fixed_step_format()
  options.read_only = True
  return ocp.CheckpointManager(directory, options=options)


def load_fiddle_json(json_str: str) -> fiddle.Config:
  """Loads a Fiddle config from a JSON string."""
  # Use replacements to map from old to new module names, e.g.,
  # 'google3.research.simulation.neuralgcm.something': 'neuralgcm.something'
  replacements = {}
  for k, v in replacements.items():
    json_str = json_str.replace(k, v)
  return serialization.load_json(json_str)


def untrained_model_from_training_dir(
    checkpoints_dir: str,
) -> api.Model:
  """Loads model without parameters."""
  logging.info('loading model metadata')
  with read_only_manager(checkpoints_dir) as manager:
    model_config_dict = manager.metadata().custom_metadata[_MODEL_CONFIG_KEY]
  # TODO(shoyer): avoid unnecessary serialization to/from JSON. This may require
  # refactoring fiddle.experiment.serialization.
  logging.info('loading model config into fiddle')
  model_config = load_fiddle_json(json.dumps(model_config_dict))
  logging.info('building model')
  return api.Model.from_fiddle_config(model_config)


def model_from_training_checkpoint(
    checkpoints_dir: str,
    step: int | None = None,
    use_ema_params: bool = True,
) -> api.Model:
  """Load a checkpoint into a Model."""
  logging.info(f'Loading checkpoint for step {step} from {checkpoints_dir}')
  model = untrained_model_from_training_dir(checkpoints_dir)
  abstract_state = model_checkpointing.split_model_state_for_saving(model)

  param_key = 'ema_params' if use_ema_params else 'params'
  restore_args = ocp.args.Composite(
      metadata=ocp.args.JsonRestore(),
      non_params=ocp.args.StandardRestore(abstract_state.non_params),
      **{param_key: ocp.args.StandardRestore(abstract_state.params)},
  )

  with read_only_manager(checkpoints_dir) as manager:
    if step is None:
      step = manager.latest_step()
      if step is None:
        raise ValueError(f'Found no checkpoints in {checkpoints_dir}')
    logging.info(f'Loading checkpoint for step {step} from {checkpoints_dir}')
    ckpt = manager.restore(step, args=restore_args)

  params = ckpt[param_key]
  nnx.update(model, params)
  nnx.update(model, ckpt.non_params)
  return model


def export_training_to_model_checkpoint(
    train_checkpoints_dir: str,
    model_checkpoint_path: str,
    train_step: int | None = None,
    use_ema_params: bool = True,
) -> None:
  """Load a training checkpoint and save it as a model checkpoint.

  Args:
    train_checkpoints_dir: The directory containing training checkpoints.
    model_checkpoint_path: The path to save the model checkpoint to.
    train_step: The step of the training checkpoint to load. If None, the
      latest checkpoint will be loaded.
    use_ema_params: Whether to use the EMA params or the non-EMA params.
  """
  model = model_from_training_checkpoint(
      train_checkpoints_dir, step=train_step, use_ema_params=use_ema_params
  )
  model_checkpointing.save_checkpoint(model, model_checkpoint_path)
