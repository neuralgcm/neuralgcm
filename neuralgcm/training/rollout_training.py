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

"""Entry point for rollout training of NeuralGCM models with Fiddle."""

from collections.abc import Sequence
import copy
import os

from absl import app
from absl import flags
import fiddle as fdl
from fiddle import absl_flags as fdl_flags
from fiddle.experimental import serialization
import jax
# Import model and metric modules so classes referenced in Fiddle configs are
# defined when configurations are parsed or deserialized.
from neuralgcm.hybrid import atmosphere  # pylint: disable=unused-import
from neuralgcm.ml import learned_update  # pylint: disable=unused-import
from terrax.core import parallelism
# pylint: disable=unused-import
from terrax.core import spherical_harmonics
from terrax.metrics import aggregation
from terrax.metrics import base
from terrax.metrics import deterministic_losses
from terrax.metrics import deterministic_metrics
from terrax.metrics import evaluators
from terrax.metrics import probabilistic_losses
from terrax.metrics import probabilistic_metrics
from terrax.metrics import weighting
# pylint: enable=unused-import
from terrax.training import trainer


_FIDDLE_CONFIG = fdl_flags.DEFINE_fiddle_config(
    'fiddle_config',
    default=None,
    help_string='Pure Fiddle config for RolloutTrainer.',
    allow_imports=True,
)
_EXPERIMENT_DIR = flags.DEFINE_string(
    'experiment_dir',
    default=None,
    help='Optional directory override for RolloutTrainer.experiment_dir.',
)


def run_training(
    cfg: fdl.Buildable[trainer.RolloutTrainer],
    experiment_dir: str | None = None,
) -> None:
  """Finalizes runtime fields on a copy of `cfg` and runs rollout training."""
  cfg = copy.deepcopy(cfg)
  if experiment_dir is not None:
    cfg.experiment_dir = experiment_dir
  if (
      'experiment_dir' not in fdl.ordered_arguments(cfg)
      or not cfg.experiment_dir
  ):
    raise ValueError(
        '`experiment_dir` must be non-empty on the RolloutTrainer config or '
        'specified via --experiment_dir.'
    )

  # Serialize an unsharded copy of the model config for checkpoint metadata.
  no_sharding_model_cfg = parallelism.update_mesh_properties(
      cfg.model, mesh=None
  )
  cfg.checkpoint_config.model_config_str = serialization.dump_json(
      no_sharding_model_cfg
  )

  default_metrics_dir = os.path.join(cfg.experiment_dir, 'metrics')
  if (
      'online_metrics_saver' not in fdl.ordered_arguments(cfg)
      or cfg.online_metrics_saver is None
  ):
    cfg.online_metrics_saver = fdl.Config(
        trainer.JsonMetricsSaver,
        metrics_dir=default_metrics_dir,
    )
  else:
    saver_cfg = cfg.online_metrics_saver
    if (
        isinstance(saver_cfg, fdl.Buildable)
        and fdl.get_callable(saver_cfg) is trainer.JsonMetricsSaver
        and (
            'metrics_dir' not in fdl.ordered_arguments(saver_cfg)
            or not saver_cfg.metrics_dir
        )
    ):
      saver_cfg.metrics_dir = default_metrics_dir

  # Dinosaur uses shmap over full SPMD mesh specs while RolloutTrainer also
  # vmaps over batch and ensemble axes.
  jax.config.update('jax_disable_vmap_shmap_error', True)
  rollout_trainer = fdl.build(cfg)
  rollout_trainer.run_training()


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  cfg = _FIDDLE_CONFIG.value
  if cfg is None:
    raise app.UsageError('Must specify --fiddle_config.')
  run_training(cfg, experiment_dir=_EXPERIMENT_DIR.value)


if __name__ == '__main__':
  jax.config.update('jax_traceback_filtering', 'off')
  app.run(main)
