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
"""Defines API for synthetic tasks used for benchmarking learned components."""

from __future__ import annotations

import abc
import typing

import coordax as cx
from flax import nnx
import jax
import jax.numpy as jnp
from neuralgcm.experimental.core import module_utils
from neuralgcm.experimental.core import transforms
from neuralgcm.experimental.metrics import base as metrics_base
from neuralgcm.experimental.metrics import evaluators
import numpy as np
import optax
import tqdm

LearnedTransformFactory = typing.Callable[
    [dict[str, cx.Field], dict[str, cx.Coordinate]], nnx.Module
]


class SyntheticTask(nnx.Module, abc.ABC):
  """Base class for synthetic learning tasks."""

  @property
  @abc.abstractmethod
  def batch_axis(self) -> cx.Coordinate:
    """Returns the batch axis for the task."""

  @property
  @abc.abstractmethod
  def input_shapes(self) -> dict[str, cx.Field]:
    """Returns shapes of input fields."""

  @property
  @abc.abstractmethod
  def target_split_axes(self) -> dict[str, cx.Coordinate]:
    """Returns split axes for target fields."""

  @abc.abstractmethod
  def sample_batch(
      self, rng: jax.Array
  ) -> tuple[dict[str, cx.Field], dict[str, cx.Field]]:
    """Generates a batch of inputs and targets.

    Args:
      rng: Random number generator key.

    Returns:
      A tuple containing 'inputs' and 'targets' dictionaries.
    """

  @property
  def generalization_tasks(self) -> dict[str, SyntheticTask]:
    """Returns a dictionary of generalization tasks."""
    return {}

  @property
  def loss_evaluator(self) -> evaluators.Evaluator | None:
    """Returns the evaluator used for computing training loss."""
    return None

  @property
  def metrics_evaluator(self) -> evaluators.Evaluator | None:
    """Returns the evaluator used for computing evaluation metrics."""
    return None


def run_training(
    task: SyntheticTask,
    model_factory: LearnedTransformFactory,
    optimizer_def: optax.GradientTransformation,
    train_steps: int,
    eval_every: int,
    n_eval_batches: int,
    rng: jax.Array,
    loss_evaluator: evaluators.Evaluator | None = None,
    metrics_evaluator: evaluators.Evaluator | None = None,
    run_full_evaluation: bool = False,
    select_generalizations: typing.Sequence[str] | None = None,
    calibration_steps: int = 0,
) -> tuple[nnx.Module, dict[str, typing.Any]]:
  """Runs training loop and evaluation.

  Args:
    task: SyntheticTask instance.
    model_factory: Factory function to create the model.
    optimizer_def: Optax optimizer definition.
    train_steps: Number of training steps.
    eval_every: Frequency of evaluation.
    n_eval_batches: Number of batches for evaluation.
    rng: Random number generator key.
    loss_evaluator: Optional evaluator for computing training loss. If None,
      task.loss_evaluator is used.
    metrics_evaluator: Optional evaluator for computing evaluation metrics. If
      None, task.metrics_evaluator is used.
    run_full_evaluation: If True, include generalization tasks in eval runs.
    select_generalizations: Optional sequence of generalization tasks to run.
    calibration_steps: Number of steps for calibration.

  Returns:
    Trained model and metrics history.
  """
  model = model_factory(task.input_shapes, task.target_split_axes)

  if calibration_steps > 0:
    stats_modules = module_utils.retrieve_subclass_modules(
        model, transforms.StreamNorm
    )
    if stats_modules:
      for m in stats_modules:
        m.update_stats = True

      @nnx.jit
      def calibrate_step(model, inputs):
        model(inputs)

      for i in range(calibration_steps):
        step_rng = jax.random.fold_in(rng, i)
        inputs, _ = task.sample_batch(step_rng)
        calibrate_step(model, inputs)
      for m in stats_modules:
        m.update_stats = False

  optimizer = nnx.Optimizer(model, optimizer_def, wrt=nnx.Param)

  loss_evaluator = loss_evaluator or task.loss_evaluator
  metrics_evaluator = metrics_evaluator or task.metrics_evaluator

  if loss_evaluator is None:
    raise ValueError(
        'loss_evaluator must be provided either as argument or task property.'
    )
  if metrics_evaluator is None:
    metrics_evaluator = loss_evaluator

  def compute_loss(model, inputs, targets):
    predictions = model(inputs)
    loss = loss_evaluator.evaluate_total(predictions, targets)
    return loss.data, predictions

  @nnx.jit
  def train_step(model, optimizer, inputs, targets):
    grad_fn = nnx.value_and_grad(compute_loss, has_aux=True)
    (loss, predictions), grads = grad_fn(model, inputs, targets)
    optimizer.update(model, grads)
    return loss, predictions

  tasks_to_eval = {'train': task}
  if run_full_evaluation:
    generalizations = task.generalization_tasks
    if select_generalizations is not None:
      generalizations = {
          k: v
          for k, v in generalizations.items()
          if k in select_generalizations
      }
    tasks_to_eval.update(generalizations)

  eval_steps = {}
  evaluators_dict = {}
  for t_name, t_instance in tasks_to_eval.items():
    t_evaluator = t_instance.metrics_evaluator or metrics_evaluator
    evaluators_dict[t_name] = t_evaluator

    def make_eval_step(evaluator):
      @nnx.jit
      def step_fn(model, inputs, targets):
        predictions = model(inputs)
        if isinstance(predictions, cx.Field) and len(targets) == 1:
          predictions = {next(iter(targets.keys())): predictions}
        agg_states = evaluator.evaluate(predictions, targets)
        loss_agg_states = loss_evaluator.evaluate(predictions, targets)
        return agg_states, loss_agg_states

      return step_fn

    eval_steps[t_name] = make_eval_step(t_evaluator)

  train_rng, eval_rng = jax.random.split(rng)

  train_loss_history = []
  eval_metrics_history = {}

  for step in tqdm.tqdm(range(train_steps), desc='Training'):
    step_rng = jax.random.fold_in(train_rng, step)
    inputs, targets = task.sample_batch(step_rng)
    loss, _ = train_step(model, optimizer, inputs, targets)

    # Record training loss
    train_loss_history.append(float(loss))

    if step % eval_every == 0:
      for t_name, t_instance in tasks_to_eval.items():
        agg_states = None
        loss_agg_states = None
        if t_name == 'train':
          t_eval_rng = eval_rng
        else:
          t_eval_rng = jax.random.fold_in(eval_rng, hash(t_name) % 1_000_000)

        for i in range(n_eval_batches):  # aggregate metrics.
          eval_step_rng = jax.random.fold_in(
              t_eval_rng, step * n_eval_batches + i
          )
          eval_inputs, eval_targets = t_instance.sample_batch(eval_step_rng)
          step_agg_states, step_loss_agg_states = eval_steps[t_name](
              model, eval_inputs, eval_targets
          )

          if agg_states is None:
            agg_states = step_agg_states
            loss_agg_states = step_loss_agg_states
          else:
            for k in agg_states:
              agg_states[k] = agg_states[k] + step_agg_states[k]
            for k in loss_agg_states:
              loss_agg_states[k] = loss_agg_states[k] + step_loss_agg_states[k]

        metric_values = evaluators_dict[t_name].evaluate_metrics(
            {}, {}, agg_states
        )
        eval_loss = loss_evaluator.evaluate_total({}, {}, loss_agg_states)
        metric_values['loss'] = {'total': eval_loss}

        prefix = f'{t_name}/'
        for metric_key, metric_dict in metric_values.items():  # record metrics.
          for var_name, value in metric_dict.items():
            history_key = f'{prefix}{metric_key}/{var_name}'
            if history_key not in eval_metrics_history:
              eval_metrics_history[history_key] = []
            eval_metrics_history[history_key].append(value)

  # Stack metrics into cx.Field.
  metrics_fields = {}
  train_step_axis = cx.LabeledAxis('train_step', np.arange(train_steps))
  metrics_fields['train_loss'] = cx.field(
      np.stack(train_loss_history), train_step_axis
  )
  eval_steps = np.arange(0, train_steps, eval_every)
  train_step_evals_axis = cx.LabeledAxis('eval_step', eval_steps)

  stack_fn = cx.cmap(jnp.stack)
  for k, history in eval_metrics_history.items():
    stacked_field = stack_fn(history)
    metrics_fields[k] = stacked_field.tag(train_step_evals_axis)

  return model, metrics_fields


def run_evaluation(
    task: SyntheticTask,
    model: nnx.Module,
    n_batches: int,
    rng: jax.Array,
    metrics_evaluator: evaluators.Evaluator | None = None,
    select_generalizations: typing.Sequence[str] | None = None,
    n_examples_to_save: int | None = None,
) -> dict[str, typing.Any]:
  """Runs evaluation loop.

  Args:
    task: SyntheticTask instance.
    model: Trained model.
    n_batches: Number of batches for evaluation.
    rng: Random number generator key.
    metrics_evaluator: Optional evaluator. If None, task.metrics_evaluator used.
    select_generalizations: Optional sequence of generalization tasks to run.
    n_examples_to_save: If set, optionally save up to this many examples of
      inputs, targets and predictions for each task.

  Returns:
    Dictionary of computed metrics and optionally saved examples.
  """
  generalizations = task.generalization_tasks
  if select_generalizations is not None:
    generalizations = {
        k: v for k, v in generalizations.items() if k in select_generalizations
    }
  tasks = {'train': task} | generalizations
  result = {}

  for task_name, task_instance in tasks.items():
    metrics_evaluator_to_use = (
        metrics_evaluator or task_instance.metrics_evaluator
    )
    if metrics_evaluator_to_use is None:
      raise ValueError(
          f'metrics_evaluator must be provided for task {task_name}.'
      )

    @nnx.jit
    def eval_step(model, inputs, targets):
      predictions = model(inputs)
      if isinstance(predictions, cx.Field) and len(targets) == 1:
        predictions = {next(iter(targets.keys())): predictions}
      agg_states = metrics_evaluator_to_use.evaluate(  # pylint: disable=cell-var-from-loop
          predictions, targets
      )
      return agg_states

    agg_states = None
    task_rng = jax.random.fold_in(rng, hash(task_name) % 1_000_000)

    for i in range(n_batches):
      step_rng = jax.random.fold_in(task_rng, i)
      inputs, targets = task_instance.sample_batch(step_rng)
      step_agg_states = eval_step(model, inputs, targets)
      if agg_states is None:
        agg_states = step_agg_states
      else:
        for k in agg_states:
          agg_states[k] = agg_states[k] + step_agg_states[k]

    metrics = metrics_evaluator_to_use.evaluate_metrics({}, {}, agg_states)

    prefix = f'{task_name}/'
    for metric_key, metric_dict in metrics.items():
      for var_name, value in metric_dict.items():
        result[f'{prefix}{metric_key}/{var_name}'] = value
      metric_obj = metrics_evaluator_to_use.metrics[metric_key]
      if isinstance(metric_obj, metrics_base.Loss):
        result[f'{prefix}{metric_key}/total'] = metric_obj.total(metric_dict)

    if n_examples_to_save is not None:
      has_batch = task.batch_axis.ndim > 0
      batch_size = 1 if not has_batch else task.batch_axis.shape[0]
      if has_batch:
        combine_fn = cx.cmap(jnp.concatenate)
      else:
        combine_fn = cx.cmap(jnp.stack)

      @nnx.jit
      def predict(model, inputs):
        return model(inputs)

      saved_inputs, saved_targets, saved_predictions = [], [], []
      collected = 0
      i = 0
      example_rng = jax.random.fold_in(task_rng, 1_000_000)

      while collected < n_examples_to_save:
        step_rng = jax.random.fold_in(example_rng, i)
        inputs, targets = task_instance.sample_batch(step_rng)
        predictions = predict(model, inputs)

        saved_inputs.append(inputs)
        saved_targets.append(targets)
        saved_predictions.append(predictions)
        collected += batch_size
        i += 1

      def concat_and_slice(fields_list):
        if not fields_list:
          return {}
        truncate = lambda f: f.isel(batch=slice(0, n_examples_to_save))
        untag_batch = lambda f: cx.untag(f, task.batch_axis)
        fn = lambda *args: truncate(combine_fn(untag_batch(args)).tag('batch'))
        return jax.tree.map(fn, *fields_list, is_leaf=cx.is_field)

      for k, v in concat_and_slice(saved_inputs).items():
        result[f'{prefix}inputs/{k}'] = v
      for k, v in concat_and_slice(saved_targets).items():
        result[f'{prefix}targets/{k}'] = v
      for k, v in concat_and_slice(saved_predictions).items():
        result[f'{prefix}predictions/{k}'] = v

  return result
