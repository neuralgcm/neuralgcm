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
"""Helper methods for constructing augmented trajectory functions."""

import dataclasses
import functools
from typing import Any, Callable, Sequence, Tuple
from dinosaur import coordinate_systems
from dinosaur import pytree_utils
from dinosaur import typing
import haiku as hk
import jax
import jax.numpy as jnp

Array = typing.Array
DynamicalSystem = Any  # to prevent circular dependency on model_builder
Pytree = typing.Pytree

tree_map = jax.tree_util.tree_map
tree_leaves = jax.tree_util.tree_leaves

# Linter confused by wrapped functions
# pylint: disable=g-bare-generic


def with_preprocessing(
    fn: Callable[..., Pytree],
    preprocess_fn: Callable,
) -> Callable[..., Pytree]:
  """Generates a function that computes `fn` on `preprocess_fn(x)`."""
  @functools.wraps(fn)
  def apply_fn(x, *args, **kwargs):
    return fn(preprocess_fn(x), *args, **kwargs)

  return apply_fn


def with_post_processing(
    fn: Callable[..., Pytree],
    post_process_fn: Callable,
) -> Callable[..., Pytree]:
  """Generates a function that applies `post_process_fn` to outputs of `fn`."""
  @functools.wraps(fn)
  def apply_fn(*args, **kwargs):
    return post_process_fn(*fn(*args, **kwargs))

  return apply_fn


def with_forcing(
    fn: Callable[..., Pytree],
    forcing_fn: typing.ForcingFn,
    forcing_data: typing.ForcingData,
) -> Callable[..., Pytree]:
  """Converts fn(x, forcing) to fn(x)."""
  # evaluates forcing=forcing_fn(forcing_data, x.sim_time)
  # when x does not have sim_time, forcing_fn will get sim_time=None
  @functools.wraps(fn)
  def wrapped(x, forcing_fn=forcing_fn):
    # handle dataclass or dict for state data
    if dataclasses.is_dataclass(x):
      if isinstance(x, typing.ModelState):
        sim_time = dataclasses.asdict(x.state).get('sim_time', None)
      else:
        sim_time = dataclasses.asdict(x).get('sim_time', None)
    else:
      sim_time = x.get('sim_time', None)
    # handle sim_time of ndim 0 or 1
    if sim_time is not None:
      sim_time = jax.numpy.asarray(sim_time)
      if sim_time.ndim:
        forcing_fn = jax.vmap(forcing_fn, in_axes=(None, 0))
    forcing = forcing_fn(forcing_data, sim_time)  # pyrefly: ignore[bad-argument-type]
    return fn(x, forcing=forcing)
  return wrapped


def with_split_input(
    fn: Callable[..., Pytree],
    split_index: int,
    time_axis: int = 0,
) -> Callable[..., Pytree]:
  """Decorates `fn` to be evaluated on first `split_index` time slices.

  The returned function is a generalization to pytrees of the function:
  `fn(x[:split_index], *args, **kwargs)`

  Args:
    fn: function to be transformed.
    split_index: number of input elements along the time axis to use.
    time_axis: axis corresponding to time dimension in `x` to decorated `fn`.

  Returns:
    decorated `fn` that is evaluated on only `split_index` first time slices of
    provided inputs.
  """
  @functools.wraps(fn)
  def apply_fn(x, *args, **kwargs):
    init, _ = pytree_utils.split_along_axis(x, split_index, axis=time_axis)
    return fn(init, *args, **kwargs)

  return apply_fn


def with_input_included(
    trajectory_fn: typing.TrajectoryFn,
    time_axis: int = 0,
    num_last_input_frames_to_trim: int = 0,
) -> typing.TrajectoryFn:
  """Returns a `trajectory_fn` that concatenates inputs `x` to trajectory."""
  if num_last_input_frames_to_trim > 0:
    num_last_input_frames_to_trim = -num_last_input_frames_to_trim
  else:
    num_last_input_frames_to_trim = None  # pyrefly: ignore[bad-assignment]
  inputs_time_slice = slice(None, num_last_input_frames_to_trim)
  @functools.wraps(trajectory_fn)
  def _trajectory(x, *args, **kwargs):
    final, unroll = trajectory_fn(x, *args, **kwargs)
    x_concat = pytree_utils.slice_along_axis(x, time_axis, inputs_time_slice)
    return final, pytree_utils.concat_along_axis([x_concat, unroll], time_axis)

  return _trajectory


def trajectory_with_inputs_and_forcing(
    model: DynamicalSystem,
    num_init_frames: int,
    start_with_input: bool = False,
) -> typing.TrajectoryFn:
  """Returns trajectory_fn that comuptes model trajectory from target data.

  Wraps the default model.trajectory_fn to operate on data representation. It
  corresponds to slicing `num_init_frames` from the inputs, encoding and
  unrolling the trajectory.

  Args:
    model: model of a dynamical system used to obtain the trajectory.
    num_init_frames: number of time frames used from the physics trajectory to
      initialize the model state.
    start_with_input: whether the firest decoded step in the output trajectory
      should correspond to last input time or first future output.

  Returns:
    Trajectory function that operates on target data trajectory by encoding
    the `initial_frames` inputs and unrolls trajectory in a model space.
  """
  def _trajectory_fn(x, forcing_data, outer_steps, inner_steps=1):

    # configure the model.trajectory function with a decoder on the output.
    trajectory_fn = functools.partial(
        model.trajectory,
        outer_steps=outer_steps,
        inner_steps=inner_steps,
        forcing_data=forcing_data,
        start_with_input=start_with_input)
    # add preprocessing to encode input to model state.
    encode_fn = with_forcing(model.encode, model.forcing_fn, forcing_data)
    trajectory_fn = with_preprocessing(trajectory_fn, encode_fn)
    trajectory_fn = with_split_input(trajectory_fn, num_init_frames)
    return trajectory_fn(x)

  return _trajectory_fn


def trajectory_with_inputs_and_forcing_and_stop_gradients(
    model: DynamicalSystem,
    num_init_frames: int,
    start_with_input: bool = False,
    stop_gradient_outer_steps: Sequence[int] = (),
) -> typing.TrajectoryFn:
  """Returns trajectory_fn that comuptes model trajectory from target data.

  This extension of `trajectory_with_inputs_and_forcing` allows adding stop
  gradients to the trajectory at designated steps. For example, if
  `stop_gradient_outer_steps = [2]`, then gradients along the trajectory stop
  at t=2. This does not mean that gradients with respect to X[2] will be zero.
  It simply means that, for t > 2, gradients of X[t] with respect to X[2] will
  be zero.

  Wraps the default model.trajectory_fn to operate on data representation. It
  corresponds to slicing `num_init_frames` from the inputs, encoding and
  unrolling the trajectory.

  Args:
    model: model of a dynamical system used to obtain the trajectory.
    num_init_frames: number of time frames used from the physics trajectory to
      initialize the model state.
    start_with_input: whether the firest decoded step in the output trajectory
      should correspond to last input time or first future output.
    stop_gradient_outer_steps: Tuple (possibly empty) indicating outer steps at
      which to place stop gradients.

  Returns:
    Trajectory function that operates on target data trajectory by encoding
    the `initial_frames` inputs and unrolls trajectory in a model space.
    Decoding is not done by this function.
  """
  stop_gradient_outer_steps = list(sorted(stop_gradient_outer_steps))
  if num_init_frames != 1:
    raise ValueError(f'{num_init_frames=} is not supported yet.')

  if stop_gradient_outer_steps and min(stop_gradient_outer_steps) <= 0:
    raise ValueError(
        f'{stop_gradient_outer_steps=} contained non-positive values'
    )

  expand_dim0 = lambda tree: tree_map(lambda x_i: x_i[jnp.newaxis], tree)
  concat_dim0 = lambda trees: pytree_utils.concat_along_axis(trees, axis=0)
  slice_dim0 = lambda tree, idx: pytree_utils.slice_along_axis(
      tree, axis=0, idx=idx
  )

  def concat_trajectories_with_stop_grads(
      x, forcing_data, outer_steps, inner_steps=1
  ):
    if (
        stop_gradient_outer_steps
        and max(stop_gradient_outer_steps) > outer_steps
    ):
      raise ValueError(
          f'{stop_gradient_outer_steps=} contained values > {outer_steps=}'
      )
    outer_steps_seq = list(stop_gradient_outer_steps)
    if not outer_steps_seq or outer_steps_seq[-1] != outer_steps:
      outer_steps_seq.append(outer_steps)

    # The first leg needs to encode the input. So use
    # trajectory_with_inputs_and_forcing, which does the encoding.
    final_state, first_leg = trajectory_with_inputs_and_forcing(
        model,
        num_init_frames=num_init_frames,
        start_with_input=start_with_input,
    )(
        x,
        forcing_data=forcing_data,
        outer_steps=outer_steps_seq[0],
        inner_steps=inner_steps,
    )

    # At this point, sections contains times [0, ..., outer_steps_seq[0]]
    sections = [
        first_leg,
    ]

    # Subsequent legs do not need encoding, so use model.trajectory directly.
    trajectory_fn = functools.partial(
        model.trajectory,
        inner_steps=inner_steps,
        forcing_data=forcing_data,
        start_with_input=start_with_input,
    )
    for i in range(1, len(outer_steps_seq)):
      # outer_steps_seq[-1] may or may not be in stop_gradient_outer_steps.
      # The other steps will be by construction.
      assert set(outer_steps_seq[:-1]).issubset(stop_gradient_outer_steps)
      stop_grad_at_start = outer_steps_seq[i - 1] in stop_gradient_outer_steps

      initial_state = final_state

      # this_leg contains times [outer_steps_seq[0]+1, ..., outer_steps_seq[1]]
      final_state, this_leg = trajectory_fn(
          jax.lax.stop_gradient(initial_state)
          if stop_grad_at_start
          else initial_state,
          outer_steps=outer_steps_seq[i] - outer_steps_seq[i - 1],
      )

      if stop_grad_at_start and start_with_input:
        # Replace the initial point that had a stop gradient on it.
        this_leg = concat_dim0([
            expand_dim0(initial_state),
            slice_dim0(this_leg, idx=slice(1, None)),
        ])
      sections.append(this_leg)

    return final_state, concat_dim0(sections)

  return concat_trajectories_with_stop_grads


def decoded_trajectory_with_forcing(
    model: DynamicalSystem,
    start_with_input: bool = False,
) -> typing.TrajectoryFn:
  """Returns trajectory_fn that comuptes decoded trajectory values.

  Args:
    model: model of a dynamical system used to obtain the trajectory.
    start_with_input: whether the firest decoded step in the output trajectory
      should correspond to last input time or first future output.

  Returns:
    Trajectory function that additionally decodes trajectory values.
  """
  def _trajectory_fn(x, forcing_data, outer_steps, inner_steps=1):

    # configure the model.trajectory function with a decoder on the output.
    trajectory_fn = functools.partial(
        model.trajectory,
        forcing_data=forcing_data,
        post_process_fn=with_forcing(model.decode,
                                     model.forcing_fn, forcing_data),
        start_with_input=start_with_input)
    return trajectory_fn(x, outer_steps, inner_steps)

  return _trajectory_fn


def decoded_trajectory_with_inputs_and_forcing(
    model: DynamicalSystem,
    num_init_frames: int,
    start_with_input: bool = False,
) -> typing.TrajectoryFn:
  """Returns trajectory_fn operating on decoded input and forcing data.

  The returned function uses `num_init_frames` of the physics space trajectory
  provided as an input to model.encode_fn to initialize the model state, then
  unrolls the trajectory of specified length that is decoded to the physics
  space using `model.decode_fn`.

  Args:
    model: model of a dynamical system used to obtain the trajectory.
    num_init_frames: number of time frames used from the physics trajectory to
      initialize the model state.
    start_with_input: whether the firest decoded step in the output trajectory
      should correspond to last input time or first future output.

  Returns:
    Trajectory function that operates on physics space trajectories
    and returns unrolls in physics space.
  """
  def _trajectory_fn(x, forcing_data, outer_steps, inner_steps=1):

    # configure the model.trajectory function with a decoder on the output.
    trajectory_fn = decoded_trajectory_with_forcing(model, start_with_input)
    trajectory_fn = functools.partial(
        trajectory_fn,
        forcing_data=forcing_data,
        outer_steps=outer_steps,
        inner_steps=inner_steps)
    # add preprocessing to encode input to model state.
    trajectory_fn = with_preprocessing(
        trajectory_fn, with_forcing(model.encode,
                                    model.forcing_fn, forcing_data))
    # concatenate input trajectory to output trajectory for easier comparison.
    trajectory_fn = with_input_included(
        trajectory_fn, num_last_input_frames_to_trim=int(start_with_input))
    # make trajectories operate on full examples by splitting the init.
    trajectory_fn = with_split_input(trajectory_fn, num_init_frames)
    return trajectory_fn(x)

  return _trajectory_fn


def process_trajectory(
    input_trajectory: Pytree,
    process_fn: Callable[[Pytree], Pytree],
) -> Pytree:
  """Processes trajectory by applying `process_fn` along time axis."""
  step_fn = lambda c, x: tuple([None, hk.remat(process_fn)(x)])
  _, out = hk.scan(step_fn, None, xs=input_trajectory)
  return out


def _maybe_to_nodal_with_physics_sharding(x, /, coords):
  x = coordinate_systems.maybe_to_nodal(x, coords)
  x = coords.with_physics_sharding(x)
  return x


def _maybe_to_modal_with_physics_sharding(x, /, coords):
  x = coordinate_systems.maybe_to_modal(x, coords)
  x = coords.with_physics_sharding(x)
  return x


def compute_prediction_representations(
    predicted_trajectory: typing.Pytree,
    forcing_data: typing.ForcingData,
    model: DynamicalSystem,
) -> typing.TrajectoryRepresentations:
  """Computes TrajectoryRepresentations for predicted trajectory.

  Args:
    predicted_trajectory: predictions on `model.coords` coordinates.
    forcing_data: forcing data to be used for encode/decode transformations.
    model: model used for conversion between representations.

  Returns:
    `TrajectoryRepresentations` for predictions.
  """
  decode_fn = with_forcing(model.decode, model.forcing_fn, forcing_data)
  data_to_nodal = functools.partial(
      _maybe_to_nodal_with_physics_sharding, coords=model.output_coords)
  data_to_modal = functools.partial(
      _maybe_to_modal_with_physics_sharding, coords=model.output_coords)
  model_to_nodal = functools.partial(
      _maybe_to_nodal_with_physics_sharding, coords=model.coords)
  model_to_modal = functools.partial(
      _maybe_to_modal_with_physics_sharding, coords=model.coords)
  predicted_data_trajectory = process_trajectory(
      predicted_trajectory, decode_fn)
  # Note: we pass original prediction to the decoder, but use dict for outputs.
  if isinstance(predicted_trajectory, typing.ModelState):
    predicted_trajectory = predicted_trajectory.state
  if dataclasses.is_dataclass(predicted_trajectory):
    # Losses operate on dicts: convert struct to dict if needed.
    predicted_trajectory = predicted_trajectory.asdict()
  return typing.TrajectoryRepresentations(
      data_nodal_trajectory=process_trajectory(
          predicted_data_trajectory, data_to_nodal),
      data_modal_trajectory=process_trajectory(
          predicted_data_trajectory, data_to_modal),
      model_nodal_trajectory=process_trajectory(
          predicted_trajectory, model_to_nodal),
      model_modal_trajectory=process_trajectory(
          predicted_trajectory, model_to_modal),
  )


def compute_target_representations(
    target_trajectory: typing.Pytree,
    forcing_data: typing.ForcingData,
    model: DynamicalSystem,
) -> typing.TrajectoryRepresentations:
  """Computes TrajectoryRepresentations for target trajectory.

  Note: currently this method only supports models that use a single time slice
  for initialization.

  Args:
    target_trajectory: target trajectory on `model.output_coords` coordinates.
    forcing_data: forcing data to be used for encode/decode transformations.
    model: model used for conversion between representations.

  Returns:
    `TrajectoryRepresentations` for predictions.
  """
  encode_slice_fn = with_forcing(model.encode, model.forcing_fn, forcing_data)
  encode_fn = lambda tree: encode_slice_fn(  # pylint: disable=g-long-lambda.
      jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), tree))
  data_to_nodal = functools.partial(
      _maybe_to_nodal_with_physics_sharding, coords=model.output_coords)
  data_to_modal = functools.partial(
      _maybe_to_modal_with_physics_sharding, coords=model.output_coords)
  model_to_nodal = functools.partial(
      _maybe_to_nodal_with_physics_sharding, coords=model.coords)
  model_to_modal = functools.partial(
      _maybe_to_modal_with_physics_sharding, coords=model.coords)
  target_model_trajectory = process_trajectory(
      target_trajectory, encode_fn)
  if isinstance(target_model_trajectory, typing.ModelState):
    target_model_trajectory = target_model_trajectory.state
  if dataclasses.is_dataclass(target_model_trajectory):
    # Losses operate on dicts: convert struct to dict if needed.
    target_model_trajectory = target_model_trajectory.asdict()
  return typing.TrajectoryRepresentations(
      data_nodal_trajectory=process_trajectory(
          target_trajectory, data_to_nodal),
      data_modal_trajectory=process_trajectory(
          target_trajectory, data_to_modal),
      model_nodal_trajectory=process_trajectory(
          target_model_trajectory, model_to_nodal),
      model_modal_trajectory=process_trajectory(
          target_model_trajectory, model_to_modal),
  )


def compute_prediction_and_target_representations(
    predicted_model_trajectory: typing.Pytree,
    target_data_trajectory: typing.Pytree,
    forcing_data: typing.ForcingData,
    model: DynamicalSystem,
) -> Tuple[typing.TrajectoryRepresentations, typing.TrajectoryRepresentations]:
  """Computes TrajectoryRepresentations for predicted and target trajectories.

  Note: currently this method only supports models that use a single time slice
  for initialization. While computing all terms seems wasteful, once jit-ed
  all unused computations are optimized away. It is also tempting to compute
  all representations at once, but as of 2023-02-28 compiler doesn't manage to
  remove unused computation from a single primitive.

  Args:
    predicted_model_trajectory: predictions on `model.coords` coordinates.
    target_data_trajectory: target data on `model.output_coords` coordinates.
    forcing_data: forcing data to be used for encode/decode transformations.
    model: model used for conversion between representations.

  Returns:
    Tuple of `TrajectoryRepresentations` for predictions and targets.
  """
  prediction_representations = compute_prediction_representations(
      predicted_model_trajectory, forcing_data, model)
  target_representations = compute_target_representations(
      target_data_trajectory, forcing_data, model)
  return prediction_representations, target_representations


@jax.custom_jvp
def safe_sqrt(x: Array) -> jax.Array:
  """Sqrt(x) with gradient = 0 for x near 0."""
  return jnp.sqrt(x)


@safe_sqrt.defjvp
def safe_sqrt_jvp(
    primals: Array,
    tangents: Array,
) -> tuple[jax.Array, jax.Array]:
  (x,) = primals
  (x_dot,) = tangents
  primal_out = safe_sqrt(x)
  eps = jnp.finfo(x.dtype).eps
  safe_x = jnp.where(x > eps, x, 1.0)
  tangent_out = jnp.where(x > eps, x_dot / (2 * safe_sqrt(safe_x)), 0)
  return primal_out, tangent_out
