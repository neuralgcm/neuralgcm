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
"""Tests for models constructed via global_circulation.ml.model_builder.py."""

import functools
from absl.testing import absltest
from absl.testing import parameterized
from dinosaur import coordinate_systems
from dinosaur import layer_coordinates
from dinosaur import primitive_equations
from dinosaur import primitive_equations_states
from dinosaur import scales
from dinosaur import shallow_water
from dinosaur import shallow_water_states
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
from dinosaur import typing
from dinosaur import xarray_utils
import gin
import haiku as hk
import jax
import jax.numpy as jnp
from neuralgcm import model_builder
from neuralgcm import physics_specifications
import numpy as np

units = scales.units


class ShallowWaterTest(parameterized.TestCase):
  """Tests that ML-configured ShallowWater solver produces expected shapes."""

  def setUp(self):
    """Initializes gin-configs used in tests."""
    super().setUp()
    gin.enter_interactive_mode()
    self.physics_gin_config = '\n'.join([
        'get_physics_specs.construct_fn = @shallow_water_specs_constructor',
        'shallow_water_specs_constructor.density_vals = [1.]',
    ])
    self.model_gin_config = '\n'.join([
        'WhirlModel.model_cls = @ModularStepModel',
        'ModularStepModel.advance_module = @EquationStep',
        'ModularStepModel.encoder_module = %ENCODER_MODULE',
        'ModularStepModel.decoder_module = %DECODER_MODULE',
        'EquationStep.time_integrator = %INTEGRATOR_MODULE',
        'EquationStep.filter_modules = %FILTERS',
        'EquationStep.equation_module = @ShallowWaterEquations',
    ])

  @parameterized.parameters(
      # TODO(pnorgaard) Reenable after making forcing work with leapfrog
      # dict(encoder_module='@ShallowWaterLeapfrogEncoder',
      #      integrator_module='@semi_implicit_leapfrog',
      #      decoder_module='@LeapfrogSliceDictDecoder',
      #      filter_modules='(@ExponentialLeapfrogFilter,)',
      #      model_state_fn=lambda x: (x, x)),
      dict(
          encoder_module='@ShallowWaterStateEncoder',
          integrator_module='@imex_rk_sil3',
          decoder_module='@StateToDictDecoder',
          filter_modules='(@ExponentialFilter,)',
          model_state_fn=lambda x: x,
      ),
  )
  def test_shapes_of_model_function(
      self,
      encoder_module,
      integrator_module,
      decoder_module,
      filter_modules,
      model_state_fn,  # constructs expected model state from `State` snapshot.
  ):
    """Tests output shapes of encode, advance, decode, trajectory functions."""
    gin_bindings = '\n'.join([
        f'ENCODER_MODULE = {encoder_module}',
        f'INTEGRATOR_MODULE = {integrator_module}',
        f'DECODER_MODULE = {decoder_module}',
        f'FILTERS = {filter_modules}',
    ])
    gin.clear_config()
    gin.parse_config(self.physics_gin_config)
    gin.parse_config(self.model_gin_config)
    gin.parse_config(gin_bindings)

    grid = spherical_harmonic.Grid.with_wavenumbers(32)
    vertical_grid = layer_coordinates.LayerCoordinates(1)
    coords = coordinate_systems.CoordinateSystem(grid, vertical_grid)
    dt = 1e-3
    physics_specs = physics_specifications.get_physics_specs()

    initial_state_fn, aux_features = (
        shallow_water_states.barotropic_instability_tc(coords, physics_specs)
    )
    instant_state = initial_state_fn(jax.random.PRNGKey(42))
    model = model_builder.WhirlModel(coords, dt, physics_specs, aux_features)

    # model state is a pair of instant states for leapfrog integration.
    state = model_state_fn(instant_state)
    model_state = typing.ModelState(state)
    # data state is an instantenious configuration;
    # decoder should map model_state to data_state;
    data_state = instant_state.asdict()
    # state and data trajectories have extra time dimensions;
    stack_fn = lambda *args: jnp.stack(args)
    traj_length = 3
    state_trajectory = jax.tree_util.tree_map(
        stack_fn, *([model_state] * traj_length)
    )
    input_length = 2  # encoder takes trajectory as input for assimilation.
    data_trajectory = jax.tree_util.tree_map(
        stack_fn, *([data_state] * input_length)
    )
    forcing_data = None
    forcing = None

    rng_key = jax.random.PRNGKey(42)
    params = model.init_params(rng_key, data_trajectory, forcing_data)

    get_shape_fn = lambda tree: jax.tree_util.tree_map(np.shape, tree)

    with self.subTest('encoder'):
      actual = jax.tree_util.tree_leaves(
          model.encode_fn(params, rng_key, data_trajectory, forcing).state
      )
      expected = jax.tree_util.tree_leaves(model_state.state)
      self.assertSameStructure(get_shape_fn(actual), get_shape_fn(expected))

    with self.subTest('decoder'):
      actual = jax.tree_util.tree_leaves(
          model.decode_fn(params, rng_key, model_state, forcing)
      )
      expected = jax.tree_util.tree_leaves(data_state)
      self.assertSameStructure(get_shape_fn(actual), get_shape_fn(expected))

    with self.subTest('advance'):
      actual = jax.tree_util.tree_leaves(
          model.advance_fn(params, rng_key, model_state, forcing).state
      )
      expected = jax.tree_util.tree_leaves(model_state.state)
      self.assertSameStructure(get_shape_fn(actual), get_shape_fn(expected))

    with self.subTest('trajectory'):

      def trajectory_fwd(x, length, forcing_data):
        return model.model_cls().trajectory(
            x, length, forcing_data=forcing_data
        )

      trajectory_model = hk.transform(trajectory_fwd)
      actual = jax.tree_util.tree_leaves(
          trajectory_model.apply(
              params, rng_key, model_state, traj_length, forcing_data
          )
      )
      expected = jax.tree_util.tree_leaves((model_state, state_trajectory))
      self.assertSameStructure(get_shape_fn(actual), get_shape_fn(expected))

  def test_equivalence_with_explicit_integration(self):
    """Tests that configured model produces identical results to `integrate`."""
    leapfrog_gin_bindings = '\n'.join([
        'ENCODER_MODULE = @ShallowWaterLeapfrogEncoder',
        'INTEGRATOR_MODULE = @semi_implicit_leapfrog',
        'DECODER_MODULE = @LeapfrogSliceDecoder',
        'FILTERS = (@ExponentialLeapfrogFilter, @RobertAsselinLeapfrogFilter)',
    ])
    gin.clear_config()
    gin.parse_config(self.physics_gin_config)
    gin.parse_config(self.model_gin_config)
    gin.parse_config(leapfrog_gin_bindings)

    grid = spherical_harmonic.Grid.with_wavenumbers(32)
    vertical_grid = layer_coordinates.LayerCoordinates(1)
    coords = coordinate_systems.CoordinateSystem(grid, vertical_grid)
    physics_specs = physics_specifications.get_physics_specs()
    dt = physics_specs.nondimensionalize(30.0 * units.s)

    initial_state_fn, aux_features = (
        shallow_water_states.barotropic_instability_tc(coords, physics_specs)
    )
    instant_state = initial_state_fn(jax.random.PRNGKey(42))

    initial_state = (instant_state, instant_state)
    initial_model_state = typing.ModelState(initial_state)
    trajectory_length = 30
    # configured model prediction.
    model = model_builder.WhirlModel(coords, dt, physics_specs, aux_features)
    trajectory_fwd = lambda x: model.model_cls().trajectory(  # pylint: disable=g-long-lambda
        x, trajectory_length, forcing_data={}
    )
    trajectory_model = hk.without_apply_rng(hk.transform(trajectory_fwd))
    key = jax.random.PRNGKey(42)
    params = trajectory_model.init(key, initial_model_state)
    trajectory_fn = jax.jit(trajectory_model.apply)
    actual_traj = trajectory_fn(params, initial_model_state)[1]
    # prediction using `shallow_water_leapfrog_trajectory` function.
    explicit_trajectory_fn = shallow_water.shallow_water_leapfrog_trajectory(
        coords=coords,
        dt=dt,
        physics_specs=physics_specs,
        mean_potential=aux_features[xarray_utils.REF_POTENTIAL_KEY],
        inner_steps=1,
        outer_steps=trajectory_length,
        filters=shallow_water.default_filters(grid, dt),
    )
    explicit_trajectory_fn = jax.jit(explicit_trajectory_fn)

    _, expected_traj = explicit_trajectory_fn(initial_state)
    for actual, expected in zip(
        jax.tree_util.tree_leaves(actual_traj),
        jax.tree_util.tree_leaves(expected_traj),
    ):
      np.testing.assert_allclose(actual, expected, atol=5e-6)


class PrimitiveEquationsTest(parameterized.TestCase):
  """Tests ML-configured primitive equation solver."""

  def setUp(self):
    """Initializes gin-configs and initial conditions used in tests."""
    super().setUp()
    gin.enter_interactive_mode()
    self.physics_gin_config = '\n'.join([
        'get_physics_specs.construct_fn = @primitive_eq_specs_constructor',
    ])
    gin.clear_config()
    gin.parse_config(self.physics_gin_config)
    self.physics_specs = physics_specifications.get_physics_specs()
    # setting up coordinates and initial state.
    grid = spherical_harmonic.Grid.with_wavenumbers(32)
    vertical_grid = sigma_coordinates.SigmaCoordinates.equidistant(4)
    self.coords = coordinate_systems.CoordinateSystem(grid, vertical_grid)
    initial_state_fn, aux_features = (
        primitive_equations_states.isothermal_rest_atmosphere(
            self.coords, self.physics_specs
        )
    )
    self.initial_instant_state = initial_state_fn(jax.random.PRNGKey(42))
    self.aux_features = aux_features
    # setting up base model configuration.
    diagnostic_state = primitive_equations.compute_diagnostic_state(
        self.initial_instant_state, self.coords
    )
    nodal_state = jax.tree_util.tree_map(
        grid.to_nodal, self.initial_instant_state
    )
    inputs_mean = jax.tree_util.tree_map(
        lambda x: float(np.mean(x)), diagnostic_state.asdict()
    )
    inputs_std = jax.tree_util.tree_map(
        lambda x: 0.1 + float(np.std(x)), diagnostic_state.asdict()
    )
    outputs_mean = jax.tree_util.tree_map(
        lambda x: float(np.mean(x)), nodal_state.asdict()
    )
    outputs_std = jax.tree_util.tree_map(
        lambda x: 0.1 + float(np.std(x)), nodal_state.asdict()
    )
    self.model_gin_config = '\n'.join([
        'WhirlModel.model_cls = @ModularStepModel',
        'ModularStepModel.advance_module = @EquationStep',
        'ModularStepModel.encoder_module = %ENCODER_MODULE',
        'ModularStepModel.decoder_module = %DECODER_MODULE',
        'EquationStep.time_integrator = %INTEGRATOR_MODULE',
        'EquationStep.filter_modules = %FILTERS',
        'EquationStep.equation_module = %EQUATION_MODULE',
        # configuring a basic values for ML modules
        (
            'composed_equations_module.equation_modules = '
            '(@PrimitiveEquations, @DirectNeuralEquations)'
        ),
        (
            'with_vertical_diffusion/composed_equations_module.equation_modules'
            ' = (@PrimitiveEquations, @VerticalDiffusion)'
        ),
        'VerticalDiffusion.timescale = "3 hours"',
        (
            'DirectNeuralEquations.modal_to_nodal_features_module = '
            '@PrimitiveEquationsDiagnosticState'
        ),
        (
            'PrimitiveEquationsDiagnosticState.features_transform_module = '
            '@inputs/ShiftAndNormalize'
        ),
        (
            'DirectNeuralEquations.tendency_transform_module = '
            '@outputs/ShiftAndNormalize'
        ),
        'DirectNeuralEquations.nodal_mapping_module = @NodalMapping',
        'NodalMapping.tower_factory = @ColumnTower',
        'ColumnTower.column_net_factory = @MlpUniform',
        'MlpUniform.num_hidden_units = 32',
        'MlpUniform.num_hidden_layers = 2',
        f'inputs/ShiftAndNormalize.shifts = {inputs_mean}',
        f'inputs/ShiftAndNormalize.scales = {inputs_std}',
        f'outputs/ShiftAndNormalize.shifts = {outputs_mean}',
        f'outputs/ShiftAndNormalize.scales = {outputs_std}',
    ])

  @parameterized.parameters(
      # TODO(pnorgaard) Reenable after making forcing work with leapfrog
      # dict(encoder_module='@PrimitiveEquationLeapfrogEncoder',
      #      equation_module='@PrimitiveEquations',
      #      integrator_module='@semi_implicit_leapfrog',
      #      decoder_module='@LeapfrogSliceDictDecoder',
      #      filter_modules='(@ExponentialLeapfrogFilter,)',
      #      model_state_fn=lambda x: (x, x)),
      dict(
          encoder_module='@PrimitiveEquationStateEncoder',
          equation_module='@PrimitiveEquations',
          integrator_module='@imex_rk_sil3',
          decoder_module='@StateToDictDecoder',
          filter_modules='(@ExponentialFilter,)',
          model_state_fn=lambda x: x,
      ),
      dict(
          encoder_module='@PrimitiveEquationStateEncoder',
          equation_module='@composed_equations_module',
          integrator_module='@imex_rk_sil3',
          decoder_module='@StateToDictDecoder',
          filter_modules='(@ExponentialFilter,)',
          model_state_fn=lambda x: x,
      ),
      dict(
          encoder_module='@PrimitiveEquationStateEncoder',
          equation_module='@with_vertical_diffusion/composed_equations_module',
          integrator_module='@imex_rk_sil3',
          decoder_module='@StateToDictDecoder',
          filter_modules='(@ExponentialFilter,)',
          model_state_fn=lambda x: x,
      ),
  )
  def test_shapes_of_model_function(
      self,
      encoder_module,
      equation_module,
      integrator_module,
      decoder_module,
      filter_modules,
      model_state_fn,  # constructs expected model state from `State` snapshot.
  ):
    """Tests output shapes of encode, advance, decode, trajectory functions."""
    gin_bindings = '\n'.join([
        f'ENCODER_MODULE = {encoder_module}',
        f'EQUATION_MODULE = {equation_module}',
        f'INTEGRATOR_MODULE = {integrator_module}',
        f'DECODER_MODULE = {decoder_module}',
        f'FILTERS = {filter_modules}',
    ])
    gin.clear_config()
    gin.parse_config(self.model_gin_config)
    gin.parse_config(gin_bindings)
    dt = self.physics_specs.nondimensionalize(30 * units.s)
    model = model_builder.WhirlModel(
        self.coords, dt, self.physics_specs, self.aux_features
    )

    # model state is a pair of instant states for leapfrog integration.
    state = model_state_fn(self.initial_instant_state)
    model_state = typing.ModelState(state)
    # data state is an instantenious configuration;
    # decoder should map model_state to data_state;
    data_state = self.initial_instant_state.asdict()
    # state and data trajectories have extra time dimensions;
    stack_fn = lambda *args: jnp.stack(args)
    traj_length = 3
    state_trajectory = jax.tree_util.tree_map(
        stack_fn, *([model_state] * traj_length)
    )
    input_length = 2  # encoder takes trajectory as input for assimilation.
    data_trajectory = jax.tree_util.tree_map(
        stack_fn, *([data_state] * input_length)
    )
    forcing_data = None
    forcing = None

    rng_key = jax.random.PRNGKey(42)
    params = model.init_params(rng_key, data_trajectory, forcing_data)
    get_shape_fn = lambda tree: jax.tree_util.tree_map(np.shape, tree)

    with self.subTest('encoder'):
      actual = jax.tree_util.tree_leaves(
          model.encode_fn(params, rng_key, data_trajectory, forcing).state
      )
      expected = jax.tree_util.tree_leaves(model_state.state)
      self.assertSameStructure(get_shape_fn(actual), get_shape_fn(expected))

    with self.subTest('decoder'):
      actual = jax.tree_util.tree_leaves(
          model.decode_fn(params, rng_key, model_state, forcing)
      )
      expected = jax.tree_util.tree_leaves(data_state)
      self.assertSameStructure(get_shape_fn(actual), get_shape_fn(expected))

    with self.subTest('advance'):
      actual = jax.tree_util.tree_leaves(
          model.advance_fn(params, rng_key, model_state, forcing).state
      )
      expected = jax.tree_util.tree_leaves(model_state.state)
      self.assertSameStructure(get_shape_fn(actual), get_shape_fn(expected))

    with self.subTest('trajectory'):

      def trajectory_fwd(x, length):
        return model.model_cls().trajectory(x, length, forcing_data={})

      trajectory_model = hk.transform(trajectory_fwd)
      params = trajectory_model.init(rng_key, model_state, length=traj_length)
      actual = jax.tree_util.tree_leaves(
          trajectory_model.apply(
              params, rng_key, model_state, length=traj_length
          )
      )
      expected = jax.tree_util.tree_leaves((model_state, state_trajectory))
      self.assertSameStructure(get_shape_fn(actual), get_shape_fn(expected))


class NodalModalEncoderDecoderTest(parameterized.TestCase):
  """Tests nodal to nodal encode->decode roundtrip using configured modules."""

  def setUp(self):
    """Defines gin-configs used in tests."""
    super().setUp()
    gin.enter_interactive_mode()
    self.physics_gin_config = '\n'.join([
        'get_physics_specs.construct_fn = @primitive_eq_specs_constructor',
    ])
    encoder_module_name = 'PrimitiveEquationStateEncoder'
    self.model_gin_config = '\n'.join([
        'WhirlModel.model_cls = @ModularStepModel',
        'ModularStepModel.advance_module = @EquationStep',
        f'ModularStepModel.encoder_module = @{encoder_module_name}',
        'ModularStepModel.decoder_module = @StateToDictDecoder',
        f'{encoder_module_name}.transform_module = @InputNodalToModalTransform',
        'StateToDictDecoder.transform_module = @OutputModalToNodalTransform',
        'EquationStep.equation_module = @PrimitiveEquations',
    ])

  @parameterized.parameters(
      dict(
          input_grid=spherical_harmonic.Grid.with_wavenumbers(64),
          model_grid=spherical_harmonic.Grid.with_wavenumbers(64),
          output_grid=spherical_harmonic.Grid.with_wavenumbers(
              32, latitude_spacing='equiangular'
          ),
      ),
      dict(
          input_grid=spherical_harmonic.Grid.with_wavenumbers(
              64, latitude_spacing='equiangular'
          ),
          model_grid=spherical_harmonic.Grid.with_wavenumbers(64),
          output_grid=spherical_harmonic.Grid.with_wavenumbers(
              32, latitude_spacing='equiangular'
          ),
      ),
  )
  def test_nodal_to_nodal_trip(
      self,
      input_grid: spherical_harmonic.Grid,
      model_grid: spherical_harmonic.Grid,
      output_grid: spherical_harmonic.Grid,
  ):
    """Tests that models data->state->data trip produces expected results."""
    gin.clear_config()
    gin.parse_config(self.physics_gin_config)
    gin.parse_config(self.model_gin_config)
    physics_specs = physics_specifications.get_physics_specs()
    # Define coordinates for the model, and for the input and output data.
    vertical_coordinates = sigma_coordinates.SigmaCoordinates.equidistant(4)
    input_coords = coordinate_systems.CoordinateSystem(
        input_grid, vertical_coordinates
    )
    model_coords = coordinate_systems.CoordinateSystem(
        model_grid, vertical_coordinates
    )
    output_coords = coordinate_systems.CoordinateSystem(
        output_grid, vertical_coordinates
    )
    # Generate an initial state to act as the input data.
    initial_state_fn, aux_features = primitive_equations_states.steady_state_jw(
        input_coords, physics_specs
    )
    initial_modal_state = initial_state_fn()
    forcing_data = None
    forcing = None
    data_inputs = jax.tree_util.tree_map(
        lambda x: np.expand_dims(input_grid.to_nodal(x), 0),
        initial_modal_state.asdict(),
    )
    model = model_builder.WhirlModel(
        model_coords,
        1.0,
        physics_specs,
        aux_features,
        input_coords=input_coords,
        output_coords=output_coords,
    )
    # to initialize we pass a data trajectory.
    rng = jax.random.PRNGKey(42)
    params = model.init_params(rng, data_inputs, forcing_data)

    with self.subTest('model_state'):
      # check that `encode` produces an expected state.
      model_state = jax.jit(model.encode_fn)(params, rng, data_inputs, forcing)
      actual_prognostic_vars = model_state.state
      jax.tree.map(
          functools.partial(np.testing.assert_allclose, atol=1e-4),
          actual_prognostic_vars,
          initial_modal_state,
      )

    with self.subTest('decoded_state'):
      # check that `decode(encode(x))` is close to identity.
      target_lon_k, target_lat_k = output_grid.modal_shape
      target_modal_state = jax.tree_util.tree_map(
          lambda x: x[:, :target_lon_k, :target_lat_k],
          initial_modal_state.asdict(),
      )
      target_nodal = jax.tree_util.tree_map(
          output_grid.to_nodal, target_modal_state
      )

      def encode_decode_fn(p, x, forcing):
        x = model.encode_fn(p, rng, x, forcing)
        return model.decode_fn(p, rng, x, forcing)

      encode_decode_fn = jax.jit(encode_decode_fn)
      actual_nodal = encode_decode_fn(params, data_inputs, forcing)
      jax.tree.map(
          functools.partial(np.testing.assert_allclose, atol=2e-4),
          actual_nodal,
          target_nodal,
      )


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
