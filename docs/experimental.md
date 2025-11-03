# Experimental API

This page contains documentation for new NeuralGCM experimental API.
These features are subject to change and are expected to be moved into a separate repository in the future.

The experimental API introduces new features for model development and inference, built on `coordax` for coordinate-aware data representation and `nnx` for stateful model components.
Tutorials below introduce key concepts of the new API including:
* A data model using `coordax.Field` structures that are convertible to and from `xarray.Dataset` for easy inspection and serialization.
* A `Model` API for defining models by subclassing `api.Model` and implementing `assimilate`, `advance`, and `observe` methods.
* A typing system to manage simulation state components like prognostics, diagnostics, randomness, and dynamic inputs (forcings).
* An immutable and purely functional `InferenceModel` API for running forecasts, compatible with JAX transformations and scalable inference with Apache Beam.
* A `VectorizedModel` for efficient batch and ensemble simulations.

## Contents

```{toctree}
:maxdepth: 1
data_model_intro.ipynb
model_api_intro.ipynb
simulation_state_components_intro.ipynb
inference_model_api_intro.ipynb
model_vectorization_tutorial.ipynb
forced_parmeterized_coupled_l96.ipynb
```
