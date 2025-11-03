# NeuralGCM documentation

## Overview

The NeuralGCM codebase consists of a handful of different components, suitable
for reproducing and extending results from our paper,
[Neural General Circulation Models for Weather and Climate](https://arxiv.org/abs/2311.07222):

1. **Dynamics**: The atmospheric dynamical core is distributed in the separate
   [Dinosaur](https://github.com/neuralgcm/dinosaur) package.
2. **ML modules**: [Haiku](https://github.com/google-deepmind/dm-haiku) modules
   for defining neural network layers.
3. **ML training**: Pseudo-code for training NeuralGCM models can be found in
   the [`reference_code` subdirectory](https://github.com/neuralgcm/neuralgcm/tree/main/neuralgcm/reference_code).
4. **ML inference**: Code for running forecasts with pre-trained models,
   encapsulated in the {py:class}`~neuralgcm.PressureLevelModel` class.
5. **Evaluation**: Code for evaluating NeuralGCM weather forecasts can be found in the
   [WeatherBench2](https://github.com/google-research/weatherbench2) project.

The documentation here focuses mostly on our API for *inference* (i.e., running
trained NeuralGCM atmospheric models), which we believe is the most immediately
useful part of the NeuralGCM code for third parties. It is also a part of our
code that we can commit to supporting in roughly its current form.

We would love to support training, modifying and fine-tuning NeuralGCM models,
but with the present codebase based on Haiku and
[Gin](https://github.com/google/gin-config) this is much trickier than it needs
to be. We are currently (in May 2024) refactoring the modeling code to improve
usability.

```{tip}
To stay up to date on NeuralGCM, [subscribe to our mailing list](https://groups.google.com/g/neuralgcm-announce).
```

## Questions?

The best place to ask for help using NeuralGCM models or datasets is
[on GitHub](https://github.com/neuralgcm/neuralgcm/issues).

You can also reach the NeuralGCM team directly at
[neuralgcm@google.com](mailto:neuralgcm@google.com).

## Contents

```{toctree}
:maxdepth: 1
installation.md
inference_demo.ipynb
checkpoints.md
neuralgcm_datasets.ipynb
data_preparation.ipynb
deepdive_into_models.ipynb
checkpoint_modifications.ipynb
api.md
experimental.md
```