# Copyright 2024 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines the neuralgcm module."""

# pylint: disable=unused-import
# pylint: disable=g-bad-import-order

import neuralgcm.api
import neuralgcm.correctors
import neuralgcm.decoders
import neuralgcm.demo
import neuralgcm.diagnostics
import neuralgcm.embeddings
import neuralgcm.encoders
import neuralgcm.equations
import neuralgcm.features
import neuralgcm.filters
import neuralgcm.forcings
import neuralgcm.gin_utils
import neuralgcm.initializers
import neuralgcm.integrators
import neuralgcm.layers
import neuralgcm.mappings
import neuralgcm.model_builder
import neuralgcm.model_utils
import neuralgcm.optimization
import neuralgcm.orographies
import neuralgcm.parameterizations
import neuralgcm.perturbations
import neuralgcm.stochastic
import neuralgcm.towers

from neuralgcm.api import PressureLevelModel

__version__ = "1.1.1"  # keep in sync with pyproject.toml
