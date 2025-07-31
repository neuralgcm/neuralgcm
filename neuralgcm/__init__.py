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

import neuralgcm.demo
import neuralgcm.legacy.api
import neuralgcm.legacy.correctors
import neuralgcm.legacy.decoders
import neuralgcm.legacy.diagnostics
import neuralgcm.legacy.embeddings
import neuralgcm.legacy.encoders
import neuralgcm.legacy.equations
import neuralgcm.legacy.features
import neuralgcm.legacy.filters
import neuralgcm.legacy.forcings
import neuralgcm.legacy.gin_utils
import neuralgcm.legacy.initializers
import neuralgcm.legacy.integrators
import neuralgcm.legacy.layers
import neuralgcm.legacy.mappings
import neuralgcm.legacy.model_builder
import neuralgcm.legacy.model_utils
import neuralgcm.legacy.optimization
import neuralgcm.legacy.orographies
import neuralgcm.legacy.parameterizations
import neuralgcm.legacy.perturbations
import neuralgcm.legacy.stochastic
import neuralgcm.legacy.towers

from neuralgcm.legacy.api import PressureLevelModel

__version__ = "1.2.2"  # keep in sync with pyproject.toml
