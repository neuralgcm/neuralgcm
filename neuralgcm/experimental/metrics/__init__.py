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

"""Library for computing metrics for data in `coordax` representation.

This API borrows the design from the WeatherBenchX [1] project, but uses coordax
for labeled array representation instead of Xarray, making it compatible with
JAX and all of the JAX transformations. The current set of metrics implemented
here are a small subset of the metrics available in WeatherBenchX.

References:
  [1] https://github.com/google-research/weatherbenchX
"""
