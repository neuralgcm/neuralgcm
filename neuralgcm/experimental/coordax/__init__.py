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

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/jax-ml/jax/issues/7570
# pylint: disable=g-multiple-import,useless-import-alias,g-importing-member
from neuralgcm.experimental.coordax.coordinate_systems import (
    CartesianProduct as CartesianProduct,
    Coordinate as Coordinate,
    DummyAxis as DummyAxis,
    LabeledAxis as LabeledAxis,
    Scalar as Scalar,
    SizedAxis as SizedAxis,
    NoCoordinateMatch as NoCoordinateMatch,
    SelectedAxis as SelectedAxis,
    compose as compose_coordinates,
    from_xarray as coordinates_from_xarray,
)
from neuralgcm.experimental.coordax.fields import (
    Field as Field,
    is_field as is_field,
    cmap as cmap,
    get_coordinate as get_coordinate,
    wrap_like as wrap_like,
    wrap as wrap,
    tag as tag,
    untag as untag,
)
from neuralgcm.experimental.coordax.ndarrays import (
    NDArray as NDArray,
    register_ndarray as register_ndarray,
)
