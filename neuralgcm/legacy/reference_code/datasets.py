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
"""Loads datasets."""

import functools
import itertools
import json
import logging
import math
import multiprocessing
import random
from typing import Any, Callable, Iterator, Mapping, Optional, Tuple

import jax
import numpy as np
import pandas as pd
import tensorflow.compat.v2 as tf
import xarray


Pytree = Any
# pylint: disable=g-bare-generic
# pylint: disable=logging-fstring-interpolation


def drop_static_vars(dataset: xarray.Dataset) -> xarray.Dataset:
  """Drop fields that are static and do not vary with time."""
  has_sample_dim = 'sample' in dataset.coords
  vars_to_drop = []
  for name, var in dataset.items():
    if 'time' not in var.dims:
      vars_to_drop.append(name)
    elif has_sample_dim and var.dims[:2] != ('sample', 'time'):
      raise ValueError(f'dimensions for variable {name} do not start with '
                       f"'sample' and 'time': {var.dims}")
    elif not has_sample_dim and var.dims[0] != 'time':
      raise ValueError(f'dimensions for variable {name} do not start with '
                       f"'time': {var.dims}")
  return dataset.drop_vars(vars_to_drop)


def attrs_from_dataset(
    dataset: xarray.Dataset,
    time_series_length: int,
    subsample_rate: int = 1,
) -> dict:
  """Extracts attributes from `dataset`."""
  attrs = dict(dataset.attrs)
  attrs['trajectory_length'] = time_series_length
  attrs['time_subsample_rate'] = subsample_rate
  delta_t = (dataset.time[1] - dataset.time[0]).data
  if not np.issubdtype(dataset.time.dtype, np.floating):
    logging.info(f'converting non-float {delta_t=} to seconds')
    delta_t = np.timedelta64(delta_t, 's') / np.timedelta64(1, 's')
    attrs['save_dt_units'] = 's'
  else:
    attrs['save_dt_units'] = 'dimensionless'
  attrs['save_dt'] = float(delta_t) * subsample_rate
  return attrs
