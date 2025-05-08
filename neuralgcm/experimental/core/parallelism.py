# Copyright 2025 Google LLC
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
"""Utilities for implementing SPMD model parallelism."""

import collections
import dataclasses
import math
from typing import Self, Type, TypeGuard

import einops
import fiddle as fdl
from fiddle import selectors
from fiddle import tagging
import jax
import jax.numpy as jnp
from neuralgcm.experimental import coordax as cx
from neuralgcm.experimental.core import pytree_utils
from neuralgcm.experimental.core import typing
import numpy as np
import xarray

P = jax.sharding.PartitionSpec

# Type aliases for improved readability.
Schema = str
DimensionName = str
AxisPartition = str | tuple[str, ...] | None
ArrayPartition = tuple[AxisPartition, ...]
DimPartitions = dict[DimensionName, AxisPartition]

ArrayPartitions = dict[Schema, ArrayPartition]
FieldPartitions = dict[Schema, DimPartitions]

# Axis partitions specify how to partition a single dimension of an array, which
# can be either a single axis along the device mesh, specified as a `str`,
# a `tuple[str, ...]` if the axis is partitioned acorss multiple device axes
# or `None` if the dimension is not partitioned (i.e. replicated).

# Array partitions specify an AxisPartition per axis.
#
# For example, if we have a 3D array and we want to partition it across
# a device mesh `('x', 'y')`, along the first dimension, we would specify
#   `array_partition = (('x', 'y'), None, None)`.

# DimPartitions determines how each dimension of a field is split (partitioned)
# across multiple devices.
#
# You define this partitioning by associating each dimension with a device mesh.
# For example, if you have a device mesh defined as ('x', 'y', 'z') and you set
#   `dim_partitions = {'level': ('x', 'z'), 'lat': 'y'}`
# it means: (1) The 'level' dimension will be divided across the 'x' and 'z'
# devices of the mesh. (2) The 'lat' dimension will be divided across the 'y'
# device of the mesh. (3) Any dimensions not mentioned will be fully replicated
# on all devices.


def get_partition_spec(
    dims: tuple[str, ...], dim_partitions: DimPartitions
) -> P:
  """Returns partition spec for `field`."""
  return P(*[dim_partitions.get(d, None) for d in dims])


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class CoordinateShard(cx.Coordinate):
  """Coordinate that represents a shard of a full coordinate system.

  This class helps tag array shards that represent a subset of values of a
  `full_coordinare`. The most common use case is during data ingestion where
  shards are read from disk and are not yet converted to a sharded array.
  """

  coordinate: cx.Coordinate
  spmd_mesh_shape: collections.OrderedDict[str, int]
  dimension_partitions: dict[str, None | str | tuple[str, ...]]

  def __hash__(self):
    """Hash implementation that avoids un-hashable mesh and partitions attrs."""
    spmd_mesh_tuple = tuple(self.spmd_mesh_shape.items())
    partitions_tuple = tuple(self.dimension_partitions.items())
    return hash((self.coordinate, spmd_mesh_tuple, partitions_tuple))

  def __eq__(self, other):
    return (
        isinstance(other, CoordinateShard)
        and self.coordinate == other.coordinate
        and self.spmd_mesh_shape == other.spmd_mesh_shape
        and self.dimension_partitions == other.dimension_partitions
    )

  @property
  def dims(self) -> tuple[str, ...]:
    return self.coordinate.dims  # sharding only affects the shape.

  @property
  def shape(self) -> tuple[int, ...]:
    """Shape of the coordinate."""
    dims, full_shape = self.dims, self.coordinate.shape
    shard_shape = []
    for dim, full_size in zip(dims, full_shape, strict=True):
      axes = self.dimension_partitions.get(dim, None)
      if not isinstance(axes, tuple):
        axes = (axes,)
      n_shards = math.prod(self.spmd_mesh_shape.get(axis, 1) for axis in axes)
      size, reminder = divmod(full_size, n_shards)
      if reminder:
        raise ValueError(
            f'Dimension {dim} has size {full_size} which is not divisible by'
            f' the number of shards {n_shards}.'
        )
      shard_shape.append(size)
    return tuple(shard_shape)

  @property
  def fields(self) -> dict[str, cx.Field]:
    """A maps from field names to their values."""
    return {}

  @classmethod
  def from_xarray(
      cls, dims: tuple[str, ...], coords: xarray.Coordinates
  ) -> Self | cx.NoCoordinateMatch:
    return cx.NoCoordinateMatch(
        'CoordinateShard cannot be deterministically inferred from xarray.'
    )


def get_unsharded(coordinate: cx.Coordinate) -> cx.Coordinate:
  """Returns the unsharded coordinate of a CoordinateShard."""
  unsharded = []
  for c in cx.canonicalize_coordinates(coordinate):
    unsharded.append(c.coordinate if isinstance(c, CoordinateShard) else c)
  return cx.compose_coordinates(*unsharded)


def rearrange_spmd_mesh(
    spmd_mesh: jax.sharding.Mesh,
    dim_partitions: DimPartitions,
    dims_to_spmd_mesh_axes: dict[str, str],
) -> jax.sharding.Mesh:
  """Returns rearranged & renamed spmd mesh representing the same partitions.

  This transformation helps interface with existing codes that rely on
  spmd mesh with specific axis names. It is parameterized by a mapping from
  dimension names that are to be aligned with spmd_mesh axes in the output to
  the desired spmd_mesh axis names.

  Args:
    spmd_mesh: input spmd mesh specifying the devices and axis names.
    dim_partitions: mapping from dimension names to partition axes in spmd_mesh.
    dims_to_spmd_mesh_axes: mapping from dimension names to be aligned with the
      trailing axes of the output spmd_mesh to their desired axis names.

  Returns:
    A new spmd mesh with trailing axes renamed to spmd_mesh_axis_names and
    rearranged to correspond to partition data in a way consistent with
    dim_partitions and input spmd_mesh.
  """
  partitions = tuple(
      dim_partitions.get(k, None) for k in dims_to_spmd_mesh_axes
  )
  # Replace None with empty tuple to assign no partitioning to these dimensions.
  partitions = tuple(p if p is not None else tuple() for p in partitions)
  used_axes = sum(map(tuple, partitions), start=())
  retained_axes = tuple(
      axis for axis in spmd_mesh.axis_names if axis not in used_axes
  )

  def _as_rearrangement_string(x: str | tuple[str, ...]) -> str:
    """Converts partition description to rearrangement string."""
    if isinstance(x, str):
      return x
    if not x:
      return '1'  # effectively expands axis in the rearrangement.
    else:
      return '(' + ' '.join(x) + ')'  # groups multiple axes into one.

  rearrange_schema = (
      ' '.join(spmd_mesh.axis_names)  # current mesh axes.
      + '->'
      + ' '.join(retained_axes)
      + ' '
      + ' '.join(map(_as_rearrangement_string, partitions))
  )
  devices = einops.rearrange(spmd_mesh.devices, rearrange_schema)
  spmd_mesh_axis_names = tuple(dims_to_spmd_mesh_axes.values())
  return jax.sharding.Mesh(devices, retained_axes + spmd_mesh_axis_names)


# TODO(dkochkov): drop array_partitions specification when partitions can be
# specified using field_partitions.
# TODO(dkochkov): consider making Mesh not an nnx.Module.


@dataclasses.dataclass(frozen=True)
class Mesh:
  """Holds the device mesh and array/field partitioning strategies.

  Attributes:
    spmd_mesh: Jax sharding mesh object.
    array_partitions: Schemas for partitioning arrays. An array partition
      specifies how to partition a an array of a specific rank. For example, to
      partition a 3d array across a device mesh `('x', 'y')`, along the first
      dimension, we would use a `(('x', 'y'), None, None)` schema.
    field_partitions: Schemas for partitioning cx.Field. Each schema provides
      partitioning rules for array axes specified by the dimension names. For
      example, schema `{'vertical': {'level': ('z', 'x', 'y'), 'layer': 'z'}}`
      specifies that all `level` dimensions of a field will be partitioned
      across the `z`, `x`, and `y` axes of the device mesh and all `layer`
      dimensions will be partitioned only across the `z` axis.
  """

  spmd_mesh: jax.sharding.Mesh | None = None
  array_partitions: dict[Schema, ArrayPartition] = dataclasses.field(
      default_factory=dict
  )
  field_partitions: dict[Schema, DimPartitions] = dataclasses.field(
      default_factory=dict
  )

  def __post_init__(self):
    self._validate_partitions()

  def _validate_partitions(self):
    """Validates that partitioning options are compatible with the mesh."""
    if self.spmd_mesh is not None:
      for k, v in self.array_partitions.items():
        partition_axes = jax.tree.leaves(v)
        if not set(partition_axes).issubset(set(self.axis_names)):
          raise ValueError(f'Spec {k, v} use axes not in {self.axis_names}')
        if np.unique(partition_axes).size != len(partition_axes):
          raise ValueError(f'Encountered duplicate in spec {k, v}')

      for _, dim_partitions in self.field_partitions.items():
        for k, v in dim_partitions.items():
          partition_axes = jax.tree.leaves(v)
          if not set(partition_axes).issubset(set(self.axis_names)):
            raise ValueError(f'Spec {k, v} use axes not in {self.axis_names}')
          if np.unique(partition_axes).size != len(partition_axes):
            raise ValueError(f'Encountered duplicate in spec {k, v}')

  @property
  def shape(self) -> collections.OrderedDict[str, int]:
    """Shape of the mesh."""
    return self.spmd_mesh.shape if self.spmd_mesh else collections.OrderedDict()

  @property
  def axis_names(self) -> tuple[str, ...]:
    """Names of the mesh axes."""
    return self.spmd_mesh.axis_names if self.spmd_mesh else ()

  def rearrange_spmd_mesh(
      self,
      schema_name: Schema | None,
      dims_to_spmd_mesh_axes: dict[str, str],
  ) -> jax.sharding.Mesh | None:
    """Returns rearranged & renamed spmd mesh representing the same partitions."""
    if self.spmd_mesh is None or schema_name is None:
      return None
    if schema_name not in self.field_partitions:
      raise ValueError(
          f'Schema {schema_name} not found in field_partitions.'
          f' Available schemas: {self.field_partitions.keys()}'
      )
    return rearrange_spmd_mesh(
        self.spmd_mesh,
        self.field_partitions[schema_name],
        dims_to_spmd_mesh_axes=dims_to_spmd_mesh_axes,
    )

  def with_sharding_constraint(
      self, inputs: typing.PyTreeState, schema: str | tuple[str, ...]
  ) -> typing.PyTreeState:
    """Applies `schema` sharding constraint to `inputs`."""
    if isinstance(schema, tuple):
      if not schema:  # case when we processed all constraints
        return inputs
      constrained = self.with_sharding_constraint(inputs, schema[1:])
      return self.with_sharding_constraint(constrained, schema[0])

    is_field = lambda x: isinstance(x, cx.Field)
    all_leaves, tree_def = jax.tree.flatten(inputs, is_leaf=is_field)
    array_leaves = [x for x in all_leaves if not is_field(x)]
    if np.unique([x.ndim for x in array_leaves]).size > 1:
      raise ValueError(
          'All arrays in the pytree must have the same rank. Got:'
          f' {[x.ndim for x in array_leaves]=}'
      )
    if self.spmd_mesh is None:
      return inputs
    leaves = []
    for x in all_leaves:
      if is_field(x):
        leaves.append(self._with_sharding_constraint_field(x, schema))
      else:
        leaves.append(self._with_sharding_constraint_array(x, schema))
    return jax.tree.unflatten(tree_def, leaves)

  def _with_sharding_constraint_array(
      self, array: jax.Array, schema: str
  ) -> jax.Array:
    """Applies sharding constraint to `array`.

    Args:
      array: array to apply sharding constraint to.
      schema: key in `self.array_partitions` indicating sharding schema.

    Returns:
      `inputs` with sharding constraint(s) applied.
    """
    p_specs = P(*self.array_partitions[schema])
    sharding = jax.sharding.NamedSharding(self.spmd_mesh, p_specs)
    return jax.lax.with_sharding_constraint(array, sharding)

  def _get_named_sharding(
      self, dims: tuple[str, ...], schema: str
  ) -> jax.sharding.NamedSharding:
    dim_partitions = self.field_partitions[schema]
    p_specs = get_partition_spec(dims, dim_partitions)
    return jax.sharding.NamedSharding(self.spmd_mesh, p_specs)

  def _with_sharding_constraint_field(
      self, field: cx.Field, schema: str
  ) -> cx.Field:
    """Applies sharding constraint to `field`.

    Args:
      field: field to apply sharding constraint to.
      schema: key in `self.field_partitions` indicating sharding schema.

    Returns:
      Field with sharding constraint applied.
    """
    sharding = self._get_named_sharding(field.dims, schema)
    return jax.lax.with_sharding_constraint(field, sharding)

  def unshard(
      self, field_shards: dict[jax.Device, cx.Field], schema: str
  ) -> cx.Field:
    """Convert sharded fields to a single unsharded field.

    Args:
      field_shards: mapping from JAX device to coordax.Field indicating data to
        load on to that device.
      schema: key in `self.field_partitions` indicating sharding schema.

    Returns:
      A single coordax.Field with data from `field_shards` as a single JAX
      array with sharded inputs.
    """

    example_field = next(iter(field_shards.values()))
    sharding = self._get_named_sharding(example_field.dims, schema)

    coord = cx.get_coordinate(example_field)
    unsharded_coord = get_unsharded(coord)

    single_device_arrays = [
        jax.device_put(field.data, device)
        for device, field in field_shards.items()
    ]

    def make_array(*arrays: jax.Array) -> jax.Array:
      return jax.make_array_from_single_device_arrays(
          unsharded_coord.shape, sharding, list(arrays)
      )

    # We use jax.tree.map to handle the case where cx.Field.data contains a
    # pytree of JAX arrays (e.g., jax_datetime.Datetime), which are not
    # supported by jax.make_array_from_single_device_arrays.
    array = jax.tree.map(make_array, *single_device_arrays)
    return cx.Field(array).tag(unsharded_coord)


# TODO(dkochkov): Remove these temporary functions once we can rely on imposing
# sharding constraints on coordax.Field leaves.
def with_dycore_sharding(
    mesh: Mesh | None,
    inputs: typing.PyTreeState,
) -> typing.PyTreeState:
  """Applies even sharding variants to inputs depending on input shape."""

  def f(y: jax.Array) -> jax.Array:
    assert isinstance(mesh, Mesh)  # make pytype happy.
    if y.ndim == 1 and y.dtype == jnp.uint32:
      return y
    elif y.ndim == 3 and y.shape[0] != 1:
      return mesh.with_sharding_constraint(y, 'dycore_3d')
    elif y.ndim == 3 and y.shape[0] == 1:
      return mesh.with_sharding_constraint(y, 'dycore_3d_surface')
    elif y.ndim == 2:
      return mesh.with_sharding_constraint(y, 'dycore_2d')
    else:
      raise ValueError(f'Unsupported array shape: {y.shape}')

  if mesh is None:
    return inputs
  return pytree_utils.tree_map_over_nonscalars(f, inputs)


def with_physics_sharding(
    mesh: Mesh | None,
    inputs: typing.PyTreeState,
) -> typing.PyTreeState:
  """Applies horizontal sharding variants depending on input shape."""

  def f(y: jax.Array) -> jax.Array:
    assert isinstance(mesh, Mesh)  # make pytype happy.
    if y.ndim == 1 and y.dtype == jnp.uint32:
      return y
    elif y.ndim == 3:
      return mesh.with_sharding_constraint(y, 'physics_3d')
    elif y.ndim == 2:
      return mesh.with_sharding_constraint(y, 'physics_2d')
    else:
      raise ValueError(f'Unsupported array shape: {y.shape}')

  if mesh is None:
    return inputs
  return pytree_utils.tree_map_over_nonscalars(f, inputs)


def with_dycore_to_physics_sharding(
    mesh: Mesh | None,
    inputs: typing.PyTreeState,
) -> typing.PyTreeState:
  """Applies dycore sharding followed by physics sharding constraints."""
  return with_physics_sharding(mesh, with_dycore_sharding(mesh, inputs))


def with_physics_to_dycore_sharding(
    mesh: Mesh | None,
    inputs: typing.PyTreeState,
) -> typing.PyTreeState:
  """Applies physics sharding followed by dycore sharding constraints."""
  return with_dycore_sharding(mesh, with_physics_sharding(mesh, inputs))


#
# Helper functions for updating Fiddle configs with new parallelism options.
#


MeshType = Type[Mesh]
TagOrMeshType = tagging.TagType | MeshType


def update_mesh_properties(
    fiddle_config: fdl.Config,
    spmd_mesh_updates: (
        dict[TagOrMeshType, jax.sharding.Mesh | None] | None
    ) = None,
    array_partitions_updates: (
        dict[TagOrMeshType, ArrayPartitions] | None
    ) = None,
    field_partitions_updates: (
        dict[TagOrMeshType, FieldPartitions] | None
    ) = None,
) -> fdl.Config:
  """Returns a copy of the Fiddle config with updated mesh properties.

  Args:
    fiddle_config: input configuration that will be copied and updated.
    spmd_mesh_updates: mapping indicating how to update `spmd_mesh` attributes
      on Mesh configurations in `fiddle_config`. Keys specify the selector of
      Mesh objects either via tags or mesh types and values indicate the jax
      sharding mesh to use for the `spmd_mesh` attribute.
    array_partitions_updates: same as `spmd_mesh_updates` but for
      `array_partitions` attribute.
    field_partitions_updates: same as `spmd_mesh_updates` but for
      `field_partitions` attribute.

  Returns:
    A copy of `fiddle_config` with updated mesh properties.
  """
  # Update spmd_mesh via selectors of tags or mesh subclasses.
  spmd_mesh_updates = spmd_mesh_updates or {}
  array_partitions_updates = array_partitions_updates or {}
  field_partitions_updates = field_partitions_updates or {}

  def _is_tag(key: TagOrMeshType) -> TypeGuard[tagging.TagType]:
    return isinstance(key, fdl.Tag)

  def _is_mesh(key: TagOrMeshType) -> TypeGuard[Type[Mesh]]:
    return issubclass(key, Mesh)

  for key, spmd_mesh in spmd_mesh_updates.items():
    if _is_tag(key):
      for mesh in selectors.select(fiddle_config, tag=key):
        mesh.spmd_mesh = spmd_mesh
    elif _is_mesh(key):
      for mesh in selectors.select(fiddle_config, key):
        mesh.spmd_mesh = spmd_mesh

  # Update array_partitions via selectors of tags or mesh subclasses.
  for key, partition in array_partitions_updates.items():
    if _is_tag(key):
      for mesh in selectors.select(fiddle_config, tag=key):
        mesh.array_partitions = partition
    elif _is_mesh(key):
      for mesh in selectors.select(fiddle_config, key):
        mesh.array_partitions = partition

  # Update field_partitions via selectors of tags or mesh subclasses.
  for key, partition in field_partitions_updates.items():
    if _is_tag(key):
      for mesh in selectors.select(fiddle_config, tag=key):
        mesh.field_partitions = partition
    elif _is_mesh(key):
      for mesh in selectors.select(fiddle_config, key):
        mesh.field_partitions = partition
  return fiddle_config
