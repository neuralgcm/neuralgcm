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
"""Helper utilities for checkpointing Fiddle configs used in NeuralGCM."""

import asyncio
from concurrent import futures
import functools
from typing import Any, Optional, Sequence

from etils import epath
import fiddle as fdl
from fiddle.experimental import serialization
import orbax.checkpoint as ocp


ParamInfo = ocp.type_handlers.ParamInfo
Metadata = ocp.metadata.value.Metadata


class FiddleConfigHandler(ocp.type_handlers.TypeHandler):
  """A wrapper around serialization of fdl.Config to the json format."""

  def __init__(self, filename: str | None = None, primary_host: int | None = 0):
    """Initializes FiddleConfigHandler.

    Args:
      filename: optional file name given to the written file; defaults to
        'fiddle_config'
      primary_host: the host id of the primary host.  Default to 0.  If it's set
        to None, then all hosts will be considered as primary.  It's useful in
        the case that all hosts are only working with local storage.
    """
    self._filename = filename or 'fiddle_config'
    self._primary_host = primary_host
    self._executor = futures.ThreadPoolExecutor(max_workers=1)

  def _save_fn(self, x, directory):
    if ocp.utils.is_primary_host(self._primary_host):
      path = directory / self._filename
      path.write_text(x)
    return 0

  def typestr(self) -> str:
    return 'Config'

  async def serialize(
      self,
      values: Sequence[Any],
      infos: Sequence[ParamInfo],
      args: Sequence[ocp.SaveArgs] | None = None,
  ) -> list[ocp.Future]:  # pylint: disable=g-bare-generic
    """Serializes the given fdl.Config to the json format."""
    del args  # Unused in this example.

    async def write_config(value, info):
      # make sure the per-key directory is present as OCDBT doesn't create one
      info.path.mkdir(exist_ok=True)  # pylint: disable=attribute-error
      self._executor.submit(
          functools.partial(self._write_config, value, info.path)
      )

    ocp_futures = []
    for value, info in zip(values, infos):
      ocp_futures.append(
          ocp.future.CommitFutureAwaitingContractedSignals(
              write_config(value, info)
          )
      )
    return ocp_futures

  async def deserialize(
      self,
      infos: Sequence[ParamInfo],
      args: Optional[Sequence[ocp.RestoreArgs]] = None,
  ):
    del args  # Unused in this example.
    ocp_futures = []
    for info in infos:
      ocp_futures.append(
          await asyncio.get_event_loop().run_in_executor(
              self._executor,
              functools.partial(self._from_serialized, info.path),
          )
      )
    return await asyncio.gather(*ocp_futures)

  async def metadata(self, infos: Sequence[ParamInfo]) -> Sequence[Metadata]:
    return [Metadata(name=info.name, directory=info.path) for info in infos]

  def _write_config(self, config: fdl.Config, path: epath.Path):
    serialized_conifig = serialization.dump_json(config, indent=4)
    self._save_fn(serialized_conifig, path)
    return path

  async def _from_serialized(self, path: epath.Path):
    path = path / self._filename
    serialized_config = path.read_text()
    return serialization.load_json(serialized_config)


ocp.type_handlers.register_type_handler(
    fdl.Config, FiddleConfigHandler(), override=True
)
