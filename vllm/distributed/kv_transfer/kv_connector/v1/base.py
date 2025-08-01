# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
KVConnectorBase_V1 Class for Distributed KV Cache & Hidden State
communication in vLLM v1

The class provides the following primitives:
    Scheduler-side: runs in the scheduler, binds metadata, which
    is used by the worker-side to load/save KV cache.
        get_num_new_matched_tokens() - get number of new tokens 
            that exist in the remote KV cache. Might be called multiple
            times for a given request and should be side-effect free.
        update_state_after_alloc() - update KVConnector state after
            temporary buffer alloc by the CacheManager.
        request_finished() - called when a request is finished, with
            the computed kv cache blocks for the request.
            Returns whether KV cache should be freed now or will be
            freed asynchronously and optionally returns KV transfer
            params.

    Worker-side: runs in each worker, loads/saves KV cache to/from
    the Connector based on the metadata.
        start_load_kv() - starts loading all KVs (maybe async)
        wait_for_layer_load() - blocks until layer i load is done

        save_kv_layer() - starts saving KV for layer i (maybe async)
        wait_for_save() - blocks until all saves are done

        get_finished() - called with ids of finished requests, returns
            ids of requests that have completed async sending/recving.
        get_finished_loading() - called with scheduler outputs, returns
            a dictionary that the keys are request IDs and the values are
            the actual number of tokens loaded from the remote KV cache
"""

import enum
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional

import torch

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

# s_tensor_list, d_tensor_list, s_indices, d_indices, direction
CopyBlocksOp = Callable[[
    dict[str, torch.Tensor], dict[
        str, torch.Tensor], list[int], list[int], Literal["h2d", "d2h"]
], None]

logger = init_logger(__name__)


class KVConnectorRole(enum.Enum):
    # Connector running in the scheduler process
    SCHEDULER = 0

    # Connector running in the worker process
    WORKER = 1


class KVConnectorMetadata(ABC):  # noqa: B024
    """
    Abstract Metadata used to communicate between the
    Scheduler KVConnector and Worker KVConnector.
    """
    pass


class KVConnectorBase_V1(ABC):

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        logger.warning(
            "Initializing KVConnectorBase_V1. This API is experimental and "
            "subject to change in the future as we iterate the design.")
        self._connector_metadata: Optional[KVConnectorMetadata] = None
        self._vllm_config = vllm_config
        self._role = role

    @property
    def role(self) -> KVConnectorRole:
        return self._role

    # ==============================
    # Worker-side methods
    # ==============================

    def bind_connector_metadata(
            self, connector_metadata: KVConnectorMetadata) -> None:
        """Set the connector metadata from the scheduler.

        This function should be called by the model runner every time 
        before the model execution. The metadata will be used for runtime
        KV cache loading and saving.

        Args:
            connector_metadata (dict): the connector metadata.
        """
        self._connector_metadata = connector_metadata

    def clear_connector_metadata(self) -> None:
        """Clear the connector metadata.

        This function should be called by the model runner every time 
        after the model execution.
        """
        self._connector_metadata = None

    def _get_connector_metadata(self) -> KVConnectorMetadata:
        """Get the connector metadata.

        This function should only be called inside the connector.

        Returns:
            ConnectorMetadata: the connector metadata.
        """

        # Should only be called while set to valid metadata.
        assert self._connector_metadata is not None
        return self._connector_metadata

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """
        Initialize with the KV caches. Useful for pre-registering the
        KV Caches in the KVConnector (e.g. for NIXL).

        Args: kv_caches:
            dictionary of layer names, kv cache
        """
        return

    def set_host_xfer_buffer_ops(self, copy_operation: CopyBlocksOp):
        """
        Set the xPU-specific ops for copying KV between host and device.
        Needed when host buffer is used for kv transfer (e.g., in NixlConnector)
        """
        return

    @abstractmethod
    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        """
        Start loading the KV cache from the connector to vLLM's paged
        KV buffer. This is called from the forward context before the
        forward pass to enable async loading during model execution.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be 
            the same.
            
        """
        pass

    @abstractmethod
    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. This is called from within attention layer to ensure
        async copying from start_load_kv is complete.
        
        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        pass

    @abstractmethod
    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """
        Start saving a layer of KV cache from vLLM's paged buffer 
        to the connector. This is called from within attention layer to
        enable async copying during execution.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current 
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        pass

    @abstractmethod
    def wait_for_save(self):
        """
        Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.

        This prevents overwrites of paged KV buffer before saving done.
        """
        pass

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens on the worker.
        The scheduler process (via the Executors) will use this output
        to track which workers are done.

        Returns:
            ids of requests that have finished asynchronous transfer
            (requests that previously returned True from request_finished()),
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """
        return None, None

    def get_finished_loading(
            self, scheduler_output: "SchedulerOutput") -> dict[str, int]:
        """
        Retrieves the actual number of tokens loaded for requests that have
        completed the asynchronous loading process from the remote KV cache.

        This function is used by the scheduler process (via the Executors)
        to track the progress of requests and determine which requests have
        successfully finished loading their KV cache data.

        Returns:
            A dictionary where the keys are request IDs and the values are the
            corresponding number of tokens that have been successfully loaded
            for each request.
        """
        return {}

    # ==============================
    # Scheduler-side methods
    # ==============================

    @abstractmethod
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.
        
        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            A tuple with the following elements:
                - The number of tokens that can be loaded from the 
                  external KV cache beyond what is already computed.
                - `True` if external KV cache tokens will be loaded
                  asynchronously (between scheduler steps). Must be
                  'False' if the first element is 0.
        """
        pass

    @abstractmethod
    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        """
        Update KVConnector state after block allocation.

        If get_num_new_matched_tokens previously returned True for a
        request, this function may be called twice for that same request -
        first when blocks are allocated for the connector tokens to be
        asynchronously loaded into, and second when any additional blocks
        are allocated, after the load/transfer is complete.

        Args:
            request (Request): the request object.
            blocks (KVCacheBlocks): the blocks allocated for the request.
            num_external_tokens (int): the number of tokens that will be
                loaded from the external KV cache.
        """
        pass

    @abstractmethod
    def build_connector_meta(
            self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        """
        Build the connector metadata for this step.

        This function should NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        pass

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        return False, None
