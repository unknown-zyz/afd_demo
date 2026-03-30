"""
AFD Communicator - Handles async communication between attention and FFN nodes.

Implements double-buffering for pipeline overlap and async P2P communication.
"""

import logging
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
from enum import Enum

import torch
import torch.distributed as dist

from . import get_distributed_context

logger = logging.getLogger(__name__)


class BufferState(Enum):
    """State of a communication buffer."""
    FREE = 0
    SENDING = 1
    RECEIVING = 2
    READY = 3


@dataclass
class CommBuffer:
    """A communication buffer with state tracking."""
    tensor: torch.Tensor
    state: BufferState = BufferState.FREE
    handle: Optional[dist.Work] = None  # Async operation handle
    requested_shape: Optional[Tuple[int, ...]] = None  # Shape requested for recv


class AFDCommunicator:
    """
    Handles asynchronous communication between Attention and FFN nodes.
    
    Features:
    - Double buffering to hide communication latency
    - Async send/recv for pipeline overlap
    - Separate CUDA streams for communication
    """
    
    def __init__(
        self,
        hidden_size: int,
        max_seq_len: int,
        max_batch_size: int,
        num_buffers: int = 2,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize the communicator.
        
        Args:
            hidden_size: Model hidden dimension
            max_seq_len: Maximum sequence length
            max_batch_size: Maximum batch size
            num_buffers: Number of buffers for pipelining (2 for ping-pong)
            dtype: Data type for tensors
        """
        self.ctx = get_distributed_context()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.num_buffers = num_buffers
        self.dtype = dtype
        
        # Create separate CUDA stream for communication
        self.comm_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        
        # Initialize double buffers
        self.send_buffers: List[CommBuffer] = []
        self.recv_buffers: List[CommBuffer] = []
        self._init_buffers()
        
        # Buffer index for round-robin
        self._send_idx = 0
        self._recv_idx = 0
        
        # Pending operations
        self._pending_sends: List[dist.Work] = []
        self._pending_recvs: List[dist.Work] = []
        
        logger.info(
            f"AFDCommunicator initialized: "
            f"hidden_size={hidden_size}, num_buffers={num_buffers}, "
            f"role={self.ctx.role}"
        )
    
    def _init_buffers(self) -> None:
        """Initialize send and receive buffers."""
        # Buffer size needs to accommodate packed tensors (attn_output + residual = 2*hidden_size)
        buffer_shape = (self.max_batch_size, self.max_seq_len, self.hidden_size * 2)
        
        for _ in range(self.num_buffers):
            send_tensor = torch.empty(
                buffer_shape,
                dtype=self.dtype,
                device=self.ctx.device,
            )
            recv_tensor = torch.empty(
                buffer_shape,
                dtype=self.dtype,
                device=self.ctx.device,
            )
            
            self.send_buffers.append(CommBuffer(tensor=send_tensor))
            self.recv_buffers.append(CommBuffer(tensor=recv_tensor))
    
    def _get_send_buffer(self) -> Tuple[int, CommBuffer]:
        """Get the next available send buffer (round-robin)."""
        idx = self._send_idx
        self._send_idx = (self._send_idx + 1) % self.num_buffers
        return idx, self.send_buffers[idx]
    
    def _get_recv_buffer(self) -> Tuple[int, CommBuffer]:
        """Get the next available receive buffer (round-robin)."""
        idx = self._recv_idx
        self._recv_idx = (self._recv_idx + 1) % self.num_buffers
        return idx, self.recv_buffers[idx]
    
    def send_async(
        self,
        tensor: torch.Tensor,
        tag: int = 0,
    ) -> int:
        """
        Asynchronously send tensor to peer node.
        
        Args:
            tensor: Tensor to send (will be copied to send buffer)
            tag: Message tag for matching send/recv pairs
        
        Returns:
            Buffer index used for this send
        """
        idx, buffer = self._get_send_buffer()
        
        # Wait if buffer is still in use
        if buffer.handle is not None:
            buffer.handle.wait()
        
        # Copy data to send buffer (async on comm stream)
        actual_shape = tensor.shape
        if len(actual_shape) == 3:
            buffer_view = buffer.tensor[:actual_shape[0], :actual_shape[1], :actual_shape[2]]
        else:
            buffer_view = buffer.tensor[:actual_shape[0], :actual_shape[1], :]
        
        if self.comm_stream is not None:
            with torch.cuda.stream(self.comm_stream):
                buffer_view.copy_(tensor)
        else:
            buffer_view.copy_(tensor)
        
        # Sync before send
        if self.comm_stream is not None:
            self.comm_stream.synchronize()
        
        # Initiate async send
        buffer.handle = dist.isend(
            buffer_view.contiguous(),
            dst=self.ctx.peer_rank,
            tag=tag,
        )
        buffer.state = BufferState.SENDING
        self._pending_sends.append(buffer.handle)
        
        logger.debug(f"Async send initiated: buffer={idx}, tag={tag}, shape={actual_shape}")
        return idx
    
    def recv_async(
        self,
        shape: Tuple[int, ...],
        tag: int = 0,
    ) -> int:
        """
        Asynchronously receive tensor from peer node.
        
        Args:
            shape: Expected shape of received tensor
            tag: Message tag for matching send/recv pairs
        
        Returns:
            Buffer index where data will be received
        """
        idx, buffer = self._get_recv_buffer()
        
        # Wait if buffer is still in use
        if buffer.handle is not None:
            buffer.handle.wait()
        
        # Get buffer view for actual shape - need to handle last dim correctly
        if len(shape) == 3:
            buffer_view = buffer.tensor[:shape[0], :shape[1], :shape[2]]
        else:
            buffer_view = buffer.tensor[:shape[0], :shape[1], :]
        
        # Store requested shape for later use
        buffer.requested_shape = shape
        
        # Initiate async recv
        buffer.handle = dist.irecv(
            buffer_view.contiguous(),
            src=self.ctx.peer_rank,
            tag=tag,
        )
        buffer.state = BufferState.RECEIVING
        self._pending_recvs.append(buffer.handle)
        
        logger.debug(f"Async recv initiated: buffer={idx}, tag={tag}, shape={shape}")
        return idx
    
    def wait_send(self, buffer_idx: Optional[int] = None) -> None:
        """
        Wait for send operation to complete.
        
        Args:
            buffer_idx: Specific buffer to wait for. If None, waits for all.
        """
        if buffer_idx is not None:
            buffer = self.send_buffers[buffer_idx]
            if buffer.handle is not None:
                buffer.handle.wait()
                buffer.handle = None
                buffer.state = BufferState.FREE
        else:
            for handle in self._pending_sends:
                handle.wait()
            self._pending_sends.clear()
            for buffer in self.send_buffers:
                buffer.handle = None
                buffer.state = BufferState.FREE
    
    def wait_recv(self, buffer_idx: int) -> torch.Tensor:
        """
        Wait for receive operation to complete and get the data.
        
        Args:
            buffer_idx: Buffer index to wait for
        
        Returns:
            Received tensor (view of receive buffer with correct shape)
        """
        buffer = self.recv_buffers[buffer_idx]
        if buffer.handle is not None:
            buffer.handle.wait()
            buffer.handle = None
        buffer.state = BufferState.READY
        
        # Return tensor with the requested shape
        if buffer.requested_shape is not None:
            shape = buffer.requested_shape
            if len(shape) == 3:
                return buffer.tensor[:shape[0], :shape[1], :shape[2]]
            else:
                return buffer.tensor[:shape[0], :shape[1], :]
        return buffer.tensor
    
    def wait_all_recvs(self) -> List[torch.Tensor]:
        """Wait for all pending receives and return the data."""
        results = []
        for handle in self._pending_recvs:
            handle.wait()
        self._pending_recvs.clear()
        
        for buffer in self.recv_buffers:
            if buffer.state == BufferState.RECEIVING:
                buffer.state = BufferState.READY
                results.append(buffer.tensor)
            buffer.handle = None
        return results
    
    def send_sync(self, tensor: torch.Tensor, tag: int = 0) -> None:
        """Synchronous send (for debugging/testing)."""
        # Use blocking send directly
        logger.debug(f"Send sync: tensor.shape={tensor.shape}, tensor.device={tensor.device}, dst={self.ctx.peer_rank}, tag={tag}")
        dist.send(tensor.contiguous(), dst=self.ctx.peer_rank, tag=tag)
        logger.debug(f"Send sync complete: tag={tag}")
    
    def recv_sync(self, shape: Tuple[int, ...], tag: int = 0) -> torch.Tensor:
        """Synchronous receive (for debugging/testing)."""
        # Create tensor with requested shape
        tensor = torch.empty(shape, dtype=self.dtype, device=self.ctx.device)
        logger.debug(f"Recv sync: shape={shape}, src={self.ctx.peer_rank}, tag={tag}")
        dist.recv(tensor, src=self.ctx.peer_rank, tag=tag)
        logger.debug(f"Recv sync complete: tag={tag}, tensor.shape={tensor.shape}")
        return tensor
    
    def synchronize(self) -> None:
        """Synchronize all pending operations."""
        self.wait_send()
        for handle in self._pending_recvs:
            handle.wait()
        self._pending_recvs.clear()
        if self.comm_stream is not None:
            self.comm_stream.synchronize()


class LayerCommunicator:
    """
    Simplified communicator for layer-level communication.
    
    Handles the communication pattern for a single transformer layer:
    - Attention node sends to FFN node after attention computation
    - FFN node sends back to attention node after FFN computation
    """
    
    def __init__(self, base_comm: AFDCommunicator):
        self.comm = base_comm
        self.ctx = base_comm.ctx
        
    def attn_to_ffn(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        micro_batch_idx: int,
    ) -> Optional[torch.Tensor]:
        """
        Communication from attention node to FFN node.
        
        For attention node: sends hidden_states
        For FFN node: receives hidden_states
        
        Args:
            hidden_states: Tensor to send (attention node) or None (FFN node)
            layer_idx: Current layer index (used for tag)
            micro_batch_idx: Micro-batch index (used for tag)
        
        Returns:
            Received tensor (FFN node) or None (attention node)
        """
        tag = layer_idx * 1000 + micro_batch_idx * 10 + 0  # 0 = attn->ffn
        
        if self.ctx.is_attention_node:
            self.comm.send_async(hidden_states, tag=tag)
            return None
        else:
            recv_idx = self.comm.recv_async(hidden_states.shape, tag=tag)
            return self.comm.wait_recv(recv_idx)
    
    def ffn_to_attn(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        micro_batch_idx: int,
    ) -> Optional[torch.Tensor]:
        """
        Communication from FFN node to attention node.
        
        For FFN node: sends hidden_states
        For attention node: receives hidden_states
        """
        tag = layer_idx * 1000 + micro_batch_idx * 10 + 1  # 1 = ffn->attn
        
        if self.ctx.is_ffn_node:
            self.comm.send_async(hidden_states, tag=tag)
            return None
        else:
            recv_idx = self.comm.recv_async(hidden_states.shape, tag=tag)
            return self.comm.wait_recv(recv_idx)
