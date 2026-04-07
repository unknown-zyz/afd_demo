"""
NVSHMEM-based P2P communicator for AFD.

Uses NVSHMEM symmetric memory and one-sided put/get operations
to eliminate NCCL proxy thread cold-start latency.

REQUIRES: libnvshmem.so (install via NVIDIA HPC SDK or standalone package)

Key advantages over NCCL P2P:
- No proxy thread: GPU-initiated DMA directly over NVLink/PCIe
- No cold-start: symmetric memory pre-allocated at init
- One-sided: no receiver-side irecv posting needed
- GPU-side sync: signal_wait_until spins on GPU, no CPU scheduling
"""

import ctypes
import logging
import time
import struct
from typing import Optional, Tuple, List, Dict

import torch

logger = logging.getLogger(__name__)

# NVSHMEM comparison operators for signal_wait_until
NVSHMEM_CMP_EQ = 0
NVSHMEM_CMP_NE = 1
NVSHMEM_CMP_GT = 2
NVSHMEM_CMP_GE = 3
NVSHMEM_CMP_LT = 4
NVSHMEM_CMP_LE = 5

_nvshmem_lib = None
_nvshmem_available = None


def _check_nvshmem_available() -> bool:
    """Check if NVSHMEM is available on this system."""
    global _nvshmem_available
    if _nvshmem_available is not None:
        return _nvshmem_available
    try:
        ctypes.CDLL("libnvshmem.so")
        _nvshmem_available = True
    except OSError:
        _nvshmem_available = False
    return _nvshmem_available


def _get_nvshmem_lib():
    """Load and cache NVSHMEM library handle."""
    global _nvshmem_lib
    if _nvshmem_lib is not None:
        return _nvshmem_lib

    if not _check_nvshmem_available():
        raise RuntimeError(
            "NVSHMEM not available. Install via:\n"
            "  1) NVIDIA HPC SDK: apt install nvhpc-*\n"
            "  2) Standalone: https://developer.nvidia.com/nvshmem\n"
            "  3) pip install nvidia-nvshmem (if available)"
        )

    lib = ctypes.CDLL("libnvshmem.so")

    # nvshmem_init()
    lib.nvshmem_init.argtypes = []
    lib.nvshmem_init.restype = None

    # nvshmem_finalize()
    lib.nvshmem_finalize.argtypes = []
    lib.nvshmem_finalize.restype = None

    # nvshmem_my_pe() -> int
    lib.nvshmem_my_pe.argtypes = []
    lib.nvshmem_my_pe.restype = ctypes.c_int

    # nvshmem_n_pes() -> int
    lib.nvshmem_n_pes.argtypes = []
    lib.nvshmem_n_pes.restype = ctypes.c_int

    # nvshmem_malloc(size_t) -> void*
    lib.nvshmem_malloc.argtypes = [ctypes.c_size_t]
    lib.nvshmem_malloc.restype = ctypes.c_void_p

    # nvshmem_free(void*)
    lib.nvshmem_free.argtypes = [ctypes.c_void_p]
    lib.nvshmem_free.restype = None

    # nvshmem_putmem_nbi(void* dest, const void* src, size_t nelems, int pe)
    lib.nvshmem_putmem_nbi.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
    ]
    lib.nvshmem_putmem_nbi.restype = None

    # nvshmem_getmem_nbi(void* dest, const void* src, size_t nelems, int pe)
    lib.nvshmem_getmem_nbi.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
    ]
    lib.nvshmem_getmem_nbi.restype = None

    # nvshmem_fence()
    lib.nvshmem_fence.argtypes = []
    lib.nvshmem_fence.restype = None

    # nvshmem_quiet()
    lib.nvshmem_quiet.argtypes = []
    lib.nvshmem_quiet.restype = None

    # nvshmem_barrier_all()
    lib.nvshmem_barrier_all.argtypes = []
    lib.nvshmem_barrier_all.restype = None

    # nvshmem_uint64_p(uint64_t* dest, uint64_t value, int pe)
    lib.nvshmem_uint64_p.argtypes = [
        ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int
    ]
    lib.nvshmem_uint64_p.restype = None

    # nvshmem_uint64_wait_until(uint64_t* addr, int cmp, uint64_t val)
    lib.nvshmem_uint64_wait_until.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64
    ]
    lib.nvshmem_uint64_wait_until.restype = None

    _nvshmem_lib = lib
    return lib


class NVSHMEMWorkHandle:
    """
    Work handle compatible with torch.distributed.Work interface.

    For send: tracks async put completion.
    For recv: waits on signal flag from sender PE.
    """

    def __init__(self, comm: 'NVSHMEMCommunicator', op_type: str,
                 slot_idx: int, tensor: Optional[torch.Tensor] = None):
        self._comm = comm
        self._op_type = op_type  # 'send' or 'recv'
        self._slot_idx = slot_idx
        self._tensor = tensor
        self._completed = False

    def wait(self) -> bool:
        """Block until the operation completes."""
        if self._completed:
            return True

        lib = self._comm._lib

        if self._op_type == 'send':
            # Ensure all prior puts are visible to remote PE
            lib.nvshmem_quiet()
            self._completed = True

        elif self._op_type == 'recv':
            slot = self._comm._slots[self._slot_idx]
            # Wait for signal from sender
            lib.nvshmem_uint64_wait_until(
                slot['signal_ptr'],
                NVSHMEM_CMP_GE,
                ctypes.c_uint64(slot['expected_seq']),
            )
            # Copy from symmetric recv buffer to output tensor
            if self._tensor is not None:
                nbytes = self._tensor.nelement() * self._tensor.element_size()
                ctypes.memmove(
                    self._tensor.data_ptr(),
                    slot['data_ptr'],
                    nbytes,
                )
            self._completed = True

        return True

    def is_completed(self) -> bool:
        """Non-blocking check (approximate for NVSHMEM)."""
        return self._completed


class NVSHMEMCommunicator:
    """
    NVSHMEM-based point-to-point communicator.

    Drop-in replacement for dist.isend/dist.irecv using NVSHMEM
    symmetric memory and one-sided put/get operations.

    Usage:
        comm = NVSHMEMCommunicator.initialize(device=torch.device('cuda:0'))
        handle = comm.isend(tensor, dst=1, tag=42)
        handle.wait()
    """

    def __init__(self, lib, my_pe: int, n_pes: int,
                 device: torch.device,
                 max_tensor_bytes: int = 64 * 1024 * 1024,
                 num_slots: int = 8):
        self._lib = lib
        self._my_pe = my_pe
        self._n_pes = n_pes
        self._device = device
        self._max_tensor_bytes = max_tensor_bytes
        self._num_slots = num_slots
        self._slots: Dict[int, dict] = {}
        self._seq_counters: Dict[int, int] = {}  # slot_idx -> next seq num

        # Allocate symmetric memory for data slots
        slot_size = max_tensor_bytes + 64  # 64B header per slot
        total_data_size = slot_size * num_slots
        self._sym_data_ptr = lib.nvshmem_malloc(total_data_size)
        if not self._sym_data_ptr:
            raise RuntimeError("nvshmem_malloc failed for data slots")

        # Allocate symmetric memory for signal flags (uint64 per slot)
        signal_size = num_slots * 8  # 8 bytes per uint64 signal
        self._sym_signal_ptr = lib.nvshmem_malloc(signal_size)
        if not self._sym_signal_ptr:
            raise RuntimeError("nvshmem_malloc failed for signal flags")

        # Initialize signal flags to 0
        ctypes.memset(self._sym_signal_ptr, 0, signal_size)

        # Set up slot metadata
        for i in range(num_slots):
            data_offset = i * slot_size
            signal_offset = i * 8
            self._slots[i] = {
                'data_ptr': ctypes.c_void_p(self._sym_data_ptr + data_offset + 64),
                'header_ptr': ctypes.c_void_p(self._sym_data_ptr + data_offset),
                'signal_ptr': ctypes.c_void_p(self._sym_signal_ptr + signal_offset),
                'expected_seq': 1,  # next expected sequence number
                'max_bytes': max_tensor_bytes,
            }
            self._seq_counters[i] = 1

        logger.info(
            f"[NVSHMEM] initialized: pe={my_pe}/{n_pes}, "
            f"slots={num_slots}, max_bytes={max_tensor_bytes}"
        )

    def _tag_to_slot(self, tag: int) -> int:
        """Map tag to slot index."""
        return tag % self._num_slots

    def isend(self, tensor: torch.Tensor, dst: int, tag: int) -> NVSHMEMWorkHandle:
        """
        Async send via NVSHMEM one-sided put.

        1. Copy tensor to local symmetric data slot
        2. nvshmem_putmem_nbi → remote symmetric data slot
        3. nvshmem_fence (ordering)
        4. nvshmem_uint64_p → remote signal flag (notify receiver)
        5. Return handle (wait → nvshmem_quiet)
        """
        slot_idx = self._tag_to_slot(tag)
        slot = self._slots[slot_idx]
        nbytes = tensor.nelement() * tensor.element_size()

        if nbytes > slot['max_bytes']:
            raise ValueError(
                f"Tensor size {nbytes} exceeds slot capacity {slot['max_bytes']}"
            )

        # Write tensor shape as header (ndim + dims, packed uint64)
        shape = tensor.shape
        header = struct.pack(f'<Q{len(shape)}Q', len(shape), *shape)
        ctypes.memmove(slot['header_ptr'], header, len(header))

        # Copy tensor data to local symmetric buffer
        ctypes.memmove(slot['data_ptr'], tensor.data_ptr(), nbytes)

        # Remote write: data slot
        remote_data_offset = slot_idx * (self._max_tensor_bytes + 64) + 64
        self._lib.nvshmem_putmem_nbi(
            ctypes.c_void_p(self._sym_data_ptr + remote_data_offset),
            slot['data_ptr'],
            nbytes,
            dst,
        )

        # Remote write: header
        remote_header_offset = slot_idx * (self._max_tensor_bytes + 64)
        self._lib.nvshmem_putmem_nbi(
            ctypes.c_void_p(self._sym_data_ptr + remote_header_offset),
            slot['header_ptr'],
            len(header),
            dst,
        )

        # Fence to ensure data is ordered before signal
        self._lib.nvshmem_fence()

        # Write signal to remote PE
        seq = self._seq_counters[slot_idx]
        self._seq_counters[slot_idx] = seq + 1
        remote_signal_ptr = ctypes.c_void_p(
            self._sym_signal_ptr + slot_idx * 8
        )
        self._lib.nvshmem_uint64_p(remote_signal_ptr, seq, dst)

        return NVSHMEMWorkHandle(self, 'send', slot_idx)

    def irecv(self, tensor: torch.Tensor, src: int, tag: int) -> NVSHMEMWorkHandle:
        """
        Async recv via NVSHMEM signal wait.

        Returns handle; when waited:
        1. nvshmem_uint64_wait_until on local signal flag
        2. Data already in local symmetric memory (written by sender's put)
        3. Copy from symmetric buffer to output tensor
        """
        slot_idx = self._tag_to_slot(tag)
        slot = self._slots[slot_idx]

        # Update expected sequence number
        slot['expected_seq'] = self._seq_counters.get(slot_idx, 1)

        return NVSHMEMWorkHandle(self, 'recv', slot_idx, tensor)

    def cleanup(self):
        """Free symmetric memory and finalize NVSHMEM."""
        if self._sym_data_ptr:
            self._lib.nvshmem_free(self._sym_data_ptr)
            self._sym_data_ptr = None
        if self._sym_signal_ptr:
            self._lib.nvshmem_free(self._sym_signal_ptr)
            self._sym_signal_ptr = None
        self._lib.nvshmem_finalize()
        logger.info("[NVSHMEM] finalized")

    def benchmark(self, peer_pe: int, sizes: Optional[List[int]] = None,
                  num_iters: int = 10) -> dict:
        """
        Benchmark put/get latency for various tensor sizes.

        Returns dict with cold/warm latencies per size.
        """
        if sizes is None:
            sizes = [64, 1024, 4096, 65536, 1048576, 8388608]

        results = {}
        for size in sizes:
            if size > self._max_tensor_bytes:
                continue

            num_elements = max(1, size // 2)  # float16
            latencies = []

            for i in range(num_iters):
                tensor = torch.ones(num_elements, dtype=torch.float16,
                                    device=self._device)
                tag = 90000 + (size % 100)

                torch.cuda.synchronize()
                start = time.perf_counter()

                handle = self.isend(tensor, dst=peer_pe, tag=tag)
                handle.wait()

                torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - start) * 1000
                latencies.append(elapsed_ms)

            results[size] = {
                'cold_ms': latencies[0] if latencies else 0,
                'warm_avg_ms': (
                    sum(latencies[1:]) / max(1, len(latencies) - 1)
                    if len(latencies) > 1 else 0
                ),
                'all_ms': latencies,
            }
            logger.info(
                f"[NVSHMEM bench] size={size}: "
                f"cold={results[size]['cold_ms']:.3f}ms, "
                f"warm={results[size]['warm_avg_ms']:.3f}ms"
            )

        return results

    @staticmethod
    def initialize(device: torch.device,
                   max_tensor_bytes: int = 64 * 1024 * 1024,
                   num_slots: int = 8) -> 'NVSHMEMCommunicator':
        """
        Full initialization: nvshmem_init + allocate symmetric heap.

        Must be called after setting CUDA device and before any communication.
        Requires MPI or PMI launcher (mpirun / srun).
        """
        lib = _get_nvshmem_lib()
        lib.nvshmem_init()
        my_pe = lib.nvshmem_my_pe()
        n_pes = lib.nvshmem_n_pes()
        logger.info(f"[NVSHMEM] init: pe={my_pe}, n_pes={n_pes}")

        return NVSHMEMCommunicator(
            lib=lib,
            my_pe=my_pe,
            n_pes=n_pes,
            device=device,
            max_tensor_bytes=max_tensor_bytes,
            num_slots=num_slots,
        )

    @staticmethod
    def is_available() -> bool:
        """Check if NVSHMEM runtime is available."""
        return _check_nvshmem_available()
