"""Tests for NVSHMEM communicator module (import and availability checks)."""

import pytest


def test_nvshmem_module_importable():
    """Ensure the module can be imported without errors."""
    from src.distributed.nvshmem_communicator import (
        NVSHMEMCommunicator,
        NVSHMEMWorkHandle,
        _check_nvshmem_available,
    )
    assert NVSHMEMCommunicator is not None
    assert NVSHMEMWorkHandle is not None


def test_nvshmem_availability_check():
    """is_available() should return bool without crashing."""
    from src.distributed.nvshmem_communicator import NVSHMEMCommunicator
    result = NVSHMEMCommunicator.is_available()
    assert isinstance(result, bool)


def test_nvshmem_constants():
    """Verify NVSHMEM comparison operator constants."""
    from src.distributed.nvshmem_communicator import (
        NVSHMEM_CMP_EQ, NVSHMEM_CMP_GE,
    )
    assert NVSHMEM_CMP_EQ == 0
    assert NVSHMEM_CMP_GE == 3


def test_work_handle_interface():
    """NVSHMEMWorkHandle has wait() and is_completed() methods."""
    from src.distributed.nvshmem_communicator import NVSHMEMWorkHandle
    assert hasattr(NVSHMEMWorkHandle, 'wait')
    assert hasattr(NVSHMEMWorkHandle, 'is_completed')


@pytest.mark.skipif(True, reason="NVSHMEM runtime not installed in CI")
def test_nvshmem_initialize():
    """Full initialization test (requires NVSHMEM + MPI launcher)."""
    import torch
    from src.distributed.nvshmem_communicator import NVSHMEMCommunicator
    comm = NVSHMEMCommunicator.initialize(device=torch.device('cuda:0'))
    assert comm is not None
    comm.cleanup()
