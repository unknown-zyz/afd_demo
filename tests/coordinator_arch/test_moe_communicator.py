"""
Tests for MoECommunicator and factory.

This module tests the DeepEP-Ascend wrapper interface using mocks to avoid
requiring actual NPU hardware or deep_ep installation.
"""

import sys
import unittest
from unittest.mock import MagicMock, Mock, patch, call

import pytest

# Mock torch before any imports that need it
if "torch" not in sys.modules:
    mock_torch = MagicMock()
    mock_torch.Tensor = MagicMock
    mock_torch.distributed = MagicMock()
    mock_torch.distributed.ProcessGroup = MagicMock
    mock_torch.device = MagicMock
    sys.modules["torch"] = mock_torch
    sys.modules["torch.distributed"] = mock_torch.distributed


class TestMoECommunicator(unittest.TestCase):
    """Test MoECommunicator with mocked deep_ep."""

    def setUp(self):
        """Set up mocks before each test."""
        # Create a mock deep_ep module
        self.mock_deep_ep = MagicMock()
        self.mock_buffer_instance = MagicMock()
        self.mock_deep_ep.Buffer = Mock(return_value=self.mock_buffer_instance)

        # Install the mock in sys.modules before importing
        sys.modules["deep_ep"] = self.mock_deep_ep

        # Now we can import (will succeed with mock)
        from src.coordinator_arch.comm.moe_communicator import MoECommunicator

        self.MoECommunicator = MoECommunicator

        # Mock torch objects
        self.mock_pg = MagicMock()
        self.mock_device = MagicMock()
        self.mock_device.__str__ = lambda x: "npu:0"

    def tearDown(self):
        """Clean up mocks after each test."""
        # Remove the mock module
        if "deep_ep" in sys.modules:
            del sys.modules["deep_ep"]
        # Clear cached modules
        if "src.coordinator_arch.comm.moe_communicator" in sys.modules:
            del sys.modules["src.coordinator_arch.comm.moe_communicator"]

    def test_init_normal_mode(self):
        """Test initialization with normal mode."""
        comm = self.MoECommunicator(
            ep_group=self.mock_pg,
            hidden_size=4096,
            num_experts=64,
            max_tokens_per_rank=2048,
            device=self.mock_device,
            mode="normal",
            nvl_bytes=128 * 1024 * 1024,
        )

        # Buffer should not be created yet (lazy init)
        self.mock_deep_ep.Buffer.assert_not_called()
        self.assertEqual(comm.mode, "normal")
        self.assertEqual(comm.hidden_size, 4096)
        self.assertEqual(comm.num_experts, 64)

    def test_set_mode_low_latency(self):
        """Test mode switching to low_latency."""
        comm = self.MoECommunicator(
            ep_group=self.mock_pg,
            hidden_size=4096,
            num_experts=64,
            max_tokens_per_rank=2048,
            device=self.mock_device,
            mode="normal",
        )

        # Force buffer creation by calling dispatch
        mock_hidden = MagicMock()
        mock_hidden.shape = (100, 4096)
        mock_topk_indices = MagicMock()
        mock_topk_weights = MagicMock()

        self.mock_buffer_instance.dispatch.return_value = (
            MagicMock(shape=(150, 4096)),
            MagicMock(),
        )

        comm.dispatch(mock_hidden, mock_topk_indices, mock_topk_weights)

        # Buffer should be created with low_latency_mode=False
        self.mock_deep_ep.Buffer.assert_called_once()
        first_call = self.mock_deep_ep.Buffer.call_args
        self.assertEqual(first_call.kwargs["low_latency_mode"], False)

        # Now switch to low_latency mode
        comm.set_mode("low_latency")

        # Buffer should be recreated with low_latency_mode=True
        self.assertEqual(self.mock_deep_ep.Buffer.call_count, 2)
        second_call = self.mock_deep_ep.Buffer.call_args
        self.assertEqual(second_call.kwargs["low_latency_mode"], True)
        self.assertEqual(comm.mode, "low_latency")

    def test_dispatch_combines_forward(self):
        """Test dispatch and combine forward calls to buffer."""
        comm = self.MoECommunicator(
            ep_group=self.mock_pg,
            hidden_size=512,
            num_experts=8,
            max_tokens_per_rank=256,
            device=self.mock_device,
        )

        # Mock inputs
        mock_hidden = MagicMock()
        mock_hidden.shape = (64, 512)
        mock_topk_indices = MagicMock()
        mock_topk_weights = MagicMock()

        # Mock dispatch output
        mock_dispatched = MagicMock()
        mock_dispatched.shape = (80, 512)
        mock_expert_nums = MagicMock()
        self.mock_buffer_instance.dispatch.return_value = (
            mock_dispatched,
            mock_expert_nums,
        )

        # Test dispatch
        handle = comm.dispatch(mock_hidden, mock_topk_indices, mock_topk_weights)

        # Verify buffer.dispatch was called
        self.mock_buffer_instance.dispatch.assert_called_once_with(
            mock_hidden, mock_topk_indices, mock_topk_weights
        )

        # Verify handle structure
        self.assertEqual(handle["recv_hidden"], mock_dispatched)
        self.assertEqual(handle["expert_token_nums"], mock_expert_nums)
        self.assertEqual(handle["topk_indices"], mock_topk_indices)
        self.assertEqual(handle["topk_weights"], mock_topk_weights)

        # Mock FFN outputs
        mock_ffn_outputs = MagicMock()
        mock_ffn_outputs.shape = (80, 512)
        mock_combined = MagicMock()
        mock_combined.shape = (64, 512)
        self.mock_buffer_instance.combine.return_value = mock_combined

        # Test combine
        result = comm.combine(mock_ffn_outputs, handle)

        # Verify buffer.combine was called with correct args
        self.mock_buffer_instance.combine.assert_called_once_with(
            mock_ffn_outputs, mock_topk_indices, mock_topk_weights
        )
        self.assertEqual(result, mock_combined)

    def test_update_routing_table(self):
        """Test routing table update."""
        comm = self.MoECommunicator(
            ep_group=self.mock_pg,
            hidden_size=512,
            num_experts=8,
            max_tokens_per_rank=256,
            device=self.mock_device,
            mode="normal",
        )

        table = {
            "version": 1,
            "expert_to_rank": [0, 0, 1, 1, 2, 2, 3, 3],
            "mode": "normal",
        }

        with patch("torch.tensor") as mock_tensor:
            comm.update_routing_table(table)

        self.assertEqual(comm.version, 1)
        # Mode should remain normal (no change)
        self.assertEqual(comm.mode, "normal")


class TestFactory(unittest.TestCase):
    """Test factory function with mocked modules."""

    def tearDown(self):
        """Clean up modules after each test."""
        modules_to_remove = [
            "deep_ep",
            "src.coordinator_arch.comm.moe_communicator",
            "src.coordinator_arch.comm.factory",
        ]
        for mod in modules_to_remove:
            if mod in sys.modules:
                del sys.modules[mod]

    def test_factory_with_deepep_available(self):
        """Test factory returns MoECommunicator when deep_ep is available."""
        # Mock deep_ep
        mock_deep_ep = MagicMock()
        sys.modules["deep_ep"] = mock_deep_ep

        # Import factory (will see the mocked deep_ep)
        from src.coordinator_arch.comm.factory import build_communicator

        # Mock torch.distributed
        with patch("torch.distributed.ProcessGroup"):
            mock_pg = MagicMock()
            mock_device = MagicMock()

            comm = build_communicator(
                prefer_deepep=True,
                ep_group=mock_pg,
                hidden_size=512,
                num_experts=8,
                max_tokens_per_rank=256,
                device=mock_device,
            )

            # Should return MoECommunicator
            self.assertEqual(
                type(comm).__name__, "MoECommunicator"
            )

    def test_factory_with_deepep_unavailable(self):
        """Test factory returns FallbackMoECommunicator when deep_ep is unavailable."""
        # Ensure deep_ep is NOT available
        if "deep_ep" in sys.modules:
            del sys.modules["deep_ep"]

        # Import factory
        from src.coordinator_arch.comm.factory import build_communicator

        # Mock torch.distributed methods
        with patch("torch.distributed.get_world_size", return_value=4), \
             patch("torch.distributed.get_rank", return_value=0):
            mock_pg = MagicMock()
            mock_device = MagicMock()

            comm = build_communicator(
                prefer_deepep=True,  # Even if preferred, should fallback
                ep_group=mock_pg,
                hidden_size=512,
                num_experts=8,
                max_tokens_per_rank=256,
                device=mock_device,
            )

            # Should return FallbackMoECommunicator
            self.assertEqual(
                type(comm).__name__, "FallbackMoECommunicator"
            )

    def test_factory_prefer_fallback(self):
        """Test factory returns FallbackMoECommunicator when prefer_deepep=False."""
        # Mock deep_ep (available but not preferred)
        mock_deep_ep = MagicMock()
        sys.modules["deep_ep"] = mock_deep_ep

        from src.coordinator_arch.comm.factory import build_communicator

        with patch("torch.distributed.get_world_size", return_value=4), \
             patch("torch.distributed.get_rank", return_value=0):
            mock_pg = MagicMock()
            mock_device = MagicMock()

            comm = build_communicator(
                prefer_deepep=False,  # Explicitly prefer fallback
                ep_group=mock_pg,
                hidden_size=512,
                num_experts=8,
                max_tokens_per_rank=256,
                device=mock_device,
            )

            # Should return FallbackMoECommunicator
            self.assertEqual(
                type(comm).__name__, "FallbackMoECommunicator"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
