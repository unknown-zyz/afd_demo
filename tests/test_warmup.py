"""Tests for NCCL P2P warmup and keepalive module."""

import time
import threading
import unittest
from unittest.mock import patch, MagicMock

import torch


class TestP2PKeepaliveLogic(unittest.TestCase):
    """Test P2PKeepalive start/stop/notify_comm without real NCCL."""

    def _make_keepalive(self, **kwargs):
        """Create a P2PKeepalive with mocked NCCL internals."""
        from src.distributed.warmup import P2PKeepalive

        defaults = dict(
            peer_rank=1,
            device=torch.device("cpu"),
            interval_s=0.1,
            tensor_size=64,
            dtype=torch.float16,
        )
        defaults.update(kwargs)
        return P2PKeepalive(**defaults)

    def test_init_defaults(self):
        ka = self._make_keepalive()
        self.assertEqual(ka.peer_rank, 1)
        self.assertEqual(ka.interval_s, 0.1)
        self.assertEqual(ka._heartbeat_count, 0)
        self.assertIsNone(ka._thread)

    def test_notify_comm_updates_time(self):
        ka = self._make_keepalive()
        old_time = ka._last_comm_time
        time.sleep(0.01)
        ka.notify_comm()
        self.assertGreater(ka._last_comm_time, old_time)

    def test_start_stop_lifecycle(self):
        """Start and stop should not raise, thread should be created and joined."""
        ka = self._make_keepalive()

        # Patch _run so the thread doesn't actually do NCCL ops
        ka._run = lambda: ka._stop_event.wait()

        ka.start()
        self.assertIsNotNone(ka._thread)
        self.assertTrue(ka._thread.is_alive())

        ka.stop()
        self.assertFalse(ka._thread.is_alive())

    def test_stop_without_start(self):
        """Stop without start should not raise."""
        ka = self._make_keepalive()
        ka.stop()  # should be a no-op

    def test_multiple_notify_comm(self):
        """Multiple notify_comm calls should always update monotonically."""
        ka = self._make_keepalive()
        times = []
        for _ in range(5):
            ka.notify_comm()
            times.append(ka._last_comm_time)
            time.sleep(0.005)

        for i in range(1, len(times)):
            self.assertGreaterEqual(times[i], times[i - 1])

    @patch("src.distributed.warmup.dist")
    def test_heartbeat_runs_when_idle(self, mock_dist):
        """When idle long enough, heartbeat should fire."""
        mock_dist.get_rank.return_value = 0

        # Mock isend/irecv to return mock handles
        mock_handle = MagicMock()
        mock_handle.wait.return_value = None
        mock_dist.isend.return_value = mock_handle
        mock_dist.irecv.return_value = mock_handle

        ka = self._make_keepalive(interval_s=0.05)
        # Force last comm to be old
        ka._last_comm_time = time.monotonic() - 10

        ka.start()
        time.sleep(0.2)  # let a few heartbeats fire
        ka.stop()

        self.assertGreater(ka._heartbeat_count, 0)
        self.assertTrue(mock_dist.isend.called)


class TestWarmupP2PSignature(unittest.TestCase):
    """Test warmup_p2p function interface (no real NCCL)."""

    def test_import(self):
        from src.distributed.warmup import warmup_p2p
        self.assertTrue(callable(warmup_p2p))

    def test_keepalive_import(self):
        from src.distributed.warmup import P2PKeepalive
        self.assertTrue(callable(P2PKeepalive))

    def test_distributed_context_warmup_method(self):
        """DistributedContext should have a warmup method."""
        from src.distributed import DistributedContext
        self.assertTrue(hasattr(DistributedContext, 'warmup'))


if __name__ == "__main__":
    unittest.main()
