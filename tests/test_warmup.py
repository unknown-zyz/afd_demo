"""Tests for NCCL P2P warmup public interfaces."""

import unittest


class TestWarmupP2PSignature(unittest.TestCase):
    """Interface checks that do not require a real distributed process group."""

    def test_import(self):
        from src.distributed.warmup import warmup_p2p

        self.assertTrue(callable(warmup_p2p))

    def test_distributed_context_warmup_method(self):
        from src.distributed import DistributedContext

        self.assertTrue(hasattr(DistributedContext, "warmup"))


if __name__ == "__main__":
    unittest.main()
