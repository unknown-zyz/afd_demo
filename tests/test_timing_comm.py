import torch
import pytest

from src.utils.timing import EventType, TimingTracker


class NoFutureHandle:
    def get_future(self):
        raise RuntimeError("future unsupported")


def test_record_send_enqueue_records_payload_and_source():
    tracker = TimingTracker("attention", num_layers=1, num_micro_batches=1, comm_timing_mode="enqueue")
    tensor = torch.zeros(2, 3, dtype=torch.float32)

    tracker.record_send(NoFutureHandle(), layer_idx=0, mb_idx=0, start_time=tracker.start_time, tensor=tensor)
    timing = tracker.finish().to_dict()

    event = timing["events"][0]
    assert event["type"] == EventType.SEND_TRANSFER.value
    assert event["tensor_bytes"] == 2 * 3 * 4
    assert event["completion_source"] == "enqueue"
    assert timing["comm_timing_mode"] == "enqueue"


def test_record_send_completion_falls_back_to_observed_wait():
    handle = NoFutureHandle()
    tracker = TimingTracker("attention", num_layers=1, num_micro_batches=1, comm_timing_mode="completion")
    tensor = torch.zeros(4, dtype=torch.float16)

    tracker.record_send(handle, layer_idx=0, mb_idx=0, start_time=tracker.start_time, tensor=tensor)
    tracker.observe_send_completion(handle)
    timing = tracker.finish().to_dict()

    event = timing["events"][0]
    assert event["type"] == EventType.SEND_TRANSFER.value
    assert event["tensor_bytes"] == 4 * 2
    assert event["completion_source"] == "observed_wait"
    assert timing["comm_timing_mode"] == "completion"


def test_ep_overlap_timing_aggregates_wait_and_hidden_time():
    tracker = TimingTracker("ffn_coordinator", num_layers=1, num_micro_batches=2)
    start = tracker.start_time

    tracker.record_event(EventType.EP_DISPATCH_WAIT, 0, 0, start, start + 0.001)
    tracker.record_event(EventType.EP_REDUCE_WAIT, 0, 0, start, start + 0.002)
    tracker.record_event(EventType.EP_OVERLAP_HIDDEN, 0, 0, start, start + 0.003)
    timing = tracker.finish().to_dict()

    assert timing["total_ep_dispatch_wait_ms"] == pytest.approx(1.0)
    assert timing["total_ep_reduce_wait_ms"] == pytest.approx(2.0)
    assert timing["total_ep_overlap_hidden_ms"] == pytest.approx(3.0)
