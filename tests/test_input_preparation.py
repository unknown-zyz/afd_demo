import torch

from src.main import get_effective_prefill_seq_len, tokenize_batch_prompts


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, prompts, return_tensors, padding, truncation, max_length):
        self.calls.append(
            {
                "prompts": prompts,
                "return_tensors": return_tensors,
                "padding": padding,
                "truncation": truncation,
                "max_length": max_length,
            }
        )
        if padding == "max_length":
            seq_len = max_length
        else:
            seq_len = min(max(len(prompt.split()) for prompt in prompts), max_length)
        shape = (len(prompts), seq_len)
        return {
            "input_ids": torch.zeros(shape, dtype=torch.long),
            "attention_mask": torch.ones(shape, dtype=torch.long),
        }


def test_effective_prefill_seq_len_prefers_explicit_value():
    assert get_effective_prefill_seq_len(512, 128) == 512


def test_effective_prefill_seq_len_falls_back_to_max_seq_len():
    assert get_effective_prefill_seq_len(None, 128) == 128


def test_tokenize_batch_prompts_uses_fixed_padding_for_experiment_seq():
    tokenizer = FakeTokenizer()

    inputs, target_seq_len = tokenize_batch_prompts(
        tokenizer,
        ["hello world"] * 2,
        prefill_seq_len=512,
        max_seq_len=128,
    )

    assert target_seq_len == 512
    assert inputs["input_ids"].shape == (2, 512)
    assert tokenizer.calls[-1]["padding"] == "max_length"
    assert tokenizer.calls[-1]["max_length"] == 512


def test_tokenize_batch_prompts_keeps_dynamic_padding_without_experiment_seq():
    tokenizer = FakeTokenizer()

    inputs, target_seq_len = tokenize_batch_prompts(
        tokenizer,
        ["hello world"] * 2,
        prefill_seq_len=None,
        max_seq_len=128,
    )

    assert target_seq_len == 128
    assert inputs["input_ids"].shape == (2, 2)
    assert tokenizer.calls[-1]["padding"] is True
    assert tokenizer.calls[-1]["max_length"] == 128
