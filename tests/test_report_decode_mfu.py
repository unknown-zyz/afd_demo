import csv
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "report_decode_mfu.py"
SPEC = importlib.util.spec_from_file_location("report_decode_mfu", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_qwen3_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "model_type": "qwen3_moe",
                "hidden_size": 2048,
                "intermediate_size": 6144,
                "moe_intermediate_size": 768,
                "num_hidden_layers": 48,
                "num_attention_heads": 32,
                "num_key_value_heads": 4,
                "num_experts": 128,
                "num_experts_per_tok": 8,
                "decoder_sparse_step": 1,
                "vocab_size": 151936,
            }
        )
    )


class ReportDecodeMFUTest(unittest.TestCase):
    def test_build_row_infers_ep_topology_and_mfu(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            decode_dir = tmp_path / "decode-dbo"
            decode_dir.mkdir()
            timing_path = decode_dir / "timing_attention_decode-dbo_npu_ep7_b2_s128_t20.json"
            timing_path.write_text(
                json.dumps(
                    {
                        "decode_tpot_ms": 10.0,
                        "decode_steps": 19,
                        "actual_prompt_len": 128,
                        "routing_backend": "coordinator",
                        "routing_update_mode": "oneshot",
                        "routing_table_version": 1,
                        "routing_poll_count": 0,
                        "routing_poll_ms": 0.0,
                    }
                )
            )
            model_config = tmp_path / "config.json"
            _write_qwen3_config(model_config)

            shape = MODULE.load_model_shape(model_config)
            row = MODULE.build_row(
                timing_path,
                shape,
                default_attn_devices=1,
                default_ffn_devices=1,
                peak_tflops_per_device=800.0,
            )

            self.assertEqual(row["ffn_devices"], 7)
            self.assertEqual(row["attn_devices"], 1)
            self.assertEqual(row["total_devices"], 8)
            self.assertAlmostEqual(row["throughput_tok_s"], 200.0)
            self.assertGreater(row["system_achieved_tflops"], 0)
            self.assertIsNotNone(row["system_mfu"])
            self.assertEqual(row["routing_backend"], "coordinator")

    def test_main_writes_summary_csv(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root = tmp_path / "run"
            decode_dir = root / "decode-dbo"
            decode_dir.mkdir(parents=True)
            (decode_dir / "timing_attention_decode-dbo_npu_ep7_b2_s128_t20.json").write_text(
                json.dumps(
                    {
                        "decode_tpot_ms": 20.0,
                        "decode_steps": 19,
                        "actual_prompt_len": 128,
                    }
                )
            )
            model_dir = tmp_path / "model"
            model_dir.mkdir()
            _write_qwen3_config(model_dir / "config.json")
            out = tmp_path / "decode_mfu_summary.csv"
            argv = sys.argv[:]
            try:
                sys.argv = [
                    "report_decode_mfu.py",
                    "--root",
                    str(root),
                    "--model-name",
                    str(model_dir),
                    "--peak-tflops-per-device",
                    "800",
                    "--out",
                    str(out),
                ]
                self.assertEqual(MODULE.main(), 0)
            finally:
                sys.argv = argv

            with out.open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["ffn_devices"], "7")
            self.assertNotEqual(rows[0]["system_mfu"], "")


if __name__ == "__main__":
    unittest.main()
