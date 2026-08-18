from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from agimaze_predict.baselines.aux_transformer.config import load_training_config


class AuxTrainConfigTest(unittest.TestCase):
    def test_loads_auxiliary_model_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "aux.toml"
            path.write_text(
                """
[data]
train_files = ["train.jsonl"]
validation_files = ["valid.jsonl"]

[model]
aux_latents_per_token = 2
aux_gate_mode = "learned"
aux_gate_init = 0.1
aux_scale = 0.3
aux_target_weight = 0.1
aux_target_decoder_layers = 2

[run]
output = "run.pt"
""",
                encoding="utf-8",
            )
            values = load_training_config(path)

        self.assertEqual(values["aux_latents_per_token"], 2)
        self.assertEqual(values["aux_gate_mode"], "learned")
        self.assertEqual(values["aux_gate_init"], 0.1)
        self.assertEqual(values["aux_scale"], 0.3)
        self.assertEqual(values["aux_target_weight"], 0.1)
        self.assertEqual(values["aux_target_decoder_layers"], 2)

    def test_rejects_invalid_auxiliary_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "aux.toml"
            path.write_text('[model]\naux_gate_mode = "invalid"\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "aux_gate_mode must be one of"):
                load_training_config(path)


if __name__ == "__main__":
    unittest.main()
