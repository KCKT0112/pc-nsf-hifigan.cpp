"""Regression tests for restricted checkpoint loading and generator extraction."""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.dont_write_bytecode = True

CONVERTER = Path(__file__).resolve().parents[1] / "converter/convert_hifigan.py"
spec = importlib.util.spec_from_file_location("convert_hifigan", CONVERTER)
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


class ForbiddenObject:
    def __reduce__(self):
        # Deserializing with unrestricted pickle raises this unique exception.
        return (eval, ("(_ for _ in ()).throw(RuntimeError('PICKLE_EXECUTED'))",))


class ConverterTests(unittest.TestCase):
    def run_converter(self, root, checkpoint):
        ckpt = root / "model.ckpt"
        torch.save(checkpoint, ckpt)
        config = root / "config.json"
        config.write_text(json.dumps(next(iter(converter.OFFICIAL_PRESETS.values()))))
        output = root / "model_f32.gguf"
        process = subprocess.run(
            [sys.executable, str(CONVERTER), "--ckpt", str(ckpt), "--config", str(config),
             "--out", str(output), "--dtype", "F32"], capture_output=True, text=True,
        )
        return process, output

    def test_native_and_lightning(self):
        state = {"conv_pre.weight": torch.ones(2, 128, 3), "conv_pre.bias": torch.zeros(2)}
        for checkpoint in [
            {"generator": state},
            {"state_dict": {"generator." + k: v for k, v in state.items()}, "epoch": 10},
        ]:
            with tempfile.TemporaryDirectory() as directory:
                process, output = self.run_converter(Path(directory), checkpoint)
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertTrue(output.is_file())

    def test_pickle_object_rejected_without_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            process, output = self.run_converter(Path(directory), {"generator": {}, "extra": ForbiddenObject()})
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("Weights only load failed", process.stderr)
            self.assertNotIn("RuntimeError: PICKLE_EXECUTED", process.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
