"""Export to the Hugging Face Hub."""

import tempfile
import unittest
from pathlib import Path

import skops.io as sio

from opendecision.hub import export
from opendecision.i18n import LocalizedError


class ExportTest(unittest.TestCase):
    def test_writes_loadable_skops_model_and_model_card(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "flights"
            result = export("passagens", destination, repo="user/test", license_id="mit")

            self.assertEqual(result["files"], ["README.md", "config.toml", "model.skops", "training_data.csv"])
            model = sio.load(destination / "model.skops")
            state = "Tenho origem GRU destino REC ida 12/11 e ainda não consultei voos"
            self.assertEqual(model.predict([state])[0], "search")

            card = (destination / "README.md").read_text(encoding="utf-8")
            self.assertTrue(card.startswith("---\nlanguage: [pt]\n"))
            self.assertIn("license: mit", card)
            self.assertIn('hf_hub_download("user/test", "model.skops")', card)
            self.assertIn("## Decision policy", card)
            for section in ("## Model details", "## Uses", "## Input format", "## Actions",
                            "## Training details", "## Bias, risks and limitations", "## Citation"):
                self.assertIn(section, card)
            self.assertIn("model-index:", card)
            self.assertIn("| `confirm` | medium | 0.8 |", card)

    def test_rejects_unknown_scenario(self):
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(LocalizedError):
            export("hotels", Path(folder))


if __name__ == "__main__":
    unittest.main()
