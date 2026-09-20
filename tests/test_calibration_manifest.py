import json
import tempfile
import unittest
from pathlib import Path

from app.core.calibration_manifest import (
    sha256_file,
    update_calibration_manifest,
    verify_calibration_manifest,
)


class CalibrationManifestTests(unittest.TestCase):
    def test_manifest_records_artifacts_sources_and_stable_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "blank.tif"
            source.write_bytes(b"source pixels")
            artifact = root / "fit_parameters.csv"
            artifact.write_text("Channel,a,b,c\nR,1,2,3\n", encoding="utf-8")

            manifest = update_calibration_manifest(
                root,
                "dose_calibration",
                {"model": "test"},
                [source],
            )
            loaded = json.loads((root / "calibration_manifest.json").read_text(encoding="utf-8"))

            self.assertEqual(loaded["calibration_id"], manifest["calibration_id"])
            self.assertEqual(
                loaded["artifacts"]["fit_parameters.csv"]["sha256"],
                sha256_file(artifact),
            )
            self.assertEqual(loaded["dose_calibration"]["sources"][0]["sha256"], sha256_file(source))
            self.assertEqual(list(root.glob("calibration_manifest_*.json")), [])
            self.assertTrue(verify_calibration_manifest(root)["fit_parameters.csv"])

            artifact.write_text("tampered\n", encoding="utf-8")
            self.assertFalse(verify_calibration_manifest(root)["fit_parameters.csv"])


if __name__ == "__main__":
    unittest.main()
