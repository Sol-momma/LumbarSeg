from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.hash_validation_cohort import build_rows, read_manifest, write_manifest


class ValidationCohortHashTests(unittest.TestCase):
    def test_manifest_detects_mask_content_change(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "images").mkdir()
            (root / "masks").mkdir()
            filename = "sample.npz"
            np.savez_compressed(root / "images" / filename, image=np.array([[1.0, 2.0]], dtype=np.float32))
            np.savez_compressed(root / "masks" / filename, mask=np.array([[0, 1]], dtype=np.uint8))

            original = build_rows(root, [filename])
            manifest = root / "manifest.tsv"
            write_manifest(manifest, original)
            self.assertEqual(read_manifest(manifest), original)

            np.savez_compressed(root / "masks" / filename, mask=np.array([[0, 2]], dtype=np.uint8))
            self.assertNotEqual(build_rows(root, [filename]), original)

    def test_logical_hash_ignores_npz_container_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "images").mkdir()
            (root / "masks").mkdir()
            filename = "sample.npz"
            image = np.array([[1.0, 2.0]], dtype=np.float32)
            mask = np.array([[0, 1]], dtype=np.uint8)
            np.savez_compressed(root / "images" / filename, image=image)
            np.savez_compressed(root / "masks" / filename, mask=mask)
            original = build_rows(root, [filename])

            np.savez_compressed(root / "images" / filename, image=image)
            np.savez_compressed(root / "masks" / filename, mask=mask)
            self.assertEqual(build_rows(root, [filename]), original)


if __name__ == "__main__":
    unittest.main()
