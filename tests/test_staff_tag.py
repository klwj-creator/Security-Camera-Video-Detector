import tempfile
from pathlib import Path
import unittest

import cv2
import numpy as np

from cctv_detector.staff_tag import TagTemplates, restore_box


class StaffTagTests(unittest.TestCase):
    def test_rotation_boxes_return_to_original_frame(self):
        original = [10, 20, 40, 70]
        rotated = {0: [10, 20, 40, 70], 1: [20, 60, 70, 90],
                   2: [60, 130, 90, 180], 3: [130, 10, 180, 40]}
        for rotation, box in rotated.items():
            with self.subTest(rotation=rotation):
                np.testing.assert_allclose(restore_box(box, rotation, 100, 200), original)

    def test_reference_pattern_is_localized_without_padding_penalty(self):
        with tempfile.TemporaryDirectory() as folder:
            template = np.full((12, 24), 190, dtype=np.uint8)
            for x in (3, 10, 17):
                template[1:8, x:x+3] = 20
            path = Path(folder) / "tag.png"
            cv2.imwrite(str(path), template)
            image = np.full((80, 100), 190, dtype=np.uint8)
            image[30:42, 50:74] = template
            matcher = TagTemplates(path, widths=(24,), ratios=(0.5,), angle_step=90)
            match = matcher.best(image)
            self.assertGreater(match.score, 0.99)
            self.assertEqual(match.box, (50, 30, 74, 42))
            self.assertEqual(match.center, (62, 36))
            # Searching an ROI must restore full-frame coordinates.
            self.assertEqual(matcher.best(image, (40, 20, 90, 70)).center, (62, 36))

    def test_upside_down_pattern_is_supported(self):
        with tempfile.TemporaryDirectory() as folder:
            pattern = np.full((12, 24), 220, dtype=np.uint8)
            for x in (2, 10, 18):
                pattern[1:9, x:x+3] = 10
            path = Path(folder) / "tag.png"
            cv2.imwrite(str(path), pattern)
            image = np.full((80, 100), 220, dtype=np.uint8)
            image[30:42, 50:74] = np.rot90(pattern, 2)
            match = TagTemplates(path, widths=(24,), ratios=(0.5,), angle_step=90).best(image)
            self.assertGreater(match.score, 0.99)
            # Repeated bars on a uniform background can match after a small
            # translation at more than one angle. Localization, not a unique
            # orientation, is the required behavior.
            self.assertLessEqual(np.linalg.norm(np.array(match.center) - [62, 36]), 3)

    def test_empty_or_constant_template_cannot_count_as_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "constant.png"
            cv2.imwrite(str(path), np.full((20, 20), 100, dtype=np.uint8))
            with self.assertRaises(ValueError):
                TagTemplates(path)


if __name__ == "__main__":
    unittest.main()
