from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from cctv_detector.vision import YoloPipeline, clip_box


class Tensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


def result(coords, ids=None, scores=None):
    return SimpleNamespace(boxes=SimpleNamespace(
        xyxy=Tensor(coords), id=None if ids is None else Tensor(ids),
        conf=Tensor(scores or [])))


class VisionTests(unittest.TestCase):
    def make_pipeline(self, folder, people, logos, class_name="target_logo"):
        path = Path(folder) / "logo.pt"
        path.touch()
        person_model = Mock(task="detect", names={0: "person"})
        logo_model = Mock(task="detect", names={0: "target_logo"})
        person_model.track.return_value = [people]
        logo_model.predict.return_value = logos
        factory = Mock(side_effect=[person_model, logo_model])
        with patch.dict("sys.modules", {"ultralytics": SimpleNamespace(YOLO=factory)}):
            pipeline = YoloPipeline("people.pt", str(path), class_name,
                                    shirt_top=0.2, shirt_bottom=0.8)
        return pipeline, person_model, logo_model

    def test_original_resolution_crops_map_back_to_correct_person(self):
        with tempfile.TemporaryDirectory() as folder:
            people = result([[-5, 0, 80, 200], [100, 20, 180, 120]], [7, 8])
            logos = [result([[-2, 2, 20, 30]], scores=[0.9]), result([[3, 4, 13, 14]], scores=[0.8])]
            pipeline, person_model, logo_model = self.make_pipeline(folder, people, logos)
            frame = np.zeros((150, 200, 3), dtype=np.uint8)
            frame[30:120, :80] = 42
            detections = pipeline.detect(frame)
            self.assertEqual(detections[0].box, (0, 0, 80, 150))
            self.assertEqual(detections[0].logos[0].box, (0, 32, 20, 60))
            self.assertEqual(detections[1].logos[0].box, (103, 44, 113, 54))
            crops = logo_model.predict.call_args.args[0]
            self.assertEqual(crops[0].shape, (90, 80, 3))
            self.assertTrue(np.all(crops[0] == 42))
            self.assertTrue(person_model.track.call_args.kwargs["persist"])
            self.assertEqual(person_model.track.call_args.kwargs["tracker"], "bytetrack.yaml")

    def test_untracked_detections_do_not_create_alert_candidates(self):
        with tempfile.TemporaryDirectory() as folder:
            pipeline, _, model = self.make_pipeline(folder, result([[0, 0, 10, 10]]), [])
            self.assertEqual(pipeline.detect(np.zeros((20, 20, 3), dtype=np.uint8)), [])
            model.predict.assert_not_called()

    def test_unknown_logo_class_fails_instead_of_using_first_class(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "absent"):
                self.make_pipeline(folder, result([], []), [], class_name="wrong_logo")

    def test_degenerate_and_nonfinite_boxes_are_discarded(self):
        self.assertIsNone(clip_box((10, 10, 0, 0), 100, 100))
        self.assertIsNone(clip_box((float("nan"), 0, 10, 10), 100, 100))
        self.assertIsNone(clip_box((110, 0, 120, 10), 100, 100))


if __name__ == "__main__":
    unittest.main()
