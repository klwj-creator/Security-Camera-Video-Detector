from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from cctv_detector.cli import main
from cctv_detector.evidence import ConfirmationConfig
from cctv_detector.runner import run_session
from cctv_detector.video import Frame, VideoSource
from cctv_detector.vision import Logo, Person


def make_video(path, frames=10):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    if not writer.isOpened():
        raise RuntimeError("Test video encoder unavailable")
    for i in range(frames):
        writer.write(np.full((48, 64, 3), i * 10, dtype=np.uint8))
    writer.release()


class Frames:
    live = False

    def __init__(self, items):
        self.items = items
        self.closed = False

    def __iter__(self):
        return iter(self.items)

    def close(self):
        self.closed = True


class VideoAndRunnerTests(unittest.TestCase):
    def test_real_video_decode_and_spaced_extraction(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.avi"
            make_video(path)
            with VideoSource(str(path)) as source:
                frames = list(source)
                self.assertEqual(len(frames), 10)
                self.assertAlmostEqual(frames[-1].timestamp, 0.9)
                self.assertEqual(frames[0].image.shape, (48, 64, 3))
            with redirect_stdout(io.StringIO()):
                code = main(["extract", str(path), "--output", str(Path(folder) / "frames"),
                             "--every", "0.3", "--max-images", "3"])
            self.assertEqual(code, 0)
            manifest_path = next((Path(folder) / "frames").glob("*/manifest.json"))
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual([row["frame_index"] for row in manifest["images"]], [0, 3, 6])
            self.assertTrue(all((manifest_path.parent / row["file"]).is_file() for row in manifest["images"]))

    def test_confirmed_event_has_readable_snapshot_and_no_duplicates(self):
        with tempfile.TemporaryDirectory() as folder:
            image = np.zeros((48, 64, 3), dtype=np.uint8)
            source = Frames([Frame(image, i, i * 0.2) for i in range(6)])
            person = Person(3, (0, 0, 50, 48), (0, 5, 50, 35), [Logo((10, 10, 20, 20), 0.9)])
            pipeline = Mock()
            pipeline.detect.return_value = [person]
            with redirect_stdout(io.StringIO()):
                output, summary = run_session(source, pipeline, folder, "target_logo", ConfirmationConfig())
            events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["video_seconds"], 0.4)
            self.assertEqual(events[0]["track_id"], 3)
            self.assertIsNotNone(cv2.imread(str(output / events[0]["snapshot"])))
            self.assertEqual(summary["processed_frames"], 6)
            self.assertEqual(summary["status"], "finished")
            self.assertTrue(source.closed)

    def test_reconnect_resets_tracking_and_partial_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            image = np.zeros((48, 64, 3), dtype=np.uint8)
            source = Frames([Frame(image, i, i * 0.2, int(i >= 2)) for i in range(4)])
            pipeline = Mock()
            pipeline.detect.return_value = [Person(1, (0, 0, 40, 40), (0, 5, 40, 30),
                                                    [Logo((10, 10, 20, 20), 0.9)])]
            output, summary = run_session(source, pipeline, folder, "target_logo", ConfirmationConfig())
            self.assertEqual(summary["alerts"], 0)
            self.assertEqual(summary["observed_reconnects"], 1)
            pipeline.reset.assert_called_once()
            self.assertEqual((output / "events.jsonl").read_text(), "")

    def test_failed_inference_writes_failure_summary_and_closes_source(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Frames([Frame(np.zeros((48, 64, 3), dtype=np.uint8), 0, 0)])
            pipeline = Mock()
            pipeline.detect.side_effect = RuntimeError("inference failed")
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                run_session(source, pipeline, folder, "target_logo", ConfirmationConfig())
            summary = json.loads(next(Path(folder).glob("*/summary.json")).read_text())
            self.assertEqual(summary["status"], "failed")
            self.assertTrue(source.closed)

    def test_empty_source_cannot_report_success(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, "no readable frames"):
                run_session(Frames([]), Mock(), folder, "target_logo", ConfirmationConfig())

    def test_live_reader_drops_backlog_and_reports_disconnect(self):
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        capture = Mock()
        capture.get.side_effect = lambda prop: {cv2.CAP_PROP_FPS: 10, cv2.CAP_PROP_FRAME_COUNT: 0,
                                                cv2.CAP_PROP_FRAME_WIDTH: 64, cv2.CAP_PROP_FRAME_HEIGHT: 48}.get(prop, 0)
        capture.read.side_effect = [(True, image) for _ in range(10)] + [(False, None)]
        with patch.object(VideoSource, "_open", return_value=capture):
            source = VideoSource("rtsp://example.invalid/live", reconnects=0)
        # Run the producer to completion before consuming: only the latest frame should remain.
        source._pump()
        self.assertEqual(source.queue.qsize(), 1)
        self.assertEqual(source.queue.get_nowait().index, 9)
        self.assertTrue(source.done.is_set())
        self.assertIsNotNone(source.error)
        source.close()


if __name__ == "__main__":
    unittest.main()
