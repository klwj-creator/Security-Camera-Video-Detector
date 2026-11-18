"""Optional real-library smoke check, NOT a shirt-logo accuracy test.

Uses the bundled Ultralytics bus image and the COCO person checkpoint in both
stages. Downloads pretrained person weights when missing. No logo is trained.
Run from the project root: .venv/bin/python scripts/smoke_inference.py
"""

import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".runtime/ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime/matplotlib"))


def main():
    import cv2
    from ultralytics import YOLO
    from cctv_detector.evidence import ConfirmationConfig
    from cctv_detector.runner import run_session, write_json
    from cctv_detector.video import VideoSource
    from cctv_detector.vision import YoloPipeline

    weights = ROOT / "models/yolo11n.pt"
    weights.parent.mkdir(exist_ok=True)
    YOLO(str(weights))
    package = Path(importlib.util.find_spec("ultralytics").origin).parent
    image = cv2.imread(str(package / "assets/bus.jpg"))
    if image is None:
        raise RuntimeError("Bundled reference image is unavailable.")
    image = cv2.resize(image, (480, 640))
    pipeline = YoloPipeline(str(weights), str(weights), "person", device="cpu", imgsz=320)
    people = pipeline.detect(image)
    if not people:
        raise RuntimeError("Pretrained person model did not produce tracked people.")
    pipeline.reset()
    fixture = ROOT / ".runtime/person-smoke.avi"
    fixture.parent.mkdir(exist_ok=True)
    writer = cv2.VideoWriter(str(fixture), cv2.VideoWriter_fourcc(*"MJPG"), 5, (480, 640))
    if not writer.isOpened():
        raise RuntimeError("Cannot create smoke-test video.")
    for _ in range(6):
        writer.write(image)
    writer.release()
    with VideoSource(str(fixture)) as source:
        output, summary = run_session(source, pipeline, ROOT / "outputs/integration-smoke", "person",
                                      ConfirmationConfig(), max_frames=6)
    if summary["processed_frames"] != 6:
        raise RuntimeError("Smoke run did not process all six frames.")
    report = {
        "test": "real-library integration smoke, not logo accuracy",
        "person_tracks_on_reference_image": len(people),
        "second_stage": "COCO person detector used as a stand-in; not a custom logo model",
        "output_directory": str(output.relative_to(ROOT)), "summary": summary,
    }
    write_json(ROOT / ".runtime/inference-smoke.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
