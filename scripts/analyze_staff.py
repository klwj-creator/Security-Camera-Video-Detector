"""Offline reference-tag baseline; exports evidence and person tracks for review.

No sample frame ranges, clothing colors, or staff identities are hardcoded.
The PDF's two low-resolution tag exemplars are used as templates.
"""

import argparse
from dataclasses import asdict
import json
import hashlib
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".runtime/ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime/matplotlib"))


def main():
    import cv2
    import numpy as np
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.byte_tracker import BYTETracker
    from cctv_detector.staff_tag import OverheadPeople, TagTemplates

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("--output", default="outputs/staff-baseline")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--tag-every", type=int, default=3, help="Check tags every N decoded frames; track on every processed frame")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.78)
    parser.add_argument("--motion-min", type=float, default=0.12)
    parser.add_argument("--warmup-seconds", type=float, default=4)
    args = parser.parse_args()
    if args.stride < 1 or args.tag_every < 1 or not 0 <= args.motion_min <= 1 or not 0 < args.threshold <= 1:
        parser.error("Invalid stride, motion fraction, or correlation threshold.")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "observations.json"
    if cache.exists():
        parser.error("Output already contains observations; choose a new output directory.")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError("Video must have a usable nominal frame rate.")
    width, height = int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    backgrounds = []
    for index in np.linspace(0, max(0, min(total-1, int(fps*args.warmup_seconds))), 9).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, image = cap.read()
        if ok:
            backgrounds.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    if not backgrounds:
        raise RuntimeError("Video has no readable frames.")
    background = np.median(backgrounds, axis=0).astype(np.uint8)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    people = OverheadPeople(str(ROOT / "models/yolo11n.pt"), device=args.device)
    templates = TagTemplates([ROOT / "configs/staff-tag-dark.png", ROOT / "configs/staff-tag-light.png"])
    tracker = BYTETracker(SimpleNamespace(track_high_thresh=0.2, track_low_thresh=0.1,
                                         new_track_thresh=0.2, track_buffer=max(1, round(fps/args.stride)),
                                         match_thresh=0.9, fuse_score=True))
    observations = []
    index = 0
    started = time.monotonic()
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if index % args.stride:
                index += 1
                continue
            raw = people.detect(image)
            tracks = tracker.update(Boxes(raw, image.shape[:2]), image)
            difference = cv2.absdiff(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), background)
            frame_rows = []
            for row in tracks:
                x1, y1, x2, y2 = row[:4]
                a, b = max(0, int(x1)), max(0, int(y1))
                c, d = min(width, int(x2)), min(height, int(y2))
                region = difference[b:d, a:c]
                motion = float(np.mean(region > 30)) if region.size else 0
                tag = templates.best(image, (a, b, c, d)) if motion >= args.motion_min and index % args.tag_every == 0 else None
                frame_rows.append({"track_id": int(row[4]), "box": [float(v) for v in row[:4]],
                                   "person_score": float(row[5]), "motion_fraction": motion,
                                   "tag": asdict(tag) if tag else None,
                                   "tag_positive": bool(tag and tag.score >= args.threshold and tag.contrast >= 12)})
            observations.append({"frame": index, "seconds": index/fps, "people": frame_rows,
                                 "raw_detections": raw.tolist()})
            if index % 100 == 0:
                positives = [p["track_id"] for p in frame_rows if p["tag_positive"]]
                print(f"Frame {index}/{total}; direct tag candidates {positives}", flush=True)
            index += 1
    finally:
        cap.release()
    report = {"video_name": Path(args.video).name,
              "video_sha256": hashlib.sha256(Path(args.video).read_bytes()).hexdigest(),
              "fps": fps, "width": width, "height": height,
              "total_frames": index, "stride": args.stride, "threshold": args.threshold,
              "tag_every": args.tag_every,
              "motion_min": args.motion_min, "warmup_seconds": args.warmup_seconds,
              "elapsed_seconds": time.monotonic()-started, "observations": observations,
              "method": "YOLO original+180, ByteTrack, rotated/scaled PDF tag exemplars; offline baseline"}
    cache.write_text(json.dumps(report, indent=2, allow_nan=False))
    positives = [(o["frame"], p["track_id"], round(p["tag"]["score"], 3))
                 for o in observations for p in o["people"] if p["tag_positive"]]
    print(json.dumps({"output": str(cache), "positive_observations": positives}, indent=2))


if __name__ == "__main__":
    main()
