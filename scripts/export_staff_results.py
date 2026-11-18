"""Re-track cached person boxes, rescore reference tags, and export frame results.

Outputs are automatic candidates, not hand-labeled ground truth. Offline
backfilling marks earlier observations of a subsequently confirmed track.
"""

import argparse
from collections import defaultdict
from dataclasses import asdict
import csv
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".runtime/ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime/matplotlib"))


def intervals(indices):
    result = []
    for index in sorted(set(indices)):
        if result and index == result[-1][1] + 1:
            result[-1][1] = index
        else:
            result.append([index, index])
    return result


def main():
    import cv2
    import numpy as np
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.byte_tracker import BYTETracker
    from cctv_detector.staff_tag import TagTemplates

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations")
    parser.add_argument("video")
    parser.add_argument("--output", default="outputs/staff-results")
    parser.add_argument("--threshold", type=float, default=0.82)
    parser.add_argument("--min-hits", type=int, default=2)
    parser.add_argument("--tag-every", type=int, default=2)
    parser.add_argument("--motion-min", type=float, default=0.12)
    args = parser.parse_args()
    source = json.loads(Path(args.observations).read_text())
    if source["stride"] != 1:
        parser.error("Use full-frame observations (--stride 1) to avoid ambiguous missing-frame labels.")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    fps, width, height = source["fps"], source["width"], source["height"]
    if not cap.isOpened() or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != source["total_frames"]:
        raise ValueError("Video does not match the cached observations.")
    backgrounds = []
    for index in np.linspace(0, min(source["total_frames"]-1, round(fps*4)), 9).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, image = cap.read()
        if ok:
            backgrounds.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    background = np.median(backgrounds, axis=0).astype(np.uint8)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    tracker = BYTETracker(SimpleNamespace(track_high_thresh=0.15, track_low_thresh=0.1,
                                         new_track_thresh=0.2, track_buffer=round(fps),
                                         match_thresh=0.8, fuse_score=False))
    templates = TagTemplates([ROOT / "configs/staff-tag-dark.png", ROOT / "configs/staff-tag-light.png"])
    by_track = defaultdict(list)
    observations = []
    try:
        for observation in source["observations"]:
            ok, image = cap.read()
            if not ok:
                raise RuntimeError("Video decode ended before the observations.")
            index = observation["frame"]
            raw = np.array(observation["raw_detections"], dtype=np.float32).reshape(-1, 6)
            rows = tracker.update(Boxes(raw, image.shape[:2]), image)
            difference = cv2.absdiff(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), background)
            people = []
            for row in rows:
                x1, y1, x2, y2 = [float(v) for v in row[:4]]
                # Approximate inner body region, invariant to 180-degree facing.
                dx, dy = (x2-x1)*0.1, (y2-y1)*0.1
                region = difference[max(0,int(y1)):min(height,int(y2)), max(0,int(x1)):min(width,int(x2))]
                motion = float(np.mean(region > 30)) if region.size else 0
                tag = templates.best(image, (x1+dx, y1+dy, x2-dx, y2-dy)) if index % args.tag_every == 0 and motion >= args.motion_min else None
                person = {"frame": index, "track_id": int(row[4]), "box": [x1, y1, x2, y2],
                          "tag": asdict(tag) if tag else None,
                          "tag_positive": bool(tag and tag.score >= args.threshold and tag.contrast >= 12)}
                by_track[person["track_id"]].append(person)
                people.append(person)
            observations.append({"frame": index, "people": people})
            if index % 100 == 0:
                print(f"Rescored {index}/{source['total_frames']}", flush=True)
    finally:
        cap.release()
    confirmed = {}
    for track_id, history in by_track.items():
        hits = [row["frame"] for row in history if row["tag_positive"]]
        # Repeated evidence within one second, not unrelated hits far apart.
        if any(sum(0 <= last-first <= fps for first in hits) >= args.min_hits for last in hits):
            confirmed[track_id] = {"first": history[0]["frame"], "last": history[-1]["frame"],
                                   "hits": hits, "best_score": max(row["tag"]["score"] for row in history if row["tag"])}
    fields = ["frame_index", "frame_number", "seconds", "staff_candidate_present", "status",
              "track_ids", "xy_centers", "direct_tag_track_ids"]
    positive_frames = []
    with (output / "frame-results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for observation in observations:
            index = observation["frame"]
            people = [p for p in observation["people"] if p["track_id"] in confirmed]
            direct = [p["track_id"] for p in people if p["tag_positive"]]
            xy = [[round((p["box"][0]+p["box"][2])/2, 2), round((p["box"][1]+p["box"][3])/2, 2)] for p in people]
            if people:
                positive_frames.append(index)
            writer.writerow({"frame_index": index, "frame_number": index+1, "seconds": round(index/fps, 3),
                             "staff_candidate_present": int(bool(people)),
                             "status": "direct_tag_evidence" if direct else "track_inferred" if people else "not_identified",
                             "track_ids": json.dumps([p["track_id"] for p in people]),
                             "xy_centers": json.dumps(xy), "direct_tag_track_ids": json.dumps(direct)})
    summary = {"method": "automatic reference-template baseline with offline track backfill",
               "total_frames": source["total_frames"], "candidate_frames": len(positive_frames),
               "candidate_intervals_zero_based_inclusive": intervals(positive_frames),
               "confirmed_candidate_tracks": confirmed, "threshold": args.threshold,
               "min_hits": args.min_hits, "fps": fps,
               "coordinates": "person bounding-box center in pixels; origin top-left; not calibrated floor coordinates",
               "caveat": "not_identified does not establish staff absence; no independent ground truth accuracy measured"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "evidence.json").write_text(json.dumps(observations))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
