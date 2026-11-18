"""Produce conservative frame CSV, evidence stills, and review video from a cache."""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import cv2
    import numpy as np
    from cctv_detector.staff_results import summarize

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence")
    parser.add_argument("video")
    parser.add_argument("--output", default="outputs/staff-conservative")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--inference-seconds", type=float, default=0.6)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    evidence = json.loads(Path(args.evidence).read_text())
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(3)), int(cap.get(4))
    rows = summarize(evidence, fps, args.threshold, max_inference_seconds=args.inference_seconds)
    fields = ["frame_index", "frame_number", "seconds", "staff_candidate_present", "status", "track_ids", "person_xy", "tag_xy"]
    with (output / "frame-results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({"frame_index": row["frame"], "frame_number": row["frame"]+1,
                             "seconds": round(row["frame"]/fps, 3), "staff_candidate_present": int(bool(row["people"])),
                             "status": row["status"], "track_ids": json.dumps([p["track_id"] for p in row["people"]]),
                             "person_xy": json.dumps([[round((p["box"][0]+p["box"][2])/2, 2),
                                                       round((p["box"][1]+p["box"][3])/2, 2)] for p in row["people"]]),
                             "tag_xy": json.dumps([p["tag"]["center"] for p in row["people"] if p["direct"]])})
    writer = cv2.VideoWriter(str(output / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Cannot open output video encoder.")
    direct_frames = []
    last_still = -100
    tiles = []
    try:
        for row in rows:
            ok, image = cap.read()
            if not ok:
                raise RuntimeError("Video ended before evidence rows.")
            index = row["frame"]
            cv2.rectangle(image, (0, 0), (width, 45), (20, 20, 20), -1)
            label = f"Frame {index} ({index/fps:.2f}s): {row['status']}"
            cv2.putText(image, label, (12, 28), 0, .65, (255, 255, 255), 2)
            for person in row["people"]:
                a, b, c, d = [int(v) for v in person["box"]]
                color = (0, 220, 0) if person["direct"] else (0, 180, 255)
                cv2.rectangle(image, (a, b), (c, d), color, 2)
                cv2.putText(image, f"Candidate {person['track_id']}", (a, max(60, b-5)), 0, .5, color, 1)
                if person["direct"]:
                    ta, tb, tc, td = [int(v) for v in person["tag"]["box"]]
                    cv2.rectangle(image, (ta, tb), (tc, td), (255, 0, 255), 2)
            if row["status"] == "direct_tag_evidence":
                direct_frames.append(index)
                if index-last_still >= 12:
                    cv2.imwrite(str(output / f"evidence-{index:04d}.jpg"), image)
                    person = next(p for p in row["people"] if p["direct"])
                    tx, ty = [int(v) for v in person["tag"]["center"]]
                    crop = image[max(0, ty-55):min(height,ty+55), max(0,tx-65):min(width,tx+65)]
                    tile = cv2.resize(crop, (260, 220))
                    cv2.rectangle(tile, (0, 0), (260, 30), (0, 0, 0), -1)
                    cv2.putText(tile, f"Frame {index}, score {person['tag']['score']:.3f}", (5, 20), 0, .5, (255,255,255), 1)
                    tiles.append(tile)
                    last_still = index
            writer.write(image)
    finally:
        cap.release()
        writer.release()
    if tiles:
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(output / "evidence-gallery.jpg"), np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)]))
    intervals = []
    for index in (row["frame"] for row in rows if row["people"]):
        if intervals and index == intervals[-1][1]+1:
            intervals[-1][1] = index
        else:
            intervals.append([index, index])
    report = {"total_frames": len(rows), "fps": fps, "width": width, "height": height,
              "candidate_frames": sum(bool(row["people"]) for row in rows),
              "direct_tag_frames": len(direct_frames), "candidate_intervals_zero_based_inclusive": intervals,
              "direct_frame_indices": direct_frames, "threshold": args.threshold,
              "max_inference_seconds": args.inference_seconds,
              "warning": "Automatic candidates, not ground truth. not_identified does not establish absence. Offline inference uses future evidence."}
    (output / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
