"""Detect the illustrated staff tag in a video and export per-frame predictions.

Usage: python scripts/detect_staff.py details/sample.mp4 --output outputs/staff-model
Optionally reuse --people-cache outputs/staff-full-sample/observations.json.
This is a development model trained on the supplied clip, not a validated
general-purpose staff identification system.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT/".runtime/ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT/".runtime/matplotlib"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    import cv2
    import numpy as np
    from ultralytics import YOLO
    from cctv_detector.staff_tag import OverheadPeople, TagTemplates
    from cctv_detector.tag_tracks import associate_body, build_trajectories, center

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("--model", default=str(ROOT/"models/staff-tag-sample-v2.pt"))
    parser.add_argument("--person-model", default=str(ROOT/"models/yolo11n.pt"))
    parser.add_argument("--people-cache")
    parser.add_argument("--output", default="outputs/staff-model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--pattern-threshold", type=float, default=0.83)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--max-gap", type=int, default=5)
    args = parser.parse_args()
    if not 0 < args.confidence <= 1 or not 0 <= args.pattern_threshold <= 1 or args.max_gap < 0 or args.imgsz < 32:
        parser.error("Invalid confidence, gap, or image size.")
    if not Path(args.video).is_file() or not Path(args.model).is_file():
        parser.error("Video and trained tag model must exist locally.")
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/"summary.json").exists():
        parser.error("Output already has a completed run; choose another directory.")
    model = YOLO(args.model)
    ids = [key for key,value in model.names.items() if value == "staff_tag"]
    if model.task != "detect" or not ids:
        parser.error("Tag model must be a detector with a staff_tag class.")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("Cannot read the video.")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width,height = int(cap.get(3)),int(cap.get(4))
    patterns = TagTemplates([ROOT/"configs/staff-tag-dark.png",ROOT/"configs/staff-tag-light.png"],
                           widths=tuple(max(8,round(w*width/960)) for w in (18,22,26,30)))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Video must have a usable nominal frame rate.")
    video_hash = sha256(args.video)
    cache = None
    if args.people_cache:
        cache = json.loads(Path(args.people_cache).read_text())
        if cache.get("video_sha256") != video_hash or cache["stride"] != 1:
            raise ValueError("Person cache does not have the matching video hash and stride 1.")
    people_model = None if cache else OverheadPeople(args.person_model,device=args.device)
    detections, raw_counts = [], []
    started = time.monotonic()
    index = 0
    try:
        while True:
            ok,image = cap.read()
            if not ok:
                break
            result = model.predict(image,classes=ids,imgsz=args.imgsz,conf=args.confidence,
                                   device=args.device,verbose=False)[0]
            raw_counts.append(len(result.boxes))
            if cache:
                people = cache["observations"][index]["raw_detections"]
            else:
                people = people_model.detect(image).tolist() if len(result.boxes) else []
            accepted = []
            for box,score in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist()):
                person_box = associate_body(box,people)
                if person_box is not None:
                    a,b,c,d = box
                    pattern = patterns.best(image,(a-8,b-8,c+8,d+8))
                    if pattern is None or pattern.score < args.pattern_threshold:
                        continue
                    if sum((x-y)**2 for x,y in zip(center(box),pattern.center)) > max(8,(c-a)/2,(d-b)/2)**2:
                        continue
                    accepted.append({"tag_box":box,"person_box":person_box,"score":float(score),
                                     "pattern_score":pattern.score})
            detections.append(accepted)
            if index%100 == 0:
                print(f"Frame {index}: {len(accepted)} person-associated tag detections",flush=True)
            index += 1
    finally:
        cap.release()
    inference_elapsed = time.monotonic()-started
    if not detections:
        raise RuntimeError("Video yielded no frames.")
    rows = build_trajectories(detections,width,fps,args.max_gap)
    with (output/"frame-results.csv").open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=["frame_index","frame_number","seconds","staff_present_prediction",
                                                  "evidence","staff_xy","person_centers","track_ids"])
        writer.writeheader()
        for index,people in enumerate(rows):
            writer.writerow({"frame_index":index,"frame_number":index+1,"seconds":round(index/fps,3),
                             "staff_present_prediction":int(bool(people)),
                             "evidence":"direct" if any(p["evidence"]=="direct" for p in people) else "interpolated" if people else "not_identified",
                             "staff_xy":json.dumps([[round(v,2) for v in center(p["tag_box"])] for p in people]),
                             "person_centers":json.dumps([[round(v,2) for v in center(p["person_box"])] if p["person_box"] else None for p in people]),
                             "track_ids":json.dumps([p["track_id"] for p in people])})
    (output/"detections.json").write_text(json.dumps(detections))
    (output/"trajectories.json").write_text(json.dumps(rows))
    render_video(args.video,output,rows,width,height,fps)
    positive = [i for i,p in enumerate(rows) if p]
    intervals = []
    for index in positive:
        if intervals and index==intervals[-1][1]+1:
            intervals[-1][1]=index
        else:
            intervals.append([index,index])
    report = {"video_name":Path(args.video).name,"video_sha256":video_hash,"model_sha256":sha256(args.model),
              "total_frames":len(rows),"fps":fps,"width":width,"height":height,
              "predicted_present_frames":len(positive),
              "direct_frames":sum(any(p["evidence"]=="direct" for p in people) for people in rows),
              "interpolated_only_frames":sum(bool(people) and all(p["evidence"]=="interpolated" for p in people) for people in rows),
              "intervals_zero_based_inclusive":intervals,"confidence":args.confidence,"max_gap_frames":args.max_gap,
              "pattern_threshold":args.pattern_threshold,"imgsz":args.imgsz,
              "inference_seconds":inference_elapsed,"processing_fps":len(rows)/inference_elapsed,
              "used_person_cache":bool(cache),"raw_tag_detections":sum(raw_counts),
              "person_associated_tag_detections":sum(len(p) for p in detections),
              "coordinate_definition":"staff_xy = tag/chest center proxy in image pixels; top-left origin, x right, y down. Not floor coordinates.",
              "limitations":"Development model trained on this clip; not independent accuracy validation. Not identified does not establish absence. Short gaps use offline interpolation, not direct tag visibility."}
    (output/"summary.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


def render_video(video,output,rows,width,height,fps):
    import cv2
    cap = cv2.VideoCapture(video)
    writer = cv2.VideoWriter(str(output/"annotated.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height))
    if not writer.isOpened():
        raise RuntimeError("Video encoder unavailable.")
    stills = []
    last_still = -100
    try:
        for index,people in enumerate(rows):
            ok,image = cap.read()
            if not ok:
                raise RuntimeError("Source decode ended during rendering.")
            cv2.rectangle(image,(0,0),(width,42),(20,20,20),-1)
            state = "tag detected" if any(p["evidence"]=="direct" for p in people) else "short gap inferred" if people else "staff not identified"
            cv2.putText(image,f"Frame {index} | {index/fps:.2f}s | {state}",(10,28),0,.65,(255,255,255),2)
            for person in people:
                a,b,c,d = [int(v) for v in person["tag_box"]]
                color = (0,220,0) if person["evidence"]=="direct" else (0,180,255)
                cv2.rectangle(image,(a,b),(c,d),color,2)
            if people and index-last_still >= 25:
                filename=f"evidence-{index:04d}.jpg"
                cv2.imwrite(str(output/filename),image)
                stills.append({"frame":index,"file":filename})
                last_still=index
            writer.write(image)
    finally:
        cap.release()
        writer.release()
    (output/"evidence-stills.json").write_text(json.dumps(stills,indent=2))


if __name__ == "__main__":
    main()
