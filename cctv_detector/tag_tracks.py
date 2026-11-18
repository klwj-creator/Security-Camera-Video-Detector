"""Short tag trajectories with bounded, explicit interpolation."""

import math


def center(box):
    return ((box[0]+box[2])/2, (box[1]+box[3])/2)


def associate_body(tag_box, people):
    tx, ty = center(tag_box)
    candidates = []
    for row in people:
        x1,y1,x2,y2 = row[:4]
        if x1 <= tx <= x2 and y1 <= ty <= y2:
            area = max(1,(x2-x1)*(y2-y1))
            if (tag_box[2]-tag_box[0])*(tag_box[3]-tag_box[1]) > area*0.2:
                continue
            cx,cy = center(row[:4])
            distance = math.hypot((tx-cx)/max(1,x2-x1), (ty-cy)/max(1,y2-y1))
            candidates.append((distance, list(row[:4])))
    return min(candidates, key=lambda c:c[0])[1] if candidates else None


def build_trajectories(frames, width, fps, max_gap=5, min_hits=2):
    """Associate by short-distance motion; never reuse long-lived person IDs."""
    tracks = []
    for index, detections in enumerate(frames):
        active = [(i, t) for i,t in enumerate(tracks) if index-t[-1][0] <= max_gap+1]
        pairs = []
        for ti, history in active:
            previous_frame, previous = history[-1]
            previous_center = center(previous["tag_box"])
            dt = index-previous_frame
            for di, detection in enumerate(detections):
                distance = math.dist(previous_center, center(detection["tag_box"]))
                if distance <= width*2*dt/fps:
                    pairs.append((distance,ti,di))
        used_tracks, used_detections = set(), set()
        for _, ti, di in sorted(pairs):
            if ti in used_tracks or di in used_detections:
                continue
            tracks[ti].append((index,detections[di]))
            used_tracks.add(ti)
            used_detections.add(di)
        for di, detection in enumerate(detections):
            if di not in used_detections:
                tracks.append([(index,detection)])
    rows = [[] for _ in frames]
    kept = 0
    for history in tracks:
        if len(history) < min_hits:
            continue
        kept += 1
        for index, detection in history:
            rows[index].append({**detection,"track_id":kept,"evidence":"direct"})
        for (left, a),(right,b) in zip(history,history[1:]):
            if right-left > max_gap+1:
                continue
            for index in range(left+1,right):
                fraction = (index-left)/(right-left)
                box = [x+(y-x)*fraction for x,y in zip(a["tag_box"],b["tag_box"])]
                rows[index].append({"tag_box":box,"person_box":None,"score":None,
                                    "track_id":kept,"evidence":"interpolated"})
    return rows
