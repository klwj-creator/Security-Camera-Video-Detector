"""Conservative offline staff evidence aggregation.

A tracker ID is not an identity. Never label its entire history merely because
a tag appears once; bound inference in time and require repeated evidence.
"""

from collections import defaultdict
import math


def distance_to_body_center(person):
    x1, y1, x2, y2 = person["box"]
    tx, ty = person["tag"]["center"]
    return math.hypot((tx-(x1+x2)/2)/max(1, x2-x1), (ty-(y1+y2)/2)/max(1, y2-y1))


def select_tag_owners(people, threshold=0.9, cluster_pixels=14):
    candidates = [p for p in people if p.get("tag") and p["tag"]["score"] >= threshold
                  and p["tag"]["contrast"] >= 12]
    # One visible tag must not label several overlapping person boxes.
    selected = []
    for person in sorted(candidates, key=distance_to_body_center):
        center = person["tag"]["center"]
        if any(math.dist(center, p["tag"]["center"]) < cluster_pixels for p in selected):
            continue
        selected.append(person)
    return selected


def summarize(observations, fps, threshold=0.9, min_hits=2, max_inference_seconds=0.6):
    if not math.isfinite(fps) or fps <= 0 or not 0 < threshold <= 1 or min_hits < 1:
        raise ValueError("A positive frame rate, valid threshold, and positive hit count are required.")
    if not math.isfinite(max_inference_seconds) or max_inference_seconds < 0:
        raise ValueError("Inference duration must be finite and nonnegative.")
    hits = defaultdict(list)
    direct = {}
    for observation in observations:
        index = observation["frame"]
        owners = select_tag_owners(observation["people"], threshold)
        direct[index] = {p["track_id"] for p in owners}
        for person in owners:
            hits[person["track_id"]].append(index)
    # A positive cluster must have at least min_hits observations within 0.4s.
    confirmed_hits = {}
    for track_id, indices in hits.items():
        supported = [i for i in indices if sum(abs(j-i) <= fps*0.4 for j in indices) >= min_hits]
        if supported:
            confirmed_hits[track_id] = supported
    rows = []
    for observation in observations:
        index = observation["frame"]
        people = []
        for person in observation["people"]:
            track_id = person["track_id"]
            nearby = any(abs(index-hit) <= fps*max_inference_seconds for hit in confirmed_hits.get(track_id, []))
            if not nearby:
                continue
            people.append({**person, "direct": track_id in direct[index]})
        rows.append({"frame": index, "people": people,
                     "status": "direct_tag_evidence" if any(p["direct"] for p in people)
                     else "track_inferred" if people else "not_identified"})
    return rows
