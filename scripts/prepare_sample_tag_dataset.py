"""Build exploratory tag training data from reviewed sample evidence.

This intentionally uses the supplied sample for development. It is not an
independent test dataset or a source of generalization accuracy claims.
"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import cv2
    from cctv_detector.staff_results import distance_to_body_center

    observations = json.loads((ROOT / "outputs/staff-results/evidence.json").read_text())
    target = ROOT / "data/sample-tag-development"
    if target.exists():
        raise RuntimeError("Development dataset already exists; preserve annotations rather than overwrite.")
    positives = {
        "train": {506, 508, 846, 854, 856, 858, 860, 862,
                  972, 974, 976, 978, 980, 1116, 1118},
        "val": {430, 432, 434, 892, 894, 896, 898, 900, 1014, 1016, 1018, 1020, 1022, 1024, 1026},
        "test": {1198, 1200, 1202, 1204, 1206},
    }
    for split in positives:
        (target / "images" / split).mkdir(parents=True)
        (target / "labels" / split).mkdir(parents=True)
    cap = cv2.VideoCapture(str(ROOT / "details/sample.mp4"))
    records = []
    annotation_tiles = []
    try:
        for observation in observations:
            ok, image = cap.read()
            if not ok:
                raise RuntimeError("Source ended before annotations.")
            index = observation["frame"]
            split = next((s for s, indices in positives.items() if index in indices), None)
            examples = []
            if split:
                candidates = [p for p in observation["people"] if p["tag"]
                              and 400 < p["tag"]["center"][0] < 620]
                if not candidates:
                    raise RuntimeError(f"Missing tag candidate for reviewed frame {index}.")
                # Duplicate person proposals sometimes contain the same physical tag.
                best = max(candidates, key=lambda p: p["tag"]["score"])
                center = best["tag"]["center"]
                duplicates = [p for p in candidates if sum((a-b)**2 for a,b in zip(p["tag"]["center"], center)) < 25]
                person = min(duplicates, key=distance_to_body_center)
                examples.append((split, person, True))
            # Reviewed non-wearers at desks, away from the walking staff path.
            if index < 300 and index % 10 == 0:
                split = "train" if index < 100 else "val" if index < 200 else "test"
                candidates = [p for p in observation["people"]
                              if (p["box"][0]+p["box"][2])/2 > 650
                              or 250 < (p["box"][0]+p["box"][2])/2 < 390]
                examples.extend((split, p, False) for p in candidates[:4])
            for split, person, positive in examples:
                x1, y1, x2, y2 = [int(v) for v in person["box"]]
                x1, y1 = max(0,x1-8), max(0,y1-8)
                x2, y2 = min(image.shape[1],x2+8), min(image.shape[0],y2+8)
                crop = image[y1:y2, x1:x2]
                if crop.shape[0] < 30 or crop.shape[1] < 30:
                    continue
                name = f"frame-{index:04d}-track-{person['track_id']}-{'tag' if positive else 'negative'}"
                label = ""
                if positive:
                    # Visually checked tag extents; do not label the padded
                    # template-search square as the physical object.
                    tx,ty = person["tag"]["center"]
                    a,b,c,d = tx-12,ty-8,tx+12,ty+8
                    a,b,c,d = max(x1,a),max(y1,b),min(x2,c),min(y2,d)
                    w,h = x2-x1,y2-y1
                    label = f"0 {((a+c)/2-x1)/w:.6f} {((b+d)/2-y1)/h:.6f} {(c-a)/w:.6f} {(d-b)/h:.6f}\n"
                    preview = crop.copy()
                    cv2.rectangle(preview,(int(a-x1),int(b-y1)),(int(c-x1),int(d-y1)),(0,255,0),1)
                    tile = cv2.resize(preview,(150,210))
                    cv2.rectangle(tile,(0,0),(150,22),(0,0,0),-1)
                    cv2.putText(tile,f"{index} {split}",(4,16),0,.45,(255,255,255),1)
                    annotation_tiles.append(tile)
                if not cv2.imwrite(str(target/"images"/split/f"{name}.jpg"),crop):
                    raise RuntimeError("Could not save development image.")
                (target/"labels"/split/f"{name}.txt").write_text(label)
                records.append({"frame":index,"split":split,"positive":positive,"file":name,
                                "person_box":person["box"],"tag":person["tag"] if positive else None})
            if index < 300 and index % 10 == 0:
                split = "train" if index < 100 else "val" if index < 200 else "test"
                # Visually reviewed full-clip false matches: patterned shirt,
                # unrelated shirt lettering, and two pieces of static scenery.
                for name,(a,b,c,d) in {
                    "patterned-shirt":(150,420,290,560),
                    "other-shirt-logo":(300,0,425,150),
                    "floor-pattern":(0,590,130,720),
                    "paper-pattern":(110,0,215,95),
                }.items():
                    filename=f"hard-negative-{index:04d}-{name}"
                    if not cv2.imwrite(str(target/"images"/split/f"{filename}.jpg"),image[b:d,a:c]):
                        raise RuntimeError("Cannot save hard-negative crop.")
                    (target/"labels"/split/f"{filename}.txt").write_text("")
                    records.append({"frame":index,"split":split,"positive":False,
                                    "file":filename,"reviewed_hard_negative":name})
    finally:
        cap.release()
    dataset = target / "dataset.yaml"
    dataset.write_text(f"path: {json.dumps(str(target))}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: staff_tag\n")
    (target/"annotations.json").write_text(json.dumps(records,indent=2))
    import numpy as np
    while len(annotation_tiles)%8:
        annotation_tiles.append(np.zeros_like(annotation_tiles[0]))
    cv2.imwrite(str(target/"annotation-review.jpg"),np.vstack([np.hstack(annotation_tiles[i:i+8]) for i in range(0,len(annotation_tiles),8)]))
    counts = {s: {"positive":sum(r["split"]==s and r["positive"] for r in records),
                  "negative":sum(r["split"]==s and not r["positive"] for r in records)} for s in positives}
    (target/"README.txt").write_text("Development annotations from ONE supplied clip. Time-separated passes, not independent recordings.\n"
                                    "Positive tag boxes are template-assisted and visually reviewed; check annotation-review.jpg.\n"
                                    "Do not report these split metrics as independent camera or recording accuracy.\n"+json.dumps(counts,indent=2))
    print(json.dumps({"dataset":str(dataset),"counts":counts},indent=2))


if __name__ == "__main__":
    main()
