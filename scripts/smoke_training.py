"""Verify train/evaluate commands on synthetic shapes, NOT real logo accuracy.

Requires the pretrained weights already downloaded by smoke_inference.py.
All fixtures, logs, and resulting weights are isolated under .runtime/.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


def main():
    import cv2
    import numpy as np

    weights = ROOT / "models/yolo11n.pt"
    if not weights.is_file():
        raise RuntimeError("Run smoke_inference.py first to download the person checkpoint.")
    output = ROOT / ".runtime" / f"training-smoke-{uuid4().hex[:8]}"
    output.mkdir(parents=True)
    for split_index, split in enumerate(("train", "val", "test")):
        images = output / "dataset/images" / split
        labels = output / "dataset/labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for index in range(2):
            image = np.full((128, 128, 3), 30 + 15 * split_index, dtype=np.uint8)
            x, y, size = 20 + 12 * index, 24 + 8 * split_index, 32
            cv2.rectangle(image, (x, y), (x + size - 1, y + size - 1), (10, 180, 240), -1)
            if not cv2.imwrite(str(images / f"shape-{index}.jpg"), image):
                raise RuntimeError("Could not create synthetic test image.")
            (labels / f"shape-{index}.txt").write_text(
                f"0 {(x + size / 2) / 128} {(y + size / 2) / 128} {size / 128} {size / 128}\n")
    dataset = output / "dataset.yaml"
    dataset.write_text(f"path: {json.dumps(str(output / 'dataset'))}\n"
                       "train: images/train\nval: images/val\ntest: images/test\n"
                       "names:\n  0: synthetic_marker\n")
    env = os.environ.copy()
    env["YOLO_CONFIG_DIR"] = str(ROOT / ".runtime/ultralytics")
    env["MPLCONFIGDIR"] = str(ROOT / ".runtime/matplotlib")
    env["OMP_NUM_THREADS"] = "2"
    prefix = [sys.executable, "-m", "cctv_detector"]
    with (output / "training.log").open("w") as log:
        subprocess.run(prefix + ["train", "--data", str(dataset), "--model", str(weights),
                                 "--epochs", "1", "--batch", "2", "--imgsz", "128",
                                 "--device", "cpu", "--output", str(output / "training")],
                       cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    trained = output / "training/train/weights/best.pt"
    if not trained.is_file():
        raise RuntimeError("Training did not produce the expected checkpoint.")
    with (output / "evaluation.log").open("w") as log:
        subprocess.run(prefix + ["evaluate", "--data", str(dataset), "--model", str(trained),
                                 "--split", "test", "--imgsz", "128", "--device", "cpu",
                                 "--output", str(output / "evaluation")],
                       cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    report = {"test": "synthetic train/evaluate integration only; not logo accuracy",
              "directory": str(output.relative_to(ROOT)),
              "checkpoint": str(trained.relative_to(ROOT)), "train_exit_code": 0, "evaluate_exit_code": 0}
    (ROOT / ".runtime/training-smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
