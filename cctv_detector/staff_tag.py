"""Reference-pattern baseline for the supplied overhead staff-tag task.

This is a template matcher, not a trained logo model. Scores are normalized
correlations, not probabilities. See details/README.md for the source pattern.
"""

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass
class TagMatch:
    score: float
    center: tuple
    box: tuple
    angle: float
    contrast: float


def restore_box(box, rotation, width, height):
    """Undo np.rot90 using continuous image-edge coordinates."""
    x1, y1, x2, y2 = box
    points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=float)
    x, y = points[:, 0].copy(), points[:, 1].copy()
    if rotation == 1:
        points[:, 0], points[:, 1] = width - y, x
    elif rotation == 2:
        points[:, 0], points[:, 1] = width - x, height - y
    elif rotation == 3:
        points[:, 0], points[:, 1] = y, height - x
    return [float(points[:, 0].min()), float(points[:, 1].min()),
            float(points[:, 0].max()), float(points[:, 1].max())]


class OverheadPeople:
    def __init__(self, weights, device="cpu", imgsz=960, rotations=(0, 2)):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        if self.model.task != "detect":
            raise ValueError("Person weights must be a detection model.")
        self.person_classes = [key for key, name in self.model.names.items() if name == "person"]
        if not self.person_classes:
            raise ValueError("Person weights must contain a class named person.")
        self.device, self.imgsz, self.rotations = device, imgsz, rotations

    def detect(self, image):
        height, width = image.shape[:2]
        inputs = [np.ascontiguousarray(np.rot90(image, k)) for k in self.rotations]
        results = self.model.predict(inputs, classes=self.person_classes, conf=0.12, imgsz=self.imgsz,
                                     device=self.device, verbose=False)
        boxes, scores = [], []
        for rotation, result in zip(self.rotations, results):
            for box, score in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()):
                restored = restore_box(box, rotation, width, height)
                x1, y1, x2, y2 = restored
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                scores.append(float(score))
        kept = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.12, nms_threshold=0.45)
        rows = []
        for index in np.asarray(kept).reshape(-1):
            x, y, w, h = boxes[int(index)]
            rows.append([x, y, x + w, y + h, scores[int(index)], 0])
        return np.array(rows, dtype=np.float32).reshape(-1, 6)


class TagTemplates:
    def __init__(self, template_path, widths=(18, 22, 26, 30), ratios=(0.45, 0.65, 0.85), angle_step=30):
        self.bank = []
        paths = template_path if isinstance(template_path, (list, tuple)) else [template_path]
        for path in paths:
            reference = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if reference is None or reference.std() < 5:
                raise ValueError("A readable, contrasting tag template is required.")
            self._add_reference(reference, widths, ratios, angle_step)

    def _add_reference(self, reference, widths, ratios, angle_step):
        for width in widths:
            for ratio in ratios:
                small = cv2.resize(reference, (width, max(5, round(width * ratio))), interpolation=cv2.INTER_AREA)
                for angle in range(0, 360, angle_step):
                    # A mean-colored square around a right-angle template dilutes
                    # correlation on light shirts. Preserve its exact support.
                    if angle % 90 == 0:
                        self.bank.append((np.ascontiguousarray(np.rot90(small, angle // 90)), angle))
                        continue
                    h, w = small.shape
                    side = math.ceil(math.hypot(w, h)) + 2
                    base = np.full((side, side), float(small.mean()), dtype=np.float32)
                    ox, oy = (side - w) // 2, (side - h) // 2
                    base[oy:oy+h, ox:ox+w] = small
                    matrix = cv2.getRotationMatrix2D((side / 2, side / 2), angle, 1)
                    rotated = cv2.warpAffine(base, matrix, (side, side), borderValue=float(small.mean()))
                    rotated = np.uint8(np.clip(rotated, 0, 255))
                    self.bank.append((rotated, angle))

    def best(self, image, box=None):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        height, width = gray.shape
        x1, y1, x2, y2 = (0, 0, width, height) if box is None else box
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(width, int(x2)), min(height, int(y2))
        crop = gray[y1:y2, x1:x2]
        best = None
        for template, angle in self.bank:
            h, w = template.shape
            if crop.shape[0] < h or crop.shape[1] < w:
                continue
            response = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(response)
            if best is None or score > best.score:
                x, y = loc
                contrast = float(crop[y:y+h, x:x+w].std())
                best = TagMatch(float(score), (x1+x+w/2, y1+y+h/2),
                                (x1+x, y1+y, x1+x+w, y1+y+h), angle, contrast)
        return best
