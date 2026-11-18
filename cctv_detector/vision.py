"""YOLO adapter: persistent person tracks and batched shirt-crop inference."""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import List, Optional, Tuple

Box = Tuple[int, int, int, int]


def clip_box(box, width: int, height: int) -> Optional[Box]:
    if not all(math.isfinite(float(v)) for v in box):
        return None
    x1, y1, x2, y2 = box
    result = (max(0, min(width, math.floor(x1))), max(0, min(height, math.floor(y1))),
              max(0, min(width, math.ceil(x2))), max(0, min(height, math.ceil(y2))))
    return result if result[2] > result[0] and result[3] > result[1] else None


def shirt_box(person: Box, top: float, bottom: float) -> Box:
    x1, y1, x2, y2 = person
    height = y2 - y1
    return x1, int(y1 + height * top), x2, max(int(y1 + height * bottom), int(y1 + height * top) + 1)


@dataclass(frozen=True)
class Logo:
    box: Box
    confidence: float


@dataclass
class Person:
    track_id: int
    box: Box
    shirt: Box
    logos: List[Logo]


class YoloPipeline:
    def __init__(self, person_model: str, logo_model: str, logo_class: str,
                 person_conf: float = 0.35, logo_conf: float = 0.6,
                 device: Optional[str] = None, imgsz: int = 640,
                 shirt_top: float = 0.15, shirt_bottom: float = 0.7):
        if not Path(logo_model).is_file():
            raise ValueError("Logo model file does not exist. Train a custom detector first.")
        from ultralytics import YOLO

        self.person_model = YOLO(person_model)
        self.logo_model = YOLO(logo_model)
        if self.person_model.task != "detect" or self.logo_model.task != "detect":
            raise ValueError("Both models must be object detection models.")
        self.person_ids = [key for key, value in self.person_model.names.items() if value == "person"]
        target_ids = [key for key, value in self.logo_model.names.items() if value == logo_class]
        if not self.person_ids:
            raise ValueError("The person model must contain a class named 'person'.")
        if not target_ids:
            available = ", ".join(self.logo_model.names.values())
            raise ValueError(f"Logo class '{logo_class}' is absent from the model. Available: {available}")
        self.target_ids = target_ids
        self.person_conf, self.logo_conf = person_conf, logo_conf
        self.options = {"verbose": False, "imgsz": imgsz}
        if device is not None:
            self.options["device"] = device
        self.shirt_top, self.shirt_bottom = shirt_top, shirt_bottom

    def reset(self):
        for tracker in getattr(self.person_model.predictor, "trackers", []):
            tracker.reset()

    def detect(self, frame) -> List[Person]:
        height, width = frame.shape[:2]
        result = self.person_model.track(frame, persist=True, tracker="bytetrack.yaml",
                                         classes=self.person_ids, conf=self.person_conf,
                                         **self.options)[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return []
        people = []
        crops = []
        for coords, track_id in zip(boxes.xyxy.cpu().tolist(), boxes.id.cpu().tolist()):
            box = clip_box(coords, width, height)
            if box is None:
                continue
            shirt = shirt_box(box, self.shirt_top, self.shirt_bottom)
            x1, y1, x2, y2 = shirt
            crops.append(frame[y1:y2, x1:x2])
            people.append(Person(int(track_id), box, shirt, []))
        if not crops:
            return []
        results = self.logo_model.predict(crops, classes=self.target_ids, conf=self.logo_conf, **self.options)
        for person, logos in zip(people, results):
            if logos.boxes is None:
                continue
            ox, oy, right, bottom = person.shirt
            for coords, score in zip(logos.boxes.xyxy.cpu().tolist(), logos.boxes.conf.cpu().tolist()):
                local = clip_box(coords, right - ox, bottom - oy)
                if local is not None:
                    a, b, c, d = local
                    person.logos.append(Logo((a + ox, b + oy, c + ox, d + oy), float(score)))
        return people


def annotate(frame, people: List[Person], logo_class: str, alerted_ids=()):
    import cv2

    canvas = frame.copy()
    for person in people:
        color = (0, 0, 255) if person.track_id in alerted_ids else (0, 200, 0)
        x1, y1, x2, y2 = person.box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, f"Track {person.track_id}", (x1, max(18, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        sx1, sy1, sx2, sy2 = person.shirt
        cv2.rectangle(canvas, (sx1, sy1), (sx2, sy2), (180, 180, 180), 1)
        for logo in person.logos:
            lx1, ly1, lx2, ly2 = logo.box
            cv2.rectangle(canvas, (lx1, ly1), (lx2, ly2), (0, 220, 255), 2)
            cv2.putText(canvas, f"{logo_class} {logo.confidence:.2f}", (lx1, max(18, ly1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
    return canvas
