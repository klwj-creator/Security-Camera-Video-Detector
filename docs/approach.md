# CCTV shirt-logo detection approach

Recorded from the project discussion on 14 November 2026.

This document preserves the original general-logo proposal. The supplied evaluation now has a concrete three-bar staff name tag and an overhead clip; see `details/README.md` and the current staff workflow in `README.md`. The evaluation-specific prototype uses a trained tag proposal model, local reference-pattern verification, person-region checks, and short tag trajectories. These changes address the actual camera view and reduce false matches without treating long person track IDs as identities.

## Intended outcome

Find people wearing a specified shirt logo in recorded or live CCTV video. Associate logo evidence with a temporary person track, confirm it over multiple observations, and save a local alert with a snapshot. Recognition is of a visible logo, not a person's identity.

## Pipeline

1. Inspect original CCTV frames to establish whether the logo is distinguishable. Distance, angle, occlusion, lighting, and motion blur may make detection impossible. Cropping or enlarging cannot recover uncaptured detail.
2. Decode recorded video or the camera's RTSP stream using OpenCV initially. FFmpeg or GStreamer can be used for more demanding capture requirements.
3. Run a pretrained YOLO person detector and ByteTrack to obtain person boxes and temporary track IDs.
4. Crop a configurable shirt region from each person box in the original-resolution image. Run a separate custom YOLO logo detector on those crops and translate its boxes back to the full frame.
5. Accumulate positive and negative observations in a bounded time window for each track. Confirm repeated confident detections before alerting. Suppress duplicate alerts during that track's lifetime.
6. Save an annotated snapshot and structured event with the logo name, confidence, track ID, and video timestamp. Keep footage and alerts local by default.

The shirt region is an approximation from a person box, not a clothing segmentation model. Bags, overlapping people, or nearby signage can still cause incorrect attribution. Evaluate those cases; use pose or clothing segmentation if real footage shows this is necessary.

## Data and training

- Fine-tune pretrained weights for the target logo, rather than training from scratch or assuming a general detector knows the logo.
- Collect real shirt examples under representative camera conditions, with varied distances, folds, angles, and partial visibility.
- Label visible logo bounding boxes using CVAT. Include plain shirts, other logos, and confusing patterns as negatives.
- Prefer person/shirt crops resembling inference inputs when preparing training images. Annotate coordinates relative to the saved crop.
- Separate whole recording sessions into training, validation, and test sets before extracting frames. Adjacent frames from one recording must not leak between splits.
- Use validation data to tune confidence and confirmation settings. Reserve the test set for final reporting.

## Initial tools

Python, OpenCV, Ultralytics YOLO, CVAT, and ByteTrack. YOLO11 nano is an initial person detector and training starting point, not a claim that this model is optimal. NVIDIA DeepStream is a possible later option for many cameras on NVIDIA hardware.

## Acceptance evidence

- Functional checks: decoding, crop coordinate mapping, tracking integration, temporal confirmation, duplicate suppression, snapshot writing, and clean shutdown.
- Logo evaluation: detector precision/recall on held-out labeled shirt crops.
- End-to-end evaluation: manually labeled target-person appearances, missed appearances, false alerts per camera-hour, and duplicate alerts.
- Performance: processing rate and alert delay on intended hardware. File processing rate alone does not establish live-stream latency.
- Difficult cases: low light, small logos, overlapping people, back views, partial visibility, and stream interruptions.

Thresholds for acceptable accuracy and delay must be agreed using the user's operational needs. Synthetic tests establish software behavior, not real logo recognition accuracy.

## Inputs still needed

The target logo name and appearance; representative positive and negative CCTV recordings; the live camera connection when ready; and intended deployment hardware. A logo image alone is not a verified trained detector.
