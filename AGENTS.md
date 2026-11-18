# Security Camera Video Detector

## Project context

The original use case is detecting a person wearing a specific shirt logo in CCTV footage. The user requested implementation after saving the approach as source documents. Read `docs/approach.md` and `docs/sources.md` for the original design and references. The user has since supplied a concrete staff-name-tag evaluation and footage, described below; real-world generalization remains unverified.

Update: the user supplied `details/AI Evaluation Test.pdf` and `details/sample.mp4`. Read `details/README.md` and inspect the original PDF. The concrete target is the illustrated three-bar staff name tag, and the immediate task is per-frame staff presence, optional xy localization, runnable code, and a 1–2 page explanation. Adapt the original logo proposal to the ceiling-mounted camera and tag visibility/tracking requirements. Do not continue asking for source footage that is now present. One clip is not an independent generalization test; preserve distinctions between reviewed sample answers and automatic predictions.

Exploratory training on this one supplied clip is allowed for developing the evaluation solution. `scripts/prepare_sample_tag_dataset.py` records reviewed positive frames and desk non-wearer negatives, with time-separated passes for development splits. These are not independent recordings and cannot satisfy deployment accuracy validation. Inspect the annotation review image before training. Never backfill an entire long person track after a later tag hit: tracker IDs can switch between people. Keep inferred evidence time-bounded and avoid assigning one physical tag to several overlapping person boxes.

The initial Python implementation is in `cctv_detector/`; setup and commands are in `README.md`. Read `docs/verification.md` for tested behavior and remaining gaps. Run `.venv/bin/python -m unittest discover -s tests -v` for the software checks. The optional inference smoke script uses a person detector as a second-stage stand-in and must never be presented as verified logo recognition.

Current staff entry point: `scripts/detect_staff.py`, using `models/staff-tag-sample-v2.pt`. This evaluation-specific path proposes tag boxes at high resolution, checks the reference pattern and person context, and tracks only short tag trajectories with explicit interpolation. It can use a hash-matched person cache for repeat experiments, but must run the person model for a new video. Never quote cached tag-stage speed as full-pipeline or live-camera performance. The original generic cropped-person logo pipeline remains available separately.

## Recommended approach

1. Inspect recorded footage from the actual CCTV camera at its original resolution. Establish whether the logo is distinguishable at the expected distance, lighting, camera angle, and motion blur. Enlarging or cropping an image cannot restore detail the camera did not capture.
2. Read video with OpenCV, FFmpeg, or GStreamer. Live cameras commonly expose an RTSP stream. Start with recorded clips for repeatable experiments, then connect the same detection pipeline to the live feed.
3. Use a pretrained YOLO person detector to locate people. Crop each detected person from the original-resolution frame and pass the crop to a second detector for the target logo. Associate each logo detection with the corresponding person.
4. Fine-tune an existing object detector for the particular logo. Do not assume a general pretrained person detector recognizes arbitrary logos. Starting from pretrained weights is preferred over training from scratch.
5. Use ByteTrack or BoT-SORT to follow people across frames with temporary track IDs. Track IDs describe continuity within video; they do not establish a person's real-world identity.
6. Combine logo evidence over multiple observations of a tracked person before triggering an alert. Tune confidence, observation requirements, and repeat-alert suppression using actual footage. Save a snapshot or short clip as evidence when alerts are implemented.

## Training data

- Collect representative examples of the logo on actual shirts across distances, angles, lighting conditions, folds, occlusion, and motion blur.
- Use CVAT to draw bounding boxes around visible target logos and export labels in a format compatible with the training library.
- Include negative examples: other logos, plain shirts, and visually similar patterns.
- Keep entire recordings separate between training, validation, and testing. Avoid randomly splitting adjacent frames across these sets, which can make results look misleadingly strong.

## Suggested tools

- Python: application and detection pipeline.
- OpenCV: frame capture, cropping, and image processing; FFmpeg or GStreamer are alternatives for video handling.
- Ultralytics YOLO: pretrained person detection and custom logo-detector training.
- CVAT: training-data annotation.
- ByteTrack: initial tracking option; BoT-SORT is an alternative.
- NVIDIA DeepStream: consider for a larger deployment with many cameras on NVIDIA hardware.

These are starting recommendations, not mandatory dependencies. Confirm current compatibility, licensing, and hardware suitability when implementation is requested.

## Verification expectations

- Evaluate on held-out recordings from the real camera, including footage without the target logo.
- Measure missed detections and false alerts, as well as processing speed and end-to-end alert delay on the intended hardware.
- Test challenging conditions such as small logos, partial visibility, people overlapping, and low light.
- Do not claim reliable logo detection from person detection alone, a running video feed, or tests without representative labeled footage.

## Reference documentation

- Ultralytics modes: https://docs.ultralytics.com/modes/
- Custom training: https://docs.ultralytics.com/modes/train/
- Tracking: https://docs.ultralytics.com/modes/track/
- CVAT dataset formats: https://docs.cvat.ai/docs/dataset_management/formats/
- NVIDIA DeepStream: https://developer.nvidia.com/deepstream-sdk
