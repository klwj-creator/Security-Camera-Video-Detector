# Security Camera Video Detector

A local Python pipeline for finding a specific shirt logo on tracked people in CCTV footage. It reads recordings, webcams, and RTSP streams; checks shirt crops with a custom logo model; and writes confirmed alerts with annotated snapshots.

**Status:** a generic logo pipeline and an experimental detector for the supplied staff-name-tag evaluation are implemented. The staff model was developed using the supplied sample, so its results are not independent accuracy validation. No footage is uploaded by this application; pretrained person weights may be downloaded on first use.

## Supplied staff-tag evaluation

The original video and evaluation brief are preserved in `details/`, with matching source hashes in `details/source-manifest.json`. The concrete target is the illustrated three-bar name tag, on light or dark clothing. The clip has 1,341 frames at 25 FPS, 960 x 720 pixels.

Run the current staff prototype from this folder:

```sh
source .venv/bin/activate
python scripts/detect_staff.py details/sample.mp4 --output outputs/my-staff-run
```

Use another local recording in place of `details/sample.mp4` to run a new video. The default local weights are `models/staff-tag-sample-v2.pt`; the detector does not contain hardcoded sample frame ranges or clothing-color rules. It proposes tags with YOLO, verifies the local reference pattern, requires person context, and confirms short tag trajectories. Confidence and pattern thresholds are tunable. This is an offline prototype; real-time end-to-end operation is not established.

Each run saves `frame-results.csv`, `summary.json`, raw detections, trajectories, evidence stills, and `annotated.mp4`. Frame indices are zero-based; the CSV also includes one-based numbers and nominal video seconds. `staff_xy` is a tag/chest-center proxy in original image pixels (top-left origin), not a floor coordinate. Person-box centers are reported separately where available.

Green boxes are retained direct tag detections. Orange boxes bridge short gaps between detections, up to five frames. `not_identified` means the software did not identify staff in that frame; it does not prove no staff were present. Hidden tags and lengthy occlusion remain a limitation. The output is a set of predictions requiring review, not an exhaustive ground-truth answer.

For a repeat run of this exact sample, `--people-cache outputs/staff-full-sample/observations.json` can reuse previously computed person boxes. A source hash prevents applying that cache to a different video. Cached-run speed excludes person inference. The cache is optional and is not used on a new recording.

The training data and annotation review image are in `data/sample-tag-development/`. Positive examples use separate walking passes across development splits, with reviewed hard negatives from patterned clothes, another shirt logo, and scenery. All splits still come from one clip. See `docs/verification.md` for scope and limitations. The earlier template-only and first-model outputs are retained for comparison; use the current staff entry point above for new runs.

## Source documents

- [Detection approach](docs/approach.md): architecture, data requirements, limitations, and acceptance evidence.
- [Official references](docs/sources.md): source links and implementation rationale.
- [Agent instructions](AGENTS.md): project context for future work.

## Set up

Python 3.9 or newer. Run from the project folder:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
python -m cctv_detector --help
```

An environment has already been prepared in `.venv` on this machine. `requirements-tested.txt` records the versions installed during initial verification; it is a macOS Apple Silicon/Python 3.9 snapshot, not a portable cross-platform lockfile. The small initial detector is YOLO11 nano. Use `--device cpu`, `--device mps` on supported Macs, or `--device 0` on CUDA hardware; measure performance before choosing.

## Prepare footage and labels

First inspect a local recording:

```sh
python -m cctv_detector inspect /path/to/recording.mp4
python -m cctv_detector extract /path/to/recording.mp4 --every 2 --max-images 100
```

Extraction saves full-resolution JPEG frames and a timestamp manifest in a new directory. Inspect the logo at the expected viewing distance. For training, crop shirts/persons to resemble inference inputs and annotate logo boxes relative to those saved crops in CVAT. Export YOLO detection labels. Include negative images; use empty label files for images with no target logo.

Split whole recordings into train, validation, and test groups **before** extracting frames. A suggested directory layout is:

```text
logo-dataset/
  images/train/    images/val/    images/test/
  labels/train/    labels/val/    labels/test/
```

Each image has a matching `.txt` label with the same stem. Each logo line is `class_id center_x center_y width height`, with coordinates normalized to the image dimensions. Copy `configs/logo-dataset.example.yaml`, set its absolute dataset path, and set the actual class name.

## Train and evaluate the logo model

```sh
python -m cctv_detector train --data configs/my-logo.yaml --epochs 50 --device cpu
python -m cctv_detector evaluate --data configs/my-logo.yaml --model runs/logo/train/weights/best.pt --split test --device cpu
```

Training output directories increment if they already exist; use the actual `best.pt` path printed by training. The epoch count is a starting setting, not an accuracy guarantee. Evaluation reports detector metrics on labeled images; end-to-end alert accuracy still needs held-out CCTV recordings and manually checked appearances.

## Run detection

```sh
python -m cctv_detector run --source /path/to/recording.mp4 --logo-model /path/to/best.pt --logo-class target_logo --show
```

`--logo-class` must exactly match the class name stored in your trained model. A missing model or class fails clearly. For a webcam use `--source 0`. For a live camera, set `CCTV_SOURCE` to its stream URL in your local environment and omit `--source`; keep camera credentials out of project files.

```sh
python -m cctv_detector run --logo-model /path/to/best.pt --logo-class target_logo
```

Press Q/Escape in the preview or Ctrl-C in the terminal to stop. Omit `--show` for operation without a preview. `--max-frames 300` bounds a test run.

Default confirmation requires three positive observations within two seconds, a positive ratio of at least 0.6, and a positive span of at least 0.2 seconds. These are tunable starting values. Negatives count against confirmation. One alert is saved per active track. After a person disappears long enough or a stream reconnects, they may receive a new track and another alert. Tracking does not identify a person across visits.

Shirt search defaults to 15–70% of the person box's height. Adjust `--shirt-top` and `--shirt-bottom` to your camera angle; `0` and `1` search the whole person. The region is a geometric approximation and can contain bags, another person, or background. A visible logo detection does not independently prove it is printed on the shirt.

Each run creates a unique directory under `outputs/` with:

- `events.jsonl`: confirmed events, video timestamps, boxes, confidence, and snapshot filenames.
- `alert-*.jpg`: annotated evidence snapshots.
- `settings.json`: confirmation settings and source type, without camera URLs.
- `summary.json`: completion status, processed frames, alerts, and processing rate.

Live capture uses a latest-frame queue so inference does not build an application-level backlog. It attempts reconnection on read failures and resets evidence and tracking after reconnecting. Real RTSP compatibility, buffering, and latency depend on the camera and backend and remain unverified. File timestamps use frame index / nominal FPS, so variable-frame-rate recordings should be transcoded to constant frame rate for accurate timing. Event timestamps refer to the current confirmation frame, not the first sighting.

## Verify

```sh
python -m unittest discover -s tests -v
```

The test suite uses synthetic frames and controlled detector responses to check confirmation, crop mapping, decoding, and saved events. It does not measure logo accuracy. Before deployment, measure missed target appearances, false alerts per camera-hour, duplicate alerts, and camera-to-alert delay using real footage and intended hardware. Keep validation and final test recordings separate.

Optional real-library checks are available as `python scripts/smoke_inference.py` and then `python scripts/smoke_training.py`. The first downloads the small person checkpoint if needed and uses it as a stand-in for both detection stages. The second trains for one epoch on synthetic shapes and evaluates the resulting checkpoint. They verify integration only. Their test footage, logs, and weights are kept separate from real training data and must not be treated as a trained shirt-logo detector. See `docs/verification.md` for results and remaining gaps.
