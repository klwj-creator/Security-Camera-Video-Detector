# Verification record

14 November 2026, Asia/Kuala_Lumpur.

## Verified software behavior

- Installed project successfully with `python -m pip install .`; the packaged command displays help.
- Dependency consistency check passed (`python -m pip check`).
- All 30 current tests passed (`python -m unittest discover -s tests -v`). Tests cover evidence expiry, negative observations, duplicate timestamps, independent tracks, stale ID cleanup, original-resolution crop mapping, unknown target classes, real video decoding and extraction, saved alert snapshots, reconnect resets, latest-frame queue behavior, overhead rotation mapping, tag-template support, duplicate tag ownership, prevention of long-history backfill, and bounded tag interpolation.
- Ran the actual Ultralytics YOLO11 nano person detector with ByteTrack on the reference bus image bundled in the dependency. It produced three person tracks.
- Processed a six-frame generated recording through both inference stages, tracking, confirmation, and snapshot/event output. All six frames were processed successfully.
- Ran the actual training command for one epoch on a tiny synthetic shape dataset. It produced `best.pt`; the evaluation command then successfully loaded that checkpoint and evaluated the separate synthetic test split. This checks command/API compatibility, not training quality for any logo. Reproduce with `scripts/smoke_training.py` after the inference smoke check; `.runtime/training-smoke.json` identifies the latest report, checkpoint, and logs.

The second detector in this integration check was deliberately another COCO person detector, with target class `person`, because no trained target-logo model has been supplied. Its three generated events are test artifacts and are not shirt-logo detections. `scripts/smoke_inference.py` documents and reproduces this check; `.runtime/inference-smoke.json` stores the latest local report. Smoke-test outputs belong under `outputs/integration-smoke/`.

## Not established

- Exhaustive staff presence on every sample frame, including long periods when the tag is hidden. The current output is conservative tag evidence and short-gap interpolation, not a fully reviewed ground-truth presence annotation.
- Generalization to independent recordings: the new staff-tag model and all development splits use the supplied clip.
- Real RTSP camera connectivity, recovery, buffering, and camera-to-alert latency: no camera supplied.
- Reliable attribution to a shirt in crowded/overlapping scenes: geometric crop assumptions require real footage evaluation.

The processing rate in the integration report is a short CPU smoke measurement with a small static fixture, not a production benchmark or a live latency measurement.

## Supplied staff-tag evaluation

The user supplied the evaluation PDF and `sample.mp4`. Both are preserved unchanged in `details/`, with SHA-256 hashes matching the Downloads originals. Embedded reference images and extracted text are saved alongside them. The video decodes to 1,341 frames at 25 FPS and 960 x 720 pixels.

The template-only baseline found clear tags but also showed false matches and serious long-track backfill contamination. It is retained as an experiment, not the recommended answer. The current staff entry point is `scripts/detect_staff.py` with `models/staff-tag-sample-v2.pt`.

Actual tag training used reviewed, template-assisted boxes from the supplied clip. The refined development splits contain 15 positive + 52 negative training crops, 15 positive + 44 negative validation crops, and 5 positive + 56 negative test crops. Positive passes are temporally separated across splits. Full-clip review identified hard negatives on a patterned shirt, unrelated shirt lettering, and scenery. All splits remain correlated through the same camera, actors, and recording.

The refined model's standard crop-level evaluation on 61 development test images (5 annotated tags, 56 negative images) reported precision 0.993, recall 1.0, mAP50 0.995, and mAP50-95 0.542. These figures concern that tiny annotated crop set under the evaluator's thresholds, not whole-video staff-presence accuracy, not runtime threshold performance, and not independent deployment validation. Logs are in `.runtime/staff-tag-refined-test.log` and `runs/staff-tag-refined-test/`.

The current full-clip run is `outputs/staff-tag-sample/`:

- Model confidence threshold 0.25; local pattern correlation threshold 0.83; input size 1280.
- 147 predicted-positive frames: 133 with retained direct tag evidence and 14 with bounded interpolation.
- CSV contains all 1,341 frame decisions, zero- and one-based frame numbering, timestamps, tag/chest xy proxies, and associated body-box centers where available.
- Video and evidence stills show direct boxes in green and interpolated boxes in orange. `not_identified` is not a verified absence label.
- Short trajectories require repeated observations and fill at most five missing frames. No entire long person-ID history is relabeled after a tag hit.
- Reused source-hash-matched person boxes for this repeat experiment. The approximately 13 FPS tag-stage measurement excludes person inference and must not be presented as real-time full-pipeline performance.

A separate 16-frame re-encoded excerpt was run through the complete standalone command without any person cache. It processed all 16 frames and detected the expected clear light-shirt tag over four frames. This verifies the no-cache code path and support for a different input file, not independent-scene accuracy. Results are under `.runtime/staff-standalone-results/`.

A two-page prototype solution note is saved as `output/pdf/staff-detection-solution.pdf` and visually checked after rendering. The full supplied task is not proven solved: hidden-tag staff appearances, a complete presence ground truth, and independent recordings remain necessary for stronger claims.
