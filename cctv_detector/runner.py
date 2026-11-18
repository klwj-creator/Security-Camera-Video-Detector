"""Run a detection session and persist local evidence."""

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from uuid import uuid4

from .evidence import Confirmation, ConfirmationConfig
from .vision import annotate


def new_run(root, prefix="run"):
    path = Path(root) / f"{prefix}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_session(source, pipeline, output, logo_class, config: ConfirmationConfig,
                show=False, max_frames=None):
    import cv2

    output = new_run(output)
    confirmation = Confirmation(config)
    frames, alerts, reconnects = 0, 0, 0
    generation = 0
    started = time.monotonic()
    status = "running"
    last_timestamp = None
    max_capture_age = 0.0
    write_json(output / "settings.json", {
        "logo_class": logo_class, "confirmation": asdict(config),
        "source_type": "live" if source.live else "file",
        "timestamp_basis": "monotonic capture seconds" if source.live else "frame index / nominal FPS",
        "note": "Settings and outputs do not establish real-world recognition accuracy.",
    })
    try:
        with (output / "events.jsonl").open("w", encoding="utf-8") as events:
            for item in source:
                if item.generation != generation:
                    pipeline.reset()
                    confirmation = Confirmation(config)
                    reconnects += 1
                    generation = item.generation
                confirmation.advance(item.timestamp)
                people = pipeline.detect(item.image)
                confirmed = []
                for person in people:
                    if confirmation.observe(person.track_id, item.timestamp, bool(person.logos)):
                        confirmed.append(person)
                canvas = annotate(item.image, people, logo_class, {p.track_id for p in confirmed})
                for person in confirmed:
                    snapshot = f"alert-{alerts + 1:06d}.jpg"
                    if not cv2.imwrite(str(output / snapshot), canvas):
                        raise RuntimeError("Could not save the alert snapshot.")
                    payload = {
                        "event": "logo_confirmed", "logo_class": logo_class,
                        "track_id": person.track_id, "stream_generation": generation,
                        "frame_index": item.index, "video_seconds": round(item.timestamp, 3),
                        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                        "person_box": person.box,
                        "logos": [asdict(logo) for logo in person.logos], "snapshot": snapshot,
                    }
                    events.write(json.dumps(payload, allow_nan=False) + "\n")
                    events.flush()
                    alerts += 1
                    print(f"Logo confirmed on track {person.track_id} at {item.timestamp:.2f}s: {output / snapshot}", flush=True)
                frames += 1
                last_timestamp = item.timestamp
                if source.live and item.captured_monotonic is not None:
                    # Excludes unknown latency inside the camera and decoder.
                    max_capture_age = max(max_capture_age, time.monotonic() - item.captured_monotonic)
                if show:
                    cv2.imshow("CCTV logo detector - Q to stop", canvas)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        status = "stopped_by_user"
                        break
                if max_frames is not None and frames >= max_frames:
                    status = "frame_limit"
                    break
            else:
                if frames == 0:
                    raise RuntimeError("The source yielded no readable frames.")
                status = "finished"
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception:
        status = "failed"
        raise
    finally:
        source.close()
        if show:
            cv2.destroyAllWindows()
        elapsed = time.monotonic() - started
        summary = {
            "status": status, "processed_frames": frames, "alerts": alerts,
            "elapsed_seconds": round(elapsed, 3),
            "processing_fps": round(frames / elapsed, 3) if elapsed else 0,
            "last_video_seconds": last_timestamp, "observed_reconnects": reconnects,
            "max_delivery_to_result_seconds_approx": round(max_capture_age, 3) if source.live else None,
            "latency_note": "Does not measure camera-to-capture latency. Live reader drops queued frames to favor freshness.",
        }
        write_json(output / "summary.json", summary)
    return output, summary
