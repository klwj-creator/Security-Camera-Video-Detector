"""Command line workflow. Help and confirmation tests need no ML packages."""

import argparse
import math
import os
from pathlib import Path
import sys

from .evidence import ConfirmationConfig


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Must be a finite positive number.")
    return number


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer.")
    return number


def probability(value):
    number = positive(value)
    if number > 1:
        raise argparse.ArgumentTypeError("Must be at most 1.")
    return number


def add_model_options(parser):
    parser.add_argument("--device", help="cpu, mps (Apple GPU), or CUDA device such as 0")
    parser.add_argument("--imgsz", type=positive_int, default=640)


def parser():
    root = argparse.ArgumentParser(description="Local CCTV shirt-logo detector. A custom logo model is required.")
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Detect and track a target logo in video")
    run.add_argument("--source", help="Video path, webcam index, or stream URL; alternatively set CCTV_SOURCE")
    run.add_argument("--person-model", default="yolo11n.pt")
    run.add_argument("--logo-model", required=True, help="Local custom logo detection checkpoint (.pt)")
    run.add_argument("--logo-class", required=True, help="Exact target class name stored in your model")
    run.add_argument("--person-conf", type=probability, default=0.35)
    run.add_argument("--logo-conf", type=probability, default=0.6)
    run.add_argument("--min-hits", type=positive_int, default=3)
    run.add_argument("--window-seconds", type=positive, default=2.0)
    run.add_argument("--min-ratio", type=probability, default=0.6)
    run.add_argument("--min-span", type=float, default=0.2)
    run.add_argument("--track-ttl", type=positive, default=2.0)
    run.add_argument("--shirt-top", type=float, default=0.15)
    run.add_argument("--shirt-bottom", type=float, default=0.7)
    run.add_argument("--output", default="outputs")
    run.add_argument("--show", action="store_true", help="Open an annotated video preview")
    run.add_argument("--max-frames", type=positive_int)
    run.add_argument("--reconnects", type=int, default=3)
    add_model_options(run)
    inspect = commands.add_parser("inspect", help="Read a local video's dimensions and frame rate")
    inspect.add_argument("video")
    extract = commands.add_parser("extract", help="Save spaced full frames from a local video for annotation")
    extract.add_argument("video")
    extract.add_argument("--output", default="data/extracted")
    extract.add_argument("--every", type=positive, default=2.0, help="Seconds between saved frames")
    extract.add_argument("--max-images", type=positive_int, default=100)
    train = commands.add_parser("train", help="Fine-tune a logo detector on a labeled YOLO dataset")
    train.add_argument("--data", required=True)
    train.add_argument("--model", default="yolo11n.pt")
    train.add_argument("--epochs", type=positive_int, default=50)
    train.add_argument("--batch", type=positive_int, default=8)
    train.add_argument("--output", default="runs/logo")
    train.add_argument("--degrees", type=float, default=0, help="Rotation augmentation range; overhead tags may need 180")
    train.add_argument("--flipud", type=float, default=0, help="Vertical flip probability")
    add_model_options(train)
    evaluate = commands.add_parser("evaluate", help="Evaluate the logo detector on held-out labeled images")
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--split", choices=("val", "test"), default="test")
    evaluate.add_argument("--output", default="runs/evaluation")
    add_model_options(evaluate)
    return root


def execute(args):
    # Keep third-party settings inside the project and out of the user's home config.
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".runtime/ultralytics").resolve()))
    if args.command == "run":
        source_name = args.source or os.environ.get("CCTV_SOURCE")
        if not source_name:
            raise ValueError("Provide --source or set CCTV_SOURCE.")
        if not 0 <= args.shirt_top < args.shirt_bottom <= 1:
            raise ValueError("Shirt bounds must satisfy 0 <= top < bottom <= 1.")
        if args.reconnects < 0:
            raise ValueError("Reconnect count cannot be negative.")
        config = ConfirmationConfig(args.min_hits, args.window_seconds, args.min_ratio,
                                    args.min_span, args.track_ttl)
        from .vision import YoloPipeline
        from .video import VideoSource
        from .runner import run_session

        pipeline = YoloPipeline(args.person_model, args.logo_model, args.logo_class,
                                args.person_conf, args.logo_conf, args.device, args.imgsz,
                                args.shirt_top, args.shirt_bottom)
        with VideoSource(source_name, args.reconnects) as source:
            path, summary = run_session(source, pipeline, args.output, args.logo_class, config,
                                        args.show, args.max_frames)
        print(f"{summary['status']}: {summary['processed_frames']} frames, {summary['alerts']} alerts. Results: {path}")
        return 130 if summary["status"] == "interrupted" else 0
    if args.command in ("inspect", "extract"):
        if not Path(args.video).is_file():
            raise ValueError("Provide a local video file for inspection or extraction.")
        import cv2
        from .video import VideoSource
        from .runner import new_run, write_json

        with VideoSource(args.video) as source:
            if args.command == "inspect":
                iterator = iter(source)
                if next(iterator, None) is None:
                    raise ValueError("The video has no readable frames.")
                print(f"{source.width} x {source.height}, {source.fps:.3f} FPS, "
                      f"{source.frame_count or 'unknown'} frames")
                return 0
            output = new_run(args.output, "frames")
            manifest = {"source_file": Path(args.video).name, "images": []}
            next_time = 0.0
            for item in source:
                if item.timestamp + 1e-9 < next_time:
                    continue
                name = f"frame-{item.index:08d}.jpg"
                if not cv2.imwrite(str(output / name), item.image):
                    raise RuntimeError("Could not save an extracted image.")
                manifest["images"].append({"file": name, "frame_index": item.index, "seconds": item.timestamp})
                next_time = item.timestamp + args.every
                if len(manifest["images"]) >= args.max_images:
                    break
            if not manifest["images"]:
                raise RuntimeError("The video yielded no images.")
            write_json(output / "manifest.json", manifest)
            print(f"Saved {len(manifest['images'])} full frames to {output}. Split recordings before labeling.")
            return 0
    if not Path(args.data).is_file():
        raise ValueError("Dataset YAML does not exist. See configs/logo-dataset.example.yaml.")
    if args.command == "evaluate" and not Path(args.model).is_file():
        raise ValueError("Trained model file does not exist.")
    from ultralytics import YOLO

    model = YOLO(args.model)
    if model.task != "detect":
        raise ValueError("A detection model is required.")
    options = {"data": str(Path(args.data).resolve()), "imgsz": args.imgsz,
               "project": str(Path(args.output).resolve())}
    if args.device is not None:
        options["device"] = args.device
    if args.command == "train":
        if not math.isfinite(args.degrees) or not 0 <= args.degrees <= 180 or not 0 <= args.flipud <= 1:
            raise ValueError("Training degrees must be 0–180 and flipud must be 0–1.")
        model.train(epochs=args.epochs, batch=args.batch, workers=0,
                    degrees=args.degrees, flipud=args.flipud, **options)
    else:
        model.val(split=args.split, **options)
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return execute(args)
    except ImportError as exc:
        print(f"Missing dependency: {exc.name}. Install this project's requirements first.", file=sys.stderr)
        return 2
    except (ValueError, RuntimeError, OSError) as exc:
        # Do not echo library exceptions containing stream credentials.
        if args.command == "run" and (args.source or os.environ.get("CCTV_SOURCE", "")).lower().startswith(("rtsp:", "rtsps:", "http:", "https:")):
            print("Detection failed. Check camera connectivity, model files, and output permissions.", file=sys.stderr)
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 2
