"""Sequential file decoding and a latest-frame reader for live sources."""

from dataclasses import dataclass
import math
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
import time
from typing import Optional


@dataclass
class Frame:
    image: object
    index: int
    timestamp: float
    generation: int = 0
    captured_monotonic: Optional[float] = None


class VideoSource:
    def __init__(self, source: str, reconnects: int = 3):
        import cv2

        self.cv2 = cv2
        self.source = source
        self.network = source.lower().startswith(("rtsp://", "rtsps://", "http://", "https://"))
        self.camera = source.isdecimal()
        self.live = self.network or self.camera
        if not self.live and not Path(source).is_file():
            raise ValueError("Video file does not exist.")
        # Backends can otherwise include camera passwords in diagnostic output.
        if self.network:
            cv2.setLogLevel(0)
        self.reconnects = reconnects
        self.cap = self._open()
        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.fps = fps if math.isfinite(fps) and fps > 0 else None
        count = float(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_count = int(count) if math.isfinite(count) and count > 0 else None
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not self.live and self.fps is None:
            self.cap.release()
            raise ValueError("The video has no usable frame rate; transcode it before processing.")
        self.stop = Event()
        self.done = Event()
        self.queue = Queue(maxsize=1)
        self.thread: Optional[Thread] = None
        self.error: Optional[str] = None

    def _open(self):
        cv2 = self.cv2
        if self.network:
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG,
                                   [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])
        else:
            cap = cv2.VideoCapture(int(self.source) if self.camera else self.source)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError("Cannot open video source. Check the file or camera connection.")
        return cap

    def _pump(self):
        started = time.monotonic()
        index, generation, failures = 0, 0, 0
        try:
            while not self.stop.is_set():
                ok, image = self.cap.read()
                if not ok:
                    self.cap.release()
                    while not self.stop.is_set():
                        if failures >= self.reconnects:
                            raise RuntimeError("Live video disconnected; reconnect attempts exhausted.")
                        failures += 1
                        if self.stop.wait(1.0):
                            return
                        try:
                            self.cap = self._open()
                            generation += 1
                            break
                        except RuntimeError:
                            continue
                    continue
                failures = 0
                captured = time.monotonic()
                item = Frame(image, index, captured - started, generation, captured)
                index += 1
                try:
                    self.queue.put_nowait(item)
                except Full:
                    try:
                        self.queue.get_nowait()
                    except Empty:
                        pass
                    self.queue.put_nowait(item)
        except Exception:
            self.error = "Live video read failed or reconnect attempts were exhausted."
        finally:
            self.cap.release()
            self.done.set()

    def __iter__(self):
        if self.live:
            self.thread = Thread(target=self._pump, daemon=True, name="camera-reader")
            self.thread.start()
            while not self.stop.is_set():
                try:
                    yield self.queue.get(timeout=0.2)
                except Empty:
                    if self.done.is_set():
                        if self.error:
                            raise RuntimeError(self.error)
                        return
        else:
            index = 0
            while not self.stop.is_set():
                ok, image = self.cap.read()
                if not ok:
                    # For indexed files, an early decode failure is not a clean EOF.
                    if self.frame_count is not None and index < self.frame_count - 1:
                        raise RuntimeError("Video decoding stopped before the advertised end of the file.")
                    return
                yield Frame(image, index, index / self.fps)
                index += 1

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=6.0)
        else:
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
