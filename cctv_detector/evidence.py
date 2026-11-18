"""Time-bounded confirmation independent of video and ML dependencies."""

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Deque, Dict, Tuple


@dataclass(frozen=True)
class ConfirmationConfig:
    min_hits: int = 3
    window_seconds: float = 2.0
    min_ratio: float = 0.6
    min_span: float = 0.2
    track_ttl: float = 2.0

    def __post_init__(self):
        values = (self.window_seconds, self.min_ratio, self.min_span, self.track_ttl)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Confirmation settings must be finite.")
        if self.min_hits < 1 or self.window_seconds <= 0 or self.track_ttl <= 0:
            raise ValueError("Hit count, evidence window, and track lifetime must be positive.")
        if not 0 < self.min_ratio <= 1:
            raise ValueError("Minimum positive ratio must be in (0, 1].")
        if not 0 <= self.min_span <= self.window_seconds:
            raise ValueError("Minimum evidence span must fit within the evidence window.")


@dataclass
class TrackEvidence:
    last_seen: float
    samples: Deque[Tuple[float, bool]] = field(default_factory=deque)
    alerted: bool = False


class Confirmation:
    def __init__(self, config: ConfirmationConfig):
        self.config = config
        self.tracks: Dict[int, TrackEvidence] = {}
        self._last_time = -math.inf

    def advance(self, timestamp: float):
        if not math.isfinite(timestamp) or timestamp < self._last_time:
            raise ValueError("Video timestamps must be finite and nondecreasing.")
        self._last_time = timestamp
        expired = [key for key, state in self.tracks.items()
                   if timestamp - state.last_seen > self.config.track_ttl]
        for key in expired:
            del self.tracks[key]

    def observe(self, track_id: int, timestamp: float, positive: bool) -> bool:
        """Return true once per active track, only on a positive observation."""
        self.advance(timestamp)
        state = self.tracks.setdefault(track_id, TrackEvidence(timestamp))
        # One observation per track and frame; repeated calls must not inflate votes.
        if state.samples and timestamp == state.samples[-1][0]:
            return False
        state.last_seen = timestamp
        state.samples.append((timestamp, positive))
        while state.samples and timestamp - state.samples[0][0] > self.config.window_seconds:
            state.samples.popleft()
        hits = [t for t, hit in state.samples if hit]
        enough = (len(hits) >= self.config.min_hits
                  and len(hits) / len(state.samples) >= self.config.min_ratio
                  and hits[-1] - hits[0] >= self.config.min_span) if hits else False
        if positive and enough and not state.alerted:
            state.alerted = True
            return True
        return False
