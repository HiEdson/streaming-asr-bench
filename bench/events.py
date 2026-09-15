from dataclasses import dataclass, asdict, field
from typing import List, Optional
import json


@dataclass
class EmissionEvent:
    """One observation of the model's output during streaming."""
    chunk_index: int
    audio_consumed: float      # seconds of audio fed to the model so far
    compute_elapsed: float     # cumulative seconds of wall-clock compute
    hypothesis: str            # FULL partial hypothesis at this moment

    @property
    def realtime_emission(self) -> float:
        """When this output could exist in a live system.

        Audio arrives in real time, so you cannot emit before the audio
        has been spoken; and you cannot emit before compute has finished.
        Whichever is later governs.
        """
        return max(self.audio_consumed, self.compute_elapsed)


@dataclass
class EmissionLog:
    model: str
    config: dict
    audio_path: str
    audio_duration: float
    reference: str = ""
    events: List[EmissionEvent] = field(default_factory=list)
    final_hypothesis: str = ""

    def add(self, event: EmissionEvent) -> None:
        self.events.append(event)

    @property
    def realtime_factor(self) -> float:
        if not self.events:
            return 0.0
        return self.events[-1].compute_elapsed / self.audio_duration

    def save(self, path: str) -> None:
        payload = asdict(self)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "EmissionLog":
        with open(path) as f:
            d = json.load(f)
        events = [EmissionEvent(**e) for e in d.pop("events")]
        return cls(events=events, **d)