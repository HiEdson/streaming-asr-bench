from abc import ABC, abstractmethod
from typing import Optional
from bench.stream import AudioChunk


class StreamingASR(ABC):
    """One streaming ASR model under one configuration.

    Contract:
      - push() returns the FULL hypothesis so far, never a delta.
        Revision rate is computed by diffing consecutive hypotheses,
        which is impossible if adapters return increments.
      - push() returns None when the model produced no new output
        for this chunk. Empty string means "decoded, result is empty" —
        these are different events and the metrics code distinguishes them.
      - The adapter owns its own buffering and internal state. The runner
        feeds chunks and records outputs; it knows nothing about how the
        model consumes them.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier for results, e.g. 'faster-whisper/tiny.en'."""

    @property
    @abstractmethod
    def config(self) -> dict:
        """Every parameter that could affect WER or latency.

        Stamped into the EmissionLog so a result can always be traced
        back to the exact setup that produced it.
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear all state. Called before each utterance."""

    @abstractmethod
    def push(self, chunk: AudioChunk) -> Optional[str]:
        """Feed one chunk. Return the full hypothesis, or None."""

    @abstractmethod
    def finalize(self) -> str:
        """Audio has ended. Return the final hypothesis."""