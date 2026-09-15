import numpy as np
from typing import Optional
from faster_whisper import WhisperModel

from bench.adapters.base import StreamingASR
from bench.stream import AudioChunk


class FasterWhisperAdapter(StreamingASR):
    def __init__(self, model_size="tiny.en", decode_every=1, compute_type="int8"):
        self.model_size = model_size
        self.decode_every = decode_every
        self.compute_type = compute_type
        self.model = WhisperModel(model_size, compute_type=compute_type)

        segments, _ = self.model.transcribe(np.zeros(16000, dtype=np.float32))
        warmup_text = " ".join(s.text for s in segments)
        print(f"[warmup] silence produced: {warmup_text!r}")
        
        self.reset()

    @property
    def name(self) -> str:
        return f"faster-whisper/{self.model_size}"

    @property
    def config(self) -> dict:
        return {
            "model_size": self.model_size,
            "decode_every": self.decode_every,
            "compute_type": self.compute_type,
        }

    def reset(self) -> None:
        self.buffer = np.array([], dtype=np.float32)
        self.n_pushed = 0

    def push(self, chunk: AudioChunk) -> Optional[str]:
        self.buffer = np.concatenate([self.buffer, chunk.samples])
        self.n_pushed += 1
        if self.n_pushed % self.decode_every != 0:
            return None
        segments, _ = self.model.transcribe(self.buffer)
        return " ".join(s.text for s in segments).strip()

    def finalize(self) -> str:
        segments, _ = self.model.transcribe(self.buffer)
        return " ".join(s.text for s in segments).strip()