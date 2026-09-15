from dataclasses import dataclass
from typing import Iterator
import numpy as np
import soundfile as sf


@dataclass
class AudioChunk:
    index: int
    samples: np.ndarray
    audio_consumed: float   # seconds of audio delivered INCLUDING this chunk
    is_final: bool


def load_audio(path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    samples, sr = sf.read(path, dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if sr != target_sr:
        raise ValueError(f"{path} is {sr} Hz, expected {target_sr}")
    return samples, sr


def chunk_audio(path: str, chunk_ms: int = 500,
                target_sr: int = 16000) -> Iterator[AudioChunk]:
    """Yield an audio file as if it were arriving from a live microphone."""
    samples, sr = load_audio(path, target_sr)
    chunk_samples = int(sr * chunk_ms / 1000)
    total = len(samples)

    for i, start in enumerate(range(0, total, chunk_samples)):
        end = min(start + chunk_samples, total)
        yield AudioChunk(
            index=i,
            samples=samples[start:end],
            audio_consumed=end / sr,
            is_final=(end >= total),
        )


def audio_duration(path: str) -> float:
    info = sf.info(path)
    return info.frames / info.samplerate