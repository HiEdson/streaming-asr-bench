from typing import Optional
import jiwer

from bench.events import EmissionLog
from bench.reference import normalize


def _tokens(text: str) -> list[str]:
    return normalize(text).split()


def wer(log: EmissionLog) -> Optional[float]:
    if not log.reference:
        return None
    ref, hyp = normalize(log.reference), normalize(log.final_hypothesis)
    if not ref:
        return None
    return jiwer.wer(ref, hyp)


def ttft(log: EmissionLog) -> Optional[float]:
    """Realtime seconds until the first non-empty hypothesis."""
    for e in log.events:
        if normalize(e.hypothesis):
            return e.realtime_emission
    return None


def revision_rate(log: EmissionLog) -> Optional[float]:
    """Fraction of updates that rewrote earlier output rather than extending it."""
    if len(log.events) < 2:
        return None
    revisions = 0
    comparisons = 0
    prev = _tokens(log.events[0].hypothesis)
    for e in log.events[1:]:
        cur = _tokens(e.hypothesis)
        comparisons += 1
        if cur[:len(prev)] != prev:
            revisions += 1
        prev = cur
    return revisions / comparisons if comparisons else None


def compute_all(log: EmissionLog) -> dict:
    return {
        "model": log.model,
        "audio_path": log.audio_path,
        "audio_duration": log.audio_duration,
        "wer": wer(log),
        "ttft": ttft(log),
        "rtf": log.realtime_factor,
        "revision_rate": revision_rate(log),
        **log.config,
    }