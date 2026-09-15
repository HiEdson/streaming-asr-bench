import sys
import time
import json
from pathlib import Path
from bench.reference import get_reference
from bench.metrics import compute_all

from bench.stream import chunk_audio, audio_duration
from bench.events import EmissionEvent, EmissionLog
from bench.adapters.faster_whisper import FasterWhisperAdapter


def run(audio_path: str, adapter, chunk_ms: int = 500) -> EmissionLog:
    adapter.reset()

    log = EmissionLog(
        model=adapter.name,
        config={**adapter.config, "chunk_ms": chunk_ms},
        audio_path=audio_path,
        audio_duration=audio_duration(audio_path),
    )
    log.reference = get_reference(audio_path)
    compute_elapsed = 0.0

    for chunk in chunk_audio(audio_path, chunk_ms=chunk_ms):
        t0 = time.perf_counter()
        hypothesis = adapter.push(chunk)
        compute_elapsed += time.perf_counter() - t0

        if hypothesis is not None:
            log.add(EmissionEvent(
                chunk_index=chunk.index,
                audio_consumed=chunk.audio_consumed,
                compute_elapsed=compute_elapsed,
                hypothesis=hypothesis,
            ))

    t0 = time.perf_counter()
    log.final_hypothesis = adapter.finalize()
    compute_elapsed += time.perf_counter() - t0

    return log


if __name__ == "__main__":
    audio_path = sys.argv[1]

    adapter = FasterWhisperAdapter(model_size="tiny.en")
    log = run(audio_path, adapter)
    log.reference = get_reference(audio_path)

    Path("results").mkdir(exist_ok=True)
    out = Path("results") / (Path(audio_path).stem + ".json")
    log.save(str(out))

    m = compute_all(log)
    print(f"model: {m['model']}")
    print(f"duration: {m['audio_duration']:.2f}s")
    print(f"WER: {m['wer']:.3f}" if m['wer'] is not None else "WER: n/a")
    print(f"TTFT: {m['ttft']:.2f}s" if m['ttft'] is not None else "TTFT: n/a")
    print(f"RTF: {m['rtf']:.2f}")
    print(f"revision rate: {m['revision_rate']:.2f}" if m['revision_rate'] is not None else "revision rate: n/a")
    print()

    for e in log.events:
        print(f"  [{e.audio_consumed:5.2f}s audio | {e.compute_elapsed:6.2f}s compute] {e.hypothesis!r}")
    print()
    print(f"final: {log.final_hypothesis!r}")
    print(f"saved: {out}")
    