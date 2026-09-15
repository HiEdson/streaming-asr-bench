import csv
import sys
from pathlib import Path

from bench.adapters.faster_whisper import FasterWhisperAdapter
from bench.metrics import compute_all
from scripts.run_one import run


def find_audio(root: str, limit: int | None = None) -> list[str]:
    paths = sorted(str(p) for p in Path(root).rglob("*.flac"))
    return paths[:limit] if limit else paths

def fmt(v, spec=".3f"):
    return format(v, spec) if v is not None else "n/a"

def main():
    root = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    paths = find_audio(root, limit)
    print(f"found {len(paths)} files")

    adapter = FasterWhisperAdapter(model_size="tiny.en")

    rows = []
    for i, path in enumerate(paths, 1):
        log = run(path, adapter)
        m = compute_all(log)
        rows.append(m)

        # print(f"[{i}/{len(paths)}] {Path(path).stem} "
        #       f"WER={m['wer']:.3f} TTFT={m['ttft']:.2f} "
        #       f"RTF={m['rtf']:.2f} rev={m['revision_rate']:.2f}")

        print(f"[{i}/{len(paths)}] {Path(path).stem} "
              f"WER={fmt(m['wer'])} TTFT={fmt(m['ttft'], '.2f')} "
              f"RTF={fmt(m['rtf'], '.2f')} rev={fmt(m['revision_rate'], '.2f')}")

    Path("results").mkdir(exist_ok=True)
    out = Path("results") / "run_many.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def mean(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    print()
    print(f"n = {len(rows)}")
    print(f"mean WER:           {mean('wer'):.3f}")
    print(f"mean TTFT:          {mean('ttft'):.2f}s")
    print(f"mean RTF:           {mean('rtf'):.2f}")
    print(f"mean revision rate: {mean('revision_rate'):.2f}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()