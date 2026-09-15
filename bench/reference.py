from pathlib import Path
from whisper.normalizers import EnglishTextNormalizer

_normalizer = EnglishTextNormalizer()


def normalize(text: str) -> str:
    return _normalizer(text)


def get_reference(audio_path: str) -> str:
    """Find the LibriSpeech transcript for an audio file."""
    p = Path(audio_path)
    utt_id = p.stem                      # e.g. 1272-128104-0000
    speaker, chapter, _ = utt_id.split("-")
    trans = p.parent / f"{speaker}-{chapter}.trans.txt"

    for line in trans.read_text().splitlines():
        line_id, _, text = line.partition(" ")
        if line_id == utt_id:
            return text
    raise KeyError(f"No transcript for {utt_id} in {trans}")