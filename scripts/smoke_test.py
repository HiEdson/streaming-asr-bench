from bench.stream import chunk_audio, audio_duration
import sys

path = sys.argv[1]
print("duration:", audio_duration(path))
for c in chunk_audio(path):
    print(c.index, c.samples.shape, round(c.audio_consumed, 2), c.is_final)
