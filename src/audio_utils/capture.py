"""Mic capture smoke test for Phase 0 exit criteria.

Usage:
    python -m src.audio_utils.capture --seconds 3 --out data/raw/smoke_test.wav
"""

import argparse

import numpy as np
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16_000


def record(seconds: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    return audio.reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--out", type=str, default="data/raw/smoke_test.wav")
    args = parser.parse_args()

    print(f"Recording {args.seconds}s at {SAMPLE_RATE} Hz mono...")
    audio = record(args.seconds)
    sf.write(args.out, audio, SAMPLE_RATE, subtype="PCM_16")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
