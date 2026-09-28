"""Guided recording session for Phase 1.2 (own laptop-mic dataset).

Walks a consented speaker through a batch of prompts (random digit
challenges or free-speech), records each on the laptop mic, and appends a row
per file to data/splits/genuine_metadata.csv.

Usage:
    python -m src.data.record_session --speaker-id s01 --session 1 \\
        --condition quiet --distance-cm 50 --language en \\
        --n-digit-prompts 20 --n-free-speech 2
"""

import argparse
import csv
import os
import random
from datetime import date

import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16_000
METADATA_PATH = "data/splits/genuine_metadata.csv"
METADATA_FIELDS = [
    "speaker_id",
    "session",
    "date",
    "condition",
    "distance_cm",
    "device",
    "text",
    "language",
    "file_path",
]

DIGITS_EN = "0123456789"
DIGITS_FA = "۰۱۲۳۴۵۶۷۸۹"


def random_digit_string(n: int, language: str) -> str:
    pool = DIGITS_FA if language == "fa" else DIGITS_EN
    return "".join(random.choice(pool) for _ in range(n))


def record_clip(seconds: float, sample_rate: int = SAMPLE_RATE):
    audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    return audio.reshape(-1)


def append_metadata_row(row: dict, path: str = METADATA_PATH) -> None:
    is_new = not os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def run_session(args) -> None:
    out_dir = os.path.join("data", "raw", args.speaker_id, f"session{args.session}")
    os.makedirs(out_dir, exist_ok=True)

    prompts = []
    for _ in range(args.n_digit_prompts):
        prompts.append(("digits", random_digit_string(4, args.language)))
    for _ in range(args.n_free_speech):
        prompts.append(("free_speech", "[30s free speech — describe your day, in your own words]"))

    print(f"Speaker {args.speaker_id}, session {args.session}, {len(prompts)} prompts.")
    for i, (kind, text) in enumerate(prompts):
        seconds = 30.0 if kind == "free_speech" else 3.0
        input(f"[{i + 1}/{len(prompts)}] Say: '{text}'  (press Enter to record {seconds:.0f}s)")
        audio = record_clip(seconds)

        filename = f"{args.speaker_id}_s{args.session}_{kind}_{i:03d}.wav"
        file_path = os.path.join(out_dir, filename)
        sf.write(file_path, audio, SAMPLE_RATE, subtype="PCM_16")

        append_metadata_row(
            {
                "speaker_id": args.speaker_id,
                "session": args.session,
                "date": date.today().isoformat(),
                "condition": args.condition,
                "distance_cm": args.distance_cm,
                "device": args.device,
                "text": text,
                "language": args.language,
                "file_path": file_path,
            }
        )
        print(f"  saved {file_path}")

    print("Session complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speaker-id", required=True)
    parser.add_argument("--session", type=int, required=True)
    parser.add_argument("--condition", default="quiet", choices=["quiet", "moderate_noise", "far_mic"])
    parser.add_argument("--distance-cm", type=int, default=50)
    parser.add_argument("--device", default="laptop_mic")
    parser.add_argument("--language", default="en", choices=["en", "fa"])
    parser.add_argument("--n-digit-prompts", type=int, default=20)
    parser.add_argument("--n-free-speech", type=int, default=1)
    args = parser.parse_args()
    run_session(args)


if __name__ == "__main__":
    main()
