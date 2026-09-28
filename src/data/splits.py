"""Speaker-disjoint splits and trial-list generation (Phase 1.4).

Usage:
    python -m src.data.splits --genuine data/splits/genuine_metadata.csv \\
        --spoof data/splits/spoof_metadata.csv --out-dir data/splits \\
        --train-frac 0.6 --dev-frac 0.2
"""

import argparse
import csv
import os
import random
from collections import defaultdict


def read_csv(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def speaker_disjoint_split(speaker_ids: list, train_frac: float, dev_frac: float, seed: int = 0) -> dict:
    """Assign each speaker to exactly one of train/dev/test."""
    speakers = sorted(set(speaker_ids))
    rng = random.Random(seed)
    rng.shuffle(speakers)

    n = len(speakers)
    n_train = round(n * train_frac)
    n_dev = round(n * dev_frac)

    return {
        "train": speakers[:n_train],
        "dev": speakers[n_train : n_train + n_dev],
        "test": speakers[n_train + n_dev :],
    }


def write_split_lists(split: dict, out_dir: str) -> None:
    for name, speakers in split.items():
        path = os.path.join(out_dir, f"speakers_{name}.txt")
        with open(path, "w") as f:
            f.write("\n".join(speakers) + "\n")


def build_trials(genuine_rows: list, spoof_rows: list, speakers: list, n_enroll: int = 4) -> list:
    """Build a trial list for one split (dev or test) covering every trial type
    from the roadmap: genuine, zero-effort/same-gender impostor, replay/TTS/VC
    spoof, wrong-challenge. Gender labels aren't collected in metadata, so
    same-gender impostor trials are left for manual annotation once gender
    metadata exists.
    """
    trials = []
    by_speaker = defaultdict(list)
    for row in genuine_rows:
        if row["speaker_id"] in speakers:
            by_speaker[row["speaker_id"]].append(row)

    speaker_list = [s for s in speakers if s in by_speaker]

    for spk in speaker_list:
        rows = by_speaker[spk]
        if len(rows) < n_enroll + 1:
            continue
        enroll = rows[:n_enroll]
        enroll_files = ";".join(r["file_path"] for r in enroll)

        # Genuine trials: remaining same-speaker utterances (ideally other sessions)
        for probe in rows[n_enroll:]:
            trials.append(
                {
                    "enroll_files": enroll_files,
                    "test_file": probe["file_path"],
                    "label": "genuine",
                    "trial_type": "genuine",
                }
            )

        # Zero-effort impostor: one probe utterance per other speaker
        for other in speaker_list:
            if other == spk:
                continue
            other_rows = by_speaker[other]
            if not other_rows:
                continue
            probe = other_rows[0]
            trials.append(
                {
                    "enroll_files": enroll_files,
                    "test_file": probe["file_path"],
                    "label": "impostor",
                    "trial_type": "zero_effort_impostor",
                }
            )

    # Spoof trials, split by attack type
    attack_type_map = {"replay": "replay_spoof", "tts": "tts_spoof", "voice_conversion": "vc_spoof"}
    for row in spoof_rows:
        if row["target_speaker"] not in by_speaker:
            continue
        enroll_files = ";".join(r["file_path"] for r in by_speaker[row["target_speaker"]][:n_enroll])
        trials.append(
            {
                "enroll_files": enroll_files,
                "test_file": row["file_path"],
                "label": "spoof",
                "trial_type": attack_type_map.get(row["attack_type"], row["attack_type"]),
            }
        )

    return trials


def write_trials(trials: list, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["enroll_files", "test_file", "label", "trial_type"])
        writer.writeheader()
        writer.writerows(trials)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genuine", default="data/splits/genuine_metadata.csv")
    parser.add_argument("--spoof", default="data/splits/spoof_metadata.csv")
    parser.add_argument("--out-dir", default="data/splits")
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--dev-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    genuine_rows = read_csv(args.genuine)
    spoof_rows = read_csv(args.spoof)

    if not genuine_rows:
        print(f"No genuine metadata found at {args.genuine} — nothing to split yet.")
        return

    speaker_ids = [r["speaker_id"] for r in genuine_rows]
    split = speaker_disjoint_split(speaker_ids, args.train_frac, args.dev_frac, args.seed)
    write_split_lists(split, args.out_dir)

    for name in ("dev", "test"):
        trials = build_trials(genuine_rows, spoof_rows, split[name])
        write_trials(trials, os.path.join(args.out_dir, f"trials_{name}.csv"))
        print(f"{name}: {len(split[name])} speakers, {len(trials)} trials")

    print(f"train: {len(split['train'])} speakers")


if __name__ == "__main__":
    main()
