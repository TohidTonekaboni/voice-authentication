# Metadata Schemas (Phase 1.2 / 1.3)

## Genuine recordings — `data/splits/genuine_metadata.csv`

One row per recorded file.

| Column | Type | Notes |
|---|---|---|
| `speaker_id` | str | Assigned ID, not linked to real name in analysis files |
| `session` | int | 1, 2, ... — at least 2 sessions on different days |
| `date` | str | ISO 8601 (`YYYY-MM-DD`) |
| `condition` | str | `quiet` \| `moderate_noise` \| `far_mic` etc. |
| `distance_cm` | int | 20, 50, 100, ... |
| `device` | str | e.g. `macbook_pro_mic` |
| `text` | str | The digit sequence or free-speech prompt spoken |
| `language` | str | `fa` \| `en` |
| `file_path` | str | Relative path under `data/raw/` or `data/processed/` |

Written incrementally by `src/data/record_session.py`.

## Spoof recordings — `data/splits/spoof_metadata.csv`

One row per attack file.

| Column | Type | Notes |
|---|---|---|
| `attack_type` | str | `replay` \| `tts` \| `voice_conversion` |
| `tool` | str | e.g. `xtts_v2`, `openvoice`, `phone_speaker_replay` |
| `settings` | str | Free-text: playback device/volume/distance, or TTS model config |
| `target_speaker` | str | `speaker_id` being impersonated |
| `source_speaker` | str | For voice conversion: whose speech was converted (may equal `target_speaker` for replay/TTS) |
| `text` | str | The digit sequence generated/replayed |
| `file_path` | str | Relative path under `data/spoof/` |

Label every spoof file so entire attack systems/tools can be held out later
for the unseen-attack generalization test (Phase 3).

## Trial list — `data/splits/trials_{dev,test}.csv`

Produced by `src/data/splits.py`.

| Column | Type | Notes |
|---|---|---|
| `enroll_files` | str | `;`-separated list of 3-5 enrollment file paths |
| `test_file` | str | Path to the probe file |
| `label` | str | `genuine` \| `impostor` \| `spoof` |
| `trial_type` | str | `genuine` \| `zero_effort_impostor` \| `same_gender_impostor` \| `replay_spoof` \| `tts_spoof` \| `vc_spoof` \| `wrong_challenge` |
