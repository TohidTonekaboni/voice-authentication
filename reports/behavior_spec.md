# Behavior Spec: `verify(audio, challenge) -> decision`

## Scope

AI/data-science prototype only. No payment integration, mobile SDK, HSM, or risk
engine. Target channel: laptop microphone.

## Inputs

- **Enroll:** `user_id: str`, `audio_list: list[Audio]` — 3-5 short utterances
  (clean, 16 kHz mono) recorded during onboarding.
- **Verify:** `user_id: str`, `audio: Audio` — a single utterance responding to
  a challenge, plus `challenge: str` — the digit sequence the system asked the
  user to speak (e.g. a random 4-digit code).

`Audio` is raw waveform data (numpy array or WAV bytes) at 16 kHz mono.

## Outputs

`VerifyResult`:

- `decision: "ACCEPT" | "REJECT"`
- `reason_codes: list[str]` — zero or more of the codes below, present only on
  `REJECT` (or when a soft warning applies)
- `scores: dict` — `asv`, `pad`, `asr_match`, `quality` (internal/debug only,
  never shown to the end user)
- `latency_ms: dict` — per-stage timing (`vad`, `asv`, `pad`, `asr`, `fusion`,
  `total`)

## Reason codes

| Code | Meaning |
|---|---|
| `SPEAKER_MISMATCH` | ASV score below threshold — voice doesn't match the enrolled template |
| `SPOOF_SUSPECTED` | PAD flags the audio as replay, synthetic, or converted speech |
| `WRONG_CHALLENGE` | ASR transcript doesn't exactly match the requested digit sequence |
| `LOW_QUALITY` | Audio fails the quality gate (too short, clipped, too noisy) before scoring |

Multiple codes may co-occur (e.g. a replay of the genuine user's voice can
trigger both `SPOOF_SUSPECTED` and `WRONG_CHALLENGE` if a stale challenge was
replayed).

## Decision logic (baseline, refined in Phase 5)

`ACCEPT` iff: quality passes AND `asv >= t_asv` AND `pad >= t_pad` AND ASR
transcript matches `challenge` exactly. Otherwise `REJECT` with the relevant
reason code(s). Prefer false rejects over false accepts.

## Mock API contract

```python
from dataclasses import dataclass, field

@dataclass
class VerifyResult:
    decision: str                    # "ACCEPT" | "REJECT"
    reason_codes: list[str] = field(default_factory=list)
    scores: dict = field(default_factory=dict)      # internal only
    latency_ms: dict = field(default_factory=dict)  # per stage + total

def enroll(user_id: str, audio_list: list) -> None: ...
def verify(user_id: str, audio, challenge: str) -> VerifyResult: ...
```

## Non-goals for this prototype

- No production auth/session handling, no HSM-backed template storage.
- No mobile capture path — laptop mic only.
- No production-grade rate limiting (a simple demo-level lockout is enough,
  per Phase 7).
