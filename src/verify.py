"""Public API contract for the voice-auth prototype.

Stage implementations (ASV/PAD/ASR/fusion) land in later phases; this module
currently defines the contract and a stub so the rest of the repo has a fixed
interface to build against.
"""

from dataclasses import dataclass, field


@dataclass
class VerifyResult:
    decision: str  # "ACCEPT" | "REJECT"
    reason_codes: list = field(default_factory=list)
    scores: dict = field(default_factory=dict)  # asv, pad, asr_match, quality (internal only)
    latency_ms: dict = field(default_factory=dict)  # per stage + total


def enroll(user_id: str, audio_list: list) -> None:
    raise NotImplementedError("Phase 2: ASV enrollment not yet implemented")


def verify(user_id: str, audio, challenge: str) -> VerifyResult:
    raise NotImplementedError("Phase 5: fusion not yet implemented")
