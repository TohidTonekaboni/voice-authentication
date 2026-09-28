"""Preprocessing pipeline (Phase 1.5): resample, normalize, VAD-trim, quality gate.

Usage:
    from src.audio_utils.preprocess import preprocess_file
    result = preprocess_file("data/raw/foo.wav", "data/processed/foo.wav")
"""

from dataclasses import dataclass

import numpy as np
import soundfile as sf
import torch
import torchaudio

TARGET_SR = 16_000
MIN_DURATION_S = 1.5
CLIPPING_THRESHOLD = 0.99  # abs sample value considered clipped
MAX_CLIPPING_RATIO = 0.001  # fraction of samples allowed to clip
MIN_SNR_DB = 5.0

_vad_model = None
_vad_utils = None


def _load_vad():
    global _vad_model, _vad_utils
    if _vad_model is None:
        _vad_model, _vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
        )
    return _vad_model, _vad_utils


@dataclass
class QualityReport:
    passed: bool
    duration_s: float
    snr_db: float
    clipping_ratio: float
    reasons: list


def load_and_resample(path: str, target_sr: int = TARGET_SR) -> tuple:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav, target_sr


def normalize_loudness(wav: torch.Tensor, target_rms: float = 0.1) -> torch.Tensor:
    rms = wav.pow(2).mean().sqrt()
    if rms > 0:
        wav = wav * (target_rms / rms)
    return wav.clamp(-1.0, 1.0)


def trim_with_vad(wav: torch.Tensor, sr: int = TARGET_SR) -> torch.Tensor:
    model, utils = _load_vad()
    get_speech_timestamps = utils[0]
    timestamps = get_speech_timestamps(wav.squeeze(0), model, sampling_rate=sr)
    if not timestamps:
        return wav[:, :0]
    start = timestamps[0]["start"]
    end = timestamps[-1]["end"]
    return wav[:, start:end]


def estimate_snr_db(wav: np.ndarray, frame_ms: float = 20.0, sr: int = TARGET_SR) -> float:
    """Rough SNR estimate: top-10% energy frames vs bottom-10% energy frames."""
    frame_len = max(1, int(sr * frame_ms / 1000))
    n_frames = len(wav) // frame_len
    if n_frames < 10:
        return 0.0
    energies = np.array(
        [np.mean(wav[i * frame_len : (i + 1) * frame_len] ** 2) for i in range(n_frames)]
    )
    energies = np.sort(energies)
    k = max(1, n_frames // 10)
    noise_floor = np.mean(energies[:k]) + 1e-12
    signal_peak = np.mean(energies[-k:]) + 1e-12
    return float(10 * np.log10(signal_peak / noise_floor))


def check_quality(wav: torch.Tensor, sr: int = TARGET_SR) -> QualityReport:
    arr = wav.squeeze(0).numpy()
    duration_s = len(arr) / sr
    clipping_ratio = float(np.mean(np.abs(arr) >= CLIPPING_THRESHOLD)) if len(arr) else 1.0
    snr_db = estimate_snr_db(arr, sr=sr) if len(arr) else 0.0

    reasons = []
    if duration_s < MIN_DURATION_S:
        reasons.append(f"duration {duration_s:.2f}s < {MIN_DURATION_S}s")
    if clipping_ratio > MAX_CLIPPING_RATIO:
        reasons.append(f"clipping ratio {clipping_ratio:.4f} > {MAX_CLIPPING_RATIO}")
    if snr_db < MIN_SNR_DB:
        reasons.append(f"SNR {snr_db:.1f} dB < {MIN_SNR_DB} dB")

    return QualityReport(
        passed=not reasons,
        duration_s=duration_s,
        snr_db=snr_db,
        clipping_ratio=clipping_ratio,
        reasons=reasons,
    )


def preprocess_file(in_path: str, out_path: str | None = None) -> tuple:
    """Resample -> normalize -> VAD-trim -> quality check.

    Returns (waveform, sample_rate, QualityReport). Writes to out_path if the
    clip passes the quality gate and out_path is given.
    """
    wav, sr = load_and_resample(in_path)
    wav = normalize_loudness(wav)
    wav = trim_with_vad(wav, sr)
    report = check_quality(wav, sr)

    if report.passed and out_path:
        sf.write(out_path, wav.squeeze(0).numpy(), sr, subtype="PCM_16")

    return wav, sr, report
