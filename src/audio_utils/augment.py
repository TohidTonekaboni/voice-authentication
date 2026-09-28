"""Augmentation pipeline (Phase 1.5): noise, reverb, codec, speed/gain.

Noise and reverb augmentation expect real corpora (MUSAN, RIRS_NOISES) under
data/raw/musan/ and data/raw/rirs_noises/ — see configs/dataset_sources.yaml
for download instructions. When those directories are empty, add_noise falls
back to synthetic white noise so the pipeline is still exercisable without
the full corpora.
"""

import glob
import os
import random
import tempfile

import numpy as np
import torch
import torchaudio

MUSAN_DIR = "data/raw/musan"
RIRS_DIR = "data/raw/rirs_noises"


def _rms(x: torch.Tensor) -> torch.Tensor:
    return x.pow(2).mean().sqrt().clamp_min(1e-12)


def add_noise(wav: torch.Tensor, snr_db: float, sr: int, noise_dir: str = MUSAN_DIR) -> torch.Tensor:
    """Mix in a MUSAN noise clip (or synthetic white noise) at the given SNR."""
    noise_files = glob.glob(os.path.join(noise_dir, "**", "*.wav"), recursive=True)
    n_samples = wav.shape[-1]

    if noise_files:
        noise_path = random.choice(noise_files)
        noise, noise_sr = torchaudio.load(noise_path)
        if noise.shape[0] > 1:
            noise = noise.mean(dim=0, keepdim=True)
        if noise_sr != sr:
            noise = torchaudio.functional.resample(noise, noise_sr, sr)
        if noise.shape[-1] < n_samples:
            reps = n_samples // noise.shape[-1] + 1
            noise = noise.repeat(1, reps)
        start = random.randint(0, noise.shape[-1] - n_samples)
        noise = noise[:, start : start + n_samples]
    else:
        noise = torch.randn(1, n_samples)

    signal_rms = _rms(wav)
    noise_rms = _rms(noise)
    target_noise_rms = signal_rms / (10 ** (snr_db / 20))
    noise = noise * (target_noise_rms / noise_rms)
    return (wav + noise).clamp(-1.0, 1.0)


def add_reverb(wav: torch.Tensor, sr: int, rir_dir: str = RIRS_DIR) -> torch.Tensor:
    """Convolve with a real RIR from RIRS_NOISES, or a small synthetic RIR."""
    rir_files = glob.glob(os.path.join(rir_dir, "**", "*.wav"), recursive=True)

    if rir_files:
        rir, rir_sr = torchaudio.load(random.choice(rir_files))
        if rir.shape[0] > 1:
            rir = rir.mean(dim=0, keepdim=True)
        if rir_sr != sr:
            rir = torchaudio.functional.resample(rir, rir_sr, sr)
    else:
        decay = torch.exp(-torch.arange(int(0.2 * sr)) / (0.05 * sr))
        rir = (decay * torch.randn(decay.shape[0])).unsqueeze(0)

    rir = rir / rir.abs().max().clamp_min(1e-12)
    wet = torchaudio.functional.fftconvolve(wav, rir)[:, : wav.shape[-1]]
    return wet.clamp(-1.0, 1.0)


def perturb_speed(wav: torch.Tensor, sr: int, factor: float) -> tuple:
    """Speed perturbation via resampling (also shifts pitch, as in Kaldi-style speed-perturb)."""
    new_sr = int(sr * factor)
    resampled = torchaudio.functional.resample(wav, sr, new_sr)
    return torchaudio.functional.resample(resampled, new_sr, sr), sr


def perturb_gain(wav: torch.Tensor, gain_db: float) -> torch.Tensor:
    return (wav * (10 ** (gain_db / 20))).clamp(-1.0, 1.0)


def simulate_codec(wav: torch.Tensor, sr: int, format: str = "mp3", bitrate: int = 32) -> torch.Tensor:
    """Round-trip through a lossy codec via torchaudio's ffmpeg-backed encoder.

    Uses a real temp file rather than BytesIO — torchcodec's encoder needs a
    file extension to pick the container/codec.
    """
    with tempfile.NamedTemporaryFile(suffix=f".{format}") as f:
        torchaudio.save(f.name, wav, sr, compression=bitrate)
        out, out_sr = torchaudio.load(f.name)
    if out_sr != sr:
        out = torchaudio.functional.resample(out, out_sr, sr)
    if out.shape[-1] < wav.shape[-1]:
        out = torch.nn.functional.pad(out, (0, wav.shape[-1] - out.shape[-1]))
    return out[:, : wav.shape[-1]]


def random_augment(wav: torch.Tensor, sr: int) -> torch.Tensor:
    """Apply a random subset of augmentations, for training-time data loading."""
    if random.random() < 0.6:
        wav = add_noise(wav, snr_db=random.uniform(0, 20), sr=sr)
    if random.random() < 0.3:
        wav = add_reverb(wav, sr=sr)
    if random.random() < 0.3:
        wav, sr = perturb_speed(wav, sr, factor=random.choice([0.9, 1.0, 1.1]))
    if random.random() < 0.3:
        wav = perturb_gain(wav, gain_db=random.uniform(-6, 6))
    if random.random() < 0.2:
        try:
            wav = simulate_codec(wav, sr)
        except RuntimeError:
            pass  # ffmpeg backend unavailable; skip codec augmentation
    return wav
