# Voice Authentication for Wallet Payments: AI & Data Science Prototype Roadmap

**Scope:** AI and data science only. No payment integration, mobile SDK, HSM, or risk engine.
**Test channel:** laptop microphone.
**Deliverable:** a working `verify(audio, challenge) -> decision` function, a live demo, a frozen evaluation protocol, and a report.
**Duration:** about 8-10 weeks solo, or 5-6 weeks with two people.

---

## 0. Overview

### 0.1 What the prototype does

1. **Enroll:** the user records 3-5 short utterances. The system stores an averaged speaker embedding (template).
2. **Verify:** the system shows a random challenge (for example, 4 random digits). The user speaks it. The system runs three checks in parallel:
   - **ASV** (automatic speaker verification): is this the enrolled speaker?
   - **PAD** (presentation attack detection / anti-spoofing): is this a live human, not a replay or a cloned voice?
   - **ASR challenge check:** did the person say the requested digits?
3. **Fusion:** the three results are combined into `ACCEPT` or `REJECT`, with reason codes.

### 0.2 System diagram

```
Mic → VAD + quality check → ┬→ ASV embedding  → cosine score ─┐
                            ├→ PAD model      → spoof score  ─┼→ Fusion + thresholds → ACCEPT / REJECT
                            └→ ASR (digits)   → text match   ─┘         + reason codes + latency
```

### 0.3 Timeline summary

| Phase | Focus | Duration |
|---|---|---|
| 0 | Setup, scope, targets | 3-4 days |
| 1 | Datasets and data pipeline | 1-2 weeks |
| 2 | Speaker verification (ASV) | 2 weeks |
| 3 | Anti-spoofing (PAD) | 2 weeks |
| 4 | Challenge check (ASR) | 1 week |
| 5 | Fusion and thresholds | 1 week |
| 6 | Latency optimization | 1 week |
| 7 | Live demo and robustness tests | 1 week |
| 8 | Report and handoff | 3-4 days |

Phase 1 data recording and spoof generation can overlap with Phases 2-3 model work.

---

## Phase 0: Setup, scope, targets (3-4 days)

### Tasks

- [ ] **Write the behavior spec** (one page): inputs, outputs, reason codes (`SPEAKER_MISMATCH`, `SPOOF_SUSPECTED`, `WRONG_CHALLENGE`, `LOW_QUALITY`), and the mock API below.
- [ ] **Set up the environment:**
  - Python 3.10+, PyTorch, torchaudio
  - `sounddevice` or `pyaudio` for mic capture
  - `onnxruntime` for optimized inference
  - Model toolkits: SpeechBrain, WeSpeaker, NeMo (speaker models), the AASIST repo, faster-whisper, Silero VAD
  - Tracking: MLflow or Weights & Biases; DVC or versioned folders for data
- [ ] **Compute plan:**
  - Laptop CPU: inference, demo, and small experiments
  - Free or rented GPU (Colab, Kaggle, or a cloud GPU with 16-24 GB) for fine-tuning
- [ ] **Create the repo structure:**

```
voice-auth/
├── data/
│   ├── raw/            # public datasets + own recordings (never commit)
│   ├── processed/      # 16 kHz mono, trimmed, normalized
│   ├── spoof/          # replay, TTS, VC attacks
│   └── splits/         # train/dev/test speaker lists + trial lists
├── src/
│   ├── audio_utils/    # capture, resample, VAD, quality checks, augmentation
│   ├── asv/            # embedding extractor, scoring, fine-tuning
│   ├── pad/            # anti-spoofing models, training
│   ├── asr/            # digit recognizer + challenge matcher
│   ├── fusion/         # gating, calibration, thresholds
│   ├── verify.py       # the public verify() function
│   └── demo/           # CLI / Gradio app
├── configs/
├── notebooks/
├── experiments/
└── reports/
```

- [ ] **Define the mock API contract:**

```python
from dataclasses import dataclass

@dataclass
class VerifyResult:
    decision: str            # "ACCEPT" | "REJECT"
    reason_codes: list[str]  # e.g. ["SPOOF_SUSPECTED"]
    scores: dict             # asv, pad, asr_match, quality (internal only)
    latency_ms: dict         # per stage + total

def enroll(user_id: str, audio_list: list) -> None: ...
def verify(user_id: str, audio, challenge: str) -> VerifyResult: ...
```

- [ ] **Set proof-of-concept targets** (adjust after the first baselines):

| Component | Metric | Target |
|---|---|---|
| ASV | EER on own laptop-mic test set | < 3-5% |
| ASV | FRR at FAR = 0.1-1% | as low as possible, report it |
| PAD | Detection of replay and cloned voice | high on seen attacks; report the unseen-attack drop |
| ASR | Digit-sequence accuracy | > 95% in quiet, report noisy |
| Latency | Model time on laptop CPU (excluding recording) | < 300-500 ms |

### Exit criteria

- Repo runs, the mic records a clean 16 kHz WAV, and targets are written in `reports/targets.md`.

---

## Phase 1: Datasets and data pipeline (1-2 weeks)

### 1.1 Public datasets

| Purpose | Dataset | Notes |
|---|---|---|
| Speaker pretraining and benchmark | VoxCeleb 1/2 | Or just use checkpoints already trained on it |
| Persian speech and speakers | Common Voice (Persian), DeepMine | Fine-tuning and Persian evaluation |
| Anti-spoofing | ASVspoof 2019 LA, ASVspoof 5 | Training baselines and benchmarks |
| Augmentation | MUSAN (noise), RIRS_NOISES (reverb) | Noise, music, babble, room impulse responses |

You do not need the full datasets if you start from pretrained checkpoints. Subsets are enough for fine-tuning and evaluation.

### 1.2 Your own laptop-mic dataset (most important)

This is your target channel, so it matters more than any public data.

- [ ] **Speakers:** yourself plus 10-30 volunteers, with mixed gender and age if possible. Get **written consent** from each (recording, cloning for attack tests, storage, deletion date).
- [ ] **Sessions:** at least 2 sessions on different days, since voice varies over time.
- [ ] **Conditions:** quiet room, moderate noise (fan, keyboard, background talk), and different distances (20 cm, 50 cm, 1 m).
- [ ] **Content per speaker:**
  - 200+ random 4-digit sequences (the challenge format)
  - 20+ free-speech samples (30 s each) for cloning and text-independent tests
  - Both Persian and English digits if you plan to support both
- [ ] **Metadata CSV** for every file: `speaker_id, session, date, condition, distance_cm, device, text, language, file_path`.

### 1.3 Spoof dataset (attacks against consented speakers)

- [ ] **Replay attacks:** play genuine recordings through a phone speaker, a laptop speaker, and a Bluetooth speaker, and re-record with the laptop mic at different volumes and distances.
- [ ] **TTS / voice cloning attacks:** clone each consented speaker's voice from 5-30 s of audio with several open-source systems (for example XTTS-v2, OpenVoice, and other current tools). Generate the *exact* challenge digits, since a real attacker would.
- [ ] **Voice conversion:** convert another person's speech into the target's voice where tools are available.
- [ ] **Label every spoof file** with `attack_type`, `tool`, `settings`, and `target_speaker`, so you can hold out entire attack systems later.
- [ ] Keep at least 1-2 attack systems **completely out of training** for the generalization test.

### 1.4 Splits and trial lists

- [ ] **Split by speaker** into train, dev, and test. A speaker never appears in more than one split.
- [ ] **Freeze the test set and protocol early.** Trial list format: `enroll_files, test_file, label, trial_type`.
- [ ] **Trial types:**
  - Genuine (same speaker, other session)
  - Zero-effort impostor (different speaker)
  - Same-gender impostor
  - Replay spoof
  - TTS/clone spoof
  - Voice-conversion spoof
  - Wrong-challenge (correct speaker, wrong digits)

### 1.5 Preprocessing and augmentation pipeline

- [ ] Resample to 16 kHz mono, normalize loudness, trim with Silero VAD.
- [ ] Quality checks: SNR estimate, clipping ratio, speech duration (reject clips under about 1.5 s).
- [ ] Augmentation: additive noise (MUSAN), reverb (RIRs), codec simulation (MP3, Opus, AMR-like), speed and gain perturbation.

### Deliverables

- Versioned datasets, frozen trial lists, and a **data card** (sources, consent status, speaker demographics, known biases).

### Exit criteria

- Train/dev/test splits and trial lists are frozen and committed. No test-speaker audio has been used for training or tuning.

---

## Phase 2: Speaker verification (2 weeks)

### Week 1: baselines

- [ ] Evaluate these pretrained models **zero-shot** on your laptop-mic test set:
  - SpeechBrain ECAPA-TDNN
  - WeSpeaker CAM++ or ResNet
  - NeMo TitaNet
  - A WavLM-based speaker model
- [ ] Compute **EER** and **minDCF** for each, and record them in an experiment table.
- [ ] Pick 1-2 candidates using accuracy first, then size and speed.

### Week 2: fine-tuning and analysis

- [ ] **Fine-tune** the best candidate:
  - Data: Persian public data plus train-split speakers from your recordings
  - Loss: AAM-Softmax (or keep the pretrained head)
  - Crop length: 2-4 s, matching the challenge duration
  - Strategy: low learning rate, or unfreeze the last layers first, to avoid forgetting
  - Augmentation: heavy (noise, reverb, codec)
- [ ] **Scoring:**
  - Template = mean of the 3-5 enrollment embeddings, length-normalized
  - Score = cosine similarity
  - Compare against AS-norm, and optionally PLDA
- [ ] **Ablations** (report each as a table):

| Factor | Values |
|---|---|
| Enrollment utterances | 1, 3, 5, 10 |
| Test duration | 1.5 s, 2 s, 3 s, 4 s |
| Noise level | quiet, moderate, high |
| Content | text-dependent digits vs free speech |
| Model | zero-shot vs fine-tuned |

### Deliverables

- Best ASV checkpoint, scoring code, and an ablation table.

### Exit criteria

- EER and minDCF reported on the frozen test set, and the best configuration is selected on **dev** results only.

---

## Phase 3: Anti-spoofing / PAD (2 weeks)

This is the highest-risk phase. Give it real time.

### Week 1: baselines and channel adaptation

- [ ] Train **AASIST** and **RawNet2** on ASVspoof and evaluate on your own laptop-mic spoof set. Expect a large accuracy drop, which is normal and is the motivation for the next steps.
- [ ] Try an **SSL front-end** (wav2vec2, XLS-R, or WavLM) with an AASIST or simple classifier head. These usually generalize better.
- [ ] **Fine-tune with your channel data:** train-split replay and cloned-voice samples, with codec, noise, and room augmentation.

### Week 2: generalization and compactness

- [ ] **Unseen-attack test (critical):** train without one attack tool (for example, train on TTS system A + replay, test on TTS system B). Report results **per attack type**.
- [ ] **Metrics:** EER, APCER and BPCER at fixed thresholds, and a per-attack breakdown table.
- [ ] Build a **compact variant** (distilled or smaller model) if the SSL model is too slow for CPU.
- [ ] Analyze failures: which attacks pass, at which distances and volumes, and with which tools.

### Deliverables

- PAD checkpoints (full and compact), per-attack results, and an honest "seen vs unseen" report.

### Exit criteria

- PAD meets targets on seen attacks. The drop on unseen attacks is measured and documented, not hidden.

---

## Phase 4: Challenge check / ASR (1 week)

- [ ] Evaluate **faster-whisper (tiny/base/small)** restricted to digits, and optionally a small CTC model. Test Persian and English digits on your laptop-mic recordings.
- [ ] Measure **digit accuracy and WER** under quiet, noisy, and far-mic conditions.
- [ ] Fine-tune only if the baseline is poor.
- [ ] Implement the matcher:
  - Normalize output (Persian, Arabic-Indic, and Western digits, spoken number words)
  - Require an exact digit-sequence match, with zero tolerance
- [ ] **Replay-of-genuine test:** a real recording of the genuine user saying a *different* challenge must be rejected for `WRONG_CHALLENGE`.

### Exit criteria

- Challenge accuracy table by noise and distance, and the matcher passes the wrong-challenge test.

---

## Phase 5: Fusion and thresholds (1 week)

- [ ] **Baseline: rule-based gating.** `ACCEPT` only if ASV >= t1 AND PAD >= t2 AND challenge matches AND quality passes.
- [ ] **Improved: calibrated fusion.** Logistic regression on ASV score, PAD score, and quality features (SNR, duration). Fit on the **dev set only**.
- [ ] **Threshold selection:** choose the operating point for a very low false-accept rate (for example FAR <= 0.1-1%) and report the resulting FRR. In a payment context, prefer rejecting over accepting; a future step-up factor handles the rejects.
- [ ] **Metrics:**
  - Tandem **t-DCF / t-EER** for ASV + PAD together
  - A confusion breakdown by trial type (genuine, impostor, replay, clone, wrong-challenge)
- [ ] **Bootstrap confidence intervals** on all headline numbers, since the test set is small.

### Deliverables

- Final `verify()` function with documented thresholds and end-to-end metrics.

### Exit criteria

- Frozen thresholds (chosen on dev), and end-to-end results on the test set with confidence bands.

---

## Phase 6: Latency optimization (1 week)

- [ ] **Profile** each stage on the laptop CPU: capture, VAD, feature extraction, ASV, PAD, ASR, fusion. Log mean and p95.
- [ ] **Export to ONNX** and apply FP16 or INT8 quantization. **Re-run accuracy** after each change and log the delta.
- [ ] **Parallelize** ASV, PAD, and ASR branches (threads or async).
- [ ] **Stream with VAD** and stop recording as soon as enough speech (2-3 s) is captured.
- [ ] **Warm up** models at startup and cache the enrollment template.
- [ ] Try smaller models where accuracy allows (ECAPA with fewer channels, Whisper-tiny, compact PAD).

### Latency report format

| Stage | Before (ms) | After (ms) | Accuracy delta |
|---|---|---|---|
| VAD | | | |
| ASV | | | |
| PAD | | | |
| ASR | | | |
| Fusion | | | |
| **Total (parallel)** | | | |

### Exit criteria

- A before/after latency table with p95, and accuracy loss from optimization is within an agreed limit (for example, less than 0.5% absolute EER).

---

## Phase 7: Live demo and robustness tests (1 week)

### Demo app

- [ ] CLI or Gradio/Streamlit app with two flows:
  - **Enroll:** record 3-5 samples, show quality feedback
  - **Verify:** show the random challenge, record, then show the decision, reason codes, scores (debug mode only), and latency

### Scenario tests (run each with enough repetitions)

| Scenario | Expected result |
|---|---|
| Genuine user, quiet room | ACCEPT |
| Genuine user, noisy room | ACCEPT (possibly higher FRR) |
| Genuine user, tired, cold, fast/slow speech | ACCEPT (measure FRR) |
| Another person says the correct challenge | REJECT (`SPEAKER_MISMATCH`) |
| Replay of a genuine recording | REJECT (`SPOOF_SUSPECTED` or `WRONG_CHALLENGE`) |
| Cloned voice reads the correct challenge | REJECT (`SPOOF_SUSPECTED`) |
| Correct voice, wrong digits | REJECT (`WRONG_CHALLENGE`) |
| Different distances and headset vs laptop mic | measure |

### Robustness and fairness

- [ ] Error rates by gender, noise level, and distance, with small-sample caveats.
- [ ] **Repeated-attempt attack:** simulate many attempts to see whether scores leak information. Add attempt limits and lockouts in the demo logic, and **never expose raw scores** to the end user.
- [ ] Log every attempt (scores, decision, latency, condition) for error analysis and iterate on the top failure cases.

### Exit criteria

- The demo runs end to end, and a scenario results table is complete.

---

## Phase 8: Report and handoff (3-4 days)

### Final report contents

1. Problem and scope
2. Data description and consent summary (the data card)
3. Model choices and why
4. Metrics: EER, minDCF, APCER/BPCER, t-DCF, with confidence intervals
5. Latency table
6. Failure analysis and threshold rationale
7. Scenario test results

### Honest limitations to state

- The test set is small and mostly laptop-mic. Real-world FAR and FRR need many more speakers and devices.
- The spoof set covers only attacks you generated. Real attackers will use newer tools.
- Laptop-mic performance does not transfer automatically to phone mics, codecs, and networks.
- Persian and English performance may differ.

### Handoff package

- [ ] Model checkpoints and ONNX files, with model cards
- [ ] The `verify()` and `enroll()` API contract
- [ ] Frozen evaluation protocol and trial lists
- [ ] Reproducible training and evaluation scripts
- [ ] Recommendations for the future integration team: device binding and attestation, transaction-bound challenges, encrypted template storage, risk engine, rate limiting, continuous spoof-data refresh

---

## Weekly plan

| Week | Deliverable |
|---|---|
| 1 | Environment, public data ready, start recording speakers |
| 2 | Spoof data generated, splits and trial lists frozen |
| 3-4 | ASV baselines, fine-tuning, ablations |
| 5-6 | PAD baselines, channel adaptation, unseen-attack test |
| 7 | ASR challenge check, fusion, thresholds |
| 8 | Latency optimization and demo app |
| 9-10 | Robustness testing, report, handoff |

---

## Experiment tracking checklist

For every experiment, log:

- [ ] Data version and split ID
- [ ] Model name, checkpoint, and hyperparameters
- [ ] Augmentation settings
- [ ] Metrics on dev (and test only for final runs)
- [ ] Random seed and commit hash

---

## Key rules for this prototype

1. **Never evaluate on speakers used in training or tuning.** This is the most common mistake and produces falsely good numbers.
2. **Touch the test set once**, for the final numbers. Use dev for all decisions.
3. **Test PAD on unseen attacks.** In-distribution PAD accuracy is misleading.
4. **Use text-dependent random digits.** They improve accuracy on short audio and defeat pre-recorded replay.
5. **Prefer false rejects over false accepts** when choosing thresholds.
6. **Get written consent** for every voice you record or clone, and delete the data when the project ends.
7. **Report uncertainty.** Use bootstrap intervals and state the sample sizes.

---

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| PAD does not generalize to new deepfakes | High | Multiple attack tools, held-out attack test, SSL front-end, plan continuous refresh |
| Too few speakers for reliable metrics | High | Recruit early, bootstrap intervals, state limitations clearly |
| Laptop-mic noise and room variation | Medium | Augmentation, quality gate, retry prompts |
| Persian data is limited | Medium | Fine-tune from pretrained models, record own Persian data |
| Latency above target on CPU | Medium | ONNX, quantization, smaller models, parallel branches |
| Overfitting to your own voice and room | Medium | Multiple speakers, separate sessions, speaker-disjoint splits |
| Consent and privacy issues | High | Written consent, access control, a deletion date |

---

## Suggested next step

Start with Phase 0 and the first recordings in Week 1. Data collection has the longest lead time because it depends on other people, so schedule the volunteer sessions first.
