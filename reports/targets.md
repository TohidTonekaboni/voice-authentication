# Proof-of-Concept Targets

Set at Phase 0. Adjust after the first baselines (Phase 2 week 1).

| Component | Metric | Target |
|---|---|---|
| ASV | EER on own laptop-mic test set | < 3-5% |
| ASV | FRR at FAR = 0.1-1% | as low as possible, report it |
| PAD | Detection of replay and cloned voice | high on seen attacks; report the unseen-attack drop |
| ASR | Digit-sequence accuracy | > 95% in quiet, report noisy |
| Latency | Model time on laptop CPU (excluding recording) | < 300-500 ms |

## Compute plan

- Laptop CPU: inference, demo, and small experiments.
- Free or rented GPU (Colab, Kaggle, or a cloud GPU with 16-24 GB) for fine-tuning.
