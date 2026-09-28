# Data Card

Fill in once the own-voice dataset (Phase 1.2) and spoof dataset (Phase 1.3)
are collected and frozen.

## Sources

- Public datasets used (from `configs/dataset_sources.yaml`): ...
- Own laptop-mic recordings: N speakers, N sessions each, dates ...

## Consent status

- Consent form: `data/consent_form.md`
- N participants signed, deletion dates on file
- Any withdrawals: ...

## Speaker demographics

| Attribute | Breakdown |
|---|---|
| Gender | ... |
| Age range | ... |
| Language(s) | ... |
| Accents/regions | ... |

## Known biases / limitations

- Sample size and what it does/doesn't support statistically
- Recording conditions covered vs. not covered (devices, noise, distance)
- Attack tools covered vs. held out for generalization testing

## Splits

- `data/splits/speakers_{train,dev,test}.txt`
- `data/splits/trials_{dev,test}.csv`
- Confirm: no speaker appears in more than one split; test set touched once.
