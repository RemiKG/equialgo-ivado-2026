# EquiAlgo experiments

Target: **97% measured accuracy**. Best verified so far: **94.78% accuracy / 94.56% macro F1**.

Open **TRIALS.txt** for trial number, name, description and actual result. Scores are evaluation-only: leaderboard publication is disabled. Desktop control is also disabled.

## Layout

- `experiments/`: independent approaches, each free to choose its own model, features, training code and dependencies. No shared model superclass or prescribed ML architecture.
- `src/equialgo/`: input/output contracts, trial records and release utilities only.
- `runs/`: immutable predictions, append-only measured results, and trial outputs.
- `docs/`: research, reasoning, governance and event handoff.
- `config/`: active release pointer, evaluation policy and environment requirements.
- `data/`: unchanged organizer inputs; excluded from Git.
- `archive/`: frozen former approaches and the pre-reorganization code.
- `delivery/`: local bundles and candidate exports; excluded from Git.
- `.local/`: private environment, browser-session helper, scratch work and logs; excluded from Git.

The four root challenge deliverables remain `predictions.csv`, `model_corrige.py`, `audit_rapport.ipynb`, and `presentation.pdf`. They are the measured fallback until a better trial is verified. `model_corrige.py` dispatches to the experiment identified in `config/active_release.json`; it does not prescribe future models.

## Experiment contract

Optional [applicant clustering](experiments/clustering/README.md) provides eight
methods for cohort exploration, with labels and diagnostics exported separately.

Any method is allowed to emit a two-column CSV containing the 4,000 evaluation IDs and binary `decision_octroi`, within the 36-44% allocation budget. Method-specific probabilities, features and training processes remain inside that experiment. Do not add a shared abstraction until different methods actually need it.

Every serious trial records a hypothesis, evidence, expected failure mode and result. Failed trials remain visible. Measured portal accuracy, historical-label agreement and simulated metrics are separate fields. New candidates cannot replace the active release based on simulations.

## Commands

```bash
python -m pip install -r config/requirements.txt
python -m pip install -e .
python model_corrige.py
python -m equialgo.trials refresh
python -m equialgo.trials register --name NAME --description DESCRIPTION --prediction FILE
python -m equialgo.trials score NUMBER --accuracy FRACTION --macro-f1 FRACTION --source SOURCE
```

The active champion uses evaluation-cohort feedback and is not independently validated on new applicants. Its exact file hash and frozen entry point are in `config/active_release.json`. Previous models and all 14 measured scores have been preserved. OpenAI Codex assistance is disclosed; only the supplied synthetic dataset is used for training.
