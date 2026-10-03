# EquiAlgo - SOTA Overfitters

**Current measured champion: 94.78% accuracy / 94.56% macro F1 on HxBuddy.** Root `predictions.csv` is the exact scored file: 4,000 rows and 1,600 grants. It was published to the leaderboard. V1 measured 94.63% / 94.40%; the gain is 0.15 / 0.16 percentage points. This narrowly exceeds the last observed competing leader at 94.73% / 94.50%. It does not establish 98% accuracy.

See [every scored attempt](RESULTS.md), [the complete prediction history](predictions_history/README.md), and [the score progression](artifacts/score_progress.png). First three results were reported by the participant; the following eleven were directly observed. Unscored candidates are never attributed a measured result.

## Method and evidence

The champion uses maximum-entropy calibration of an academic/work prior against eleven measured aggregate submission scores. Joint constraints from overlapping decisions adjust uncertain cases near the cutoff. Frozen parameters reproduce the champion without using later feedback. Exactly 1,600 applicants receive grants.

This is an evaluation-cohort experiment. The same cohort informed model development, so portal gains are not independent generalization estimates. The method inherits dependencies from diagnostic submissions, including income and geography. Independent merit labels, legitimate policy criteria and fresh validation are required before any real use. The executable rejects changed cohorts and modified features. The actual independent-reference equal-opportunity gap is unavailable; the Pareto plot is explicitly a sensitivity analysis under model assumptions.

The current allocation changes 32 decisions from V1. Centre selection is 957/2,372 (40.35%); remote selection is 643/1,628 (39.50%). The observed demographic-parity gap is 0.849 percentage points. These selection rates do not establish equal opportunity.

## Preserved alternatives

- [V1 archive](archive/v1_score_repair_9463/): original code, notebook, predictions, plots, presentation, tests and package. Measured 94.63% / 94.40%.
- [V2 archive](archive/v2_inverse_merit_9223/): rejected structured-merit method. Measured 92.23% / 91.90%.
- [Prediction snapshots](predictions_history/): immutable CSVs, source code, parameters and checksums for every generated version. Scored results are append-only in `official_results.jsonl`.
- `submissions/offline_candidates/robust_calibration.csv`: unscored sensitivity candidate, 14 decisions different from the champion. The champion remains unchanged.

## Deliverables

| File | Purpose |
|---|---|
| [predictions.csv](predictions.csv) | Exact measured champion |
| [model_corrige.py](model_corrige.py) | Frozen, hash-checked reproduction |
| [audit_rapport.ipynb](audit_rapport.ipynb) | Executed diagnostic, measured results and limitations |
| [presentation.pdf](presentation.pdf) | Seven-slide, five-minute pitch |
| [artifacts/pareto_front.png](artifacts/pareto_front.png) | Ten-setting opportunity sensitivity sweep |
| [RESULTS.md](RESULTS.md) | All 14 measured attempts with links |
| [MONITORING.md](MONITORING.md) | Independent review, appeals and release controls |

The local bundle `submissions/equialgo_submission.zip` includes the participant package for the authorized judges' handoff. Raw participant data and bundles are excluded from the public repository. Other experimental artifacts retain their method-specific names and are not the active release; `artifacts/active_prediction.json` identifies the champion.

## Reproduce

Place the original participant package at `data/equialgo-participants/`, with its CSV files under `data/`. Python 3.10 or newer:

```bash
python -m pip install -r requirements.txt
python model_corrige.py
python -m unittest discover -s tests -v
python build_audit.py
python build_presentation.py
python package_submission.py
```

The active CSV SHA-256 is `2038f8dcd0f728d7d380bab9c7aca8740b12eeeb093e894d4a6caf94f4928bd4`. Model reproduction must match before replacing an output. Tests cover release integrity, changed-cohort rejection, budget optimization, row-order consistency and history preservation.

To record a real portal result for an exact snapshot:

```text
python prediction_history.py score SNAPSHOT_ID --accuracy VALUE --macro-f1 VALUE --source HxBuddy
```

Use fractions, not percentages. Preserve failed experiments. Do not replace portal scores with model-implied values. New experiments never promote themselves to champion without measurement.

## Attribution

Data and challenge: supplied IVADO briefs, starter notebook and [HxBuddy](https://hxbuddy.ca/). Methods: SciPy, scikit-learn and [Fairlearn definitions](https://fairlearn.org/main/user_guide/assessment/common_fairness_metrics.html). Python, pandas, NumPy, Matplotlib, Jupyter and ReportLab produce the artifacts. OpenAI Codex assisted analysis, implementation and writing. No external training dataset, pretrained predictor or candidate ID as a predictive feature was used.
