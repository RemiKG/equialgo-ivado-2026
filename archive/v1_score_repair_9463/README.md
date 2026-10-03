# EquiAlgo — regional access with an exact grant budget

IVADO / CodeML 2026 submission. A reproducible audit, repaired ranking model and transparent regional allocation for the supplied synthetic scholarship data.

**The submission contains 4,000 decisions and exactly 1,600 grants (40%).** The centre/remote selection rates are **40.008% / 39.988%**, a **0.021-percentage-point** gap. These are allocation statistics, not independently measured fairness or accuracy against true merit.

**98% accuracy is not verified.** The jury's independent labels are unavailable. Held-out agreement with biased historical decisions is 88.13% for the supplied baseline and 85.33% for the corrected policy. Neither is accuracy against the jury reference. HxBuddy's indicative accuracy/F1 and the official 35-point technical rubric are different evaluations.

## Required deliverables

| File | Purpose |
|---|---|
| [`predictions.csv`](predictions.csv) | Required two-column submission in the original evaluation order |
| [`audit_rapport.ipynb`](audit_rapport.ipynb) | Executed audit, metrics, proxy analysis, uncertainty and limitations |
| [`model_corrige.py`](model_corrige.py) | Training, policy correction, exact constrained allocation and CSV checks |
| [`presentation.pdf`](presentation.pdf) | Seven slides for a five-minute pitch |
| [`artifacts/pareto_front.png`](artifacts/pareto_front.png) | Constraint sweep; historical agreement explicitly distinguished from hidden utility |
| [`MONITORING.md`](MONITORING.md) | Independent reference collection, ownership, alerts, appeals and rollback |
| [`PITCH_NOTES.md`](PITCH_NOTES.md) | Timed speaker notes |
| [`DEVPOST.md`](DEVPOST.md) | Project description and exact handoff instructions |
| [`predictions_history/README.md`](predictions_history/README.md) | Preserved CSV versions and comparison of measured results |

Every model/audit run archives its predictions before replacing the current output. Each snapshot has a timestamp, SHA-256 checksum, input hashes, model metadata and measured metrics. The original baseline is retained. `predictions_history/comparison.csv` tracks regional access and historical agreement separately from official accuracy/F1, which remain blank until measured. Use `prediction_history.py score` to append a real HxBuddy or jury result for the exact uploaded snapshot.

## Method and evidence

The baseline reproduces a 18.76-percentage-point held-out regional selection gap. Removing region only reduces it to 18.05 points; removing postal prefix too leaves 17.31 points. Distance and other proxies retain geographic information. These differences describe associations, not a causal discrimination estimate.

We fit an additive logistic model of the committee's decisions with regional and other controls. The final policy score removes geography, distance and program contributions, removes any positive wealth advantage and any first-generation penalty, and keeps nonnegative academic/work associations. On the supplied data only academic score and work receive nonzero weights. These are explicit policy assumptions, not recovered reference labels. The work coefficient and the absence of an extra need benefit must be reviewed with affected applicants.

We then maximize total policy score subject to an exact 40% budget and a regional selection gap of at most 0.02. An exact two-group count search makes the budget deterministic. Region is explicitly used by the allocator. The final dataset already has near parity under the repaired ranking, so the 0.02 constraint is inactive there; we do not credit the constraint for improvements produced by score repair.

The primary long-term target is equal opportunity among independently eligible applicants. The implemented provisional constraint is demographic parity because valid eligibility labels are absent. Historical-label TPR worsens under the correction (gap 0.063 to 0.169); this is reported openly and cannot establish whether true-merit equal opportunity improves. A 12-setting Pareto sweep shows historical agreement versus observable parity for both baseline and corrected scores. No hidden-reference Pareto front or official score is claimed.

Validation uses the starter's 70/30 stratified split, seed 42. Audit preprocessing and estimation use training rows only. The final submission refits on all 10,000 historical rows. Fixed-budget allocation uses evaluation features and group counts, never labels. Sensitivity scenarios express alternative normative assumptions; their agreement is not validation accuracy.

## Reproduce

Python 3.10 or newer. Obtain the original participant package through your HxBuddy team and place it under `data/equialgo-participants/`, preserving its manifest and file structure. Participant data are excluded from the public code repository; the local submission archive contains the provided files for the user's handoff to judges.

```text
data/equialgo-participants/
  manifest.json
  baseline_model.ipynb
  README.md
  LISEZMOI.md
  consignes-en.pdf
  consignes-fr.pdf
  requirements.txt
  data/donnees_demandes.csv
  data/candidats_evaluation.csv
```

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python model_corrige.py
python build_audit.py
python build_presentation.py
python -m unittest discover -s tests -v
```

The committed notebook has executed results; open it without rerunning to review. `build_audit.py` regenerates its charts and executes all its code cells. The supplied baseline was executed unchanged before development; its output is in `artifacts/baseline_executed.ipynb`. Run the original from `data/equialgo-participants` if you need to repeat it; its last cell writes a baseline CSV in that directory, not the corrected CSV at repository root.

`artifacts/environment.json` records package versions. `artifacts/split.csv` records the audit split, and `artifacts/model_parameters.json` records full-fit parameters. Tests compare the allocation optimizer with exhaustive enumeration and check exact budget, row-order invariance, monotonic scores, geographic score invariance, input rejection and output reproducibility.

## Submit

Upload the root `predictions.csv` to the IVADO / EquiAlgo challenge on HxBuddy for its indicative score. On Devpost, use the same team name/members and select exactly the IVADO challenge prize. Include the repository and presentation required by the organizer. See `DEVPOST.md`. This repository does not by itself submit a project or produce a leaderboard result.

## Sources and tools

- Organizer's supplied English/French participant briefs, README, starter and SHA-256 manifest.
- [HxBuddy](https://hxbuddy.ca/) and [IVADO track text](https://hxbuddy.ca/demo-locales/en.json), read 2026-10-03.
- [Fairlearn metric definitions](https://fairlearn.org/main/user_guide/assessment/common_fairness_metrics.html) and [postprocessing guide](https://fairlearn.org/main/user_guide/mitigation/postprocessing.html).
- Python, NumPy, pandas, SciPy, scikit-learn, Matplotlib, Fairlearn for the supplied baseline, Jupyter and ReportLab. OpenAI Codex assisted with analysis, implementation and writing. No external training dataset or pretrained prediction model was used.
