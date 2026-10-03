# EquiAlgo V2 — SOTA Overfitters

**A new candidate is ready in [`predictions.csv`](predictions.csv): 4,000 rows, 1,600 grants, 194 decisions changed from V1. Its official score is pending.**

V1 scored **94.63% accuracy / 94.40% macro F1**, as reported by the participant. Its complete code, audit, predictions, PDF and original ZIP are preserved under [`archive/v1_score_repair_9463`](archive/v1_score_repair_9463). The prior score is recorded against the exact V1 snapshot; it is not attributed to V2.

## A different inference problem

V1 retained the committee's work coefficient, removed selected coefficients and enforced a demographic-parity bound. V2 treats the true merit rule as uncertain. It evaluates 2,048 explicit hypotheses about academic achievement, work, financial need, first-generation support and commute burden. A hypothesis receives more weight when it is compatible with the observed V1 accuracy. The brief's baseline equal-opportunity gap is a weak additional clue because its exact reference cohort is unknown.

The primary model tests the possibility that V1's errors reflect missing merit criteria rather than mostly random reference noise. It averages compatible low-noise hypotheses and funds the 1,600 highest resulting eligibility weights. There is no direct regional/postal/program score term and no forced equality of regional selection rates. Income, first-generation status and commute effects have stated normative directions; their correctness is not established by the biased labels.

The historical data did **not** reliably identify the correct financial-need effect. We tested richer committee models, including nonlinear income, work saturation and interactions. A log-income model fit committee behavior better, but that is not evidence that it predicts independent merit better. V2 explicitly separates these claims.

**This inverse problem is underdetermined.** Aggregate feedback cannot identify a unique merit rule. The primary low-noise assumption may fail. Model-implied agreement, including values in the Pareto plot, is not a measured HxBuddy score and does not verify 98% accuracy. The evaluation cohort has influenced model development through aggregate feedback; it is not an untouched validation set.

## Candidates

| Candidate | Changed from V1 | Centre / remote selection | Official result |
|---|---:|---:|---|
| `predictions.csv` — structured merit, primary | 194 | 40.89% / 38.70% | Pending |
| `submissions/v2_candidates/broad_merit.csv` | 152 | 40.35% / 39.50% | Pending |
| `submissions/v2_candidates/academic_anchor.csv` | 170 | 42.28% / 36.67% | Pending |

All select 1,600 applicants. The second candidate allows much more uncertainty in merit outcomes. The third tests the simpler explanation that V1 rewarded work too strongly: it keeps R score and only 25% of the committee's academic-normalized work coefficient.

The primary swaps 97 recipients. Newly selected applicants average R score 28.72 versus 28.16 for displaced applicants, household income about $58,273 versus $79,265, and paid work 7.23 versus 15.32 hours. These are descriptive changes, not proof of correct labels or improved fairness.

## Files

| File | Purpose |
|---|---|
| [`model_corrige.py`](model_corrige.py) / [`merit_inference.py`](merit_inference.py) | Runnable model and inference implementation |
| [`data_contract.py`](data_contract.py) | Schema checks and deterministic budget allocator |
| [`audit_rapport.ipynb`](audit_rapport.ipynb) | Executed V2 audit with assumptions and evidence boundaries |
| [`presentation.pdf`](presentation.pdf) | Updated seven-slide pitch |
| [`artifacts/pareto_front.png`](artifacts/pareto_front.png) | Opportunity/utility sensitivity frontier under explicit assumptions |
| [`predictions_history/README.md`](predictions_history/README.md) | All preserved predictions and official results |
| [`MONITORING.md`](MONITORING.md) | Independent review, release controls, appeals and monitoring |
| [`archive/v1_score_repair_9463/README.md`](archive/v1_score_repair_9463/README.md) | Frozen prior method |

The local ZIP in `submissions/equialgo_submission.zip` includes the full handoff. Raw participant data and archives containing them are excluded from the public repository.

## Reproduce

Put the original participant package at `data/equialgo-participants/`, with its CSVs under `data/`. Python 3.10 or newer:

```bash
python -m pip install -r requirements.txt
python model_corrige.py
python build_audit.py
python build_presentation.py
python -m unittest discover -s tests -v
python package_submission.py
```

The model run writes the primary CSV and alternative candidates, parameter files and immutable prediction snapshots. The executed notebook already contains results. V1's full archive provides the scored CSV and supplied-baseline predictions used as aggregate-feedback anchors. No hidden label files or external training data are needed or accessed.

To append a real portal result for a precise snapshot:

```text
python prediction_history.py score SNAPSHOT_ID --accuracy 0.0000 --macro-f1 0.0000 --source "HxBuddy user-reported"
```

Replace the example zeroes with actual fractions; do not use model-implied values. The primary V2 snapshot is `20261003T170400518781Z_v2-structured-merit_e83a4b59`. `predictions_history/comparison.csv` keeps official scores separate from historical-label agreement.

## Evidence and attribution

Data and problem definition: supplied IVADO briefs, starter and manifest; [HxBuddy](https://hxbuddy.ca/). Independent feedback: participant-reported SOTA Overfitters result. Numerical methods: [SciPy Sobol sampling](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html), [normal CDF](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ndtr.html). Fairness definitions: [Fairlearn](https://fairlearn.org/main/user_guide/assessment/common_fairness_metrics.html).

Tools: Python, NumPy, pandas, SciPy, scikit-learn, Matplotlib, Jupyter, ReportLab; Fairlearn in the supplied starter. OpenAI Codex assisted with analysis, implementation and writing. No pretrained prediction model, external training dataset or candidate ID as a predictive feature was used.
