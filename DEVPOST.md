# Submission handoff

Project title: **EquiAlgo — auditable student financing**

Tagline: **Repair historical decision rules, enforce the grant budget, and make regional access inspectable.**

## Inspiration

A model can agree with past decisions while carrying forward unfair access. The supplied scholarship committee funded 48.4% of centre applicants and 27.3% of remote applicants. We wanted to distinguish imitation of that committee from independently justified support.

## What it does

EquiAlgo audits geographic disparities and proxy features, estimates a transparent historical model, repairs its ranking rules and allocates exactly 1,600 grants among the 4,000 evaluation applicants. The final regional selection-rate gap is 0.021 percentage points. An executed notebook, parameter files, a 12-setting fairness sweep and a monitoring plan make the result reproducible.

## How we built it

We first executed the provided baseline unchanged. A fixed held-out split measures historical agreement, selection disparities and regional proxy predictability. A controlled additive logistic model separates observed contributions. The policy score removes geographic/program effects, positive wealth advantage and first-generation penalty, retaining nonnegative academic/work coefficients. An exact two-group optimizer maximizes the stated ranking score under a 40% budget and a 0.02 parity bound. The repaired final ranking already meets this bound.

Python, NumPy, pandas, SciPy, scikit-learn, Matplotlib, Jupyter and ReportLab power the deliverable; the supplied starter uses Fairlearn. OpenAI Codex assisted with analysis, code and documentation. No external training dataset or pretrained prediction model was used.

## Challenges and limitations

Historical decisions are biased labels. Their predictive accuracy is not accuracy against true merit. We cannot validate the requested 98% or the official hidden-reference equal-opportunity score. Our corrected policy has 85.33% historical agreement, versus 88.13% for the baseline; its historical-label TPR gap increases. We disclose these facts and do not equate demographic parity with equal opportunity. Work credit, financial need and first-generation support remain policy assumptions to review with applicants.

## Accomplishments

- All 4,000 IDs retained, binary outputs and exactly 40% funding.
- Reproduced the baseline and showed why removing region is insufficient.
- Quantified proxies, academic-band differences, uncertainty and intersections.
- Published a constraint sweep, sensitivity scenarios and reproducibility checks.
- Designed independent reference collection, monitoring ownership and human appeals.

## Next steps

Create an independent, geographically representative eligibility reference including rejected applicants. Validate true-merit utility and opportunity, review the work/need policy, and run a monitored human-reviewed pilot before any live allocation.

## Required final actions in the event portals

1. Sign in to the HxBuddy team and select **IVADO - EquiAlgo**, challenge 13. Upload `predictions.csv`. Record the returned indicative accuracy and macro F1; neither is the full official 35-point grade.
2. Create the Devpost project with the same team name and members as HxBuddy. Select **exactly one prize corresponding to IVADO - EquiAlgo**, even if the field appears optional.
3. Include the repository URL, executed audit, model, predictions and `presentation.pdf`. Use `PITCH_NOTES.md` for the five-minute pitch.
4. Make sure judges can access the repository and required dataset through their organizer access. The local ZIP contains the original participant package for the authorized handoff; do not assume a private GitHub URL is visible to judges.

No HxBuddy or Devpost entry is submitted by generating these files. Team identity and event-portal authentication must come from the participant.

Sources: [HxBuddy](https://hxbuddy.ca/), [public IVADO track text](https://hxbuddy.ca/demo-locales/en.json), and the supplied participant PDFs, read 2026-10-03.
