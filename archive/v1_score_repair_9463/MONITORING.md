# EquiAlgo monitoring and decision governance

This proposal is for the fictional challenge. The delivered model is a prototype with no verified independent-merit labels. The following thresholds are proposed operating rules, not legal standards or observed production results.

## Responsibility and approval

| Owner | Responsibility | Cadence |
|---|---|---|
| Scholarship policy owner | Define academic merit, financial need, work credit and eligibility; approve the budget | Before release and each award cycle |
| Independent regional review panel | Include remote-region students, first-generation applicants and financial-aid staff; adjudicate eligibility without copying past awards | Before release, then quarterly |
| ML owner | Reproduce artifacts, schema and budget checks, model/version records and drift dashboard | Every batch |
| Fairness and privacy lead | Review regional and intersectional outcomes, purpose/access controls, appeals and proposed changes | Monthly and on alert |
| Human appeals officer | Correct input errors, assess context and record reasons for overrides | Every appeal |

No production approval is implied by the hackathon CSV. A signed policy decision is needed before treating any historical association as a merit criterion. In particular, paid work must not penalize students unable to work because of disability, care duties or unavailable jobs. Removing a wealth advantage does not establish an appropriate financial-need weight.

## Before release: establish an independent reference

1. Independently reassess at least 600 historical applications, stratifying by all five regions, historical decision, academic band and first-generation status. Oversample sparse groups, retaining sampling weights for population estimates. Include rejected applicants; examining only recipients creates selective labels.
2. Use two independent reviewers and adjudicate disagreements. Hide the previous decision and unnecessary location cues where possible; disclose that indirect geographic signals may remain. Publish the eligibility rubric and reviewer agreement, including disagreements by region.
3. Evaluate independent-reference accuracy, macro F1, TPR, FPR and uncertainty by region and intersections. The primary outcome is the centre/remote TPR difference among independently eligible applicants. Use at least 100 independently eligible applicants per block before treating a TPR estimate as decision-grade; collect more when intervals are too wide.
4. Pre-register candidate policies before validation. Keep a fresh final reference holdout. Do not tune to repeated leaderboard feedback. Compare academic-only, work-credit, need-credit and first-generation policies under the same budget.
5. Calibrate eligibility probabilities only after valid labels exist. The delivered score is a ranking utility, not an eligibility probability or a causal estimate.

The hidden jury reference was never accessed. Its stated baseline equal-opportunity gap, 0.270, must not be confused with our measured historical-label TPR gap, about 0.063. Any jury-score improvement remains unknown.

## Per-batch controls

| Signal | Measure and denominator | Trigger | Response |
|---|---|---|---|
| Schema and provenance | Missing/duplicate IDs, unknown regions/programs, nonfinite values, file hashes | Any invalid official record | Stop that release; correct or review the input |
| Budget | Grants / all applications; exact 1,600 / 4,000 for this submission | Anything except 40% for this policy | Block output; reconcile before award issuance |
| Access parity | Absolute centre/remote grant-rate difference; record both group counts | More than 0.02 after integer allocation | Block release; inspect allocator and cohort composition |
| Independent equal opportunity | Difference in TPR among independently eligible cases | Point gap >0.05 prompts review; signed 95% interval entirely outside ±0.05 escalates | Independent case review; freeze new automatic awards if persistent or severe |
| Independent utility | Accuracy and false-negative rates on representative adjudicated cases | >3-point decline from approved reference performance | Investigate drift and policy changes; revalidate |
| Regional and intersectional outcomes | All five regions; region × program × first-generation, with sample sizes and intervals | Group rate changes >5 points over two comparable cycles | Review context and data; do not force every small subgroup to a quota |
| Feature drift | Standardized mean shifts, missingness and categorical frequencies versus approved training data | Numeric shift >0.25 SD, missingness >1%, or new category | Review data collection and population change |
| Appeals and overrides | Appeals / all decisions; upheld appeals / all appeals, by region | Regional upheld-appeal gap >10 points with ≥30 appeals per group | Investigate systematic omissions and collect adjudicated outcomes |

Thresholds for five-region and intersection monitoring are investigation thresholds, not proof of unfair treatment. Flag cells below 30; pool time windows or gather more evidence. Report denominators, signed gaps and confidence intervals. Adjust multiple-comparison analysis for exploratory subgroup scans. Batch selection counts are exact; bootstrap intervals address sampling uncertainty, not the certainty of a fixed CSV.

## Explanation, appeal and budget reconciliation

Provide each applicant their own input values, the policy version, the criteria affecting priority, and a clear human appeal route. Explain that a fixed cohort budget and regional access constraint affect selection; do not present a ranking score as a probability. A changed region can change the allocation even though it does not change the individual score.

Allow data corrections and contextual evidence. A human panel reviews borderline cases and hardship without promising funding from a score alone. Track why decisions change. Hold an explicitly approved reserve in future operational budgets or use a published waitlist process; do not silently exceed the budget, withdraw an issued award, or secretly substitute a different CSV. The challenge submission itself funds exactly 1,600 applicants and has no additional reserve.

## Privacy, logging and retention

Collect only attributes needed for the approved policy and audit. Keep region and first-generation indicators under restricted access. Separate identity records from pseudonymous decision logs, encrypt storage and transport, and log access and model versions. Student identifiers are tie breakers only, using a fixed hash seed, never predictive features. Do not infer new sensitive attributes for individual decisions.

Publish aggregate results with small-cell suppression. Limit detailed applicant explanations to the applicant and authorized reviewers. Set a documented retention period covering review/appeal needs, then delete or de-identify records. Obtain institution-specific privacy and policy review before processing real applicants; no legal compliance claim is made here.

## Rollback and continuous review

On a severe schema, budget, fairness or data-quality incident, freeze new automatic awards and preserve the audit record. Route decisions to the approved human allocation process. Do not automatically return to the known-biased production baseline. The policy owner and independent panel must approve remediation and revalidation. Version data, model parameters, tie seed, constraint, code commit and output checksum so each decision can be reproduced.

Revisit the normative work, need and first-generation weights quarterly with applicant representatives. Review true outcomes without confusing access to funding with academic ability; award receipt can itself affect future outcomes. Never retrain solely on automated awards without an independently reviewed sample.
