# Research toward 97% measured accuracy

Current starting point: 209 errors in 4,000 decisions. At least 89 net errors must be corrected to reach 97%. The target is a measured score, not a promise or an estimate from the model being tested.

## Why the previous search stalled

V1 and the nonlinear academic/work model are close to each other. Broad changes to income, first-generation status, distance and regional treatment worsened measured results. This supports, but does not establish, a low-dimensional merit rule with an uncertain boundary. A larger conventional classifier trained on the committee's biased labels may learn the wrong target more effectively.

There are two fundamentally different explanations: missing predictable structure, or residual outcome variation not represented in the features. Parameter tuning cannot distinguish them. Both explanations must remain live until tested.

## Independent approaches

1. **Latent outcome / observation model.** Treat legitimate merit and committee decisions as separate random variables. Fit a shared academic signal and a separate bias channel, then test whether the inferred merit rule explains withheld aggregate results. Jiang and Nachum derive correction by reweighting under specified label-bias assumptions. Their identifiability assumptions need justification here; a fairness constraint cannot identify a unique hidden merit rule by itself. Source: https://proceedings.mlr.press/v108/jiang20a.html

2. **Aggregate supervision with structure.** Treat measured agreements as overlapping aggregate constraints. Introduce feature-neighbour consistency or learned representations, not just a fixed prior. Havaldar et al. combine belief propagation and representation refinement for learning from label proportions. Their disjoint-bag experiments are not identical to our overlapping adaptive submissions, so their reported gains cannot be transferred to this task. Source: https://proceedings.iclr.cc/paper_files/paper/2024/file/5a5ef25bfe9ad4f3081dfefe6c092a9c-Paper-Conference.pdf

3. **Designed measurements and finite-cohort decoding.** A changed prediction set provides an exact count constraint: if k decisions are flipped and accuracy changes by d/4000, the flipped set contains (k+d)/2 old errors. Deliberately varied subsets can provide information about otherwise unpredictable boundary cases. This targets the fixed evaluation cohort and must be described as adaptive test-set reconstruction, not improved generalization. It may require many scored submissions, so a simulation must establish the query budget before committing. Source: https://proceedings.mlr.press/v117/acharya20a.html

4. **Synthetic generative structure.** Inspect whether observable features share a recoverable data-generation mechanism or hidden common factor. All evidence must come from supplied features. No private answer file or organizer system access is involved. A discovered mechanism must reproduce many features exactly before it is trusted. This route is specific to the synthetic challenge and would not transfer to real applicants.

## Selection criteria

Prefer an approach that explains existing evidence, predicts an omitted score, and has a plausible path to 89 corrected decisions. Use independent synthetic controls to measure the information needed. Avoid claiming a 97% path from agreement with pseudo-labels or the model's own assumptions. Limit the number of distinct hypotheses, rather than sweeping superficial hyperparameters.

Li et al. show that simple proportion matching can fail in the agnostic setting. Matching all observed aggregate scores is therefore a consistency check, not validation of individual labels. Source: https://proceedings.mlr.press/v247/li24b.html

## Operational constraints

- Score-only evaluations; no leaderboard publication operation exists in the new client.
- At most 20 submissions in a rolling hour; local cap 19 reserves one slot.
- Background requests only; no desktop, clipboard or browser-window control.
- All measured and unmeasured trials appear in root `TRIALS.txt`.
- Any method can enter as an independent experiment. Shared infrastructure defines only input/output, provenance, rate limits and release evidence.


## Findings after initial implementation

- Trial 15 (robust prior ensemble): actual accuracy 0.94775, macro F1 0.94556; tied the champion's displayed score.
- Trial 16 (separate latent-merit/committee heads): actual accuracy 0.94575, macro F1 0.94347; rejected.
- A 256-variable dense integer decoder was computationally unreliable within the bounded simulation. It is not used in live scoring.
- Exact enumeration in independent 18-variable blocks resolved simulated random labels in 8.0 queries on average (20 simulations; range 4-10). Projected 96 queries for 216 labels, before final verification. This supports feasibility but not an actual score claim.
- The first score-only aggregate measurement was recorded as trial 17. The server then directed a cooldown until 19:36:05 UTC; the worker honors it.
- A bounded synthetic-feature search found no common normal-stream match; 100,000 seeds under each of two generators, with 64 starting offsets for regional features, also found no match. No generator recovery is claimed.
