# Five-minute pitch — seven slides

**Slide 1 — 0:00–0:35.** Our starting question is what the target measures. The committee's labels encode previous decisions, including their bias. Our deliverable scores all 4,000 evaluation applicants and funds exactly 1,600. The measured regional selection gap is 0.021 percentage points. That is an access statistic. We do not claim 98% accuracy, because only the judges possess independent reference labels.

**Slide 2 — 0:35–1:15.** We reproduced the supplied baseline before changing anything: 88.13% agreement with historical decisions, but an 18.76-point selection gap. Dropping region leaves 18.05 points. Dropping postal code too leaves 17.31. Distance alone predicts regional block almost perfectly. We also compare within academic bands; the remaining difference is an association, not a causal estimate.

**Slide 3 — 1:15–2:05.** The correction has three steps. First, estimate the historical committee's conditional associations while controlling for observed geography and other variables. Second, replace its ranking policy: remove geography and program effects, a positive wealth advantage and any first-generation penalty. Retain nonnegative academic and work effects. On these data, only academic score and work remain. These are normative assumptions; work credit and financial need require stakeholder review. Third, an exact optimizer funds 40% while allowing no more than a two-point regional selection gap. Region is explicitly used in that allocation.

**Slide 4 — 2:05–2:50.** We sweep 12 fairness tolerances for two scores, always under the same budget. The left panel shows agreement with historical decisions versus regional parity; the right shows the repaired ranking's utility cost. Some tolerances are inactive and produce identical decisions. This is an observable trade-off, not the hidden-reference Pareto front. We chose the provisional two-point guardrail without jury labels. Equal opportunity against independent merit remains the long-term target.

**Slide 5 — 2:50–3:30.** The final file contains 949 grants among 2,372 centre applicants and 651 among 1,628 remote applicants. Their rates are 40.01 and 39.99 percent. The repaired score already yields near parity, so the two-point constraint does not change this final allocation. We validate IDs, budget, monotonicity, deterministic reproduction and exact optimizer behavior. The corrected model agrees with historical decisions 85.33% of the time. Its historical-label TPR gap increases; we report that rather than interpreting it as a true-merit result.

**Slide 6 — 3:30–4:20.** Before deployment, an independent regional panel should establish a reference by reviewing funded and rejected cases. The ML owner checks every batch, while the fairness lead monitors all five regions, intersections, drift and appeals. Applicants receive understandable criteria and human review. Invalid outputs are blocked. Severe incidents freeze automatic releases and go to an approved human process, rather than automatically restoring the biased baseline. Budget reconciliation is explicit.

**Slide 7 — 4:20–5:00.** Our contribution is a complete, inspectable submission with honest limits. Demographic parity is a temporary access safeguard, not proof of equal opportunity. We still need independent eligibility evidence and an agreed balance between academics, work, need and first-generation support. We provide the model, notebook, predictions, plots and monitoring plan so judges can examine every decision. Our standard is transparent assumptions, measured evidence and a way to challenge an outcome.

## Likely questions

**Why not train a more accurate historical classifier?** Higher agreement with the committee may reproduce its bias. The official target is independently defined merit.

**Why demographic parity when scoring uses equal opportunity?** True-opportunity optimization requires valid eligibility labels. Constraining TPR using biased historical positives can retain past exclusions. We enforce an observable access safeguard and explicitly leave the true EOp result unclaimed.

**Does dropping geography remove proxies?** No. We use observed geography as an estimation control, explicitly repair the coefficient policy, and retain geography for the allocation constraint and audits. This is not causal identification.

**Why reward paid work?** It is a provisional, learned positive association and may reflect effort or financial burden. It can disadvantage applicants unable to work. We expose academic-only and need-support alternatives and require policy review before deployment.

**How certain are the results?** Counts and format are exact for this batch. Bootstrap intervals cover sampling uncertainty conditional on a model, not label bias. Training-refit stability and assumption sensitivity are separate, clearly labeled analyses.
