"""Rebuild the measured audit, charts, executed notebook and pitch artifacts."""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from nbclient import NotebookClient
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from model_corrige import (ROOT, DATA, NUMERIC, TARGET, DEFAULT_GAP, RepairedScore,
                          allocate, groups, load_data, verify_predictions)

ART = ROOT / "artifacts"
COLORS = ["#245f89", "#16a394", "#eb9c36", "#b95c69"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.facecolor": "white"})


def rates(y, p, group):
    y, p, group = np.asarray(y), np.asarray(p), np.asarray(group)
    result = {"historical_agreement": float(accuracy_score(y, p)),
              "historical_macro_f1": float(f1_score(y, p, average="macro")),
              "selection_rate": float(p.mean())}
    sr, tpr, fpr = [], [], []
    for g in [0, 1]:
        mask = group == g
        sr.append(float(p[mask].mean()))
        pos, neg = mask & (y == 1), mask & (y == 0)
        tpr.append(float(p[pos].mean()) if pos.any() else None)
        fpr.append(float(p[neg].mean()) if neg.any() else None)
    result.update({"centre_selection": sr[0], "remote_selection": sr[1],
                   "demographic_parity_gap": abs(sr[0] - sr[1]),
                   "historical_tpr_centre": tpr[0], "historical_tpr_remote": tpr[1],
                   "historical_tpr_gap": abs(tpr[0] - tpr[1]) if None not in tpr else None,
                   "historical_fpr_centre": fpr[0], "historical_fpr_remote": fpr[1]})
    return result


def bootstrap(y, p, group, repetitions=500):
    rng = np.random.default_rng(2026)
    strata = [np.flatnonzero((group == g) & (y == label)) for g in [0, 1] for label in [0, 1]]
    rows = []
    for _ in range(repetitions):
        ix = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata if len(s)])
        rows.append(rates(y[ix], p[ix], group[ix]))
    result = {"historical_agreement": [float(v) for v in np.quantile(
        [row["historical_agreement"] for row in rows], [.025, .975])]}
    # Bootstrap signed gaps first; absolute-value percentiles incorrectly exclude
    # zero at the parity boundary even when the signed interval spans zero.
    for key, left, right in [("demographic_parity_gap", "centre_selection", "remote_selection"),
                             ("historical_tpr_gap", "historical_tpr_centre", "historical_tpr_remote")]:
        lo, hi = np.quantile([row[left] - row[right] for row in rows], [.025, .975])
        result[key] = [0.0 if lo <= 0 <= hi else float(min(abs(lo), abs(hi))), float(max(abs(lo), abs(hi)))]
    return result


def wilson(successes, n):
    z = 1.959963984540054
    p = successes / n
    centre = (p + z*z/(2*n)) / (1 + z*z/n)
    radius = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return centre-radius, centre+radius


def encoded(train, test, exclude=()):
    cats = ["programme_etudes", "region_administrative", "code_postal_3"]
    cols = [c for c in train.columns if c not in ["id_candidat", TARGET] and c not in exclude]
    x = pd.get_dummies(train[cols], columns=[c for c in cats if c in cols], dtype=float)
    xt = pd.get_dummies(test[cols], columns=[c for c in cats if c in cols], dtype=float)
    return x, xt.reindex(columns=x.columns, fill_value=0)


def mark_front(frame, x="demographic_parity_gap", y="historical_agreement"):
    values = frame[[x, y]].to_numpy()
    return np.array([not np.any((values[:, 0] <= a + 1e-12) & (values[:, 1] >= b - 1e-12)
                        & ((values[:, 0] < a - 1e-12) | (values[:, 1] > b + 1e-12))) for a, b in values])


def main():
    ART.mkdir(exist_ok=True)
    history, evaluation = load_data()
    train, test = train_test_split(history, test_size=.3, random_state=42, stratify=history[TARGET])
    y, group = test[TARGET].to_numpy(), groups(test)
    test_ids = test.id_candidat.to_numpy()
    pd.DataFrame({"id_candidat": history.id_candidat,
                  "split": np.where(history.index.isin(test.index), "holdout", "train")}).to_csv(ART / "split.csv", index=False)
    print("Fitting the baseline and two feature-deletion controls...", flush=True)
    metrics, rf_prob = {}, None
    for name, excluded in [("baseline", ()), ("drop_region", ("region_administrative",)),
                           ("drop_region_postal", ("region_administrative", "code_postal_3"))]:
        x, xt = encoded(train, test, excluded)
        model = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=42, n_jobs=2)
        model.fit(x, train[TARGET])
        probability = model.predict_proba(xt)[:, 1]
        metrics[name] = rates(y, (probability >= .5).astype(int), group)
        if name == "baseline":
            rf_prob = probability
            baseline_pred = (probability >= .5).astype(int)
    repaired = RepairedScore.fit(train)
    scores = repaired.score(test)
    historical_prob = repaired.historical_probability(test)
    corrected_pred = allocate(scores, group, test_ids)
    metrics["corrected"] = rates(y, corrected_pred, group)
    metrics["controlled_logistic"] = rates(y, (historical_prob >= .5).astype(int), group)
    metrics["baseline"]["ci_95"] = bootstrap(y, baseline_pred, group)
    metrics["corrected"]["ci_95"] = bootstrap(y, corrected_pred, group)
    print("Measuring individual and combined regional proxies...", flush=True)
    proxy_rows = []
    for feature in ["code_postal_3"] + NUMERIC + ["programme_etudes", "all_without_region"]:
        if feature in ["code_postal_3", "programme_etudes"]:
            lookup = train.assign(remote=groups(train)).groupby(feature).remote.mean()
            p = test[feature].map(lookup).fillna(groups(train).mean()).to_numpy()
            proxy_rows.append({"feature": feature, "remote_prediction_auc": roc_auc_score(group, p),
                               "remote_prediction_accuracy": accuracy_score(group, p >= .5)})
            continue
        if feature == "all_without_region":
            x, xt = encoded(train, test, ("region_administrative",))
            predictor = RandomForestClassifier(n_estimators=150, min_samples_leaf=20, random_state=42, n_jobs=2)
        else:
            x = pd.get_dummies(train[[feature]], dtype=float)
            xt = pd.get_dummies(test[[feature]], dtype=float).reindex(columns=x.columns, fill_value=0)
            predictor = DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, random_state=42)
        predictor.fit(x, groups(train))
        p = predictor.predict_proba(xt)[:, 1]
        proxy_rows.append({"feature": feature, "remote_prediction_auc": roc_auc_score(group, p),
                           "remote_prediction_accuracy": accuracy_score(group, p >= .5)})
    proxies = pd.DataFrame(proxy_rows).sort_values("remote_prediction_auc", ascending=False)
    proxies.to_csv(ART / "proxy_audit.csv", index=False)
    coef = pd.DataFrame({"feature": repaired.columns, "historical_coefficient_standardized": repaired.coefficients,
                         "policy_coefficient_standardized": repaired.repaired,
                         "policy_coefficient_raw": repaired.repaired / repaired.scales})
    coef.to_csv(ART / "coefficients.csv", index=False)

    # Each allocation uses features/scores and group counts, never holdout labels.
    # Agreement is calculated only afterwards, as a descriptive historical metric.
    sweep = []
    tolerances = [1, .25, .20, .15, .10, .075, .05, .03, .02, .01, .005, .001]
    for method, utility in [("baseline score", rf_prob), ("repaired score", scores)]:
        best_utility = utility[allocate(utility, group, test_ids, max_gap=None) == 1].sum()
        for epsilon in tolerances:
            p = allocate(utility, group, test_ids, max_gap=epsilon)
            row = {"method": method, "max_gap": epsilon, **rates(y, p, group),
                   "ranking_utility_loss": float((best_utility - utility[p == 1].sum()) / len(p))}
            sweep.append(row)
    sweep = pd.DataFrame(sweep)
    sweep["historical_pareto"] = mark_front(sweep)
    sweep.to_csv(ART / "pareto.csv", index=False)

    # Explicitly hypothetical policy preferences; these are not reference labels.
    full = RepairedScore.fit(history)
    eval_scores = full.score(evaluation)
    eval_group = groups(evaluation)
    pred = allocate(eval_scores, eval_group, evaluation.id_candidat.to_numpy())
    submission = pd.DataFrame({"id_candidat": evaluation.id_candidat, TARGET: pred})
    from prediction_history import archive_prediction, preserve_existing
    preserve_existing(ROOT / "predictions.csv", evaluation)
    submission.to_csv(ROOT / "predictions.csv", index=False)
    validation = verify_predictions(evaluation, submission)
    full.save(ART / "model_parameters.json")
    (ART / "submission_validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    z = (evaluation[NUMERIC].to_numpy() - full.means[:len(NUMERIC)]) / full.scales[:len(NUMERIC)]
    r_coefficient = full.repaired[0]
    normalized = eval_scores / r_coefficient
    scenarios = {"academic_only": z[:, 0], "learned_work_credit": normalized,
                 "add_mild_financial_need": normalized - .15*z[:, 1],
                 "add_stronger_financial_need": normalized - .30*z[:, 1],
                 "add_first_generation_support": normalized + .10*z[:, 4]}
    scenario_rows = []
    for name, utility in scenarios.items():
        hypothetical = allocate(utility, eval_group, evaluation.id_candidat.to_numpy(), max_gap=None)
        met = rates(hypothetical, pred, eval_group)
        scenario_rows.append({"assumed_policy": name, "decision_overlap": met["historical_agreement"],
                              "tpr_gap_under_assumption": met["historical_tpr_gap"],
                              "changed_decisions": int((hypothetical != pred).sum())})
    scenarios_df = pd.DataFrame(scenario_rows)
    scenarios_df.to_csv(ART / "assumption_sensitivity.csv", index=False)
    stability = []
    for seed in [7, 42, 123, 2026, 8675309]:
        resample = history.sample(frac=1, replace=True, random_state=seed)
        m = RepairedScore.fit(resample)
        p = allocate(m.score(evaluation), eval_group, evaluation.id_candidat.to_numpy())
        stability.append({"bootstrap_seed": seed, "decision_overlap": float((pred == p).mean()),
                          "changed_decisions": int((pred != p).sum()), "grant_rate": float(p.mean())})
    pd.DataFrame(stability).to_csv(ART / "fit_stability.csv", index=False)

    regional_rows = []
    for name, data, decisions in [("historical decisions", history, history[TARGET].to_numpy()),
                                   ("evaluation corrected", evaluation, pred)]:
        for region in sorted(data.region_administrative.unique()):
            mask = (data.region_administrative == region).to_numpy()
            n, k = int(mask.sum()), int(decisions[mask].sum())
            low, high = wilson(k, n)
            regional_rows.append({"sample": name, "region": region, "n": n, "grants": k,
                                  "rate": k/n, "ci_low": low, "ci_high": high})
    regional = pd.DataFrame(regional_rows)
    regional.to_csv(ART / "regional_rates.csv", index=False)
    intersections = evaluation.assign(decision=pred).groupby(
        ["region_administrative", "premiere_generation_universitaire", "programme_etudes"]
    ).decision.agg(["size", "sum", "mean"]).reset_index()
    intersections["small_cell"] = intersections["size"] < 30
    intersections.to_csv(ART / "intersection_audit.csv", index=False)
    bins = [15, 24, 26, 28, 30, 32, 40.01]
    academic = history.assign(group=np.where(groups(history), "Remote", "Centre"),
                              academic_band=pd.cut(history.cote_r_equivalent, bins, right=False))
    band_rates = academic.groupby(["academic_band", "group"], observed=True)[TARGET].agg(["size", "mean"]).reset_index()
    band_rates.to_csv(ART / "academic_band_audit.csv", index=False)
    band_pivot = band_rates.pivot(index="academic_band", columns="group", values="mean")
    weights = academic.academic_band.value_counts(normalize=True).reindex(band_pivot.index)
    standardized_gap = float(((band_pivot.Centre-band_pivot.Remote)*weights).sum())
    # Quantile shifts are reported, not treated as hypothesis tests or hidden outcomes.
    drift = pd.DataFrame({"feature": NUMERIC, "history_mean": history[NUMERIC].mean().to_numpy(),
                           "evaluation_mean": evaluation[NUMERIC].mean().to_numpy(),
                           "standardized_mean_shift": ((evaluation[NUMERIC].mean()-history[NUMERIC].mean())/
                                                       history[NUMERIC].std()).to_numpy()})
    drift.to_csv(ART / "data_drift.csv", index=False)
    manifest = json.loads((DATA.parent / "manifest.json").read_text(encoding="utf-8"))
    import hashlib
    integrity = {f["path"]: hashlib.sha256((DATA.parent/f["path"]).read_bytes()).hexdigest() == f["sha256"]
                 for f in manifest["files"]}
    if not all(integrity.values()):
        raise ValueError("Original participant files failed checksum verification.")
    summary = {"historical_rows": len(history), "evaluation_rows": len(evaluation),
               "holdout_rows": len(test), "seed": 42, "max_gap_policy": DEFAULT_GAP,
               "historical_selection_centre": float(history.loc[groups(history)==0, TARGET].mean()),
               "historical_selection_remote": float(history.loc[groups(history)==1, TARGET].mean()),
               "academic_band_standardized_historical_gap": standardized_gap,
               "holdout_metrics": metrics, "submission": validation,
               "assumption_sensitivity": scenario_rows, "fit_stability": stability,
               "original_files_intact": integrity,
               "hidden_reference_accuracy": None, "hidden_reference_eop_gap": None,
               "interpretation": "Historical agreement and assumed-policy agreement are not true-merit accuracy."}
    (ART / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    archive_prediction(evaluation, submission, "corrected-policy-audited",
                       historical_agreement=metrics["corrected"]["historical_agreement"],
                       note="Full-fit evaluation predictions; historical agreement uses the separate 3,000-row holdout model.")
    import sklearn, scipy, fairlearn, reportlab
    versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
                "matplotlib": matplotlib.__version__, "fairlearn": fairlearn.__version__,
                "nbformat": nbformat.__version__, "reportlab": reportlab.Version}
    (ART / "environment.json").write_text(json.dumps(versions, indent=2) + "\n", encoding="utf-8")
    print("Rendering measured charts...", flush=True)
    plots(regional, proxies, sweep, band_pivot, summary)
    make_notebook(summary)
    print(json.dumps(summary["submission"], indent=2), flush=True)


def plots(regional, proxies, sweep, band_pivot, summary):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = {"Bas-Saint-Laurent": "Bas-Saint-Laurent", "Capitale-Nationale": "Capitale-Nationale",
              "Cote-Nord": "Cote-Nord", "Gaspesie-Iles-de-la-Madeleine": "Gaspesie / Iles", "Montreal": "Montreal"}
    for ax, sample, title, color in zip(axes, ["historical decisions", "evaluation corrected"],
            ["Historical committee decisions (10,000)", "Corrected allocation (4,000 new applicants)"], COLORS):
        d = regional[regional["sample"] == sample]
        ax.barh([labels[r] for r in d.region], d.rate*100, color=color,
                xerr=np.array([d.rate-d.ci_low, d.ci_high-d.rate])*100, capsize=3)
        ax.axvline(40, color="#607080", linestyle="--", linewidth=1)
        ax.set(xlim=(0, 60), xlabel="Grant rate (%)", title=title)
    fig.suptitle("Regional access | 95% Wilson intervals", fontsize=14)
    fig.text(.5, .01, "Different populations: descriptive comparison, not a causal estimate. Batch decisions are deterministic.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, .95)); fig.savefig(ART / "regional_access.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    d = proxies.sort_values("remote_prediction_auc")
    names = {"code_postal_3": "Postal prefix", "distance_domicile_campus_km": "Distance to campus",
             "heures_travail_semaine": "Hours worked", "revenu_familial_estime": "Household income",
             "premiere_generation_universitaire": "First generation", "cote_r_equivalent": "Academic score",
             "programme_etudes": "Program", "all_without_region": "All features except region"}
    ax.barh([names[f] for f in d.feature], d.remote_prediction_auc, color=COLORS[0])
    ax.axvline(.5, linestyle="--", color=COLORS[3]); ax.set(xlim=(0, 1.05), xlabel="Held-out regional prediction ROC AUC", title="Geography survives removal of the region column")
    fig.tight_layout(); fig.savefig(ART / "proxy_signals.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for (method, d), color in zip(sweep.groupby("method", sort=False), COLORS):
        axes[0].plot(d.demographic_parity_gap*100, d.historical_agreement*100, "o-", color=color, label=method)
        for eps in [.20, .10, .02, .001]:
            row = d[d.max_gap == eps].iloc[0]
            if method == "baseline score": axes[0].annotate(f"{eps:.3f}", (row.demographic_parity_gap*100, row.historical_agreement*100), fontsize=8, xytext=(3, 3), textcoords="offset points")
    front = sweep[sweep.historical_pareto].sort_values("demographic_parity_gap")
    axes[0].scatter(front.demographic_parity_gap*100, front.historical_agreement*100, s=100, facecolors="none", edgecolors="black", label="Nondominated historical points")
    selected = sweep[(sweep.method == "repaired score") & (sweep.max_gap == DEFAULT_GAP)].iloc[0]
    axes[0].scatter([selected.demographic_parity_gap*100], [selected.historical_agreement*100], marker="*", s=200, color=COLORS[2], zorder=10)
    axes[0].set(xlabel="Demographic parity gap (percentage points)", ylabel="Agreement with historical decisions (%)", title="Observed trade-off | fixed 40% budget")
    axes[0].legend(fontsize=8, loc="lower right")
    d = sweep[sweep.method == "repaired score"]
    axes[1].plot(d.demographic_parity_gap*100, d.ranking_utility_loss, "o-", color=COLORS[1])
    axes[1].set(xlabel="Demographic parity gap (percentage points)", ylabel="Policy score lost per applicant", title="Repaired policy score | lower loss is better")
    fig.text(.5, .015, "Labels next to blue points are parity tolerances. Historical agreement is NOT accuracy against the hidden reference.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig(ART / "pareto_front.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    band_pivot.plot.bar(ax=ax, color=COLORS[:2], rot=0)
    ax.set(xlabel="R-score band", ylabel="Historical grant rate", title="Within academic bands, regional disparities persist")
    fig.tight_layout(); fig.savefig(ART / "academic_bands.png", dpi=180); plt.close(fig)


def make_notebook(summary):
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    cells = [md("""# EquiAlgo — audit of regional access

**Decision:** a transparent policy score, exact 40% grant budget, and a maximum 2-percentage-point regional selection gap. This is a hackathon prototype, not a validated lending system.

**Evidence boundary:** the jury's independent reference is unavailable. Historical accuracy measures reproduction of a biased committee. No 98% true-merit accuracy or hidden-reference equal-opportunity claim is made.

The provided starter notebook was executed unchanged first; see `artifacts/baseline_executed.ipynb`. All eight original files pass the organizer's SHA-256 manifest. Only the supplied synthetic dataset was used."""),
    code("""from pathlib import Path
import json
import numpy as np
import pandas as pd
from IPython.display import display, Image, Markdown
from model_corrige import load_data, groups, verify_predictions
root = Path.cwd()
art = root / 'artifacts'
history, evaluation = load_data()
summary = json.loads((art / 'audit_summary.json').read_text())
display(pd.Series({k: summary[k] for k in ['historical_rows','evaluation_rows','holdout_rows','seed']}))
assert all(summary['original_files_intact'].values())
assert not (set(history.id_candidat) & set(evaluation.id_candidat))
assert history.isna().sum().sum() == evaluation.isna().sum().sum() == 0"""),
    md("""## 1. What the historical labels measure

`decision_octroi` records committee decisions, not independently established eligibility. Regional access differs despite an academic mean difference of less than one R-score point. We report the association within academic bands, without claiming causal identification or that academic score fully defines merit.

The split is 7,000 training / 3,000 held-out historical applicants, stratified by the historical decision, seed 42. Preprocessing and coefficient estimation use training rows only. Policy rules and the 0.02 parity tolerance are specified without looking at jury labels. Holding an exact batch budget uses candidate scores and group counts, never their outcomes."""),
    code("""display(history.assign(group=np.where(groups(history), 'Remote', 'Centre')).groupby('group').agg(
    n=('id_candidat','size'), grant_rate=('decision_octroi','mean'), mean_r=('cote_r_equivalent','mean')))
display(Image(filename=str(art / 'academic_bands.png')))
print('Academic-band standardized historical selection gap:', round(summary['academic_band_standardized_historical_gap'], 4))"""),
    md("""## 2. Baseline and the proxy trap

Reproduce the supplied random forest (300 trees, leaf size 20, seed 42). Deleting region or region plus postal prefix leaves substantial disparity. Separate classifiers predict remote-region membership from each feature on the same held-out rows; ROC AUC near 1 indicates strong geographic information. AUC measures association, not whether a feature is inherently illegitimate: financial need and paid work can also be legitimate support criteria."""),
    code("""metrics = pd.DataFrame({k:{a:b for a,b in v.items() if a != 'ci_95'} for k,v in summary['holdout_metrics'].items()}).T
display(metrics[['historical_agreement','demographic_parity_gap','historical_tpr_gap','selection_rate']].round(4))
display(pd.read_csv(art / 'proxy_audit.csv').round(4))
display(Image(filename=str(art / 'proxy_signals.png')))"""),
    md("""## 3. Fairness objective and what is observable

Our ultimate objective is **equal opportunity relative to independent merit**: |P(grant=1 | merit=1, centre) − P(grant=1 | merit=1, remote)|. This quantity cannot be measured without valid merit labels. Historical true-positive-rate gaps are shown only to diagnose label dependence.

For this submission, the implemented fairness constraint is **demographic parity within 0.02**, |P(grant=1 | centre) − P(grant=1 | remote)| ≤ 0.02, with exactly 40% funded. This is an observable, provisional access safeguard, not a substitute proof of equal opportunity. We choose it because constraining TPR against known-biased labels can entrench historical exclusions. Cohort profiles can justify different selection rates; this choice therefore requires human governance and sensitivity analysis.

The brief warns that parity and equal opportunity can conflict. This is a trade-off, not a universal impossibility theorem: both can hold for some distributions and classifiers. See [Fairlearn's metric definitions](https://fairlearn.org/main/user_guide/assessment/common_fairness_metrics.html)."""),
    md("""## 4. Correction and feature policy

1. Estimate a regularized additive logistic model of committee decisions, controlling for region, study program, distance, wealth, work and first-generation status.
2. Build a new ranking score from the fitted associations: academic score and work have nonnegative coefficients; wealth cannot increase priority; first-generation status cannot reduce it. Region, postal prefix, distance and program have zero ranking contribution. These are explicit normative choices, not discovered true-merit labels.
3. Enumerate feasible numbers of grants in each regional block. For each count, take the highest policy scores within the block. Choose the maximum summed score subject to 40% total grants and the parity bound. This is an exact solution to the specified two-group batch optimization. A stable seeded hash resolves exact ties.

Region remains available for auditing and allocation constraints. The system is not 'region blind'. Counterfactual score invariance does not imply counterfactual decision invariance because allocation depends on regional membership. The positive work coefficient may disadvantage students unable to work; treat it as a hypothesis to review. Removing a wealth advantage does not establish the correct financial-need benefit."""),
    code("""display(pd.read_csv(art / 'coefficients.csv').round(5))
display(Image(filename=str(art / 'regional_access.png')))
display(pd.read_csv(art / 'regional_rates.csv').round(4))"""),
    md("""## 5. Pareto sweep and uncertainty

The 12 parity tolerances are 1.0, .25, .20, .15, .10, .075, .05, .03, .02, .01, .005 and .001, each with the exact 40% budget. We sweep both the baseline probability score and the repaired score. Circles mark nondominated points in the measured historical-agreement/parity plane. Historical agreement is a descriptive comparator, not our optimization target or a hidden-reference utility estimate. Several constraints can produce identical decisions when they are inactive.

The second panel measures loss of the stated policy ranking objective. Neither panel establishes the unknown jury Pareto front. The default .02 guardrail is a policy choice, not a holdout-selected optimum.

Intervals below are 500 stratified bootstrap replicates, conditional on fitted models and fixed outcomes/group strata. They do not cover label bias, causal uncertainty or retraining variability. Wilson regional-rate intervals illustrate sampling uncertainty for future applicants; the current batch's counts are exact."""),
    code("""display(Image(filename=str(art / 'pareto_front.png')))
display(pd.read_csv(art / 'pareto.csv').round(4))
display(pd.DataFrame({name: summary['holdout_metrics'][name]['ci_95'] for name in ['baseline','corrected']}))"""),
    md("""## 6. Sensitivity, intersections and remaining risks

The following scenarios are explicit alternative value judgments: academic priority alone, the learned work credit, additional financial-need credit, and first-generation support. We compare selections and TPR under these **hypothetical** eligibility rules. They are not independent validation labels and must not be presented as merit accuracy.

Five training bootstrap refits measure decision stability, not correctness. Regional × program × first-generation rates expose harms hidden by two-group averages. Cells below 30 are flagged; use wider time windows and uncertainty intervals before intervention. Academic bands are coarse, hidden determinants may remain, and association cannot identify discrimination by itself. Increasing a feature that has zero weight leaves the individual score unchanged; increasing academic score cannot reduce it, holding other inputs fixed."""),
    code("""display(pd.read_csv(art / 'assumption_sensitivity.csv'))
display(pd.read_csv(art / 'fit_stability.csv'))
display(pd.read_csv(art / 'intersection_audit.csv'))
display(pd.read_csv(art / 'data_drift.csv'))"""),
    md("""## 7. Production governance and appeals

See `MONITORING.md` for owners, denominators, alert thresholds, independent relabeling, review workflows, privacy controls and rollback. Before deployment, a geographically diverse panel must independently adjudicate a representative sample, blinded to previous decisions where feasible. Both funded and rejected applicants must be sampled to reduce selective-label bias. Only then can true-reference accuracy, calibration and equal opportunity be estimated.

An appeal is reviewed by a human who can inspect data corrections and contextual barriers; a review must not silently increase the fixed grant budget or automatically revoke another student's award. Freeze releases on severe issues and use an approved human allocation process, not an automatic return to the biased baseline."""),
    code("""submission = pd.read_csv(root / 'predictions.csv')
display(pd.Series(verify_predictions(evaluation, submission)))
assert submission.decision_octroi.sum() == 1600
assert verify_predictions(evaluation, submission)['demographic_parity_gap'] <= 0.02
print('Hidden-reference accuracy: unavailable. The requested 98% is not verified.')"""),
    md("""## Sources and reproducibility

- Organizer: supplied `README.md`, `consignes-en.pdf`, `consignes-fr.pdf`, starter notebook and manifest.
- [HxBuddy IVADO track](https://hxbuddy.ca/), [public track text](https://hxbuddy.ca/demo-locales/en.json), retrieved 2026-10-03: indicative macro F1/accuracy; final jury uses the supplied rubric; Devpost requires the matching prize and team identity.
- [Fairlearn metric definitions](https://fairlearn.org/main/user_guide/assessment/common_fairness_metrics.html) and [postprocessing](https://fairlearn.org/main/user_guide/mitigation/postprocessing.html).
- Tools: Python, NumPy, pandas, SciPy, scikit-learn, Matplotlib, Fairlearn (starter), Jupyter and ReportLab. OpenAI Codex assisted with analysis, code and documentation. No external training dataset or pretrained predictive model was used.

Run `python model_corrige.py`, `python build_audit.py`, `python build_presentation.py`, and `python -m unittest discover -s tests`. The checked-in notebook already contains results. Exact package versions and split IDs are in `artifacts/`.""")]
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbformat.write(nb, ROOT / "audit_rapport.ipynb")
    # Use the running environment explicitly, without installing a global kernel.
    from jupyter_client import KernelManager
    km = KernelManager(kernel_name="python3")
    km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    NotebookClient(nb, timeout=180, km=km, resources={"metadata": {"path": str(ROOT)}}).execute()
    nbformat.write(nb, ROOT / "audit_rapport.ipynb")


if __name__ == "__main__":
    main()
