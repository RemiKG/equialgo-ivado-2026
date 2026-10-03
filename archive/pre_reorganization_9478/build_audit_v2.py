"""Rebuild the V2 audit, keeping measured and model-implied results separate."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
from nbclient import NotebookClient
from scipy.special import expit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from jupyter_client import KernelManager

from merit_inference import (ROOT,V1,TARGET,MeritEnsemble,load_data,groups,fit_committee,
    historical_design,equal_opportunity_allocate,verify_predictions,OBSERVED_ACCURACY,OBSERVED_F1)

ART=ROOT/"artifacts"


def load_ensemble(name):
    folder=ART/name
    params=json.loads((folder/"model_parameters.json").read_text())
    arrays=np.load(folder/"merit_ensemble.npz")
    return MeritEnsemble(params["committee"],arrays["coefficients"],arrays["noise"],
        arrays["weights"],arrays["thresholds"],pd.read_csv(folder/"merit_hypotheses.csv"),name)


def implied_metrics(p,pred,g):
    utility=float(np.mean(pred*p+(1-pred)*(1-p)))
    tpr=[float(np.sum(p[g==j]*pred[g==j])/np.sum(p[g==j])) for j in [0,1]]
    return {"model_implied_agreement":utility,"model_implied_eop_gap":abs(tpr[0]-tpr[1]),
            "measured_dp_gap":abs(float(pred[g==0].mean()-pred[g==1].mean()))}


def main():
    history,evaluation=load_data();g=groups(evaluation);ids=evaluation.id_candidat.to_numpy()
    old=pd.read_csv(V1/"predictions.csv")[TARGET].to_numpy()
    current=pd.read_csv(ROOT/"predictions.csv");pred=current[TARGET].to_numpy()
    validation=verify_predictions(evaluation,current)
    print("Checking historical model families with five held-out folds...",flush=True)
    cv=[]
    for link in ["raw","log"]:
        for fold,(train,test) in enumerate(StratifiedKFold(5,shuffle=True,random_state=87).split(history,history[TARGET])):
            model=fit_committee(history.iloc[train],link)
            p=expit(historical_design(history.iloc[test],link)@np.array(model["coefficients"]))
            cv.append({"income_link":link,"fold":fold,"historical_log_loss":log_loss(history[TARGET].iloc[test],p),
                       "historical_agreement":accuracy_score(history[TARGET].iloc[test],p>=.5)})
    cv=pd.DataFrame(cv);cv.to_csv(ART/"historical_cross_validation.csv",index=False)
    print("Sweeping opportunity constraints under both explicit merit assumptions...",flush=True)
    sweep=[];models={name:load_ensemble(name) for name in ["structured_merit","broad_merit"]}
    probabilities={name:model.probabilities(evaluation) for name,model in models.items()}
    tolerances=[1,.10,.075,.05,.03,.02,.01,.005,.002,.001]
    for name,p in probabilities.items():
        for tolerance in tolerances:
            selected=equal_opportunity_allocate(p,g,ids,tolerance=tolerance)
            sweep.append({"model":name,"eop_tolerance":tolerance,
                          **implied_metrics(p,selected,g),"changed_from_v1":int(np.sum(selected!=old))})
    sweep=pd.DataFrame(sweep)
    sweep.to_csv(ART/"pareto.csv",index=False)
    changes=evaluation.loc[pred!=old].copy()
    changes["v1_decision"]=old[pred!=old];changes["v2_decision"]=pred[pred!=old]
    changes["model_merit_weight"]=probabilities["structured_merit"][pred!=old]
    # Keep this sensitive row-level analysis local rather than publish covariates.
    local=ROOT/"work"/"v2_analysis";local.mkdir(parents=True,exist_ok=True)
    changes.to_csv(local/"changed_candidates.csv",index=False)
    description=[]
    for label,mask in [("new_grantees",(pred==1)&(old==0)),("no_longer_selected",(pred==0)&(old==1))]:
        data=evaluation.loc[mask]
        description.append({"group":label,"count":int(mask.sum()),"mean_r":float(data.cote_r_equivalent.mean()),
             "mean_work_hours":float(data.heures_travail_semaine.mean()),"mean_income":float(data.revenu_familial_estime.mean()),
             "first_generation_rate":float(data.premiere_generation_universitaire.mean()),"remote_rate":float(groups(data).mean())})
    pd.DataFrame(description).to_csv(ART/"decision_change_summary.csv",index=False)
    intersections=evaluation.assign(decision=pred).groupby(["region_administrative","premiere_generation_universitaire","programme_etudes"]).decision.agg(["size","mean"]).reset_index()
    intersections["small_cell"]=intersections["size"]<30
    intersections.to_csv(ART/"intersection_audit.csv",index=False)
    regions=evaluation.assign(v1=old,v2=pred).groupby("region_administrative").agg(n=("id_candidat","size"),v1_rate=("v1","mean"),v2_rate=("v2","mean"))
    regions.to_csv(ART/"regional_rates.csv")
    candidates=json.loads((ART/"v2_candidates.json").read_text())
    summary={"version":2,"team":"SOTA Overfitters","primary":"structured_merit",
             "v1_user_reported_accuracy":OBSERVED_ACCURACY,"v1_user_reported_macro_f1":OBSERVED_F1,
             "v2_official_accuracy":None,"v2_official_macro_f1":None,"submission":validation,
             "changed_from_v1":int((pred!=old).sum()),"scenario_count":len(models["structured_merit"].noise),
             "candidates":candidates,"historical_cv":cv.groupby("income_link").mean(numeric_only=True).drop(columns="fold").to_dict("index"),
             "selected_model_implied_metrics":implied_metrics(probabilities["structured_merit"],pred,g),
             "decision_changes":description,
             "caveat":"Uses reported aggregate evaluation feedback. Model-implied improvement is not validated. Low-noise assumption is experimental."}
    (ART/"audit_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    plots(sweep,regions,models)
    notebook(summary)
    print(json.dumps({"changes":summary["changed_from_v1"],"v2_official_accuracy":None,"budget":validation["grant_rate"]}))


def plots(sweep,regions,models):
    plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False})
    fig,ax=plt.subplots(figsize=(10,4.6))
    for name,color in [("structured_merit","#149c8d"),("broad_merit","#235e8c")]:
        d=sweep[sweep.model==name].sort_values("model_implied_eop_gap")
        ax.plot(d.model_implied_eop_gap*100,d.model_implied_agreement*100,"o-",label=name,color=color)
    ax.set(xlabel="MODEL-IMPLIED equal-opportunity gap (percentage points)",
           ylabel="MODEL-IMPLIED agreement (%)",title="Opportunity / utility sweep under stated hypotheses")
    ax.legend();fig.text(.5,.01,"These are hypothesis-dependent simulations, not HxBuddy scores. Every point funds exactly 40%.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.035,1,1));fig.savefig(ART/"pareto_front.png",dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    for name,color in [("structured_merit","#149c8d"),("broad_merit","#235e8c")]:
        m=models[name]
        axes[0].hist(m.coefficients[:,1],weights=m.weights,bins=np.linspace(0,.22,25),histtype="step",linewidth=2,color=color,label=name)
        axes[1].hist(m.noise,weights=m.weights,bins=np.linspace(.04,.65,25),histtype="step",linewidth=2,color=color,label=name)
    axes[0].axvline(.1458,color="#b95d64",linestyle="--",label="V1 work weight")
    axes[0].set(xlabel="Work weight (R-score points / hour)",ylabel="Hypothesis weight",title="Vary how work contributes to merit")
    axes[1].set(xlabel="Assumed merit noise (R-score units)",ylabel="Hypothesis weight",title="Different uncertainty assumptions")
    axes[0].legend(fontsize=8);axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(ART/"hypothesis_weights.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.5));d=regions.copy();d.index=[s.replace("Gaspesie-Iles-de-la-Madeleine","Gaspesie / Iles") for s in d.index]
    (d[["v1_rate","v2_rate"]]*100).plot.barh(ax=ax,color=["#8294a9","#149c8d"])
    ax.set(xlabel="Observed selection rate (%)",title="Allocation changes on the same 4,000 applicants")
    ax.legend(["V1 (scored)","V2 (pending score)"]);fig.tight_layout();fig.savefig(ART/"regional_access.png",dpi=180);plt.close(fig)


def notebook(summary):
    md,code=nbformat.v4.new_markdown_cell,nbformat.v4.new_code_cell
    cells=[md("""# EquiAlgo V2 — SOTA Overfitters

## What changed

The participant reports **94.63% accuracy / 94.40% macro F1** for V1 on HxBuddy. That is the only measured independent-reference result currently recorded. The entire prior method is preserved in `archive/v1_score_repair_9463/`. V2 changes the inference problem: model plausible merit criteria instead of simply clipping the committee's coefficients and enforcing demographic parity.

**V2 is experimental and unscored.** Its model-implied agreement is not an accuracy estimate validated against private labels. The requested 98% has not been established. We use aggregate evaluation feedback in model development and disclose this; the evaluated cohort is no longer an untouched validation set."""),
    code("""from pathlib import Path
import json, numpy as np, pandas as pd
from IPython.display import display, Image
from data_contract import load_data, groups, verify_predictions
root=Path.cwd(); art=root/'artifacts'
history,evaluation=load_data()
summary=json.loads((art/'audit_summary.json').read_text())
display(pd.Series({'historical_rows':len(history),'evaluation_rows':len(evaluation),'scenario_count':summary['scenario_count'],'changed_from_v1':summary['changed_from_v1']}))
display(pd.DataFrame(summary['candidates']).T)
assert set(history.id_candidat).isdisjoint(evaluation.id_candidat)"""),
    md("""## 1. Original diagnosis remains valid

The supplied random forest reproduces biased decisions: 88.13% historical agreement and a held-out regional selection gap of 18.76 points. Removing region leaves 18.05 points; removing postal prefix too leaves 17.31. Postal prefix perfectly identifies the regional block in the supplied data, while distance gives ROC AUC 0.997. These results and the unchanged executed starter are preserved in the V1 archive.

Region, distance, wealth and work can carry regional information, but correlation alone does not identify which contributions belong to merit and which to committee bias. Confusing the two was the weakness of simple coefficient deletion."""),
    code("""v1=root/'archive'/'v1_score_repair_9463'/'artifacts'
display(pd.read_csv(v1/'proxy_audit.csv'))
display(Image(filename=str(v1/'academic_bands.png')))
display(pd.read_csv(art/'historical_cross_validation.csv').groupby('income_link').mean(numeric_only=True))"""),
    md("""## 2. Structural investigation, including rejected ideas

We tested linear versus nonlinear income effects, additive splines, work saturation, interactions and logistic/probit noise. A pooled logistic model with log income improves historical log loss, but nonlinear decompositions do **not** reliably identify a legitimate financial-need benefit. We therefore do not claim the historical data recovered the reference's need coefficient.

The logged five-fold historical cross-validation compares raw and log income on the same folds (seed 87). Only training rows fit each diagnostic. These metrics validate a model of committee behavior, not a model of independent merit. The full-fit work association provides a scale for a broad prior, not a fixed entitlement to work credit."""),
    md("""## 3. New method: inverse merit hypotheses

Each hypothesis defines a latent score from R score, paid work, log household income, first-generation status and commute burden. R-score coefficient is 1. Work ranges from 0 to 1.4 times the learned academic-normalized committee coefficient (bounded at .25 R points/hour). Log-income coefficient ranges from -.8 to 0, first-generation benefit from 0 to .6 R points, and commute benefit from 0 to .16 points per 100 km. These directions and ranges are explicit normative priors.

We draw 2,048 Sobol hypotheses, seed 20261003. Each also assumes Gaussian merit noise of .04–.65 R-score units and an intercept giving 40% expected eligibility in this cohort. We weight hypotheses by compatibility with V1's reported accuracy. The brief's .270 baseline opportunity gap contributes a **weak** compatibility term (width .04) because its exact cohort/model provenance is unknown. Macro F1 is recorded but is not treated as an independent likelihood observation of the same predictions.

The primary `structured_merit` candidate restricts noise to at most .30 R-score units, testing whether missing criteria explain most of V1's errors. This is an ambitious assumption, not a learned fact. `broad_merit` allows the full noise range. `academic_anchor` is a falsification alternative retaining only one quarter of the committee's work coefficient alongside R score.

The final primary ranking averages eligibility probabilities across compatible hypotheses and takes the top 1,600. There is no direct region, postal or program score term, and no forced regional selection parity. Commute burden still carries regional information. Zero direct regional coefficients do not prove counterfactual or causal fairness."""),
    code("""display(pd.Series(json.loads((art/'model_parameters.json').read_text())))
display(Image(filename=str(art/'hypothesis_weights.png')))
display(pd.read_csv(art/'decision_change_summary.csv'))"""),
    md("""## 4. Fairness objective and Pareto sweep

Our target is equal opportunity among independently eligible applicants, with a fixed budget. Those eligibility labels are unavailable. The sweep therefore uses **model-implied** opportunity: sum of selected eligibility weights / sum of all eligibility weights within each regional block. We enumerate feasible grant counts and maximize summed eligibility weight under ten opportunity tolerances, always selecting 1,600.

The plot is a sensitivity frontier under stated hypotheses, **not a verified hidden-reference Pareto frontier**. Its vertical values are expected agreement under the same assumed model; they cannot be presented as validation accuracy. Demographic parity is reported descriptively. The primary default is unconstrained ranking; the alternative constraints show the cost and instability of acting as if the inferred labels were known."""),
    code("""display(Image(filename=str(art/'pareto_front.png')))
display(pd.read_csv(art/'pareto.csv'))
display(Image(filename=str(art/'regional_access.png')))
display(pd.read_csv(art/'intersection_audit.csv'))"""),
    md("""## 5. Uncertainty, failure modes and governance

The inverse problem is not identifiable from one aggregate score. Many distinct merit rules fit it. The low-noise prior may be wrong; monetary-need, first-generation and commute benefits are not identified by the biased labels. Gaussian noise, additive scoring and a 40% latent-eligibility rate can all be misspecified. Finite Sobol scenarios approximate the prior; repeat with a larger set to assess numerical sensitivity. Aggregate-feedback tuning risks leaderboard overfitting and does not certify out-of-sample generalization.

Use the archived V1 as a scored fallback. Record every evaluated CSV by hash. Do not call a candidate better until its official result arrives. A representative independent review sample, including rejected cases, is required before production. Paid-work credit can disadvantage students unable to work; need estimates and first-generation indicators need consent, data quality review and human appeals. See `MONITORING.md` for owners, controls and escalation.

Regional and intersectional rates include counts; flag cells under 30 and use larger review samples rather than declaring small-cell gaps conclusive. Expected-opportunity constraints on assumed labels must never be treated as production safety guarantees."""),
    code("""submission=pd.read_csv(root/'predictions.csv')
display(pd.Series(verify_predictions(evaluation,submission)))
assert submission.decision_octroi.sum()==1600
display(pd.read_csv(root/'predictions_history'/'comparison.csv')[['model','official_accuracy','official_macro_f1','changed_decisions_from_previous']])
print('V2 official accuracy: not yet measured. V1 score must not be transferred to V2.')"""),
    md("""## Sources and tools

Supplied IVADO briefs, README, baseline notebook and data; [HxBuddy](https://hxbuddy.ca/); participant-reported SOTA Overfitters score; [SciPy Sobol documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html); [SciPy normal CDF](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ndtr.html); [Fairlearn metrics](https://fairlearn.org/main/user_guide/assessment/common_fairness_metrics.html).

Python, NumPy, pandas, SciPy, scikit-learn, Matplotlib, Jupyter and ReportLab. OpenAI Codex assisted with analysis, code and writing. No external training dataset, hidden answer key, pretrained predictor or candidate-ID predictive feature was used.

Reproduce: `python model_corrige.py`, `python build_audit.py`, `python build_presentation.py`, `python -m unittest discover -s tests`. Every CSV remains versioned. To rerun V1, use its archived files with the original participant data path.""")]
    nb=nbformat.v4.new_notebook(cells=cells);nb.metadata.kernelspec={"display_name":"Python 3","language":"python","name":"python3"}
    km=KernelManager(kernel_name="python3");km.kernel_spec.argv=[sys.executable,"-m","ipykernel_launcher","-f","{connection_file}"]
    NotebookClient(nb,timeout=180,km=km,resources={"metadata":{"path":str(ROOT)}}).execute()
    nbformat.write(nb,ROOT/"audit_rapport.ipynb")


if __name__=="__main__":
    main()
