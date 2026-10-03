"""V2: inverse modelling of plausible merit rules from aggregate feedback.

These hypothetical probabilities are not independently validated. V1's measured
score is not V2's score. The method exposes uncertainty in the merit definition.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, ndtr
from scipy.optimize import minimize
from scipy.stats import qmc
from threadpoolctl import threadpool_limits

from data_contract import (ROOT, DATA, FEATURES, NUMERIC, TARGET, REMOTE,
                           groups, load_data, validate_data, verify_predictions,
                           allocate, tie_keys)

V1 = ROOT / "archive" / "v1_score_repair_9463"
OBSERVED_ACCURACY, OBSERVED_F1, BENCHMARK_EOP = .9463, .944, .270
SEED = 20261003


def design(frame):
    """No direct region, postal or program term; support directions are priors."""
    return np.column_stack([frame.cote_r_equivalent.to_numpy()-28,
        frame.heures_travail_semaine.to_numpy()-10,
        np.log(frame.revenu_familial_estime.to_numpy()/70000),
        frame.premiere_generation_universitaire.to_numpy()-.33,
        frame.distance_domicile_campus_km.to_numpy()/100])


def historical_design(frame, income_link="log"):
    income = frame.revenu_familial_estime.to_numpy()/70000
    return np.column_stack([np.ones(len(frame)), (frame.cote_r_equivalent.to_numpy()-28)/3,
        (frame.heures_travail_semaine.to_numpy()-10)/4,
        np.log(income) if income_link == "log" else income,
        frame.distance_domicile_campus_km.to_numpy()/200,
        frame.premiere_generation_universitaire.to_numpy(), groups(frame)])


def fit_committee(history, income_link="log"):
    """Estimate committee associations for diagnostics and a work prior scale."""
    x, y = historical_design(history, income_link), history[TARGET].to_numpy()
    def objective(beta):
        z = x@beta
        return (np.logaddexp(0,z).sum()-y@z+.005*(beta[1:]@beta[1:]),
                x.T@(expit(z)-y)+np.r_[0,.01*beta[1:]])
    with threadpool_limits(limits=1):
        fit = minimize(objective,np.zeros(x.shape[1]),jac=True,method="L-BFGS-B",
                       options={"ftol":1e-12,"maxiter":1500})
        if not fit.success:
            raise RuntimeError(f"Committee diagnostic did not converge: {fit.message}")
        p=expit(x@fit.x)
        se=np.sqrt(np.diag(np.linalg.pinv(x.T@((p*(1-p))[:,None]*x))))
    return {"income_link":income_link,"features":["intercept","academic_per_3","work_per_4",
            "income","distance_per_200","first_generation","remote"],
            "coefficients":fit.x.tolist(),"standard_errors":se.tolist(),
            "work_per_r_point":float((fit.x[2]/4)/(fit.x[1]/3))}


def scenario_probabilities(x, coefficients, noise, target_rate=.4):
    """Set expected eligibility to the challenge capacity in each hypothesis."""
    scores=x@coefficients.T
    low=scores.min(axis=0)-6*noise; high=scores.max(axis=0)+6*noise
    for _ in range(24):
        threshold=(low+high)/2
        up=ndtr((scores-threshold)/noise).mean(axis=0)>target_rate
        low=np.where(up,threshold,low);high=np.where(up,high,threshold)
    thresholds=(low+high)/2
    return ndtr((scores-thresholds)/noise),thresholds


@dataclass
class MeritEnsemble:
    committee: dict
    coefficients: np.ndarray
    noise: np.ndarray
    weights: np.ndarray
    thresholds: np.ndarray
    scenarios: pd.DataFrame
    method: str

    def probabilities(self,frame):
        """Posterior mean under assumptions; not a validated probability."""
        x=design(frame); result=np.zeros(len(frame))
        with threadpool_limits(limits=1):
            for start in range(0,len(self.noise),128):
                sl=slice(start,start+128)
                p=ndtr((x@self.coefficients[sl].T-self.thresholds[sl])/self.noise[sl])
                result+=p@self.weights[sl]
        return result

    def save(self,folder):
        folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/"merit_ensemble.npz",coefficients=self.coefficients,
                            noise=self.noise,weights=self.weights,thresholds=self.thresholds)
        metadata={"method":self.method,"scenario_count":len(self.noise),"seed":SEED,
                  "committee":self.committee,"effective_scenarios":float(1/(self.weights@self.weights)),
                  "weighted_mean_coefficients":(self.weights@self.coefficients).tolist(),
                  "weighted_noise_r_units":float(self.weights@self.noise),
                  "feedback":{"accuracy":OBSERVED_ACCURACY,"macro_f1":OBSERVED_F1,
                              "source":"User-reported HxBuddy V1 result; rounded display"},
                  "accuracy_for_this_model":None,
                  "warning":"Aggregate evaluation feedback used. Model-implied results are not validation."}
        (folder/"model_parameters.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
        self.scenarios.to_csv(folder/"merit_hypotheses.csv",index=False)


def infer_models(history,evaluation,previous,baseline,power=11):
    """Approximate Bayesian inversion with unidentifiable merit rules.

    One aggregate score cannot identify the correct rule. Both a broad model and
    a low-noise alternative are retained. The latter tests whether omitted merit
    criteria, rather than random noise, explain most V1 errors. Prior ranges are
    assumptions, not coefficients recovered from the jury's private labels.
    """
    committee=fit_committee(history)
    work_max=min(.25,max(.08,committee["work_per_r_point"]*1.4))
    u=qmc.Sobol(d=5,scramble=True,seed=SEED).random_base2(power)
    coefficients=np.column_stack([np.ones(len(u)),u[:,0]*work_max,-u[:,1]*.8,u[:,2]*.6,u[:,3]*.16])
    noise=.04+.61*u[:,4]
    x,g=design(evaluation),groups(evaluation).astype(bool)
    previous,baseline=np.asarray(previous),np.asarray(baseline)
    columns=[];all_thresholds=[]
    with threadpool_limits(limits=1):
        for start in range(0,len(noise),128):
            sl=slice(start,start+128)
            p,thresholds=scenario_probabilities(x,coefficients[sl],noise[sl])
            all_thresholds.append(thresholds)
            agreement=(previous@p+(1-previous)@(1-p))/len(evaluation)
            variance=(p*(1-p)).sum(axis=0)/len(evaluation)**2
            baseline_gap=(baseline[~g]@p[~g])/p[~g].sum(axis=0)-(baseline[g]@p[g])/p[g].sum(axis=0)
            loglik=-.5*(agreement-OBSERVED_ACCURACY)**2/(variance+.0015**2)
            # Exact cohort/model for the brief's 0.270 gap is unknown: weak term.
            loglik-=.5*((baseline_gap-BENCHMARK_EOP)/.04)**2
            columns.append(np.column_stack([agreement,baseline_gap,loglik]))
    measurements=np.concatenate(columns);thresholds=np.concatenate(all_thresholds)
    df=pd.DataFrame(coefficients,columns=["academic","work","log_income","first_generation","commute_per_100km"])
    df["noise_r_units"]=noise;df["implied_v1_accuracy"]=measurements[:,0]
    df["implied_baseline_eop_gap"]=measurements[:,1]
    models={}
    for name,allowed in [("structured_merit",noise<=.30),("broad_merit",np.ones(len(noise),bool))]:
        logw=np.where(allowed,measurements[:,2],-np.inf)
        weights=np.exp(logw-np.max(logw));weights/=weights.sum()
        df[name+"_weight"]=weights
        models[name]=MeritEnsemble(committee,coefficients,noise,weights,thresholds,df,name)
    return models


def equal_opportunity_allocate(probability,group,ids,tolerance=None,rate=.4):
    """Exact budget; optional EOp constraint is under inferred, not true, merit."""
    p,g,ids=np.asarray(probability),np.asarray(group),np.asarray(ids)
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError("Merit weights must be finite and between zero and one.")
    if tolerance is None:
        return allocate(p,g,ids,grant_rate=rate,max_gap=None)
    if not 0<=tolerance<=1:
        raise ValueError("Opportunity tolerance must be between 0 and 1.")
    if not .36<=rate<=.44 or len(p)!=len(g) or len(p)!=len(ids) or len(set(ids))!=len(p):
        raise ValueError("Valid budget, aligned arrays and unique IDs required.")
    order=np.lexsort((tie_keys(ids),-p));a,b=order[g[order]==0],order[g[order]==1]
    if not len(a) or not len(b) or p[a].sum()==0 or p[b].sum()==0:
        raise ValueError("Both groups require positive inferred eligibility mass.")
    k=round(len(p)*rate)
    kb=np.arange(max(0,k-len(a)),min(k,len(b))+1);ka=k-kb
    ca,cb=np.r_[0,np.cumsum(p[a])],np.r_[0,np.cumsum(p[b])]
    valid=np.abs(ca[ka]/ca[-1]-cb[kb]/cb[-1])<=tolerance+1e-12
    if not valid.any():
        raise ValueError("Opportunity tolerance is infeasible for the integer batch.")
    objective=ca[ka]+cb[kb];objective[~valid]=-np.inf;best=int(np.argmax(objective))
    pred=np.zeros(len(p),dtype=int);pred[a[:ka[best]]]=1;pred[b[:kb[best]]]=1
    return pred


def read_prior_files(evaluation):
    previous=pd.read_csv(V1/"predictions.csv")
    candidates=sorted((V1/"predictions_history").glob("*baseline-original*/predictions.csv"))
    if len(candidates)!=1:
        raise FileNotFoundError("An unambiguous archived baseline is required.")
    baseline=pd.read_csv(candidates[0])
    verify_predictions(evaluation,previous);verify_predictions(evaluation,baseline)
    return previous,baseline


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"predictions.csv")
    parser.add_argument("--scenario-power",type=int,default=11)
    args=parser.parse_args()
    if not 7<=args.scenario_power<=14:
        parser.error("scenario-power must be 7..14")
    history,evaluation=load_data();previous,baseline=read_prior_files(evaluation)
    print("Fitting committee diagnostics and evaluating merit hypotheses...",flush=True)
    models=infer_models(history,evaluation,previous[TARGET],baseline[TARGET],args.scenario_power)
    output=args.output.parent;output.mkdir(parents=True,exist_ok=True)
    artifacts=output/"artifacts";artifacts.mkdir(exist_ok=True)
    candidates=output/"submissions"/"v2_candidates";candidates.mkdir(parents=True,exist_ok=True)
    from prediction_history import archive_prediction,preserve_existing
    preserve_existing(args.output,evaluation,output/"predictions_history")
    reports={};scores=pd.DataFrame({"id_candidat":evaluation.id_candidat})
    for name,model in models.items():
        probability=model.probabilities(evaluation);scores[name]=probability
        pred=equal_opportunity_allocate(probability,groups(evaluation),evaluation.id_candidat.to_numpy())
        csv=pd.DataFrame({"id_candidat":evaluation.id_candidat,TARGET:pred})
        report=verify_predictions(evaluation,csv)
        report.update({"changed_from_v1":int((pred!=previous[TARGET]).sum()),"method":name,
                       "official_accuracy":None,"v1_user_reported_accuracy":OBSERVED_ACCURACY})
        csv.to_csv(candidates/f"{name}.csv",index=False);model.save(artifacts/name)
        archive_prediction(evaluation,csv,f"v2-{name}",history_dir=output/"predictions_history",
            note="Experimental inverse-merit model; aggregate V1 feedback used; V2 official score unmeasured.",
            model_files=[artifacts/name/"model_parameters.json",artifacts/name/"merit_ensemble.npz"])
        reports[name]=report
        if name=="structured_merit":
            csv.to_csv(args.output,index=False);model.save(artifacts)
            (artifacts/"submission_validation.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
        print(name,json.dumps(report),flush=True)
    anchor=evaluation.cote_r_equivalent.to_numpy()+.25*models["structured_merit"].committee["work_per_r_point"]*evaluation.heures_travail_semaine.to_numpy()
    pred=allocate(anchor,groups(evaluation),evaluation.id_candidat.to_numpy(),max_gap=None)
    csv=pd.DataFrame({"id_candidat":evaluation.id_candidat,TARGET:pred})
    csv.to_csv(candidates/"academic_anchor.csv",index=False)
    archive_prediction(evaluation,csv,"v2-academic-anchor",history_dir=output/"predictions_history",
        note="Falsification alternative: 25% of committee's academic-normalized work effect; no income/geography term.")
    reports["academic_anchor"]={**verify_predictions(evaluation,csv),"changed_from_v1":int((pred!=previous[TARGET]).sum()),"official_accuracy":None}
    scores.to_csv(artifacts/"candidate_merit_probabilities.csv",index=False)
    (artifacts/"v2_candidates.json").write_text(json.dumps(reports,indent=2)+"\n",encoding="utf-8")
    print(f"Primary: {args.output}; alternatives: {candidates}",flush=True)


if __name__=="__main__":
    main()
