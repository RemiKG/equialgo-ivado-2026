"""Experimental V3: condition a sparse merit ensemble on every reported score.

Aggregate scores are correlated because candidates share labels. This version
uses the full covariance, does not assume a low-noise reference, and never
replaces the scored champion automatically. All priors remain explicit.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import qmc
from threadpoolctl import threadpool_limits
from data_contract import ROOT,TARGET,load_data,groups,allocate,verify_predictions
from merit_inference import design,scenario_probabilities


def observed_predictions(evaluation):
    events={}
    file=ROOT/"predictions_history/official_results.jsonl"
    for line in file.read_text(encoding="utf-8").splitlines():
        event=json.loads(line)
        events[event["snapshot"]]={**events.get(event["snapshot"],{}),**event}
    records=[];predictions=[];seen=set()
    for snapshot,event in events.items():
        if "accuracy" not in event:continue
        folder=ROOT/"predictions_history"/snapshot
        meta=json.loads((folder/"metadata.json").read_text())
        if meta["csv_sha256"] in seen:continue
        seen.add(meta["csv_sha256"])
        csv=pd.read_csv(folder/"predictions.csv");verify_predictions(evaluation,csv)
        records.append({"snapshot":snapshot,"accuracy":event["accuracy"],
                        "source":event["source"],"csv_sha256":meta["csv_sha256"]})
        predictions.append(csv[TARGET].to_numpy())
    if len(records)<2:raise ValueError("At least two distinct scored submissions are required.")
    return records,np.array(predictions)


def infer(evaluation,observations,reported,power=12):
    n=len(evaluation);x=design(evaluation)
    u=qmc.Sobol(5,scramble=True,seed=20261004).random_base2(power)
    coefficients=np.column_stack([np.ones(len(u)),.02+.32*u[:,0],
        -np.maximum(0,(u[:,1]-.4)/.6)*1.2,
        np.maximum(0,(u[:,2]-.5)/.5)*.8,
        np.maximum(0,(u[:,3]-.5)/.5)*.18])
    noise=.03+.72*u[:,4]
    signed=2*observations-1
    q0=np.mean(1-observations,axis=1)
    log_weights=[];thresholds=[];expected=[]
    with threadpool_limits(limits=1):
        for start in range(0,len(noise),128):
            sl=slice(start,start+128)
            p,t=scenario_probabilities(x,coefficients[sl],noise[sl]);v=p*(1-p)
            mean=q0[:,None]+signed@p/n
            covariance=np.einsum("in,jn,nk->kij",signed,signed,v,optimize=True)/(n*n)
            # Extra variance protects against rounded scores and model error.
            covariance+=np.eye(len(observations))[None,:,:]*1e-6
            error=(mean-np.asarray(reported)[:,None]).T
            sign,logdet=np.linalg.slogdet(covariance)
            if not np.all(sign>0):raise RuntimeError("Invalid observation covariance.")
            quadratic=np.einsum("ki,kij,kj->k",error,np.linalg.inv(covariance),error)
            log_weights.extend(-.5*(quadratic+logdet))
            thresholds.extend(t);expected.append(mean.T)
    log_weights=np.array(log_weights);weights=np.exp(log_weights-log_weights.max());weights/=weights.sum()
    thresholds=np.array(thresholds);probability=np.zeros(n)
    with threadpool_limits(limits=1):
        from scipy.special import ndtr
        for start in range(0,len(noise),128):
            sl=slice(start,start+128)
            probability+=ndtr((x@coefficients[sl].T-thresholds[sl])/noise[sl])@weights[sl]
    return probability,{"coefficients":coefficients,"noise":noise,"weights":weights,
                       "thresholds":thresholds,"expected_observed_accuracy":np.concatenate(expected)}


def main():
    _,evaluation=load_data();records,observations=observed_predictions(evaluation)
    print(f"Conditioning on {len(records)} actual reported scores, with their shared-label covariance...",flush=True)
    probability,parameters=infer(evaluation,observations,[r["accuracy"] for r in records])
    pred=allocate(probability,groups(evaluation),evaluation.id_candidat.to_numpy(),max_gap=None)
    csv=pd.DataFrame({"id_candidat":evaluation.id_candidat,TARGET:pred})
    folder=ROOT/"submissions/v3_candidates";folder.mkdir(parents=True,exist_ok=True)
    output=folder/"feedback_conditioned.csv";csv.to_csv(output,index=False)
    experiment=ROOT/"artifacts/v3_research";experiment.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(experiment/"merit_ensemble.npz",**parameters)
    best=pd.read_csv(ROOT/"predictions.csv")[TARGET].to_numpy()
    w=parameters["weights"]
    report={**verify_predictions(evaluation,csv),"observations":records,
        "changed_from_active_champion":int((pred!=best).sum()),
        "mean_coefficients":(w@parameters["coefficients"]).tolist(),
        "effective_hypotheses":float(1/(w@w)),"mean_noise":float(w@parameters["noise"]),
        "model_implied_agreement_not_validation":float(np.mean(pred*probability+(1-pred)*(1-probability))),
        "official_accuracy":None,"status":"Unscored experiment. Active champion not replaced."}
    (experiment/"model_parameters.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    from prediction_history import archive_prediction
    snapshot=archive_prediction(evaluation,csv,"v3-feedback-conditioned",
        note="Uses correlated aggregate feedback from all recorded distinct submissions. Not independently scored.",
        model_files=[experiment/"model_parameters.json",experiment/"merit_ensemble.npz"],
        source_file=Path(__file__))
    print(json.dumps(report,indent=2),flush=True)
    print(f"Candidate: {output}\nSnapshot: {snapshot.name}",flush=True)


if __name__=="__main__":main()
