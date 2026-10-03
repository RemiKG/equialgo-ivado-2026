"""Background-only sensitivity analysis. Never submits or replaces a champion.

Leave-one-submission-out checks assess aggregate feedback prediction, not
generalization to new applicants. All resulting candidates remain unscored.
"""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr,expit,logit
from threadpoolctl import threadpool_limits
from data_contract import ROOT,TARGET,load_data,groups,allocate
from experiment_feedback import observed_predictions
from prediction_history import archive_prediction


def projection(z,observations,correct):
    n=len(z);base=observations[0]
    a=np.vstack([np.ones(n),2*base-1,observations[1:]-base])
    b=np.r_[1601.,correct[0]-(1-base).sum(),(correct[1:]-correct[0])/2]
    scale=np.sqrt((a*a).sum(axis=1));a=a/scale[:,None];b=b/scale
    def objective(w):
        linear=z+a.T@w;p=expit(linear)
        return np.logaddexp(0,linear).sum()-b@w,a@p-b
    fit=minimize(objective,np.zeros(len(b)),jac=True,method='BFGS',options={'gtol':1e-7,'maxiter':800})
    p=expit(z+a.T@fit.x)
    residual=float(np.max(np.abs(a@p-b)*scale))
    if residual>.005:raise RuntimeError('Unconverged projection: '+str(residual))
    return p


def main():
    started=time.monotonic();_,evaluation=load_data();n=len(evaluation)
    records,observations=observed_predictions(evaluation)
    correct=np.round(np.array([r['accuracy'] for r in records])*n)
    folder=ROOT/'artifacts/offline_research';folder.mkdir(parents=True,exist_ok=True)
    rows=[];posteriors=[]
    with threadpool_limits(limits=1):
        for work in [.11,.14,.17]:
            for noise in [.4,.56,.72]:
                s=evaluation.cote_r_equivalent.to_numpy()+work*evaluation.heures_travail_semaine.to_numpy()
                lo,hi=s.min(),s.max()
                for _ in range(40):
                    t=(lo+hi)/2
                    if ndtr((s-t)/noise).sum()>1601:lo=t
                    else:hi=t
                prior=ndtr((s-(lo+hi)/2)/noise);z=logit(np.clip(prior,1e-8,1-1e-8))
                errors=[]
                for omitted in range(1,len(records)):
                    keep=np.arange(len(records))!=omitted
                    p=projection(z,observations[keep],correct[keep])
                    predicted=((1-observations[omitted])+(2*observations[omitted]-1)*p).sum()
                    errors.append(float(predicted-correct[omitted]))
                posterior=projection(z,observations,correct);posteriors.append(posterior)
                row={'work':work,'noise':noise,'loo_score_count_rmse':float(np.sqrt(np.mean(np.array(errors)**2))),
                     'omitted_score_errors_counts':errors}
                rows.append(row);print(json.dumps(row),flush=True)
    weights=np.exp(-np.array([r['loo_score_count_rmse'] for r in rows])/4);weights/=weights.sum()
    probability=weights@np.array(posteriors)
    pred=allocate(probability,groups(evaluation),evaluation.id_candidat.to_numpy(),max_gap=None)
    frame=pd.DataFrame({'id_candidat':evaluation.id_candidat,TARGET:pred})
    destination=ROOT/'submissions/offline_candidates';destination.mkdir(exist_ok=True,parents=True)
    frame.to_csv(destination/'robust_calibration.csv',index=False)
    report={'status':'Unscored offline candidate; active champion unchanged.','observations':records,
            'prior_comparisons':rows,'weights':weights.tolist(),'elapsed_seconds':time.monotonic()-started,
            'changed_from_champion':int((pred!=pd.read_csv(ROOT/'predictions.csv')[TARGET]).sum()),
            'warning':'Submission-level cross-validation is not independent applicant-level validation.'}
    params=folder/'model_parameters.json';params.write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(folder/'posteriors.npz',posteriors=posteriors,weights=weights,probability=probability)
    snapshot=archive_prediction(evaluation,frame,'offline-robust-calibration',source_file=Path(__file__),model_files=[params],note=report['warning'])
    print(json.dumps({'snapshot':snapshot.name,'changed_from_champion':report['changed_from_champion'],'elapsed_seconds':report['elapsed_seconds']}),flush=True)


if __name__=='__main__':main()
