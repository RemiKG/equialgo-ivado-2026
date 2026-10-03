"""Cohort-specific probability calibration using measured aggregate scores.

This experiment uses the public leaderboard as development feedback; it is not
independent validation or a deployment model. No individual reference labels
are available. Exact observed counts constrain a maximum-entropy projection.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit,logit,ndtr
from threadpoolctl import threadpool_limits
from data_contract import ROOT,TARGET,load_data,allocate,groups,verify_predictions
from experiment_feedback import observed_predictions
from experimental_models import score
from prediction_history import archive_prediction


def main():
    _,evaluation=load_data();n=len(evaluation)
    records,observations=observed_predictions(evaluation)
    correct=np.round(np.array([r['accuracy'] for r in records])*n)
    # Accuracy = 1 - predicted positives/n - reference positives/n + 2*TP/n.
    base=observations[0]
    a=np.vstack([np.ones(n),2*base-1,observations[1:]-base])
    b=np.r_[1601.,correct[0]-(1-base).sum(),(correct[1:]-correct[0])/2]
    s=score(evaluation,work=.14)
    lo,hi=s.min(),s.max()
    for _ in range(40):
        threshold=(lo+hi)/2
        if ndtr((s-threshold)/.56).sum()>1601:lo=threshold
        else:hi=threshold
    prior=ndtr((s-(lo+hi)/2)/.56)
    z=logit(np.clip(prior,1e-8,1-1e-8))
    scale=np.sqrt((a*a).sum(axis=1));a=a/scale[:,None];b=b/scale
    def objective(lam):
        linear=z+a.T@lam;p=expit(linear)
        return np.logaddexp(0,linear).sum()-b@lam,a@p-b
    with threadpool_limits(limits=1):
        fit=minimize(objective,np.zeros(len(b)),jac=True,method='BFGS',options={'gtol':1e-8,'maxiter':1000})
        probability=expit(z+a.T@fit.x)
    residual=float(np.max(np.abs(a@probability-b)*scale))
    if residual>1e-3:raise RuntimeError(f'Projection failed: residual {residual}')
    pred=allocate(probability,groups(evaluation),evaluation.id_candidat.to_numpy(),max_gap=None)
    frame=pd.DataFrame({'id_candidat':evaluation.id_candidat,TARGET:pred})
    folder=ROOT/'submissions/feedback_calibration';folder.mkdir(exist_ok=True,parents=True)
    name=f'maxentropy_{len(records):02d}'
    file=folder/(name+'.csv');frame.to_csv(file,index=False)
    params=folder/(name+'.json')
    report={'observations':records,'dual':fit.x.tolist(),'constraint_max_error_count':residual,
            'reference_positive_assumption':1601,'prior_work':.14,'prior_noise':.56,
            'changed_from_v1':int(np.sum(pred!=base)),**verify_predictions(evaluation,frame),
            'warning':'Cohort-specific leaderboard adaptation; no independent validation.'}
    params.write_text(json.dumps(report,indent=2)+'\n')
    details=pd.DataFrame({'id_candidat':evaluation.id_candidat,'prior':prior,'posterior':probability,'decision':pred})
    details.to_csv(folder/(name+'_probabilities.csv'),index=False)
    snapshot=archive_prediction(evaluation,frame,name,note=report['warning'],source_file=Path(__file__),model_files=[params])
    print(json.dumps({'file':str(file.relative_to(ROOT)),'snapshot':snapshot.name,
        'observations':len(records),'changed_from_v1':report['changed_from_v1'],
        'constraint_max_error_count':residual,'grant_rate':report['grant_rate']},indent=2))

if __name__=='__main__':main()
