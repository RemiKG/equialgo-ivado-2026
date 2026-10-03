"""Distinct latent-merit model with an explicit historical observation channel.

Fit a probit structural score for committee behavior. Use aggregate reference
agreements to estimate a separate merit head in a nonlinear feature basis.
No fixed coefficients from the previous model are reused as the final model.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr
from sklearn.preprocessing import SplineTransformer,StandardScaler
from threadpoolctl import threadpool_limits
from equialgo.contracts import ROOT,TARGET,load_data,groups,allocate
from equialgo.trials import records,register


def main():
    h,e=load_data();observed=[r for r in records() if r.get('accuracy') is not None]
    predictions=np.array([pd.read_csv(ROOT/r['prediction'])[TARGET].to_numpy() for r in observed])
    correct=np.round(4000*np.array([r['accuracy'] for r in observed]))
    s=e.cote_r_equivalent.to_numpy()+.14*e.heures_travail_semaine.to_numpy()
    # Separate merit effects and the committee's nuisance channel.
    def design(d):
        r=(d.cote_r_equivalent.to_numpy()-28)/3;w=(d.heures_travail_semaine.to_numpy()-10)/4
        return np.c_[np.ones(len(d)),r,w,w*w,r*w,
                     np.log(d.revenu_familial_estime/70000),d.premiere_generation_universitaire,
                     np.log1p(d.distance_domicile_campus_km)/3,groups(d),
                     pd.get_dummies(d.programme_etudes,dtype=float).to_numpy()[:,1:]]
    x=design(e);xh=design(h);y=h[TARGET].to_numpy();dimension=x.shape[1]
    def historical_loss(beta):
        z=xh@beta;p=np.clip(ndtr(z),1e-9,1-1e-9)
        normal=np.exp(-z*z/2)/np.sqrt(2*np.pi)
        loss=-np.sum(y*np.log(p)+(1-y)*np.log1p(-p))+.1*sum(beta[1:]**2)
        grad=xh.T@(normal*(p-y)/(p*(1-p)))+np.r_[0,.2*beta[1:]]
        return loss,grad
    with threadpool_limits(limits=1):
        fitted=minimize(historical_loss,np.zeros(dimension),jac=True,method='BFGS',options={'maxiter':1000})
    merit_prior=np.zeros(dimension);merit_prior[1:5]=fitted.x[1:5]
    # Numeric priority is estimated, not fixed to the original work coefficient.
    merit_prior/=max(merit_prior[1],1e-6)
    signed=2*predictions-1;constant=(1-predictions).sum(axis=1)
    def probability(beta):return ndtr(x@beta)
    initial=np.zeros(dimension);initial[:5]=merit_prior[:5]*5.4
    initial[0]=-(np.quantile(x@initial,.6))
    scales=np.array([1e-7,.05,.4,2,2,3,3,3,3,4,4,4,4])
    def fit(obs,counts):
        signs=2*obs-1;offset=(1-obs).sum(axis=1)
        def objective(beta):
            z=x@beta;p=ndtr(z);density=np.exp(-z*z/2)/np.sqrt(2*np.pi)
            residual=offset+signs@p-counts
            # Shared-label covariance retains the dependence across submissions.
            covariance=(signs*(p*(1-p)))@signs.T+np.eye(len(obs))*4
            solve=np.linalg.solve(covariance,residual)
            prior_delta=beta-initial
            value=.5*residual@solve+.5*sum(scales*prior_delta**2)
            # Covariance update is held fixed in this scoring step.
            gradient=x.T@(density*(signs.T@solve))+scales*prior_delta
            return value,gradient
        best=None
        for step in range(3):
            result=minimize(objective,initial if best is None else best.x,jac=True,method='L-BFGS-B',options={'maxiter':400,'ftol':1e-10})
            best=result
        return best.x
    with threadpool_limits(limits=1):
        beta=fit(predictions,correct);p=probability(beta)
    selected=allocate(p,groups(e),e.id_candidat.to_numpy(),max_gap=None)
    folder=ROOT/'experiments/latent_merit';output=folder/'candidate.csv'
    pd.DataFrame({'id_candidat':e.id_candidat,TARGET:selected}).to_csv(output,index=False)
    report={'historical_coefficients':fitted.x.tolist(),'merit_coefficients':beta.tolist(),
        'expected_existing_accuracy':((constant+signed@p)/4000).tolist(),
        'observed_accuracy':[r['accuracy'] for r in observed],
        'changed_from_champion':int(sum(selected!=pd.read_csv(ROOT/'predictions.csv')[TARGET].to_numpy())),
        'warning':'Aggregate feedback developed model; no independent validation.'}
    (folder/'fit.json').write_text(json.dumps(report,indent=2)+'\n')
    trial=register('latent-merit-observation-channel','Nonlinear probit merit head estimated from aggregate reference feedback; separate historical committee observation model provides only a structural prior.',output,'experiments/latent_merit')
    print(json.dumps({'trial':trial['trial'],**report},indent=2))


if __name__=='__main__':main()
