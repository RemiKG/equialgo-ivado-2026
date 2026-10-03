"""Graph-regularized learning from aggregate labels, as an independent method.

This adapts the principle of covariate consistency from LLP research. It uses
overlapping count constraints rather than claiming a reproduction of BP-LLP.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr,logit,expit
from scipy.sparse import coo_matrix,diags
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from equialgo.contracts import ROOT,TARGET,load_data,groups,allocate
from equialgo.trials import records,register


def main():
    _,e=load_data();n=len(e);observations=[r for r in records() if r.get('accuracy') is not None and r['trial']<=16]
    y=np.array([pd.read_csv(ROOT/r['prediction'])[TARGET].to_numpy() for r in observations]);correct=np.round(n*np.array([r['accuracy'] for r in observations]))
    # Outcome geometry is learned independently of the frozen feedback release.
    x=np.c_[e.cote_r_equivalent/3,e.heures_travail_semaine/4,np.log(e.revenu_familial_estime/70000)*.15,
            e.premiere_generation_universitaire*.1,np.log1p(e.distance_domicile_campus_km)*.03]
    neighbours=NearestNeighbors(n_neighbors=9).fit(x);distance,index=neighbours.kneighbors(x)
    rows=np.repeat(np.arange(n),8);cols=index[:,1:].ravel();weights=np.exp(-distance[:,1:].ravel()**2/.06)
    adj=coo_matrix((weights,(rows,cols)),shape=(n,n)).tocsr();adj=(adj+adj.T)*.5
    laplacian=diags(np.asarray(adj.sum(axis=1)).ravel())-adj
    a=np.vstack([np.ones(n),2*y[0]-1,y[1:]-y[0]])
    b=np.r_[1597.,correct[0]-(1-y[0]).sum(),(correct[1:]-correct[0])/2]
    score=e.cote_r_equivalent.to_numpy()+.14*e.heures_travail_semaine.to_numpy();threshold=np.quantile(score,.6)
    prior=ndtr((score-threshold)/.55);z=logit(np.clip(prior,1e-7,1-1e-7))
    scale=np.sqrt((a*a).sum(axis=1));a=a/scale[:,None];b=b/scale
    # Alternating graph consistency and a count-constrained entropy projection.
    p=prior.copy();dual=np.zeros(len(b));history=[]
    with threadpool_limits(limits=1):
        for step in range(12):
            graph_gradient=laplacian@p
            base=z-.7*graph_gradient
            def objective(lam):
                linear=base+a.T@lam;q=expit(linear)
                return np.logaddexp(0,linear).sum()-b@lam,a@q-b
            fit=minimize(objective,dual,jac=True,method='BFGS',options={'gtol':1e-7,'maxiter':600})
            dual=fit.x;q=expit(base+a.T@dual)
            delta=float(np.max(abs(q-p)));p=.5*p+.5*q
            history.append({'iteration':step,'max_change':delta})
    pred=allocate(p,groups(e),e.id_candidat.to_numpy(),max_gap=None)
    folder=ROOT/'experiments/graph_merit';file=folder/'candidate.csv'
    pd.DataFrame({'id_candidat':e.id_candidat,TARGET:pred}).to_csv(file,index=False)
    report={'iterations':history,'constraint_residual_count':float(np.max(abs(a@p-b)*scale)),
            'changed_from_champion':int(sum(pred!=pd.read_csv(ROOT/'predictions.csv')[TARGET].to_numpy())),
            'warning':'Geometry is a hypothesis; aggregate fit is not independent validation.'}
    (folder/'fit.json').write_text(json.dumps(report,indent=2)+'\n')
    trial=register('graph-consistent-aggregate-merit','Nearest-neighbour consistency with aggregate count constraints; independent graph hypothesis rather than a fixed linear merit architecture.',file,'experiments/graph_merit')
    print(json.dumps({'trial':trial['trial'],**report},indent=2))


if __name__=='__main__':main()
