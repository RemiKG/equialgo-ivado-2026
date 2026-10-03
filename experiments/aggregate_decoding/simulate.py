"""Simulation of designed aggregate supervision; no portal queries here."""
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import Bounds,LinearConstraint,milp
from scipy.special import ndtr,logit
from threadpoolctl import threadpool_limits
from equialgo.contracts import ROOT,load_data,TARGET


def decode(a,b,prior,time_limit=10):
    objective=-logit(np.clip(prior,.01,.99))
    result=milp(objective,integrality=np.ones(len(prior)),bounds=Bounds(0,1),
                constraints=LinearConstraint(a,b,b),options={'time_limit':time_limit,'mip_rel_gap':0})
    if result.x is None:return None,{'status':result.message}
    pred=np.rint(result.x).astype(int)
    return pred,{'status':result.message,'gap':float(result.mip_gap),'constraint_residual':float(np.max(abs(a@pred-b)))}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pool',type=int,default=256);args=parser.parse_args()
    _,e=load_data();champ=pd.read_csv(ROOT/'predictions.csv')[TARGET].to_numpy()
    s=e.cote_r_equivalent.to_numpy()+.14*e.heures_travail_semaine.to_numpy()
    t=np.quantile(s,.6);p=ndtr((s-t)/.56)
    pool=np.argsort(abs(p-.5))[:args.pool]
    rng=np.random.default_rng(1743);a=rng.binomial(1,.5,(90,args.pool)).astype(float)
    rows=[]
    with threadpool_limits(limits=1):
        for seed in [17,51,104]:
            truth=np.random.default_rng(seed).binomial(1,p[pool]);b=a@truth
            original_errors=int(np.sum(champ[pool]!=truth))
            for queries in [20,40,60,80]:
                start=time.monotonic();pred,info=decode(a[:queries],b[:queries],p[pool],time_limit=8)
                errors=None if pred is None else int((pred!=truth).sum())
                row={'pool':args.pool,'seed':seed,'queries':queries,'original_pool_errors':original_errors,
                     'remaining_pool_errors':errors,'net_corrected':None if errors is None else original_errors-errors,
                     'elapsed_seconds':time.monotonic()-start,**info}
                rows.append(row);print(json.dumps(row),flush=True)
    folder=ROOT/'runs/research';folder.mkdir(exist_ok=True,parents=True)
    (folder/f'aggregate_decoding_{args.pool}.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
