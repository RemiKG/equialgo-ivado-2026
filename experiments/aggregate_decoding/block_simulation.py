"""Query-count simulation with exact enumeration of small uncertain blocks."""
import json,time
import numpy as np
from scipy.special import logit
from equialgo.contracts import ROOT


def choose_query(possible,rng):
    # Balanced outcome buckets give high expected information per evaluation.
    bits=possible.shape[1];best=None;best_entropy=-1
    masks=rng.integers(0,2,(64,bits),dtype=np.int8)
    outcomes=possible@masks.T
    for j,mask in enumerate(masks):
        hist=np.bincount(outcomes[:,j].astype(int),minlength=bits+1);p=hist[hist>0]/len(possible)
        entropy=-float(p@np.log(p))
        if entropy>best_entropy:best_entropy=entropy;best=mask
    return best,best_entropy


def main():
    size=18;integers=np.arange(2**size,dtype=np.uint32)
    all_labels=((integers[:,None]>>np.arange(size,dtype=np.uint32))&1).astype(np.int8)
    rng=np.random.default_rng(9923);results=[];started=time.monotonic()
    for trial in range(20):
        truth=rng.integers(0,2,size,dtype=np.int8);possible=all_labels;queries=0
        while len(possible)>1:
            query,_=choose_query(possible,rng)
            answer=int(query@truth);possible=possible[(possible@query)==answer];queries+=1
        assert np.array_equal(possible[0],truth)
        results.append(queries)
    report={'block_size':size,'simulation_runs':20,'mean_queries':float(np.mean(results)),
            'min_queries':min(results),'max_queries':max(results),
            'projected_queries_for_216_labels':float(np.mean(results)*12),
            'elapsed_seconds':time.monotonic()-started,
            'caveat':'Synthetic random labels only; no claim about actual evaluation accuracy.'}
    (ROOT/'runs/research/exact_block_simulation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
