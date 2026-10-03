"""Check exact feature-generation structure, without reference labels."""
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
from equialgo.contracts import ROOT,load_data,groups


def main():
    h,e=load_data();d=pd.concat([h,e]).sort_values('id_candidat');r=d.cote_r_equivalent.to_numpy()
    seeds=[0,1,7,13,17,21,42,64,87,100,123,456,789,2024,2025,2026,1337,1234,12345,31415,4242,20261003]
    started=time.monotonic();matches=[]
    # First five candidates share a region, so their affine normal signature
    # avoids assuming a group shift or exact mean and standard deviation.
    for kind in ['generator','randomstate']:
        for seed in seeds:
            rng=np.random.default_rng(seed) if kind=='generator' else np.random.RandomState(seed)
            stream=rng.standard_normal(750000)
            delta=stream[1:]-stream[:-1]
            good=(delta<-.4)&(delta>-3)
            pos=np.flatnonzero(good[:-5]);scale=(r[1]-r[0])/delta[pos]
            keep=(scale>2.8)&(scale<3.2);pos=pos[keep];scale=scale[keep]
            shift=r[0]-scale*stream[pos]
            keep=(shift>27.5)&(shift<28.5);pos=pos[keep];scale=scale[keep];shift=shift[keep]
            for index in range(2,5):
                keep=np.abs(shift+scale*stream[pos+index]-r[index])<.018
                pos=pos[keep];scale=scale[keep];shift=shift[keep]
            for offset,sd,mean in zip(pos,scale,shift):
                matches.append({'generator':kind,'seed':seed,'normal_offset':int(offset),'mean':float(mean),'sd':float(sd)})
    result={'elapsed_seconds':time.monotonic()-started,'matches':matches,
            'conclusion':'No identifiable shared normal stream in the bounded search.' if not matches else 'Candidate streams require exact feature validation.'}
    folder=ROOT/'runs/research';folder.mkdir(exist_ok=True,parents=True)
    (folder/'synthetic_structure.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
