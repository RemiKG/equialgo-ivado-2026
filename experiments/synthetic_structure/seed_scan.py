"""Bounded seed/offset diagnostic using only observed regional feature values."""
import argparse,json,time
import numpy as np
import pandas as pd
from equialgo.contracts import ROOT,load_data


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stop',type=int,default=100000);parser.add_argument('--offsets',type=int,default=64);args=parser.parse_args()
    h,e=load_data();d=pd.concat([h,e]).sort_values('id_candidat')
    regions=['Montreal','Capitale-Nationale','Bas-Saint-Laurent','Cote-Nord','Gaspesie-Iles-de-la-Madeleine']
    target=np.array([regions.index(r) for r in d.region_administrative]);weights=np.array([.4,.2,.13,.12,.15]);edges=np.r_[0,weights.cumsum()]
    started=time.monotonic();matches=[]
    for kind in ['generator','randomstate']:
        for seed in range(args.stop):
            rng=np.random.default_rng(seed) if kind=='generator' else np.random.RandomState(seed)
            draws=rng.random(args.offsets+32);positions=np.arange(args.offsets)
            for i,label in enumerate(target[:24]):
                u=draws[positions+i]
                positions=positions[(u>=edges[label])&(u<edges[label+1])]
                if not len(positions):break
            for offset in positions:
                matches.append({'generator':kind,'seed':seed,'offset':int(offset)})
                print(json.dumps(matches[-1]),flush=True)
            if seed%25000==0:print(json.dumps({'generator':kind,'processed':seed,'elapsed':time.monotonic()-started}),flush=True)
    result={'matches':matches,'seed_stop':args.stop,'offsets':args.offsets,'elapsed_seconds':time.monotonic()-started}
    (ROOT/'runs/research/seed_scan.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
