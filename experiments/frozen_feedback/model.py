"""Reproduce the measured champion using its frozen feedback calibration.

This is an evaluation-cohort experiment, not a validated deployment model.
The prior and observation set are frozen: later results cannot alter a release.
V1 and rejected V2 methods are preserved under archive/.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit, ndtr
from equialgo.contracts import (ROOT, DATA, TARGET, groups, load_data, allocate,
                           validate_data, verify_predictions)


def csv_digest(frame):
    return hashlib.sha256(frame.to_csv(index=False,lineterminator='\n').encode()).hexdigest()


def feature_digest(frame):
    return csv_digest(frame.sort_values('id_candidat').reset_index(drop=True))


def champion_probabilities(evaluation, release=None):
    """Replay the frozen calibration, rejecting unvalidated cohorts or features."""
    validate_data(evaluation)
    release=release or json.loads((ROOT/'config/active_release.json').read_text())
    if feature_digest(evaluation)!=release['evaluation_feature_sha256']:
        raise ValueError('This calibration is valid only for the unchanged evaluation cohort.')
    folder=ROOT/'runs/predictions'/release['snapshot']
    metadata=json.loads((folder/'metadata.json').read_text())
    if metadata['csv_sha256']!=release['csv_sha256']:
        raise ValueError('Release and archived champion hashes differ.')
    parameter_files=list(folder.glob('maxentropy_*.json'))
    if len(parameter_files)!=1:raise ValueError('One frozen calibration parameter file is required.')
    parameter_file=parameter_files[0]
    if hashlib.sha256(parameter_file.read_bytes()).hexdigest()!=release['parameter_sha256']:
        raise ValueError('Frozen calibration parameters have changed.')
    params=json.loads(parameter_file.read_text())
    observations=[]
    for record in params['observations']:
        frame=pd.read_csv(ROOT/'runs/predictions'/record['snapshot']/'predictions.csv')
        if csv_digest(frame)!=record['csv_sha256']:
            raise ValueError('A feedback observation has changed: '+record['snapshot'])
        if set(frame.id_candidat)!=set(evaluation.id_candidat):
            raise ValueError('Feedback and evaluation candidate IDs differ.')
        observations.append(frame.set_index('id_candidat').loc[evaluation.id_candidat,TARGET].to_numpy())
    observations=np.array(observations);base=observations[0]
    a=np.vstack([np.ones(len(evaluation)),2*base-1,observations[1:]-base])
    a=a/np.sqrt((a*a).sum(axis=1))[:,None]
    score=evaluation.cote_r_equivalent.to_numpy()+params['prior_work']*evaluation.heures_travail_semaine.to_numpy()
    lo,hi=score.min(),score.max()
    for _ in range(40):
        threshold=(lo+hi)/2
        if ndtr((score-threshold)/params['prior_noise']).sum()>params['reference_positive_assumption']:
            lo=threshold
        else:hi=threshold
    prior=ndtr((score-(lo+hi)/2)/params['prior_noise'])
    z=logit(np.clip(prior,1e-8,1-1e-8))
    probability=expit(z+a.T@np.array(params['dual']))
    return probability


def predict(evaluation,release=None):
    probability=champion_probabilities(evaluation,release)
    selected=allocate(probability,groups(evaluation),evaluation.id_candidat.to_numpy(),max_gap=None)
    return pd.DataFrame({'id_candidat':evaluation.id_candidat.to_numpy(),TARGET:selected})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,default=DATA)
    parser.add_argument('--output',type=Path,default=ROOT/'predictions.csv')
    args=parser.parse_args();_,evaluation=load_data(args.data_dir)
    release=json.loads((ROOT/'config/active_release.json').read_text())
    frame=predict(evaluation,release);validation=verify_predictions(evaluation,frame)
    if csv_digest(frame)!=release['csv_sha256']:
        raise RuntimeError('Reproduction differs from the measured champion; output not replaced.')
    if args.output.exists() and csv_digest(pd.read_csv(args.output))!=release['csv_sha256']:
        raise RuntimeError('Existing output differs from the release; preserve it before replacing it.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_bytes(frame.to_csv(index=False,lineterminator='\n').encode())
    print(json.dumps({'snapshot':release['snapshot'],'measured_accuracy':release['accuracy'],
                     'measured_macro_f1':release['macro_f1'],'reproduced_csv_sha256':csv_digest(frame),
                     'validation':validation},indent=2))


if __name__=='__main__':main()
