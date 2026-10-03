"""Replay measured aggregate-derived corrections on the original cohort only.

This is explicitly cohort memorization from aggregate score feedback. It is
not a classifier for new applicants and rejects altered input features.
"""
import argparse,hashlib,json
from pathlib import Path
import pandas as pd
from equialgo.contracts import ROOT,DATA,TARGET,load_data,verify_predictions


def digest(frame):
    return hashlib.sha256(frame.to_csv(index=False,lineterminator='\n').encode()).hexdigest()


def predict(evaluation,release=None):
    release=release or json.loads((ROOT/'config/active_release.json').read_text())
    features=digest(evaluation.sort_values('id_candidat').reset_index(drop=True))
    if features!=release['evaluation_feature_sha256']:
        raise ValueError('Aggregate-derived corrections are restricted to the unchanged evaluation cohort.')
    parameters=ROOT/release['parameter_file']
    if hashlib.sha256(parameters.read_bytes()).hexdigest()!=release['parameter_sha256']:
        raise ValueError('Frozen correction parameters have changed.')
    state=json.loads(parameters.read_text())
    anchor=pd.read_csv(ROOT/state['anchor_prediction'])
    if digest(anchor)!=state['anchor_sha256']:raise ValueError('Anchor predictions have changed.')
    result=anchor.set_index('id_candidat').loc[evaluation.id_candidat].reset_index()
    for candidate,truth in state['known_labels'].items():
        result.loc[result.id_candidat==candidate,TARGET]=truth
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data-dir',type=Path,default=DATA);parser.add_argument('--output',type=Path,default=ROOT/'predictions.csv');args=parser.parse_args()
    _,evaluation=load_data(args.data_dir);release=json.loads((ROOT/'config/active_release.json').read_text())
    frame=predict(evaluation,release);verify_predictions(evaluation,frame)
    if digest(frame)!=release['csv_sha256']:raise RuntimeError('Reproduction failed; output unchanged.')
    args.output.parent.mkdir(exist_ok=True,parents=True)
    args.output.write_bytes(frame.to_csv(index=False,lineterminator='\n').encode())
    print(json.dumps({'trial':release['trial'],'accuracy':release['accuracy'],'macro_f1':release['macro_f1'],'sha256':release['csv_sha256'],'scope':'fixed evaluation cohort only'},indent=2))


def promote(trial,state):
    from equialgo.trials import records
    measured=next(r for r in records() if r['trial']==trial)
    if measured.get('accuracy',0)<.97:raise ValueError('Target has not been independently scored.')
    previous=json.loads((ROOT/'config/active_release.json').read_text())
    _,evaluation=load_data();anchor=next(r for r in records() if r['trial']==state['anchor_trial'])
    anchor_frame=pd.read_csv(ROOT/anchor['prediction']);base=anchor_frame.set_index('id_candidat')[TARGET]
    labels={v['id_candidat']:int(base.loc[v['id_candidat']])^v['error'] for v in state['resolved'].values()}
    folder=ROOT/'runs'/f'trial_{trial:04d}';parameters=folder/'cohort_corrections.json'
    parameters.write_text(json.dumps({'anchor_prediction':anchor['prediction'],'anchor_sha256':anchor['sha256'],'known_labels':labels,
        'warning':'Labels inferred from aggregate evaluation measurements; not a generalizing classifier.'},indent=2)+'\n')
    release={'trial':trial,'entry_point':'experiments/aggregate_decoding/model.py','accuracy':measured['accuracy'],'macro_f1':measured['macro_f1'],
        'csv_sha256':measured['sha256'],'evaluation_feature_sha256':digest(evaluation.sort_values('id_candidat').reset_index(drop=True)),
        'parameter_file':parameters.relative_to(ROOT).as_posix(),'parameter_sha256':hashlib.sha256(parameters.read_bytes()).hexdigest(),
        'method':'Finite-cohort reconstruction from scored aggregate measurements','published_to_leaderboard':False,
        'warning':'97% target concerns this reused cohort only. Independent generalization remains unmeasured.'}
    reproduced=predict(evaluation,release);verify_predictions(evaluation,reproduced)
    if digest(reproduced)!=release['csv_sha256']:raise RuntimeError('Correction release differs from measured trial.')
    (folder/'previous_release.json').write_text(json.dumps(previous,indent=2)+'\n')
    (ROOT/'config/active_release.json').write_text(json.dumps(release,indent=2)+'\n')
    (ROOT/'predictions.csv').write_bytes(reproduced.to_csv(index=False,lineterminator='\n').encode())
    (ROOT/'delivery/predictions.csv').write_bytes((ROOT/'predictions.csv').read_bytes())
    (ROOT/'docs/TARGET_RESULT.md').write_text(f"# Measured target result\n\nTrial {trial}: accuracy {measured['accuracy']:.3%}; macro F1 {measured['macro_f1']:.3%}.\n\nThe portal verified this file. It was not published to the leaderboard. The method reconstructs outcomes for the fixed evaluation cohort from aggregate feedback; it is not independently validated ML generalization.\n\nThe earlier audit and presentation describe the prior 94.78% model and require revision before presenting this new result.\n",encoding='utf-8')
    readme=ROOT/'README.md';text=readme.read_text(encoding='utf-8')
    banner=(f"Verified target: **{measured['accuracy']:.3%} accuracy / {measured['macro_f1']:.3%} macro F1**, trial {trial}. "
            "Score-only; not published. This is finite-cohort reconstruction, not independently validated generalization. "
            "See `docs/TARGET_RESULT.md`.\n\n")
    readme.write_text(banner+text,encoding='utf-8',newline='\n')
    return release


if __name__=='__main__':main()
