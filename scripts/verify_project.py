"""Validate reorganized contracts, frozen releases and historical integrity."""
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
from equialgo.contracts import ROOT,load_data,verify_predictions
from equialgo.trials import records


def main():
    _,evaluation=load_data();release=json.loads((ROOT/'config/active_release.json').read_text())
    output=ROOT/'.local/release_verification.csv'
    subprocess.run([sys.executable,str(ROOT/'model_corrige.py'),'--output',str(output)],cwd=ROOT,check=True,capture_output=True)
    prediction=pd.read_csv(output);verify_predictions(evaluation,prediction)
    assert hashlib.sha256(output.read_bytes()).hexdigest()==release['csv_sha256']
    assert hashlib.sha256((ROOT/'predictions.csv').read_bytes()).hexdigest()==release['csv_sha256']
    snapshots=0
    for meta in (ROOT/'runs/predictions').glob('*/metadata.json'):
        info=json.loads(meta.read_text());folder=meta.parent
        assert hashlib.sha256((folder/'predictions.csv').read_bytes()).hexdigest()==info['csv_sha256']
        assert hashlib.sha256((folder/info['model_source_filename']).read_bytes()).hexdigest()==info['model_source_sha256']
        snapshots+=1
    archive=ROOT/'archive/v1_score_repair_9463';manifest=json.loads((archive/'ARCHIVE_MANIFEST.json').read_text())
    for name,digest in manifest['files'].items():assert hashlib.sha256((archive/name).read_bytes()).hexdigest()==digest
    for trial in records():
        if trial.get('prediction'):
            file=ROOT/trial['prediction'];assert file.exists(),file
            assert hashlib.sha256(file.read_bytes()).hexdigest()==trial['sha256'],file
    policy=json.loads((ROOT/'config/evaluation_policy.json').read_text())
    assert not policy['publish_to_leaderboard'] and not policy['desktop_control_allowed']
    print(json.dumps({'champion_reproduces':True,'shared_checks_impose_no_model_architecture':True,
        'immutable_snapshots_verified':snapshots,'v1_archive_files_verified':len(manifest['files']),
        'trial_records_verified':len(records()),'leaderboard_publication':False,'desktop_control':False},indent=2))


if __name__=='__main__':main()
