"""Append-only experiment ledger; no assumptions about a model's architecture."""
import argparse
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import shutil
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd
from .contracts import ROOT,load_data,verify_predictions

LEDGER=ROOT/'runs/trials.jsonl'


def records():
    events=[json.loads(line) for line in LEDGER.read_text(encoding='utf-8').splitlines()] if LEDGER.exists() else []
    latest={}
    for event in events:latest[event['trial']]=dict(latest.get(event['trial'],{}),**event)
    return sorted(latest.values(),key=lambda r:r['trial'])


def record(event):
    event={'recorded_utc':datetime.now(timezone.utc).isoformat(),**event}
    LEDGER.parent.mkdir(exist_ok=True,parents=True)
    with LEDGER.open('a',encoding='utf-8',newline='\n') as f:f.write(json.dumps(event,ensure_ascii=False)+'\n')
    refresh()


def refresh():
    lines=['EQUIALGO TRIAL LOG','Target: at least 97.00% measured accuracy. Scores are evaluation-only; never publish to leaderboard.',
           'Format: trial number | name | description | actual result | status | preserved prediction','']
    for r in records():
        if r.get('accuracy') is not None:
            accuracy=(Decimal(str(r['accuracy']))*100).quantize(Decimal('.01'),rounding=ROUND_HALF_UP)
            f1=(Decimal(str(r['macro_f1']))*100).quantize(Decimal('.01'),rounding=ROUND_HALF_UP)
            result=f"Accuracy {accuracy}%; macro F1 {f1}% (raw {r['accuracy']}, {r['macro_f1']})"
        else:result='UNSCORED'
        lines += [f"TRIAL {r['trial']:04d} | {r['name']}",f"Description: {r['description']}",
                  f"Result: {result}",f"Status: {r['status']}",f"Prediction: {r.get('prediction','not generated')}",'']
    (ROOT/'TRIALS.txt').write_text('\n'.join(lines),encoding='utf-8',newline='\n')


def register(name,description,prediction=None,experiment=None,status='awaiting_score'):
    existing=records();number=max([r['trial'] for r in existing],default=0)+1
    folder=ROOT/'runs'/f'trial_{number:04d}';folder.mkdir(exist_ok=False)
    event={'trial':number,'name':name,'description':description,'status':status,'accuracy':None,'macro_f1':None}
    if prediction:
        _,evaluation=load_data();frame=pd.read_csv(prediction);validation=verify_predictions(evaluation,frame)
        payload=frame.to_csv(index=False,lineterminator='\n').encode()
        target=folder/'predictions.csv';target.write_bytes(payload)
        event.update(prediction=str(target.relative_to(ROOT)).replace('\\','/'),sha256=hashlib.sha256(payload).hexdigest(),validation=validation)
    if experiment:event['experiment']=str(experiment)
    (folder/'trial.json').write_text(json.dumps(event,indent=2)+'\n')
    record(event);return event


def score(trial,accuracy,macro_f1,source):
    if not 0<=accuracy<=1 or not 0<=macro_f1<=1:raise ValueError('Metrics must be fractions.')
    matching=[r for r in records() if r['trial']==trial]
    if len(matching)!=1:raise ValueError('Unknown trial.')
    record({'trial':trial,'accuracy':accuracy,'macro_f1':macro_f1,'source':source,'status':'scored'})


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('refresh')
    a=sub.add_parser('register');a.add_argument('--name',required=True);a.add_argument('--description',required=True);a.add_argument('--prediction',type=Path);a.add_argument('--experiment')
    b=sub.add_parser('score');b.add_argument('trial',type=int);b.add_argument('--accuracy',type=float,required=True);b.add_argument('--macro-f1',type=float,required=True);b.add_argument('--source',required=True)
    args=parser.parse_args()
    if args.command=='refresh':refresh()
    elif args.command=='register':print(json.dumps(register(args.name,args.description,args.prediction,args.experiment),indent=2))
    else:score(args.trial,args.accuracy,args.macro_f1,args.source)


if __name__=='__main__':main()
