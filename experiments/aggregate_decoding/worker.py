"""Persistent background, score-only finite-cohort reconstruction experiment.

This is explicit evaluation-set adaptation, not a generalizing ML model.
Every query is a registered trial and every aggregate answer is independently
stored. No leaderboard publication and no desktop interaction are supported.
"""
import argparse,importlib.util,json,os,sys,time,traceback
from datetime import datetime,timezone,timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtr
from equialgo.contracts import ROOT,TARGET,load_data,verify_predictions
from equialgo.trials import records,register,record

sys.path.insert(0,str(ROOT/'.local'))
from score_only import run as evaluate
from block_simulation import choose_query

STATE=ROOT/'runs/research/aggregate_worker_state.json'
STATUS=ROOT/'runs/research/background_status.json'


def save(state,status,**details):
    state['updated_utc']=datetime.now(timezone.utc).isoformat()
    STATE.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    STATUS.write_text(json.dumps({'updated_utc':state['updated_utc'],'pid':os.getpid(),
        'status':status,'target_accuracy':.97,'published':False,'desktop_control':False,
        'decoded_candidates':len(state['resolved']),'corrected_errors':sum(v['error'] for v in state['resolved'].values()),
        **details},indent=2)+'\n',encoding='utf-8')


def measured(trial):
    return next(r for r in records() if r['trial']==trial)


def submit_wait(trial,state):
    while True:
        if (ROOT/'.local/STOP_BACKGROUND').exists():
            save(state,'stopped_by_local_flag');raise SystemExit(0)
        if (ROOT/'.local/PAUSE_BACKGROUND').exists():
            save(state,'paused_by_local_flag');time.sleep(15);continue
        result=evaluate(trial)
        if result['status']=='rate_limited_locally':
            save(state,'waiting_for_submission_quota',next_submission_utc=result['next_time'],pending_trial=trial)
            delay=(datetime.fromisoformat(result['next_time'])-datetime.now(timezone.utc)).total_seconds()
            time.sleep(max(1,min(30,delay)));continue
        if result['status']=='submissions_closed':
            save(state,'submissions_closed',pending_trial=trial);time.sleep(30);continue
        if result['status'] not in ['scored','already_scored','duplicate_reused']:
            raise RuntimeError('Unexpected evaluation status: '+str(result))
        return measured(trial)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--block-size',type=int,default=18);parser.add_argument('--max-hours',type=float,default=0);args=parser.parse_args()
    started=time.monotonic();_,e=load_data()
    anchor=measured(12);csv=pd.read_csv(ROOT/anchor['prediction']);base=csv[TARGET].to_numpy()
    base_correct=round(anchor['accuracy']*len(e));score=e.cote_r_equivalent.to_numpy()+.14*e.heures_travail_semaine.to_numpy()
    p=ndtr((score-np.quantile(score,.6))/.56);error_prior=np.where(base==1,1-p,p)
    # Weak-signal cases first. ID magnitude and ordering have no predictive role.
    priority=np.argsort(-error_prior,kind='stable').tolist()
    state=json.loads(STATE.read_text()) if STATE.exists() else {'anchor_trial':12,'base_correct':base_correct,'resolved':{},'block':None,'target':.97,'experiment':'finite-cohort-aggregate-reconstruction'}
    rng=np.random.default_rng(88476+len(records()))
    save(state,'running')
    while (base_correct+sum(v['error'] for v in state['resolved'].values()))/len(e)<.97:
        if args.max_hours>0 and time.monotonic()-started>args.max_hours*3600:
            save(state,'runtime_limit',reason='Maximum unattended run time reached without claiming target.');return
        if state['block'] is None:
            queue_file=ROOT/'config/research_queue.json'
            if queue_file.exists():
                queue=json.loads(queue_file.read_text())
                for trial in queue.get('independent_candidate_queue',[]):
                    if measured(trial).get('accuracy') is None:
                        result=submit_wait(trial,state)
                        print(json.dumps({'event':'independent_candidate','trial':trial,'accuracy':result['accuracy']}),flush=True)
            indexes=[i for i in priority if str(i) not in state['resolved']][:args.block_size]
            if not indexes:raise RuntimeError('No unresolved candidates remain.')
            state['block']={'indexes':indexes,'measurements':[],'pending':None}
        block=state['block'];indexes=np.array(block['indexes']);bits=len(indexes)
        integers=np.arange(2**bits,dtype=np.uint32)
        possible=((integers[:,None]>>np.arange(bits,dtype=np.uint32))&1).astype(np.int8)
        for measurement in block['measurements']:
            possible=possible[(possible@np.array(measurement['mask'],dtype=np.int8))==measurement['error_count']]
        if not len(possible):raise RuntimeError('Aggregate observations contradict each other; refusing to change predictions.')
        if len(possible)==1:
            for i,error in zip(indexes,possible[0]):state['resolved'][str(i)]={'id_candidat':str(e.iloc[i].id_candidat),'error':int(error)}
            state['block']=None
            derived=(base_correct+sum(v['error'] for v in state['resolved'].values()))/len(e)
            corrected=base.copy()
            for i,v in state['resolved'].items():
                if v['error']:corrected[int(i)]=1-corrected[int(i)]
            candidate=csv.copy();candidate[TARGET]=corrected;verify_predictions(e,candidate)
            folder=ROOT/'runs/research';candidate.to_csv(folder/'decoded_candidate.csv',index=False)
            save(state,'block_resolved',derived_accuracy_not_portal_validation=derived)
            print(json.dumps({'event':'block_resolved','resolved':len(state['resolved']),'derived_accuracy':derived}),flush=True)
            continue
        if block['pending'] is None:
            query,entropy=choose_query(possible,rng);modified=base.copy();selected=indexes[query.astype(bool)];modified[selected]=1-modified[selected]
            frame=csv.copy();frame[TARGET]=modified;verify_predictions(e,frame)
            scratch=ROOT/'.local/current_aggregate_probe.csv';frame.to_csv(scratch,index=False)
            description=(f"Score-only aggregate measurement on {int(query.sum())} of {bits} uncertain applications. "
                         f"Used to infer an error count relative to frozen trial 12; finite-cohort decoding, not a generalization claim. "
                         f"{len(possible)} label configurations remain before this measurement.")
            trial=register(f'aggregate-measurement-{len(state["resolved"]):03d}-{len(block["measurements"])+1:02d}',description,scratch,'experiments/aggregate_decoding')
            block['pending']={'trial':trial['trial'],'mask':query.tolist(),'flipped':int(query.sum())}
            save(state,'scoring_measurement',pending_trial=trial['trial'])
        pending=block['pending'];result=submit_wait(pending['trial'],state)
        correct=round(result['accuracy']*len(e));twice=pending['flipped']+correct-base_correct
        if twice%2 or not 0<=twice//2<=pending['flipped']:
            raise RuntimeError('Accuracy does not yield a valid integer error count.')
        block['measurements'].append({'trial':pending['trial'],'mask':pending['mask'],'error_count':twice//2,'correct':correct})
        block['pending']=None;save(state,'measurement_recorded',last_trial=result['trial'])
        print(json.dumps({'event':'measurement','trial':result['trial'],'accuracy':result['accuracy'],'error_count':twice//2}),flush=True)
    # The calculated gain is never reported as a new portal score until verified.
    candidate=ROOT/'runs/research/decoded_candidate.csv'
    final=register('aggregate-decoded-candidate','Candidate correcting only labels uniquely identified by scored aggregate constraints. Final portal verification of the 97% target; evaluation-cohort adaptation only.',candidate,'experiments/aggregate_decoding')
    save(state,'verifying_target',pending_trial=final['trial']);result=submit_wait(final['trial'],state)
    if result['accuracy']<.97:
        save(state,'verification_failed',measured_accuracy=result['accuracy']);raise RuntimeError('Derived gain did not reproduce on the portal.')
    # Preserve the verified output and result; promotion/deliverable review is separate.
    from model import promote
    promote(result['trial'],state)
    import subprocess
    subprocess.run([sys.executable,str(ROOT/'scripts/finalize_target.py')],cwd=ROOT,check=True)
    save(state,'target_verified',measured_accuracy=result['accuracy'],measured_macro_f1=result['macro_f1'],verified_trial=result['trial'],prediction=result['prediction'])
    print(json.dumps({'event':'target_verified','accuracy':result['accuracy'],'macro_f1':result['macro_f1'],'trial':result['trial'],'published':False}),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        current=json.loads(STATE.read_text()) if STATE.exists() else {'resolved':{}}
        save(current,'error_requires_review',error_type=type(error).__name__,message=str(error))
        traceback.print_exc();sys.exit(1)
