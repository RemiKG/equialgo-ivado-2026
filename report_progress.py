"""Write an honest, linked report of every measured submission."""
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_contract import ROOT,TARGET,load_data,verify_predictions


def main():
    history=ROOT/'predictions_history'
    events=[json.loads(line) for line in (history/'official_results.jsonl').read_text().splitlines()]
    records=[]
    for i,event in enumerate(events,1):
        meta=json.loads((history/event['snapshot']/'metadata.json').read_text())
        records.append({'attempt':i,'model':meta['model'],'accuracy_percent':100*event['accuracy'],
                        'macro_f1_percent':100*event['macro_f1'],'recorded_utc':event['recorded_utc'],
                        'snapshot':event['snapshot'],'source':event['source'],'csv_sha256':meta['csv_sha256']})
    frame=pd.DataFrame(records);frame.to_csv(ROOT/'artifacts/scored_attempts.csv',index=False)
    active=json.loads((ROOT/'artifacts/active_prediction.json').read_text())
    _,evaluation=load_data();validation=verify_predictions(evaluation,pd.read_csv(ROOT/'predictions.csv'))
    lines=['# Submission results','',f"Best verified result: **{active['accuracy']:.2%} accuracy / {active['macro_f1']:.2%} macro F1**.",
           'The indicative score equalled macro F1 for every scored attempt. First three scores were reported by the participant; subsequent scores were directly observed on HxBuddy.',
           '','| # | Attempt / preserved CSV | Accuracy | Macro F1 |','|---:|---|---:|---:|']
    for r in records:
        lines.append(f"| {r['attempt']} | [{r['model']}](predictions_history/{r['snapshot']}/predictions.csv) | {r['accuracy_percent']:.2f}% | {r['macro_f1_percent']:.2f}% |")
    lines += ['',f"Active champion: `{active['snapshot']}`. Root `predictions.csv` contains {validation['grants']:,} grants among {validation['rows']:,} applications.",
              'The original method scored 94.63% / 94.40%; improvement is 0.15 percentage points accuracy and 0.16 points macro F1.',
              'The last observed competing leader scored 94.73% / 94.50%. Our lead over that observation is narrow; the leaderboard has not been rechecked since desktop control was disabled.',
              '', 'Calibration uses aggregate leaderboard feedback on this evaluation cohort. These are real portal scores, but they do not establish out-of-sample generalization or 98% accuracy.',
              '', 'Unscored candidates and historical-label diagnostics are listed separately in [the complete history](predictions_history/README.md).',
              'Every snapshot preserves predictions, code and metadata; official results are append-only. Some exploratory code snapshots require the shared root modules.',
              '', 'Desktop automation is disabled at the participant\'s request. Background work does not interact with keyboard, mouse, windows, clipboard or browser UI.']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,ax=plt.subplots(figsize=(10,4.8));ax.plot(frame.attempt,frame.macro_f1_percent,'o-',color='#1b6d99',label='Measured macro F1')
    ax.plot(frame.attempt,frame.macro_f1_percent.cummax(),color='#138b75',linewidth=2,label='Best so far')
    ax.axhline(94.50,color='#ab6348',linestyle='--',label='Last observed competing leader')
    ax.set(xlabel='Scored attempt',ylabel='HxBuddy macro F1 (%)',title='Every measured result, including regressions',xticks=frame.attempt)
    ax.spines[['top','right']].set_visible(False);ax.legend();fig.tight_layout();fig.savefig(ROOT/'artifacts/score_progress.png',dpi=180);plt.close(fig)
    print(json.dumps({'scored_attempts':len(records),'active':active['snapshot'],'validation':validation},indent=2))


if __name__=='__main__':main()
