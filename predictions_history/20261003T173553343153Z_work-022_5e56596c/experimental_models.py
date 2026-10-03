"""Preserved, explicit score hypotheses for controlled HxBuddy comparisons.

These are diagnostic candidates, not validated policy recommendations. Varying
one coefficient at a time separates failures hidden in a multi-feature change.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from data_contract import ROOT,TARGET,groups,load_data,allocate,verify_predictions
from prediction_history import archive_prediction


def score(frame,work=.1442,log_income=0.,first_generation=0.,commute=0.,remote=0.):
    return (frame.cote_r_equivalent.to_numpy()+work*frame.heures_travail_semaine.to_numpy()
            +log_income*np.log(frame.revenu_familial_estime.to_numpy()/70000)
            +first_generation*frame.premiere_generation_universitaire.to_numpy()
            +commute*frame.distance_domicile_campus_km.to_numpy()/100+remote*groups(frame))


def create(name,parameters):
    _,evaluation=load_data();s=score(evaluation,**parameters)
    pred=allocate(s,groups(evaluation),evaluation.id_candidat.to_numpy(),max_gap=None)
    frame=pd.DataFrame({'id_candidat':evaluation.id_candidat,TARGET:pred})
    folder=ROOT/'submissions/controlled_candidates';folder.mkdir(parents=True,exist_ok=True)
    file=folder/(name+'.csv');frame.to_csv(file,index=False)
    params_file=folder/(name+'.json');params_file.write_text(json.dumps(parameters,indent=2)+'\n',encoding='utf-8')
    version=archive_prediction(evaluation,frame,name,note='Controlled score hypothesis; not a policy recommendation or a scored improvement.',
                              source_file=Path(__file__),model_files=[params_file])
    champion=pd.read_csv(ROOT/'predictions.csv')[TARGET].to_numpy()
    return {'name':name,'file':str(file.relative_to(ROOT)),'parameters':parameters,'snapshot':version.name,
            'changed_from_active_champion':int((pred!=champion).sum()),**verify_predictions(evaluation,frame)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name');parser.add_argument('--work',type=float,default=.1442)
    parser.add_argument('--log-income',type=float,default=0);parser.add_argument('--first-generation',type=float,default=0)
    parser.add_argument('--commute',type=float,default=0);parser.add_argument('--remote',type=float,default=0)
    args=parser.parse_args()
    if args.name:
        print(json.dumps(create(args.name,{'work':args.work,'log_income':args.log_income,'first_generation':args.first_generation,
                                          'commute':args.commute,'remote':args.remote}),indent=2))
    else:
        configs={'work_022':{'work':.22},'income_positive':{'log_income':.8},
                 'income_negative':{'log_income':-.8},'firstgen_positive':{'first_generation':.6},
                 'firstgen_negative':{'first_generation':-.6},'commute_positive':{'commute':.15}}
        reports=[create(name,params) for name,params in configs.items()]
        (ROOT/'submissions/controlled_candidates/index.json').write_text(json.dumps(reports,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(reports,indent=2))


if __name__=='__main__':main()
