"""Additive nonlinear academic/work signal, controlling committee bias terms."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import SplineTransformer,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,cross_val_score
from threadpoolctl import threadpool_limits
from data_contract import ROOT,TARGET,load_data,groups,allocate,verify_predictions
from prediction_history import archive_prediction

def main():
    train,evaluation=load_data()
    spline=SplineTransformer(n_knots=5,degree=3,include_bias=False)
    cols=['cote_r_equivalent','heures_travail_semaine']
    z=spline.fit_transform(train[cols]);ze=spline.transform(evaluation[cols])
    def nuisance(d):
        return np.c_[np.log(d.revenu_familial_estime/70000),groups(d),
             d.distance_domicile_campus_km/200,d.premiere_generation_universitaire,
             pd.get_dummies(d.programme_etudes,dtype=float).to_numpy()[:,1:]]
    x=np.c_[z,nuisance(train)];xe=np.c_[ze,nuisance(evaluation)]
    scaler=StandardScaler().fit(x);x=scaler.transform(x)
    reports=[]
    with threadpool_limits(limits=1):
        for c in [.1,1.,10.]:
            model=LogisticRegression(C=c,max_iter=3000).fit(x,train[TARGET])
            # Remove committee income/region/program/commute/first-generation terms.
            coefficient=model.coef_[0]/scaler.scale_
            s=ze@coefficient[:z.shape[1]]
            pred=allocate(s,groups(evaluation),evaluation.id_candidat.to_numpy(),max_gap=None)
            frame=pd.DataFrame({'id_candidat':evaluation.id_candidat,TARGET:pred})
            folder=ROOT/'submissions/nonlinear';folder.mkdir(exist_ok=True,parents=True)
            name=f'additive_spline_C{c:g}';file=folder/(name+'.csv');frame.to_csv(file,index=False)
            parameters={'C':c,'coefficients':coefficient.tolist(),'knots':[b.t.tolist() for b in spline.bsplines_]}
            parameter_file=folder/(name+'.json');parameter_file.write_text(json.dumps(parameters,indent=2)+'\n')
            version=archive_prediction(evaluation,frame,name,source_file=Path(__file__),model_files=[parameter_file],
                note='Nonlinear academic/work committee signal with bias terms removed; unverified merit hypothesis.')
            reports.append({'file':str(file.relative_to(ROOT)),'snapshot':version.name,
                'changed_from_champion':int((pred!=pd.read_csv(ROOT/'predictions.csv')[TARGET]).sum()),**verify_predictions(evaluation,frame)})
    (folder/'index.json').write_text(json.dumps(reports,indent=2)+'\n')
    print(json.dumps(reports,indent=2))
if __name__=='__main__':main()
