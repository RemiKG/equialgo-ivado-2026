"""Executed audit of the frozen measured champion and complete score history."""
import json,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from model_corrige import ROOT,TARGET,load_data,groups,champion_probabilities,predict,verify_predictions
from merit_inference import equal_opportunity_allocate
from report_progress import main as report_progress


def main():
    report_progress();history,evaluation=load_data();art=ROOT/'artifacts'
    release=json.loads((art/'active_prediction.json').read_text())
    prediction=predict(evaluation);pred=prediction[TARGET].to_numpy();g=groups(evaluation)
    probability=champion_probabilities(evaluation)
    old=pd.read_csv(ROOT/'archive/v1_score_repair_9463/predictions.csv')[TARGET].to_numpy()
    v1=json.loads((ROOT/'archive/v1_score_repair_9463/artifacts/audit_summary.json').read_text())
    validation=verify_predictions(evaluation,prediction)
    sweep=[]
    for tolerance in [1,.1,.075,.05,.03,.02,.01,.005,.002,.001]:
        selected=equal_opportunity_allocate(probability,g,evaluation.id_candidat.to_numpy(),tolerance=tolerance)
        tpr=[float((selected[g==i]*probability[g==i]).sum()/probability[g==i].sum()) for i in [0,1]]
        sweep.append({'assumed_eop_tolerance':tolerance,'model_implied_eop_gap':abs(tpr[0]-tpr[1]),
            'model_implied_agreement':float(np.mean(selected*probability+(1-selected)*(1-probability))),
            'measured_dp_gap':float(abs(selected[g==0].mean()-selected[g==1].mean())),
            'grants':int(selected.sum()),'changed_from_champion':int((selected!=pred).sum())})
    sweep=pd.DataFrame(sweep);sweep.to_csv(art/'pareto.csv',index=False)
    regions=evaluation.assign(v1=old,champion=pred).groupby('region_administrative').agg(
        n=('id_candidat','size'),v1_rate=('v1','mean'),champion_rate=('champion','mean'))
    regions.to_csv(art/'regional_rates.csv')
    intersections=evaluation.assign(decision=pred).groupby(['region_administrative','premiere_generation_universitaire','programme_etudes']).decision.agg(['size','mean']).reset_index()
    intersections['small_cell']=intersections['size']<30;intersections.to_csv(art/'intersection_audit.csv',index=False)
    changes=[]
    for name,mask in [('new_grantees',(pred==1)&(old==0)),('displaced',(pred==0)&(old==1))]:
        d=evaluation.loc[mask]
        changes.append({'group':name,'n':int(mask.sum()),'r_mean':float(d.cote_r_equivalent.mean()),
            'work_mean':float(d.heures_travail_semaine.mean()),'income_mean':float(d.revenu_familial_estime.mean()),
            'remote_rate':float(groups(d).mean())})
    pd.DataFrame(changes).to_csv(art/'decision_change_summary.csv',index=False)
    summary={'version':'feedback-calibration','team':'SOTA Overfitters','release':release,'submission':validation,
             'changed_from_v1':int((pred!=old).sum()),'official_accuracy':release['accuracy'],'official_macro_f1':release['macro_f1'],
             'official_reference_eop_gap':None,'initial_diagnostic':v1,'decision_changes':changes,
             'warning':'Evaluation cohort used for model development. Simulated opportunity is not independently measured fairness.'}
    (art/'audit_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (art/'submission_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    (art/'model_parameters.json').write_text(json.dumps({'method':release['method'],'active_release':release,
        'frozen_parameters':'predictions_history/'+release['snapshot']+'/maxentropy_11.json',
        'scope':'Exact evaluation cohort only; no validated deployment model.'},indent=2)+'\n')
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(10,4.6));d=sweep.sort_values('model_implied_eop_gap')
    ax.plot(d.model_implied_eop_gap*100,d.model_implied_agreement*100,'o-',color='#159987')
    ax.set(xlabel='MODEL-IMPLIED equal-opportunity gap (percentage points)',ylabel='MODEL-IMPLIED agreement (%)',
           title='Opportunity constraint sweep: sensitivity, not validation')
    fig.text(.5,.01,'All points allocate 1,600 grants. These are not measured portal scores or verified fairness metrics.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(art/'pareto_front.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.6));d=regions.copy();d.index=d.index.str.replace('Gaspesie-Iles-de-la-Madeleine','Gaspesie / Iles')
    (100*d[['v1_rate','champion_rate']]).plot.barh(ax=ax,color=['#8999aa','#159987'])
    ax.set(xlabel='Observed selection rate (%)',title='V1 and current champion on the same applicants');fig.tight_layout()
    fig.savefig(art/'regional_access.png',dpi=180);plt.close(fig)
    notebook(summary)
    print(json.dumps({'score':release['accuracy'],'macro_f1':release['macro_f1'],'changed_from_v1':summary['changed_from_v1'],'sweep_points':len(sweep)}))


def notebook(summary):
    md,code=nbformat.v4.new_markdown_cell,nbformat.v4.new_code_cell
    cells=[md('''# EquiAlgo — SOTA Overfitters

**Measured champion: 94.78% accuracy / 94.56% macro F1.** V1 measured 94.63% / 94.40%. Fourteen portal-scored attempts, including regressions, are preserved in `RESULTS.md` and `predictions_history/`.

This audit separates historical committee behavior, measured portal scores and hypothesis-dependent fairness simulations. The evaluated cohort influenced development. Neither 98% accuracy nor out-of-sample generalization has been demonstrated. HxBuddy does not calculate the complete IVADO judging rubric.

## Data and integrity

The supplied package contains 10,000 historical decisions and 4,000 evaluation applicants. Committee decisions are biased historical observations, not the independent merit reference. No external training data or pretrained prediction model is used.'''),
    code('''import json, hashlib
from pathlib import Path
import pandas as pd
from IPython.display import display, Image
from model_corrige import load_data, predict, verify_predictions, csv_digest
root=Path.cwd()
history,evaluation=load_data()
release=json.loads((root/'artifacts/active_prediction.json').read_text())
prediction=predict(evaluation)
assert csv_digest(prediction)==release['csv_sha256']
display(pd.DataFrame([verify_predictions(evaluation,prediction)]))
print('Champion SHA-256:',release['csv_sha256'])'''),
    md('''## Diagnose the historical allocation

The original audit used a fixed stratified 70/30 split and the supplied Random Forest. Historical grant rates were 48.37% in centres and 27.30% in remote regions. Standardizing by academic bands left a 13.92 percentage-point gap. These are descriptive associations, not causal estimates.

The baseline achieved 88.13% historical agreement and an 18.76-point holdout selection gap. Removing region left 18.05 points; removing postal code as well left 17.31 points. Distance, income and work retain geographic signal. Raw labels cannot establish equal opportunity against legitimate merit.

The complete original diagnostic, bootstrap intervals, proxy investigation and code remain frozen under `archive/v1_score_repair_9463/`.'''),
    code('''original=json.loads((root/'archive/v1_score_repair_9463/artifacts/audit_summary.json').read_text())
display(pd.DataFrame(original['holdout_metrics']).T[['historical_agreement','demographic_parity_gap','historical_tpr_gap']])
display(pd.read_csv(root/'archive/v1_score_repair_9463/artifacts/proxy_audit.csv'))'''),
    md('''## Change the inference method, preserve the failed tests

The first repair retained academic achievement and an adjusted work coefficient. V2 introduced stronger financial-need and contextual assumptions and fell to 92.23%. One-variable perturbations also mostly regressed. An additive spline model reached 94.73% / 94.50%.

The champion uses maximum-entropy calibration. Begin with an academic/work prior, then minimally change individual probability estimates so that the expected aggregate agreements match eleven measured submissions. Rank the resulting estimates and allocate exactly 1,600 grants. Constraints overlap, so their joint information can differ from simply averaging their decisions.

The prior uses work weight 0.14 and noise scale 0.56. The reference-positive count is assumed to be 1,601, compatible with rounded accuracy/F1 feedback; it is not directly observed. Parameters and the observation set are frozen. The executable fails on changed input features or a different cohort instead of silently claiming the calibration transfers.

This is cohort-specific feedback adaptation. Individual private reference labels were never available. Because comparison submissions include income, first-generation, commute and region effects, the calibrated output can inherit these dependencies; it is not group-blind or automatically legitimate for real scholarship decisions.'''),
    code('''scores=pd.read_csv(root/'artifacts/scored_attempts.csv')
display(scores[['attempt','model','accuracy_percent','macro_f1_percent']])
display(Image(filename=str(root/'artifacts/score_progress.png')))'''),
    md('''## Allocation and fairness

The champion selects 957 of 2,372 centre applicants (40.35%) and 643 of 1,628 remote applicants (39.50%). It changes 32 decisions from V1. The observed regional selection gap is 0.849 percentage points.

Our normative primary metric is equal opportunity under an independent, legitimate merit assessment. Its actual gap is unavailable. Demographic parity is an observable access diagnostic, not proof of fair treatment. Budget is a hard constraint. Small intersectional cells are marked rather than interpreted as precise disparities.'''),
    code('''display(pd.read_csv(root/'artifacts/regional_rates.csv'))
display(pd.read_csv(root/'artifacts/decision_change_summary.csv'))
display(Image(filename=str(root/'artifacts/regional_access.png')))
intersections=pd.read_csv(root/'artifacts/intersection_audit.csv')
display(intersections)
print('Small cells:',int(intersections.small_cell.sum()))'''),
    md('''## Opportunity / utility sensitivity

Ten tolerances constrain the true-positive-rate gap under the calibrated probability assumptions. Every point respects the same 40% budget. The plot is a Pareto sensitivity exercise, not a measured comparison of hidden-reference fairness or additional HxBuddy results. Cohort adaptation may make these probabilities optimistic. The independent equity component of the final rubric remains unknown.'''),
    code('''display(pd.read_csv(root/'artifacts/pareto.csv'))
display(Image(filename=str(root/'artifacts/pareto_front.png')))'''),
    md('''## Governance and monitoring

An independent panel must define legitimate merit and review geographic and socioeconomic effects before real use. Validate on fresh applicants with independently adjudicated outcomes; measure opportunity, false-positive rates, selection rates and intersections with uncertainty. Keep an appeal and data-correction channel, review rejected applicants as well as recipients, and monitor distribution shift. Paid work must not penalize disability or caring obligations.

Freeze releases with data/code hashes, reject schema/budget violations, and keep rollback artifacts. The current model deliberately rejects new cohorts. No deployment is proposed from leaderboard feedback alone. `MONITORING.md` defines responsibilities and release conditions.

## Reproduce and sources

Run `python model_corrige.py`, `python -m unittest discover -s tests -v`, `python build_audit.py`, `python build_presentation.py`, then `python package_submission.py`. Required input is the unmodified IVADO participant package.

Sources: supplied IVADO briefs and baseline notebook; participant-reported first three scores; directly observed HxBuddy scores thereafter; [HxBuddy](https://hxbuddy.ca/); [Fairlearn metric definitions](https://fairlearn.org/main/user_guide/assessment/common_fairness_metrics.html). Python/SciPy/scikit-learn/ReportLab/Jupyter used. OpenAI Codex assisted implementation, analysis and writing. No pretrained predictor or external training dataset was used.''')]
    nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    km=KernelManager(kernel_name='python3');km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    NotebookClient(nb,timeout=180,resources={'metadata':{'path':str(ROOT)}},km=km).execute()
    nbformat.write(nb,ROOT/'audit_rapport.ipynb')


if __name__=='__main__':main()
