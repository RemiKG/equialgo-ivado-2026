"""Build the challenge handoff without imposing an experiment architecture."""
import hashlib,importlib.util,json,sys,zipfile
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
import pandas as pd
from equialgo.contracts import ROOT,load_data,verify_predictions


def notebook():
    nb=nbformat.read(ROOT/'audit_rapport.ipynb',as_version=4)
    for cell in nb.cells:
        if cell.cell_type=='code':
            cell.source=cell.source.replace('from model_corrige import load_data, predict, verify_predictions, csv_digest',
                'from equialgo.contracts import load_data, verify_predictions\nfrom experiments.frozen_feedback.model import predict, csv_digest')
            cell.source=cell.source.replace("root/'artifacts/active_prediction.json'","root/'experiments/frozen_feedback/release.json'")
            cell.source=cell.source.replace("root/'config/active_release.json'","root/'experiments/frozen_feedback/release.json'")
            cell.source=cell.source.replace('prediction=predict(evaluation)','prediction=predict(evaluation,release)')
            cell.source=cell.source.replace("root/'artifacts/","root/'runs/legacy_artifacts/")
        else:
            cell.source=cell.source.replace('`RESULTS.md`','`TRIALS.txt`').replace('`predictions_history/`','`runs/predictions/`')
            cell.source=cell.source.replace('`MONITORING.md`','`docs/MONITORING.md`')
            cell.source=cell.source.replace('Run `python model_corrige.py`, `python -m unittest discover -s tests -v`, `python build_audit.py`, `python build_presentation.py`, then `python package_submission.py`.',
                'Install with `python -m pip install -e .`; run `python model_corrige.py`, `python scripts/verify_project.py`, and `python scripts/build_delivery.py`. The frozen champion remains one independent experiment; new methods live separately in `experiments/`.')
    if not any('Organization and ongoing trials' in c.source for c in nb.cells):
        nb.cells.insert(1,nbformat.v4.new_markdown_cell('''## Organization and ongoing trials

The project now separates independent experiments, shared data contracts, immutable runs, research notes and frozen archives. `TRIALS.txt` is the current top-level trial ledger. This notebook documents the frozen 94.78% champion; later research results are recorded separately until a measured replacement is promoted. New evaluations are score-only and never published to the leaderboard. Desktop control is disabled.'''))
    km=KernelManager(kernel_name='python3');km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    NotebookClient(nb,timeout=180,resources={'metadata':{'path':str(ROOT)}},km=km).execute()
    nbformat.write(nb,ROOT/'audit_rapport.ipynb')


def main():
    notebook();_,evaluation=load_data();validation=verify_predictions(evaluation,pd.read_csv(ROOT/'predictions.csv'))
    files=[]
    for name in ['README.md','TRIALS.txt','pyproject.toml','predictions.csv','model_corrige.py','audit_rapport.ipynb','presentation.pdf','.gitignore','.gitattributes']:
        files.append(ROOT/name)
    for directory in ['src','experiments','config','docs','scripts','runs','archive','data/equialgo-participants']:
        files.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.pyc','.zip'] and not any(part.endswith('.egg-info') for part in p.parts))
    delivery=ROOT/'delivery';delivery.mkdir(exist_ok=True)
    manifest={'active_release':json.loads((ROOT/'config/active_release.json').read_text()),'validation':validation,
              'files':{str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (delivery/'submission_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(delivery/'equialgo_submission.zip','w',zipfile.ZIP_DEFLATED) as bundle:
        for p in files:bundle.write(p,p.relative_to(ROOT).as_posix())
        bundle.write(delivery/'submission_manifest.json','submission_manifest.json')
    print(json.dumps({'bundle':str((delivery/'equialgo_submission.zip').relative_to(ROOT)),'files':len(files),'validation':validation},indent=2))


if __name__=='__main__':main()
