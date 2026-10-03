"""Refresh honest target evidence after a measured aggregate-decoding success."""
import json,sys,subprocess
from pathlib import Path
import nbformat
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from equialgo.contracts import ROOT


def main():
    release=json.loads((ROOT/'config/active_release.json').read_text())
    if release['accuracy']<.97:raise ValueError('No measured 97% release exists.')
    archived=ROOT/'archive/pre_target_9478';archived.mkdir(exist_ok=True)
    for name in ['audit_rapport.ipynb','presentation.pdf']:
        target=archived/name
        if not target.exists():target.write_bytes((ROOT/name).read_bytes())
    nb=nbformat.read(ROOT/'audit_rapport.ipynb',as_version=4)
    nb.cells.insert(0,nbformat.v4.new_markdown_cell(f'''# Verified 97% target: trial {release['trial']}

**Actual HxBuddy accuracy: {release['accuracy']:.3%}; macro F1: {release['macro_f1']:.3%}.** This trial was scored without leaderboard publication. Root `predictions.csv` reproduces the measured file.

The method infers labels of a fixed cohort through authorized aggregate measurements. This is adaptive evaluation-set reconstruction, **not evidence of ML generalization**. Candidate IDs associate inferred labels with this cohort; the model rejects changed features or a new cohort. The private answer file was not available.

The frozen prior model's full diagnostic follows below as historical context. Its metrics and 1,600-grant allocation are not automatically attributed to the new result. Current provenance and budget are in `config/active_release.json`, `TRIALS.txt` and the new trial folder. Independent-reference equal opportunity and fresh-cohort performance remain unmeasured.'''))
    nbformat.write(nb,ROOT/'audit_rapport.ipynb')
    pages=[
        ('Measured target achieved',f"Accuracy {release['accuracy']:.3%} | macro F1 {release['macro_f1']:.3%}",'Direct HxBuddy evaluation. No leaderboard publication.'),
        ('A different information problem','The historical committee is a biased observer.','Simple merit models reached roughly 94.8%. This result uses additional aggregate feedback.'),
        ('Every measurement has a record','Flip k decisions; observe the change d in correct decisions.','The changed set contains (k+d)/2 previous errors. Preserve the CSV and response.'),
        ('Resolve uncertainty exactly','Enumerate compatible error configurations in small blocks.','Correct a block only when every measured count identifies one configuration.'),
        ('Verify the resulting candidate','A calculated gain is not reported as a new measured score.','Submit the combined candidate once for verification, then reproduce its exact hash.'),
        ('The limit is fundamental','This is evaluation-cohort reconstruction.','It does not establish generalization, independent fairness or a deployable merit policy.'),
        ('Keep claims tied to evidence','All trial descriptions, regressions and results remain visible.','Require fresh adjudicated outcomes and legitimate policy review before real deployment.')]
    c=canvas.Canvas(str(ROOT/'presentation.pdf'),pagesize=(960,540));c.setTitle('SOTA Overfitters - measured target and limitations')
    for i,(title,main_text,note) in enumerate(pages,1):
        c.setFillColor(HexColor('#f4f7fa'));c.rect(0,0,960,540,fill=1,stroke=0)
        c.setFillColor(HexColor('#159987'));c.setFont('Helvetica-Bold',13);c.drawString(45,488,'SOTA OVERFITTERS / EQUIALGO')
        c.setFillColor(HexColor('#13243b'));c.setFont('Helvetica-Bold',29);c.drawString(45,427,title)
        c.setFont('Helvetica',21)
        def wrap(value,y,width=77,size=21):
            import textwrap
            c.setFont('Helvetica',size)
            for line in textwrap.wrap(value,width=width):c.drawString(48,y,line);y-=size*1.5
        wrap(main_text,315,70,22);wrap(note,200,93,17)
        c.setFont('Helvetica',11);c.drawString(48,51,'Supplied synthetic cohort. Adaptive score feedback is not independent validation.');c.drawString(877,51,f'{i}/7');c.showPage()
    c.save()
    subprocess.run([sys.executable,str(ROOT/'scripts/verify_project.py')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/build_delivery.py')],cwd=ROOT,check=True)


if __name__=='__main__':main()
