"""Seven-slide pitch for the measured champion, including failed experiments."""
import json
from presentation_theme import *


def main():
    release=json.loads((ROOT/'artifacts/active_prediction.json').read_text())
    c=canvas.Canvas(str(ROOT/'presentation.pdf'),pagesize=(W,H));c.setTitle('SOTA Overfitters - EquiAlgo');c.setAuthor('SOTA Overfitters')
    frame(c,1,'Separate committee habits from legitimate merit.','SOTA Overfitters | Measured results, explicit assumptions and preserved experiments.')
    card(c,44,236,272,f"{release['accuracy']:.2%}",'Measured accuracy','HxBuddy champion')
    card(c,344,236,272,f"{release['macro_f1']:.2%}",'Measured macro F1','14 scored attempts preserved')
    card(c,644,236,272,'1,600','Scholarships','Exactly 40% of applicants')
    wrapped(c,47,184,'From a biased historical target to a transparent, feedback-calibrated allocation. The best measured result improves on V1 by 0.15 accuracy points.',850,22)
    text(c,47,77,'The improvement is modest. The evaluated cohort informed development; 98% is not established.',14,MUTED);c.showPage()
    frame(c,2,'Removing the sensitive column leaves the problem.','Historical decisions describe the committee. They do not certify true merit.')
    card(c,44,236,272,'18.76 pp','Baseline access gap','Historical holdout selection rates')
    card(c,344,236,272,'18.05 pp','Without region','Geographic proxies remain')
    card(c,644,236,272,'17.31 pp','Without region + postal','Income, distance and work carry signal')
    wrapped(c,47,182,'The original audit measures proxy signals, academic-band differences and uncertainty. That entire method is frozen in an archive for comparison.',850,22)
    text(c,47,76,'Equal opportunity needs independent merit labels. Historical committee labels cannot supply them.',14,MUTED);c.showPage()
    frame(c,3,'A new method has to survive a measured test.','We preserve every failed assumption as well as each improvement.')
    picture(c,ROOT/'artifacts/score_progress.png',45,88,870,312)
    text(c,47,64,'V2 fell to 91.90 F1; nonlinear merit reached 94.50; joint feedback calibration reached 94.56.',13,MUTED);c.showPage()
    frame(c,4,'Use joint evidence near the allocation cutoff.','Maximum-entropy calibration: preserve a prior while matching measured aggregate agreements.')
    for y,n,title,body in [
        (338,'1','Start with an academic / work prior','Work weight 0.14; uncertain merit boundary. Income and regional perturbations are diagnostics.'),
        (232,'2','Fit all eleven observed agreements jointly','Overlapping predictions constrain probability estimates. No individual private reference labels are available.'),
        (122,'3','Freeze parameters and allocate the fixed budget','Fund the highest 1,600 estimates. Verify the CSV hash. Reject changed cohorts until independently validated.')]:
        c.setFillColor(HexColor(TEAL));c.circle(65,y+2,20,fill=1,stroke=0);text(c,59,y-5,n,19,'#ffffff',True)
        text(c,106,y+9,title,20,NAVY,True);wrapped(c,106,y-18,body,779,16)
    c.showPage()
    frame(c,5,'Measure access. Stress-test opportunity assumptions.','The opportunity frontier is simulated; its fairness claims are not independently verified.')
    picture(c,ROOT/'artifacts/pareto_front.png',45,92,870,306)
    text(c,47,70,'Observed selection: centres 40.35%, remote 39.50%. Regional gap: 0.849 percentage points.',13,MUTED)
    text(c,47,51,'Ten constraint settings, same 40% budget. Reference equal-opportunity gap remains unavailable.',12,MUTED);c.showPage()
    frame(c,6,'A leaderboard is not a deployment validation set.','Cohort-specific adaptation can improve the score without proving generalization.')
    cards=[(44,219,'Independent merit review','Define legitimate criteria with a diverse panel. Validate on fresh applicants with outcomes independent of the committee.'),
           (494,219,'Dependency review','Calibration inherits effects from diagnostic submissions, including region and income. These need explicit policy justification.'),
           (44,59,'Appeal and correction','Review rejected and selected cases, allow corrections, and account for disability or caring constraints that limit paid work.'),
           (494,59,'Release and rollback','Monitor opportunity, false positives, intersections and drift. Enforce schema and budget; preserve data, code and prediction hashes.')]
    for x,y,title,body in cards:
        c.setFillColor(HexColor('#ffffff'));c.roundRect(x,y,421,150,12,fill=1,stroke=0)
        text(c,x+18,y+115,title,19,TEAL,True);wrapped(c,x+18,y+84,body,382,14)
    c.showPage()
    frame(c,7,'A measured improvement, with an inspectable record.','Reproducible champion, complete history, executed audit and archived alternatives.')
    wrapped(c,48,354,'What we established',396,25,TEAL)
    wrapped(c,48,306,'94.78% accuracy and 94.56% macro F1 on HxBuddy. Exactly reproduced predictions. A valid 40% budget. Every regression retained.',396,21)
    wrapped(c,515,354,'What remains to establish',396,25,TEAL)
    wrapped(c,515,306,'Independent equal opportunity, performance on fresh applicants, and legitimate policy treatment of contextual features.',396,21)
    c.setFillColor(HexColor(NAVY));c.roundRect(44,87,872,100,12,fill=1,stroke=0)
    wrapped(c,68,151,'Keep the evidence stronger than the claim.',820,27,'#ffffff')
    text(c,47,60,'Sources: IVADO package, HxBuddy, participant feedback, Fairlearn. OpenAI Codex assistance disclosed.',11,MUTED)
    c.showPage();c.save();print('Created presentation.pdf: 7 slides')


if __name__=='__main__':main()
