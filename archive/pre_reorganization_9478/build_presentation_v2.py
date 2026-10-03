"""Seven-slide V2 pitch: evidence, new hypotheses and the next measured test."""
import json
from presentation_theme import *


def main():
    summary=json.loads((ROOT/"artifacts/audit_summary.json").read_text())
    c=canvas.Canvas(str(ROOT/"presentation.pdf"),pagesize=(W,H))
    c.setTitle("SOTA Overfitters - EquiAlgo V2")
    c.setAuthor("SOTA Overfitters")
    frame(c,1,"Rethink what the committee taught us.","SOTA Overfitters | A new merit hypothesis, with a scored fallback and explicit uncertainty.")
    card(c,44,236,272,"94.63%","V1 accuracy","User-reported HxBuddy score")
    card(c,344,236,272,"194","Changed decisions","97 additions and 97 removals")
    card(c,644,236,272,"1,600","Grants in V2","Exact 40% budget")
    wrapped(c,47,183,"The old model copied a work association from a biased committee. The new model asks which merit assumptions could explain its mistakes.",850,23)
    text(c,47,80,"V2 has no official score yet. We do not transfer V1's accuracy or claim 98%.",16,MUTED)
    c.showPage()
    frame(c,2,"A proxy can survive inside the corrected model.","The original diagnostic still holds: removing region and postal code did not remove disparity.")
    card(c,44,232,272,"0.146","Old work coefficient","R-score points per extra paid hour")
    card(c,344,232,272,"0.030","V2 weighted work effect","Under the primary merit hypotheses")
    card(c,644,232,272,"2,048","Merit hypotheses","Vary work, need, context and noise")
    wrapped(c,47,188,"An estimated committee preference is not automatically a legitimate merit criterion. Paid work may reflect hardship, access to jobs or constraints outside a student's control.",850,21)
    wrapped(c,47,104,"Nonlinear income improved historical fit but did not identify a reliable hidden need benefit. V2 states its need assumptions rather than claiming to recover them.",850,15,MUTED)
    c.showPage()
    frame(c,3,"Infer several explanations. Keep them inspectable.","Aggregate feedback restricts the possibilities; it does not reveal applicant-level labels.")
    rows=[("1","Construct plausible merit rules","Positive academics, varying work credit, possible need / first-generation / commute support. No direct region, postal or program term."),
          ("2","Check compatibility with known evidence","Weight by the reported V1 accuracy. Use the supplied baseline opportunity gap weakly because its exact reference cohort is unknown."),
          ("3","Average and allocate the fixed budget","Rank averaged merit weights and award exactly 1,600 grants. Keep broad-uncertainty and academic-priority alternatives for comparison.")]
    for y,(number,title,body) in zip([340,237,121],rows):
        c.setFillColor(HexColor(TEAL));c.circle(66,y+2,21,fill=1,stroke=0)
        text(c,60,y-5,number,19,"#ffffff",True);text(c,108,y+9,title,20,NAVY,True)
        wrapped(c,108,y-19,body,780,16)
    c.showPage()
    frame(c,4,"Stress-test opportunity under each hypothesis.","This frontier is model-implied. Neither axis is an independently measured HxBuddy result.")
    picture(c,ROOT/"artifacts/pareto_front.png",35,78,890,325)
    text(c,47,59,"Ten opportunity tolerances per model; exact 40% budget. Private reference labels remain unavailable.",12,MUTED)
    c.showPage()
    frame(c,5,"The new decisions test a different policy.","Same 4,000 applicants. The primary candidate does not impose demographic parity.")
    card(c,44,236,272,"40.89%","Centre selection","970 of 2,372 applicants")
    card(c,344,236,272,"38.70%","Remote selection","630 of 1,628 applicants")
    card(c,644,236,272,"Pending","Official V2 accuracy","V1 remains the scored fallback")
    wrapped(c,47,185,"The 97 new recipients average higher academic scores, lower household income and fewer paid-work hours than those replaced.",850,22)
    wrapped(c,47,111,"This describes the policy change. Only new official scoring can tell us whether those changes match the hidden reference. The model's own agreement estimate cannot validate it.",850,16,MUTED)
    c.showPage()
    frame(c,6,"Govern assumptions, not just model weights.","A single aggregate score cannot identify a fair decision rule or justify production deployment.")
    cards=[(44,219,"Independent reference","A geographically diverse panel reviews funded and rejected applications under a published rubric, with a fresh validation sample."),
           (494,219,"Stress the priors","The primary assumes limited random merit noise. Need, work, first-generation and commute effects remain hypotheses that can be wrong."),
           (44,59,"Human appeal","Applicants can correct data and explain context. Review hardship and inability to work without treating model weights as certified merit."),
           (494,59,"Release and rollback","Block schema / budget failures; monitor independently measured opportunity, intersections and drift. Freeze unsafe releases for human review.")]
    for x,y,title,body in cards:
        c.setFillColor(HexColor("#ffffff"));c.roundRect(x,y,421,150,12,fill=1,stroke=0)
        text(c,x+18,y+115,title,19,TEAL,True);wrapped(c,x+18,y+84,body,382,14)
    c.showPage()
    frame(c,7,"A new experiment. A preserved record.","Every evaluated prediction must remain linked to its file hash and actual score.")
    wrapped(c,48,352,"Ready now",390,26,TEAL)
    wrapped(c,48,310,"A new primary CSV, two contrasting alternatives, an executed audit, a tested optimizer and a complete V1 archive.",390,21)
    wrapped(c,516,352,"Next evidence",390,26,TEAL)
    wrapped(c,516,310,"Score the new CSV on HxBuddy. Keep the observed result separate from historical agreement and hypothesis-dependent simulations.",390,21)
    c.setFillColor(HexColor(NAVY));c.roundRect(44,88,872,106,12,fill=1,stroke=0)
    wrapped(c,68,156,"Different thinking is useful when it leads to a clear, testable change.",820,25,"#ffffff")
    text(c,47,60,"Sources: IVADO briefs, HxBuddy, participant feedback, SciPy / Fairlearn. Codex assistance disclosed.",11,MUTED)
    c.showPage();c.save();print("Created V2 presentation.pdf: 7 slides")


if __name__=="__main__":main()
