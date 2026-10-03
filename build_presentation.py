"""Generate the five-minute vector PDF pitch from the measured audit artifacts."""
import json
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from model_corrige import ROOT

W, H = 960, 540
NAVY, INK, MUTED = "#101f34", "#21334b", "#657386"
TEAL, GOLD, PAPER = "#16a394", "#e5a44b", "#f4f7fa"
FONT, BOLD = "Helvetica", "Helvetica-Bold"
font_dir = Path("C:/Windows/Fonts")
if (font_dir / "arial.ttf").exists():
    pdfmetrics.registerFont(TTFont("Pitch", str(font_dir / "arial.ttf")))
    pdfmetrics.registerFont(TTFont("PitchBold", str(font_dir / "arialbd.ttf")))
    FONT, BOLD = "Pitch", "PitchBold"


def text(c, x, y, value, size=18, color=INK, bold=False):
    c.setFillColor(HexColor(color)); c.setFont(BOLD if bold else FONT, size)
    c.drawString(x, y, value)


def wrapped(c, x, y, value, width, size=18, color=INK, leading=None):
    leading = leading or size*1.4
    for para in value.split("\n"):
        line = ""
        for word in para.split():
            candidate = (line + " " + word).strip()
            if pdfmetrics.stringWidth(candidate, FONT, size) > width and line:
                text(c, x, y, line, size, color); y -= leading; line = word
            else:
                line = candidate
        if line:
            text(c, x, y, line, size, color); y -= leading
        y -= leading*.2
    return y


def frame(c, number, title, kicker):
    c.setFillColor(HexColor(PAPER)); c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(HexColor(TEAL)); c.rect(0, H-8, W, 8, fill=1, stroke=0)
    text(c, 44, 494, "EQUIALGO  /  IVADO 2026", 12, TEAL, True)
    text(c, 44, 449, title, 29, NAVY, True)
    text(c, 44, 417, kicker, 13, MUTED)
    c.setStrokeColor(HexColor("#d8e0e8")); c.line(44, 42, W-44, 42)
    text(c, 44, 23, "Supplied synthetic data only  |  Reproducible code and audit", 10, MUTED)
    text(c, 892, 23, f"{number} / 7", 10, MUTED)


def card(c, x, y, width, big, label, note="", color=TEAL):
    c.setFillColor(HexColor("#ffffff")); c.roundRect(x, y, width, 139, 12, fill=1, stroke=0)
    text(c, x+20, y+91, big, 34, color, True)
    text(c, x+20, y+61, label, 15, INK, True)
    wrapped(c, x+20, y+35, note, width-38, 11, MUTED)


def picture(c, path, x, y, width, height):
    c.drawImage(ImageReader(str(path)), x, y, width, height, preserveAspectRatio=True,
                anchor="c", mask="auto")


def main():
    summary = json.loads((ROOT / "artifacts/audit_summary.json").read_text())
    base = summary["holdout_metrics"]["baseline"]
    fixed = summary["holdout_metrics"]["corrected"]
    submission = summary["submission"]
    pdf = ROOT / "presentation.pdf"
    c = canvas.Canvas(str(pdf), pagesize=(W, H))
    c.setTitle("EquiAlgo - fairer access, explicit assumptions")
    c.setAuthor("EquiAlgo project")
    c.setSubject("IVADO challenge: diagnostic, correction, governance and limits")

    frame(c, 1, "Fairer access starts with the right target.", "Historical awards measure the committee. They do not establish who should receive support.")
    card(c, 44, 235, 272, "4,000", "Applicants scored", "All evaluation IDs preserved")
    card(c, 344, 235, 272, "1,600", "Grants allocated", "Exact 40% budget")
    card(c, 644, 235, 272, f"{submission['demographic_parity_gap']*100:.3f} pp", "Regional access gap", "Selection-rate gap; not equal opportunity")
    wrapped(c, 47, 184, "Our contribution: diagnose the proxy problem, repair the ranking criteria, and enforce an auditable allocation budget.", 850, 23)
    text(c, 47, 80, "Independent-reference accuracy is unknown. A 98% accuracy claim is not supported.", 15, MUTED)
    c.showPage()

    frame(c, 2, "Deleting the region column leaves the gap.", "Same 3,000-row holdout, seed 42. Metrics below concern historical committee decisions.")
    card(c, 44, 239, 272, f"{base['demographic_parity_gap']*100:.2f} pp", "Full baseline", f"{base['historical_agreement']*100:.2f}% historical agreement")
    card(c, 344, 239, 272, f"{summary['holdout_metrics']['drop_region']['demographic_parity_gap']*100:.2f} pp", "Without region", "Postal and other proxies remain")
    card(c, 644, 239, 272, f"{summary['holdout_metrics']['drop_region_postal']['demographic_parity_gap']*100:.2f} pp", "Without region + postal", "Distance, income and hours carry signal")
    wrapped(c, 47, 188, "Historical grant rates: 48.4% in centres versus 27.3% in remote regions. Regional differences persist within academic-score bands.", 847, 21)
    wrapped(c, 47, 110, "Distance alone predicts regional block with ROC AUC 0.997 on held-out data. This identifies a proxy; it does not establish a causal effect.", 847, 16, MUTED)
    c.showPage()

    frame(c, 3, "Repair the score. Make the allocation explicit.", "Policy choices are disclosed; no synthetic labels are passed off as independent merit.")
    sections = [
        ("1", "Estimate conditional associations", "Fit committee decisions while controlling for geography, wealth, program and other observed attributes."),
        ("2", "Apply the approved direction of effects", "Remove location/program effects, positive wealth advantage and first-generation penalty. Keep nonnegative academic and work effects."),
        ("3", "Solve the exact allocation problem", "Take the best within-group scores under 40% total grants and a maximum 2-point selection-rate gap. Stable tie breaking; no outcome labels required.")]
    for y, (number, title, body) in zip([340, 237, 121], sections):
        c.setFillColor(HexColor(TEAL)); c.circle(66, y+2, 21, fill=1, stroke=0)
        text(c, 60, y-5, number, 19, "#ffffff", True)
        text(c, 108, y+9, title, 20, NAVY, True)
        wrapped(c, 108, y-19, body, 780, 16)
    c.showPage()

    frame(c, 4, "Show the trade-off, including its limits.", "12 constraint settings per score, each with an exact 40% budget. Star = chosen policy.")
    picture(c, ROOT / "artifacts/pareto_front.png", 34, 74, 892, 326)
    text(c, 46, 57, "Historical agreement is a diagnostic comparator. Hidden-reference utility and opportunity remain unknown.", 12, MUTED)
    c.showPage()

    frame(c, 5, "A valid, reproducible allocation.", "The score repair already produces near parity; the 2-point guardrail is inactive on the final cohort.")
    card(c, 44, 234, 272, f"{submission['selection_rate_centre']*100:.2f}%", "Centre selection", "949 grants / 2,372 applicants")
    card(c, 344, 234, 272, f"{submission['selection_rate_remote']*100:.2f}%", "Remote selection", "651 grants / 1,628 applicants")
    card(c, 644, 234, 272, "40.00%", "Overall budget", "Inside the official 36%-44% range")
    wrapped(c, 47, 190, "Verified: exact IDs and row count, binary decisions, deterministic rerun, monotonic score rules, and allocator optimality against exhaustive small cases.", 850, 20)
    wrapped(c, 47, 110, f"Held-out historical agreement: {fixed['historical_agreement']*100:.2f}%. Historical-label TPR gap increases from {base['historical_tpr_gap']:.3f} to {fixed['historical_tpr_gap']:.3f}; this does not determine the hidden-merit result.", 850, 15, MUTED)
    c.showPage()

    frame(c, 6, "Govern decisions beyond the leaderboard.", "Named responsibilities, measurable triggers, a human appeal and a controlled fallback.")
    tiles = [
        (44, 218, "Independent reference", "Regional panel reviews funded and rejected applicants under a published eligibility rubric. Validate true-merit TPR and utility before release."),
        (494, 218, "Every batch", "ML owner blocks schema or budget failures. Monitor regional access, intersections, input drift and model versions with denominators."),
        (44, 58, "Appeal and correction", "Human officer reviews errors and context. Explain the score and regional constraint. Reconcile funding through an approved budget process."),
        (494, 58, "Freeze and reassess", "Escalate independently measured opportunity gaps. Freeze automatic releases on severe incidents and use the approved human process.")]
    for x, y, title, body in tiles:
        c.setFillColor(HexColor("#ffffff")); c.roundRect(x, y, 421, 150, 12, fill=1, stroke=0)
        text(c, x+18, y+115, title, 19, TEAL, True)
        wrapped(c, x+18, y+84, body, 382, 14)
    c.showPage()

    frame(c, 7, "Approve the criteria. Measure independent merit.", "The hidden reference cannot be recovered merely by fitting historical decisions more accurately.")
    wrapped(c, 48, 353, "Known", 390, 26, TEAL)
    wrapped(c, 48, 311, "A complete audit, exact-budget predictions, an inspectable model, a constraint sweep and a monitoring plan.", 390, 21)
    wrapped(c, 516, 353, "Still to establish", 390, 26, TEAL)
    wrapped(c, 516, 311, "True-merit accuracy and equal opportunity. Appropriate work and financial-need weights. Applicant-level consequences.", 390, 21)
    c.setFillColor(HexColor(NAVY)); c.roundRect(44, 88, 872, 106, 12, fill=1, stroke=0)
    wrapped(c, 68, 156, "Our standard: transparent assumptions, measured evidence, and a path to challenge a decision.", 820, 24, "#ffffff")
    text(c, 47, 60, "Sources: supplied IVADO briefs, HxBuddy track, Fairlearn docs. Tools and Codex assistance disclosed in README.", 11, MUTED)
    c.showPage()
    c.save()
    print(f"Created {pdf.name}: 7 slides")


if __name__ == "__main__":
    main()
