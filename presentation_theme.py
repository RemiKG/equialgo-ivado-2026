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
    text(c, 44, 494, "SOTA OVERFITTERS  /  EQUIALGO", 12, TEAL, True)
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


