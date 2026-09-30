#!/usr/bin/env python3
"""Presentation figures for the VERIFY-DR thesis talk: 16:9 PNGs at 3200 x 1800.

Every number is copied from a committed readout and cited by line, so each figure can
be checked against its source. Nothing is recomputed from data; the script only draws.

  U  docs/unblinding/2026-09-28_readout.txt   the unblinding, section 6
  P  docs/phase8/2026-09-29_readout.txt       Phase 8, exploratory

Colours are slots 1 and 2 of the dataviz reference palette, validated on white (all six
checks pass), with its muted ink for reference marks. Run from the repository root:

    python docs/presentation/make_figures.py
"""
import base64
import html
import io
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from statistics import mean

from PIL import ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
sys.path.insert(0, str(ROOT / "docs" / "thesis" / "figures"))
import make_fig7_1 as F71  # noqa: E402  Figure 7.1's sheet geometry and crops

W, H = 1600, 900
FONT = "'Liberation Sans', Arial, Helvetica, sans-serif"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE, WASH = "#e1e0d9", "#c3c2b7", "#ffffff", "#f3f2ee"
BLUE, ORANGE = "#2a78d6", "#eb6834"      # confidence; disagreement and the evidence pathway
SETS = ("EyePACS test", "APTOS", "Messidor-2")
NOTE = {"EyePACS test": "in-domain", "APTOS": "external", "Messidor-2": "external"}

# ------------------------------------------------------------------------------ numbers
# Coverage-accuracy AUC per seed (eyepacs_full s42, s43, s44).
# U 105-107 / 116-118 (EyePACS test), U 73-75 / 87-89 (APTOS), U 41-43 / 55-57 (Messidor-2).
AUC = {
    "EyePACS test": {"none": (0.7696, 0.7150, 0.7565), "confidence": (0.9222, 0.9055, 0.9218),
                     "registered": (0.7662, 0.7186, 0.7546), "amended": (0.8179, 0.7713, 0.8131)},
    "APTOS": {"none": (0.6846, 0.5246, 0.6352), "confidence": (0.8875, 0.8228, 0.8703),
              "registered": (0.6083, 0.4563, 0.5836), "amended": (0.7246, 0.5863, 0.7052)},
    "Messidor-2": {"none": (0.6984, 0.6416, 0.6680), "confidence": (0.8516, 0.8286, 0.8420),
                   "registered": (0.6794, 0.6462, 0.6518), "amended": (0.7354, 0.6953, 0.7166)},
}
# G1 at 80% coverage: (confidence only, disagreement only, both), means over the seeds.
# Exact-grade errors: P 253 / 270 (EyePACS test), P 328 / 345 (APTOS), P 392 / 409 (Messidor-2).
G1_EXACT = {
    "EyePACS test": {"registered": (0.609, 0.056, 0.024), "amended": (0.465, 0.087, 0.168)},
    "APTOS": {"registered": (0.399, 0.033, 0.069), "amended": (0.326, 0.160, 0.142)},
    "Messidor-2": {"registered": (0.376, 0.076, 0.059), "amended": (0.349, 0.138, 0.086)},
}
# Referral errors (refer = grade 2 or above): P 254 / 271, P 329 / 346, P 393 / 410.
G1_REFERRAL = {
    "EyePACS test": {"registered": (0.542, 0.067, 0.011), "amended": (0.411, 0.142, 0.142)},
    "APTOS": {"registered": (0.161, 0.021, 0.009), "amended": (0.139, 0.356, 0.031)},
    "Messidor-2": {"registered": (0.245, 0.043, 0.020), "amended": (0.224, 0.270, 0.041)},
}
ERRORS = {"EyePACS test": (4456, 17615), "APTOS": (1411, 3662), "Messidor-2": (577, 1744)}  # P 253, 328, 392
# G3, correct calls M3 contradicts, (registered, amended): P 265 / 282, P 340 / 357, P 404 / 421.
CONTRADICTS = {"EyePACS test": (0.649, 0.308), "APTOS": (0.465, 0.097), "Messidor-2": (0.625, 0.358)}
# G3, M1's errors above M3's ceiling (y-hat >= 3), the same in both analyses: P 264, 339, 403.
ABOVE_CEILING = {"EyePACS test": 0.077, "APTOS": 0.560, "Messidor-2": 0.074}
# E2 faithful share: mean, min, max over the seeds, and lesion images (the sum of E3's n).
# P 46-51 (EyePACS test), P 122-127 (APTOS), P 199-204 (Messidor-2).
FAITHFUL = {"EyePACS test": (0.665, 0.641, 0.704, 14158),
            "APTOS": (0.700, 0.672, 0.725, 2984),
            "Messidor-2": (0.778, 0.773, 0.789, 1574)}
# Published means (docs/report.html, chapter 6), to catch a transcription slip.
PUBLISHED_AUC = {"EyePACS test": (0.747, 0.917, 0.746, 0.801), "APTOS": (0.615, 0.860, 0.549, 0.672),
                 "Messidor-2": (0.669, 0.841, 0.659, 0.716)}


def check_numbers() -> None:
    """The confidence arm is identical in both analyses; the transcription must agree."""
    for table in (G1_EXACT, G1_REFERRAL):
        for s in SETS:
            reg, amd = table[s]["registered"], table[s]["amended"]
            assert abs((reg[0] + reg[2]) - (amd[0] + amd[2])) < 1e-9, (s, reg, amd)
    for s, published in PUBLISHED_AUC.items():
        got = [mean(AUC[s][k]) for k in ("none", "confidence", "registered", "amended")]
        assert all(abs(g - p) <= 6e-4 for g, p in zip(got, published)), (s, got, published)


# ------------------------------------------------------------------------------ drawing
_FONTS = {}


def width(s: str, size: int, bold: bool = False) -> float:
    key = (size, bold)
    if key not in _FONTS:
        name = "LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf"
        _FONTS[key] = ImageFont.truetype(f"/usr/share/fonts/truetype/liberation/{name}", size)
    return _FONTS[key].getlength(s)


def text(x, y, s, size=22, fill=INK, weight=400, anchor="start", halo=False) -> str:
    ring = (f' stroke="{SURFACE}" stroke-width="7" stroke-linejoin="round" paint-order="stroke"'
            if halo else "")
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}"{ring}>{html.escape(str(s), quote=False)}</text>')


def wrap(s: str, size: int, max_w: float, bold: bool = False) -> list:
    lines, cur = [], ""
    for word in s.split():
        trial = f"{cur} {word}".strip()
        if not cur or width(trial, size, bold) <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    return lines + ([cur] if cur else [])


def para(x, y, s, size, fill, max_w, weight=400, lh=None) -> str:
    lh = lh or round(size * 1.32)
    return "".join(text(x, y + i * lh, line_, size, fill, weight)
                   for i, line_ in enumerate(wrap(s, size, max_w, weight >= 600)))


def fit(s: str, size: int, max_w: float, bold: bool = True) -> int:
    while size > 20 and width(s, size, bold) > max_w:
        size -= 1
    return size


def line(x1, y1, x2, y2, stroke=GRID, w=1.0, dash=None, opacity=None) -> str:
    extra = (f' stroke-dasharray="{dash}"' if dash else "") + \
            (f' stroke-opacity="{opacity}"' if opacity is not None else "")
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" '
            f'stroke-width="{w}" stroke-linecap="round"{extra}/>')


def dot(x, y, colour, hollow=False, r=10) -> str:
    if hollow:
        return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r - 1.5:.1f}" fill="{SURFACE}" '
                f'stroke="{colour}" stroke-width="3"/>')
    return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{colour}" '
            f'stroke="{SURFACE}" stroke-width="2"/>')


def bar(x0, x1, yc, colour, thick=24) -> str:
    """A horizontal bar: square at the baseline, a 4 px rounded data end."""
    x1 = max(x1, x0 + 1)
    r = min(4.0, (x1 - x0) / 2)
    y0, y1 = yc - thick / 2, yc + thick / 2
    return (f'<path d="M{x0:.1f},{y0:.1f} H{x1 - r:.1f} Q{x1:.1f},{y0:.1f} {x1:.1f},{y0 + r:.1f} '
            f'V{y1 - r:.1f} Q{x1:.1f},{y1:.1f} {x1 - r:.1f},{y1:.1f} H{x0:.1f} Z" fill="{colour}"/>')


def box(x, y, w, h, stroke=AXIS, sw=1.5, fill=SURFACE, rx=10) -> str:
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"/>')


def arrow(d, colour=INK2, w=2.5) -> str:
    marker = {INK2: "ah", BLUE: "ah-b", ORANGE: "ah-o"}[colour]
    return (f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="{w}" '
            f'stroke-linejoin="round" marker-end="url(#{marker})"/>')


DEFS = "<defs>" + "".join(
    f'<marker id="{mid}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" '
    f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{colour}"/></marker>'
    for mid, colour in (("ah", INK2), ("ah-b", BLUE), ("ah-o", ORANGE))) + "</defs>"


def legend(items, x=80, y=182) -> str:
    parts = []
    for kind, colour, label in items:
        if kind == "hollow":
            parts.append(dot(x + 10, y - 7, colour, hollow=True))
        elif kind == "dash":
            parts.append(line(x, y - 7, x + 24, y - 7, colour, 2.5, "6 5"))
        elif kind == "bar":
            parts.append(bar(x, x + 22, y - 7, colour, 14))
        else:
            parts.append(dot(x + 10, y - 7, colour))
        parts.append(text(x + 32, y, label, 20, INK2))
        x += 32 + width(label, 20) + 36
    return "".join(parts)


def page(title, subtitle, body, footer="", items=()) -> str:
    size = fit(title, 40, W - 160)
    head = text(80, 84, title, size, INK, 700) + para(80, 124, subtitle, 23, INK2, W - 160)
    foot = ""
    if footer:
        lines = wrap(footer, 17, W - 160)
        foot = "".join(text(80, 872 - (len(lines) - 1 - i) * 23, ln, 17, MUTED)
                       for i, ln in enumerate(lines))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
            f'viewBox="0 0 {W} {H}" font-family="{FONT}">{DEFS}'
            f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>{head}'
            f'{legend(items) if items else ""}{body}{foot}</svg>')


def panels(n=3, left=340, right=1530, gap=64) -> list:
    w = (right - left - gap * (n - 1)) / n
    return [(left + i * (w + gap), w) for i in range(n)]


def scale(x0, w, lo, hi):
    return lambda v: x0 + (v - lo) / (hi - lo) * w


def mean3(values) -> str:
    """A mean to three decimals, rounded half up in decimal as the published tables are:
    binary floats would print 0.9165 as 0.916."""
    exact = sum(Decimal(str(v)) for v in values) / len(values)
    return str(exact.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def pct(v: float) -> str:
    return f"{100 * v:.1f}%"


def row_labels(rows, ys) -> str:
    return "".join(text(80, y - 2, label, 25, INK, 700) + text(80, y + 25, note, 19, MUTED)
                   for (label, note), y in zip(rows, ys))


def panel_head(x0, s, y=250) -> str:
    return text(x0, y, s, 25, INK, 700) + text(x0 + width(s, 25, True) + 10, y, NOTE[s], 20, MUTED)


def png_uri(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ------------------------------------------------------------------------------ figures
def fig_method() -> str:
    from PIL import Image
    sheet = Image.open(F71.SHEETS / F71.ROWS[0][1]).convert("RGB")
    whole = F71.region(sheet, F71.ROWS[0][2], F71.ROWS[0][3], False,
                       (0, 0, F71.IMAGE, F71.IMAGE)).resize((440, 440), Image.LANCZOS)
    b = [f'<image x="80" y="300" width="220" height="220" href="{png_uri(whole)}"/>',
         text(190, 552, "Fundus photograph", 20, INK2, anchor="middle"),
         text(190, 578, "512 × 512 px", 18, MUTED, anchor="middle")]
    rows = (
        (250, (("M1 · grading network", 24, INK, 700), ("EfficientNet-B0, ordinal head", 20, INK2, 400),
               ("trained on image grades (EyePACS)", 19, MUTED, 400)), AXIS,
         (("Grade ŷ", 24, INK, 700), ("and its confidence p(ŷ)", 20, INK2, 400)), AXIS,
         (("Confidence gate", 24, INK, 700), ("the baseline", 19, MUTED, 400),
          ("defer where p(ŷ) is low", 20, INK2, 400)), BLUE),
        (530, (("M2 · lesion segmenter", 24, INK, 700), ("ResNet18-UNet: MA, HE, exudates", 20, INK2, 400),
               ("trained on lesion masks (DDR)", 19, MUTED, 400)), AXIS,
         (("M3 · grading rules", 24, INK, 700), ("no learned weights", 19, MUTED, 400),
          ("evidence grade e, 0–2", 20, INK2, 400)), AXIS,
         (("Disagreement gate", 24, INK, 700), ("the arm under test", 19, MUTED, 400),
          ("defer where |ŷ − e| is high", 20, INK2, 400)), ORANGE),
    )
    for y, first, s1, second, s2, third, s3 in rows:
        for x, w, lines_, stroke in ((380, 380, first, s1), (830, 290, second, s2), (1210, 310, third, s3)):
            b.append(box(x, y, w, 140, stroke, 2.5 if stroke != AXIS else 1.5))
            for i, (s, size, fill, weight) in enumerate(lines_):
                b.append(text(x + 22, y + 44 + i * 32, s, size, fill, weight))
    b += [arrow("M300,410 H336 V320 H372"), arrow("M336,410 V600 H372"),
          arrow("M760,320 H822"), arrow("M760,600 H822"),
          arrow("M1120,320 H1202", BLUE), arrow("M1120,356 H1162 V566 H1202", ORANGE),
          arrow("M1120,636 H1202", ORANGE)]
    b += [line(380, 452, 1110, 452, AXIS, 1.5), line(380, 460, 1110, 460, AXIS, 1.5),
          f'<rect x="{745 - 205}" y="441" width="410" height="30" fill="{SURFACE}"/>',
          text(745, 464, "no shared weights · different training labels", 19, INK2, 400, "middle")]
    return page("Two pathways that never share a weight",
                "Trust the grader where independent lesion evidence agrees; defer to a human where it does not",
                "".join(b),
                "H1, pre-registered: gating on disagreement beats gating on confidence on the "
                "coverage–accuracy curve. As built, M3 stops at grade 2: without a disc–fovea frame "
                "it has no severe rule, and nothing annotates new vessels.")


def fig_timeline() -> str:
    days = range(16, 30)
    x = scale(130, 1340, 16, 29)
    y0 = 470
    b = [f'<rect x="100" y="512" width="{x(28) - 100:.1f}" height="34" rx="6" fill="{WASH}"/>',
         text(116, 535, "APTOS and Messidor-2 labels locked", 19, INK2),
         line(100, y0, 1500, y0, AXIS, 2)]
    for d in days:
        b.append(line(x(d), y0 - 6, x(d), y0 + 6, AXIS, 1.5))
        b.append(text(x(d), y0 + 30, str(d), 16, MUTED, anchor="middle"))
    events = (  # day, above?, label height, anchor, title, detail, accent
        (16, True, 318, "start", "Protocol frozen", "commit 17472a2", False),
        (22, True, 318, "start", "Final models trained", "6 runs, internal data only", False),
        (23, False, 640, "start", "Analysis plan final", "commit 1f2f9c1", False),
        (24, True, 398, "start", "Rehearsal on validation", "M3 over-calls: amend (D12, D13)", False),
        (27, False, 720, "start", "Parameters committed", "digest 0df6b029…", False),
        (28, True, 318, "end", "Locked evaluation, one label join", "every hypothesis not supported", True),
        (29, False, 640, "end", "Error analysis", "declared before it ran", False),
    )
    for d, above, ly, anchor, title, detail, accent in events:
        colour = ORANGE if accent else INK2
        top, bottom = (ly + 16, y0 - 12) if above else (y0 + 42, ly - 28)
        b.append(line(x(d), top, x(d), bottom, AXIS, 1.5))
        b.append(dot(x(d), y0, colour, r=11 if accent else 9))
        b.append(text(x(d), ly - 28, title, 22, INK, 700, anchor))
        b.append(text(x(d), ly, detail, 19, INK2, 400, anchor))
    return page("Both analyses were fixed before a test label was read",
                "September 2026. The order is on the record: the freeze hash, the parameter "
                "digest and GitHub's push times",
                "".join(b),
                "The amendments D12 and D13 were decided on validation data, before the locked "
                "evaluation, and are reported beside the registered analysis, which stays primary.")


def fig_headline() -> str:
    rows = (("No gate", "accept everything"), ("Confidence", "the baseline"),
            ("Disagreement", "the arm under test"))
    ys = (372, 502, 632)
    b = [row_labels(rows, ys)]
    for (x0, w), s in zip(panels(), SETS):
        sx = scale(x0, w, 0.5, 1.0)
        b.append(panel_head(x0, s))
        for t in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            b.append(line(sx(t), 300, sx(t), 680))
            b.append(text(sx(t), 714, f"{t:.1f}", 18, MUTED, anchor="middle"))
        m = {k: mean(v) for k, v in AUC[s].items()}
        b.append(dot(sx(m["none"]), ys[0], MUTED))
        b.append(dot(sx(m["confidence"]), ys[1], BLUE))
        b.append(text(sx(m["confidence"]), ys[1] - 22, mean3(AUC[s]["confidence"]), 21, INK, 700, "middle", halo=True))
        r, a = sx(m["registered"]), sx(m["amended"])
        b.append(line(r, ys[2], a, ys[2], ORANGE, 3, opacity=0.45))
        b.append(dot(r, ys[2], ORANGE, hollow=True))
        b.append(dot(a, ys[2], ORANGE))
        b.append(text(r, ys[2] + 40, mean3(AUC[s]["registered"]), 20, INK2, 400, "middle", halo=True))
        b.append(text(a, ys[2] - 22, mean3(AUC[s]["amended"]), 21, INK, 700, "middle", halo=True))
    b.append(text(935, 756, "area under the coverage–accuracy curve (higher is better)", 20, INK2,
                  anchor="middle"))
    return page("Confidence beat disagreement on every test set",
                "How well each rule for deciding which gradings to trust performs, over every "
                "share of cases deferred",
                "".join(b),
                "Mean of three seeds, primary variant. Pre-registered is the primary analysis; "
                "amended applies deviations D12 and D13 and is reported beside it. In every seed, "
                "set and analysis, the interval for disagreement minus confidence lies below −0.09.",
                (("dot", MUTED, "No gate"), ("dot", BLUE, "Confidence"),
                 ("hollow", ORANGE, "Disagreement, pre-registered"),
                 ("dot", ORANGE, "Disagreement, amended")))


def deferral_figure(table, title, subtitle, footer) -> str:
    rows = (("Confidence", "the baseline"), ("Disagreement", "the arm under test"))
    ys = (420, 600)
    b = [row_labels(rows, ys)]
    for (x0, w), s in zip(panels(), SETS):
        sx = scale(x0, w, 0.0, 0.7)
        b.append(panel_head(x0, s))
        for t in (0.0, 0.2, 0.4, 0.6):
            if t != 0.2:
                b.append(line(sx(t), 310, sx(t), 680))
            b.append(text(sx(t), 714, f"{100 * t:.0f}%", 18, MUTED, anchor="middle"))
        b.append(line(sx(0.2), 310, sx(0.2), 680, MUTED, 2, "6 6"))
        b.append(text(sx(0.2) + 8, 318, "chance", 17, MUTED))
        reg, amd = table[s]["registered"], table[s]["amended"]
        conf, dis_r, dis_a = reg[0] + reg[2], reg[1] + reg[2], amd[1] + amd[2]
        b.append(dot(sx(conf), ys[0], BLUE))
        b.append(text(sx(conf), ys[0] - 22, pct(conf), 21, INK, 700, "middle", halo=True))
        r, a = sx(dis_r), sx(dis_a)
        b.append(line(r, ys[1], a, ys[1], ORANGE, 3, opacity=0.45))
        b.append(dot(r, ys[1], ORANGE, hollow=True))
        b.append(dot(a, ys[1], ORANGE))
        b.append(text(r, ys[1] + 40, pct(dis_r), 20, INK2, 400, "middle", halo=True))
        b.append(text(a, ys[1] - 22, pct(dis_a), 21, INK, 700, "middle", halo=True))
    return page(title, subtitle, "".join(b), footer,
                (("dot", BLUE, "Confidence"), ("hollow", ORANGE, "Disagreement, pre-registered"),
                 ("dot", ORANGE, "Disagreement, amended"), ("dash", MUTED, "Chance: a random 20%")))


def fig_errors() -> str:
    rates = ", ".join(f"{pct(e / n)} {s}" for s, (e, n) in ERRORS.items())
    return deferral_figure(
        G1_EXACT, "The pre-registered disagreement caught fewer errors than chance",
        "Share of the grader's wrong grades sent to a human when each rule defers 20% of cases",
        f"80% coverage; mean of three seeds, primary variant. The grader's error rate: {rates}. "
        "Exploratory (Phase 8), declared before it ran.")


def fig_referral() -> str:
    return deferral_figure(
        G1_REFERRAL, "Under shift, amended disagreement caught more referral errors",
        "Share of the grader's referral errors (refer = grade 2 or above) sent to a human when "
        "each rule defers 20% of cases",
        "Found after the unblinding, in the amended analysis only. The extra catches are false "
        "referrals: disagreement raised specificity among accepted cases, not sensitivity (F3). "
        "A hypothesis for a new pre-registered study, not a finding of this one.")


def fig_why() -> str:
    ys = (382, 502, 622)
    b = ["".join(text(80, y + 8, s, 25, INK, 700) for s, y in zip(SETS, ys))]
    left, right, w = 340, 1010, 510
    sl, sr = scale(left, w, 0.0, 0.7), scale(right, w, 0.0, 0.7)
    for x0, head, sub in ((left, "M3 contradicted correct grades",
                           "% of the grader's correct calls (grades 0–2) that M3 disputes"),
                          (right, "Errors beyond M3's reach",
                           "% of the grader's errors that are calls of 3 or 4")):
        b.append(text(x0, 246, head, 25, INK, 700))
        b.append(para(x0, 276, sub, 19, MUTED, w))
    for sx in (sl, sr):
        for t in (0.0, 0.2, 0.4, 0.6):
            b.append(line(sx(t), 330, sx(t), 660))
            b.append(text(sx(t), 694, f"{100 * t:.0f}%", 18, MUTED, anchor="middle"))
    for s, y in zip(SETS, ys):
        reg, amd = CONTRADICTS[s]
        b.append(line(sl(amd), y, sl(reg), y, ORANGE, 3, opacity=0.45))
        b.append(dot(sl(reg), y, ORANGE, hollow=True))
        b.append(dot(sl(amd), y, ORANGE))
        b.append(text(sl(reg), y - 22, pct(reg), 20, INK2, 400, "middle"))
        b.append(text(sl(amd), y - 22, pct(amd), 21, INK, 700, "middle"))
        v = ABOVE_CEILING[s]
        b.append(bar(sr(0), sr(v), y, INK2 if s == "APTOS" else AXIS))
        b.append(text(sr(v) + 12, y + 7, pct(v), 21, INK, 700 if s == "APTOS" else 400))
    return page("Why disagreement lost", "The evidence pathway disputed correct grades, and it "
                "could not see severe disease",
                "".join(b),
                "M3 has no rule above grade 2: without a disc–fovea frame it cannot apply the "
                "severe (4-2-1) rule, and nothing annotates new vessels, so disagreement is zero "
                "whenever the grader says 3 or 4. Means of three seeds, primary variant; exploratory "
                "(Phase 8).",
                (("hollow", ORANGE, "Pre-registered"), ("dot", ORANGE, "Amended")))


def fig_faithfulness() -> str:
    ys = (372, 492, 612)
    x0, w = 360, 1060
    sx = scale(x0, w, 0.0, 1.0)
    b = []
    for t in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        b.append(line(sx(t), 300, sx(t), 660))
        b.append(text(sx(t), 694, f"{100 * t:.0f}%", 18, MUTED, anchor="middle"))
    b.append(line(sx(0.05), 300, sx(0.05), 660, MUTED, 2, "6 6"))
    b.append(text(sx(0.05) + 8, 292, "chance: 1 in 20", 17, MUTED))
    for s, y in zip(SETS, ys):
        m, lo, hi, n = FAITHFUL[s]
        b.append(text(80, y - 2, s, 25, INK, 700))
        b.append(text(80, y + 25, f"{n:,} lesion images", 19, MUTED))
        b.append(bar(sx(0), sx(m), y, ORANGE))
        b.append(line(sx(lo), y, sx(hi), y, INK, 2))
        b.append(line(sx(lo), y - 9, sx(lo), y + 9, INK, 2))
        b.append(line(sx(hi), y - 9, sx(hi), y + 9, INK, 2))
        b.append(text(sx(hi) + 14, y + 8, pct(m), 22, INK, 700))
    return page("Verification works: the grader uses the lesions M2 finds",
                "Share of lesion images where removing M2's lesions moved the grader more than "
                "all 19 random removals of the same area",
                "".join(b),
                "Pre-registered test E2, primary variant. Bar: mean of three seeds; whisker: the "
                "range. Faithfulness does not make disagreement a good trust signal; it shows the "
                "grader attends to the same lesions.")


def fig_examples() -> str:
    from PIL import Image
    captions = (
        ("(a) The intended case", "Messidor-2 · adjudicated grade 1", "Grader: 0, confidence 0.978",
         "M3: grade 1, microaneurysms"),
        ("(b) Outside the retina", "EyePACS · true grade 3", "Grader: 0, confidence 0.983",
         "Largest outlines off the retina"),
        ("(c) Right grade, wrong reason?", "EyePACS · true grade 2", "Grader: 0, confidence 0.982",
         "Outlines on debris-like specks"),
    )
    b = []
    for i, ((letter, name, column, row, box_), (head, l1, l2, l3)) in enumerate(zip(F71.ROWS, captions)):
        sheet = Image.open(F71.SHEETS / name).convert("RGB")
        x0 = 80 + i * 490
        crop = F71.region(sheet, column, row, True, box_).resize((880, 880), Image.LANCZOS)
        whole = F71.region(sheet, column, row, False, (0, 0, F71.IMAGE, F71.IMAGE)).resize(
            (260, 260), Image.LANCZOS)
        k = 260 / F71.IMAGE
        ImageDraw.Draw(whole).rectangle([round(v * k) for v in box_], outline=F71.BOX_COLOUR, width=4)
        b.append(text(x0, 200, head, 24, INK, 700))
        b.append(f'<image x="{x0}" y="218" width="440" height="440" href="{png_uri(crop)}"/>')
        b.append(f'<image x="{x0}" y="676" width="130" height="130" href="{png_uri(whole)}"/>')
        for j, (s, size, fill, weight) in enumerate(((l1, 19, INK, 700), (l2, 19, INK2, 400),
                                                     (l3, 19, INK2, 400))):
            b.append(text(x0 + 148, 704 + j * 30, s, fit(s, size, 290, weight >= 600), fill, weight))
    return page("The case the design was built for, and what its evidence looks like",
                "Eyes the grader called normal with high confidence, where M2 found lesions and "
                "the eye was diseased",
                "".join(b),
                "Outlines: cyan microaneurysm, green haemorrhage, magenta hard exudate; the yellow "
                "box marks the enlarged region. Chosen by eye from images meeting a rule declared "
                "in advance: 24 of 17,615 EyePACS test images, none on APTOS, 4 of 1,744 Messidor-2.")


FIGURES = {
    "01_method": fig_method, "02_timeline": fig_timeline, "03_headline_auc": fig_headline,
    "04_errors_deferred": fig_errors, "05_referral_errors": fig_referral,
    "06_why_disagreement_lost": fig_why, "07_faithfulness": fig_faithfulness,
    "08_examples": fig_examples,
}


def render(svgs: dict) -> None:
    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    exe = Path("/opt/pw-browsers/chromium")
    with sync_playwright() as p:
        browser = p.chromium.launch(**({"executable_path": str(exe)} if exe.exists() else {}))
        tab = browser.new_page(viewport={"width": W, "height": H}, device_scale_factor=2)
        for name, svg in svgs.items():
            tab.set_content("<!doctype html><meta charset='utf-8'>"
                            "<style>html,body{margin:0;background:#fff}</style>" + svg)
            tab.evaluate("document.fonts.ready")
            tab.screenshot(path=str(OUT / f"{name}.png"), clip={"x": 0, "y": 0, "width": W, "height": H})
            print(f"{(OUT / name).relative_to(ROOT)}.png")
        browser.close()


def main() -> None:
    check_numbers()
    render({name: make() for name, make in FIGURES.items()})


if __name__ == "__main__":
    main()
