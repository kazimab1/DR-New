"""Figure 7.1 -- the figure-1 case, and what its evidence looks like.

Built from the two candidate sheets in docs/phase8/, which `scripts/phase8.py figures`
rendered at 512 px and which arrived scaled by 0.86 (EyePACS) and 0.97 (Messidor-2).
Each row: the image as the models saw it, with the enlarged region boxed; the region
enlarged; the same region with M2's outlines. Run from the repository root:

    python docs/thesis/figures/make_fig7_1.py
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
SHEETS = ROOT / "docs" / "phase8"
OUT = Path(__file__).resolve().parent / "fig7_1.png"

# The sheet layout written by phase8.grid/render_panel: a 26 px legend strip, then
# panels of 1030 x 572 (two 512 px images 6 px apart, captions below), two per row.
PANEL_W, PANEL_H, LEGEND_H, IMAGE, GAP = 1030, 572, 26, 512, 6

ROWS = [  # (letter, sheet, column, row, box in the 512 px image)
    ("a", "2026-09-29_figure1_candidates_messidor2.webp", 0, 0, (240, 372, 380, 512)),
    ("b", "2026-09-29_figure1_candidates_eyepacs_test.webp", 1, 1, (0, 0, 150, 150)),
    ("c", "2026-09-29_figure1_candidates_eyepacs_test.webp", 0, 2, (40, 170, 190, 320)),
]
HEADERS = ["As the models saw it", "Enlarged", "With M2's outlines"]

CELL, COL_GAP, ROW_GAP, MARGIN, LABEL_W, HEADER_H = 480, 16, 24, 20, 44, 40
BOX_COLOUR = (255, 212, 0)          # no lesion type is drawn in yellow


def font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)
    except OSError:
        return ImageFont.load_default(size=size)


def region(sheet: Image.Image, column: int, row: int, outlined: bool, box) -> Image.Image:
    """A box in one panel's 512 px image, mapped through the sheet's scale."""
    scale = sheet.size[0] / (2 * PANEL_W)
    ox = column * PANEL_W + (IMAGE + GAP if outlined else 0)
    oy = LEGEND_H + row * PANEL_H
    x0, y0, x1, y1 = box
    return sheet.crop(tuple(round(v * scale) for v in
                            (ox + x0, oy + y0, ox + x1, oy + y1)))


def main() -> None:
    width = 2 * MARGIN + LABEL_W + 3 * CELL + 2 * COL_GAP
    height = 2 * MARGIN + HEADER_H + 3 * CELL + 2 * ROW_GAP
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    x0 = MARGIN + LABEL_W
    for i, text in enumerate(HEADERS):
        draw.text((x0 + i * (CELL + COL_GAP), MARGIN), text, fill=(20, 24, 31), font=font(22))
    for r, (letter, name, column, row, box) in enumerate(ROWS):
        sheet = Image.open(SHEETS / name).convert("RGB")
        y = MARGIN + HEADER_H + r * (CELL + ROW_GAP)
        draw.text((MARGIN, y), letter, fill=(20, 24, 31), font=font(30, bold=True))
        whole = region(sheet, column, row, False, (0, 0, IMAGE, IMAGE)).resize(
            (CELL, CELL), Image.LANCZOS)
        k = CELL / IMAGE
        ImageDraw.Draw(whole).rectangle([round(v * k) for v in box], outline=BOX_COLOUR, width=3)
        canvas.paste(whole, (x0, y))
        for i, outlined in enumerate((False, True), start=1):
            crop = region(sheet, column, row, outlined, box).resize((CELL, CELL), Image.LANCZOS)
            canvas.paste(crop, (x0 + i * (CELL + COL_GAP), y))
    canvas.save(OUT, dpi=(300, 300), optimize=True)
    print(f"{OUT.relative_to(ROOT)}: {canvas.size[0]} x {canvas.size[1]}")


if __name__ == "__main__":
    main()
