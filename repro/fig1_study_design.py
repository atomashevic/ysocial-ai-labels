#!/usr/bin/env python3
"""Fig. 1: study-design schematic, drawn as vector SVG, PDF and PNG.

The schematic reads no data; its account counts and cohort sizes are fixed text.
It is drawn from shapes and text only. PDF and PNG conversion uses
``rsvg-convert`` (librsvg); use ``--svg-only`` without it.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence
from xml.sax.saxutils import escape


from repro.paths import FIGURES_DIR

DEFAULT_OUTPUT_DIR = FIGURES_DIR
DEFAULT_STEM = "fig1_study_design"

FIGURE_WIDTH_MM = 180.0
WIDTH = 510.0  # one user unit is 180/510 mm, within 0.05% of one point
MIN_FONT_SIZE = 8.8
PNG_DPI = 300
FONT_FAMILY = "Helvetica, Arial, 'Liberation Sans', 'Nimbus Sans', sans-serif"

INK = "#1C2230"
MUTED = "#566070"
CONNECTOR = "#98A1AD"
DIVIDER = "#DCE0E6"
CARD_FILL = "#F4F6F9"
CARD_STROKE = "#D5DAE1"
SKELETON = "#D3D8DF"
AVATAR = "#C3CAD3"
THUMB = "#DDE3EA"
CHIP = "#2F3B52"
PAGE_BG = "#EEF0F3"
TITLE_BAR = "#4A5263"
PILL_BAR = "#B9C0CA"
WHITE = "#FFFFFF"
VISIBLE = "#0072B2"
HIDDEN = "#D55E00"
HUMAN = "#009E73"
AI = "#CC79A7"
AI_BADGE = "#A94C84"

# Panel frames in user units.
A_X, A_W = 8.0, 142.0
BUS_X = 158.0
B_X, B_W = 168.0, 334.0
COLUMN_GAP = 10.0
C_X, C_W = 8.0, 170.0
D_X, D_W = 192.0, 310.0
RIGHT = 502.0

# Helvetica advance widths per 1000 em for ASCII 32-126. Arial, Liberation Sans
# and Nimbus Sans share these metrics.
_REGULAR = (
    "278 278 355 556 556 889 667 191 333 333 389 584 278 333 278 278 "
    "556 556 556 556 556 556 556 556 556 556 278 278 584 584 584 556 1015 "
    "667 667 722 722 667 611 778 722 278 500 667 556 833 722 778 667 778 722 "
    "667 611 722 667 944 667 667 611 278 278 278 469 556 333 "
    "556 556 500 556 556 278 556 556 222 222 500 222 833 556 556 556 556 333 "
    "500 278 556 500 722 500 500 500 334 260 334 584"
)
_BOLD = (
    "278 333 474 556 556 889 722 238 333 333 389 584 278 333 278 278 "
    "556 556 556 556 556 556 556 556 556 556 333 333 584 584 584 611 975 "
    "722 722 722 722 667 611 778 722 278 556 722 611 833 722 778 667 778 722 "
    "667 611 722 667 944 667 667 611 333 278 333 584 556 333 "
    "556 611 556 611 556 333 611 611 278 278 556 278 889 611 611 611 611 389 "
    "556 333 611 556 778 556 556 500 389 280 389 584"
)
_EXTRA = {"·": 278, "…": 1000, "–": 556}


def _width_table(values: str) -> dict[str, int]:
    widths = [int(value) for value in values.split()]
    if len(widths) != 95:
        raise ValueError("width table must cover ASCII 32-126")
    return {chr(32 + index): width for index, width in enumerate(widths)} | _EXTRA


REGULAR_WIDTHS = _width_table(_REGULAR)
BOLD_WIDTHS = _width_table(_BOLD)


def text_width(text: str, size: float, *, bold: bool = False) -> float:
    """Return the advance width of ``text`` in user units."""

    table = BOLD_WIDTHS if bold else REGULAR_WIDTHS
    fallback = 611 if bold else 556
    return sum(table.get(char, fallback) for char in text) * size / 1000.0


def _fmt(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def _d(*tokens: object) -> str:
    """Join path tokens, formatting numbers compactly."""

    return " ".join(
        _fmt(float(token)) if isinstance(token, (int, float)) else str(token)
        for token in tokens
    )


def _attrs(pairs: dict[str, object]) -> str:
    parts = []
    for key, value in pairs.items():
        if value is None:
            continue
        name = key.rstrip("_").replace("_", "-")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = _fmt(float(value))
        parts.append(f'{name}="{escape(str(value), {chr(34): "&quot;"})}"')
    return " ".join(parts)


@dataclass(frozen=True)
class TextBox:
    text: str
    x0: float
    x1: float
    baseline: float
    size: float
    bold: bool


class Canvas:
    """Collect SVG elements and the horizontal extent of every text run."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.texts: list[TextBox] = []

    def element(self, tag: str, **attrs: object) -> None:
        self.parts.append(f"<{tag} {_attrs(attrs)}/>")

    def rect(self, x, y, w, h, *, fill=WHITE, stroke=None, stroke_width=None, rx=None) -> None:
        self.element(
            "rect", x=x, y=y, width=w, height=h, rx=rx, fill=fill,
            stroke=stroke, stroke_width=stroke_width,
        )

    def circle(self, cx, cy, r, *, fill=WHITE, stroke=None, stroke_width=None) -> None:
        self.element("circle", cx=cx, cy=cy, r=r, fill=fill, stroke=stroke, stroke_width=stroke_width)

    def line(self, x1, y1, x2, y2, *, stroke=DIVIDER, width=0.6, dash=None) -> None:
        self.element(
            "line", x1=x1, y1=y1, x2=x2, y2=y2, stroke=stroke, stroke_width=width,
            stroke_linecap="round", stroke_dasharray=dash,
        )

    def path(self, d: str, *, fill="none", stroke=None, width=None) -> None:
        self.element(
            "path", d=d, fill=fill, stroke=stroke, stroke_width=width,
            stroke_linecap="round" if stroke else None,
            stroke_linejoin="round" if stroke else None,
        )

    def polygon(self, points: Sequence[tuple[float, float]], *, fill: str) -> None:
        self.element("polygon", points=" ".join(f"{_fmt(x)},{_fmt(y)}" for x, y in points), fill=fill)

    def text(
        self, x, y, text: str, *, size=9.0, bold=False, italic=False, fill=INK,
        anchor="start", max_width: float | None = None,
    ) -> float:
        if size < MIN_FONT_SIZE - 1e-9:
            raise ValueError(f"{text!r}: font size {size} is below {MIN_FONT_SIZE}")
        width = text_width(text, size, bold=bold)
        if max_width is not None and width > max_width + 1e-9:
            raise ValueError(f"{text!r} needs {width:.1f} units but has {max_width:.1f}")
        x0 = {"start": x, "middle": x - width / 2.0, "end": x - width}[anchor]
        self.texts.append(TextBox(text, x0, x0 + width, y, size, bold))
        attrs = _attrs({
            "x": x, "y": y, "font_size": size,
            "font_weight": "bold" if bold else None,
            "font_style": "italic" if italic else None,
            "fill": fill,
            "text_anchor": None if anchor == "start" else anchor,
        })
        self.parts.append(f"<text {attrs}>{escape(text)}</text>")
        return width

    def arrow(self, x1, y1, x2, y2, *, color=CONNECTOR, width=0.9, head=4.0) -> None:
        length = math.hypot(x2 - x1, y2 - y1)
        ux, uy = (x2 - x1) / length, (y2 - y1) / length
        bx, by = x2 - ux * head, y2 - uy * head
        self.line(x1, y1, bx + ux * 0.4, by + uy * 0.4, stroke=color, width=width)
        half = head * 0.48
        self.polygon(
            [(x2, y2), (bx - uy * half, by + ux * half), (bx + uy * half, by - ux * half)],
            fill=color,
        )

    # Icons. Each is centred on (cx, cy) and scaled by ``s``.

    def person(self, cx, cy, s, color) -> None:
        self.circle(cx, cy - 0.23 * s, 0.2 * s, fill=color)
        self.path(
            _d("M", cx - 0.36 * s, cy + 0.45 * s, "A", 0.36 * s, 0.42 * s, 0, 0, 1,
               cx + 0.36 * s, cy + 0.45 * s, "Z"),
            fill=color,
        )

    def robot(self, cx, cy, s, color) -> None:
        self.line(cx, cy - 0.3 * s, cx, cy - 0.42 * s, stroke=color, width=0.07 * s)
        self.circle(cx, cy - 0.45 * s, 0.07 * s, fill=color)
        self.rect(cx - 0.44 * s, cy - 0.1 * s, 0.1 * s, 0.22 * s, fill=color, rx=0.04 * s)
        self.rect(cx + 0.34 * s, cy - 0.1 * s, 0.1 * s, 0.22 * s, fill=color, rx=0.04 * s)
        self.rect(cx - 0.36 * s, cy - 0.3 * s, 0.72 * s, 0.64 * s, fill=color, rx=0.16 * s)
        self.circle(cx - 0.15 * s, cy - 0.02 * s, 0.075 * s, fill=WHITE)
        self.circle(cx + 0.15 * s, cy - 0.02 * s, 0.075 * s, fill=WHITE)
        self.rect(cx - 0.13 * s, cy + 0.15 * s, 0.26 * s, 0.06 * s, fill=WHITE, rx=0.03 * s)

    def avatar(self, cx, cy, r) -> None:
        """Neutral forum avatar, identical for every author."""

        self.circle(cx, cy, r, fill=AVATAR)
        self.circle(cx, cy - 0.24 * r, 0.28 * r, fill=WHITE)
        self.path(
            _d("M", cx - 0.5 * r, cy + 0.62 * r, "A", 0.5 * r, 0.52 * r, 0, 0, 1,
               cx + 0.5 * r, cy + 0.62 * r, "Z"),
            fill=WHITE,
        )

    def id_card(self, cx, cy, s, color) -> None:
        w = 0.08 * s
        self.rect(cx - 0.46 * s, cy - 0.32 * s, 0.92 * s, 0.64 * s, stroke=color, stroke_width=w, rx=0.08 * s)
        self.circle(cx - 0.2 * s, cy - 0.07 * s, 0.1 * s, fill=color)
        self.path(
            _d("M", cx - 0.34 * s, cy + 0.2 * s, "A", 0.14 * s, 0.15 * s, 0, 0, 1,
               cx - 0.06 * s, cy + 0.2 * s, "Z"),
            fill=color,
        )
        self.line(cx + 0.04 * s, cy - 0.08 * s, cx + 0.3 * s, cy - 0.08 * s, stroke=color, width=w)
        self.line(cx + 0.04 * s, cy + 0.1 * s, cx + 0.22 * s, cy + 0.1 * s, stroke=color, width=w)

    def database(self, cx, cy, s, color) -> None:
        w = 0.08 * s
        rx, ry = 0.34 * s, 0.11 * s
        top, bottom = cy - 0.3 * s, cy + 0.3 * s
        self.path(
            _d("M", cx - rx, top, "V", bottom, "A", rx, ry, 0, 0, 0, cx + rx, bottom, "V", top),
            fill=WHITE, stroke=color, width=w,
        )
        self.path(_d("M", cx - rx, cy, "A", rx, ry, 0, 0, 0, cx + rx, cy), stroke=color, width=w)
        self.element("ellipse", cx=cx, cy=top, rx=rx, ry=ry, fill=WHITE, stroke=color, stroke_width=w)

    def calendar(self, cx, cy, s, color) -> None:
        w = 0.08 * s
        self.rect(cx - 0.42 * s, cy - 0.34 * s, 0.84 * s, 0.76 * s, stroke=color, stroke_width=w, rx=0.1 * s)
        self.rect(cx - 0.42 * s, cy - 0.34 * s, 0.84 * s, 0.24 * s, fill=color, rx=0.1 * s)
        self.rect(cx - 0.42 * s, cy - 0.2 * s, 0.84 * s, 0.1 * s, fill=color)
        for dx in (-0.2, 0.2):
            self.line(cx + dx * s, cy - 0.45 * s, cx + dx * s, cy - 0.28 * s, stroke=color, width=w)
        for dy in (0.04, 0.22):
            for dx in (-0.22, 0.0, 0.22):
                self.rect(cx + (dx - 0.05) * s, cy + (dy - 0.05) * s, 0.1 * s, 0.1 * s, fill=color)

    def rss(self, cx, cy, s, color) -> None:
        w = 0.14 * s
        self.circle(cx - 0.3 * s, cy + 0.3 * s, 0.1 * s, fill=color)
        self.path(
            _d("M", cx - 0.4 * s, cy - 0.05 * s, "A", 0.36 * s, 0.36 * s, 0, 0, 1, cx + 0.05 * s, cy + 0.4 * s),
            stroke=color, width=w,
        )
        self.path(
            _d("M", cx - 0.4 * s, cy - 0.4 * s, "A", 0.72 * s, 0.72 * s, 0, 0, 1, cx + 0.4 * s, cy + 0.4 * s),
            stroke=color, width=w,
        )

    def doc_stack(self, cx, cy, s, color) -> None:
        w = 0.07 * s
        self.rect(cx - 0.2 * s, cy - 0.44 * s, 0.58 * s, 0.74 * s, stroke=color, stroke_width=w, rx=0.06 * s)
        self.rect(cx - 0.38 * s, cy - 0.3 * s, 0.58 * s, 0.74 * s, stroke=color, stroke_width=w, rx=0.06 * s)
        for dy, length in ((-0.12, 0.34), (0.04, 0.34), (0.2, 0.22)):
            self.line(cx - 0.28 * s, cy + dy * s, cx + (length - 0.28) * s, cy + dy * s, stroke=color, width=0.06 * s)

    def tag(self, cx, cy, s, color) -> None:
        self.path(
            _d("M", cx - 0.45 * s, cy - 0.27 * s, "H", cx + 0.12 * s, "L", cx + 0.45 * s, cy,
               "L", cx + 0.12 * s, cy + 0.27 * s, "H", cx - 0.45 * s, "Z"),
            fill=color,
        )
        self.circle(cx + 0.12 * s, cy, 0.07 * s, fill=WHITE)

    def monitor(self, cx, cy, s, color) -> None:
        w = 0.07 * s
        self.rect(cx - 0.46 * s, cy - 0.36 * s, 0.92 * s, 0.58 * s, stroke=color, stroke_width=w, rx=0.06 * s)
        self.path(
            _d("M", cx - 0.1 * s, cy + 0.22 * s, "L", cx - 0.14 * s, cy + 0.38 * s,
               "M", cx + 0.1 * s, cy + 0.22 * s, "L", cx + 0.14 * s, cy + 0.38 * s,
               "M", cx - 0.22 * s, cy + 0.4 * s, "H", cx + 0.22 * s),
            stroke=color, width=w,
        )
        self.rect(cx - 0.34 * s, cy - 0.25 * s, 0.22 * s, 0.17 * s, fill=color, rx=0.02 * s)
        self.line(cx - 0.04 * s, cy - 0.21 * s, cx + 0.32 * s, cy - 0.21 * s, stroke=color, width=0.05 * s)
        self.line(cx - 0.04 * s, cy - 0.11 * s, cx + 0.22 * s, cy - 0.11 * s, stroke=color, width=0.05 * s)
        self.line(cx - 0.34 * s, cy + 0.04 * s, cx + 0.32 * s, cy + 0.04 * s, stroke=color, width=0.05 * s)

    def chevron(self, cx, cy, s, color, *, up: bool) -> None:
        d = -1.0 if up else 1.0
        self.path(
            _d("M", cx - 0.4 * s, cy - d * 0.2 * s, "L", cx, cy + d * 0.2 * s, "L", cx + 0.4 * s, cy - d * 0.2 * s),
            stroke=color, width=0.9,
        )

    def external_link(self, cx, cy, s, color) -> None:
        w = 0.1 * s
        self.path(
            _d("M", cx + 0.02 * s, cy - 0.4 * s, "H", cx - 0.4 * s, "V", cy + 0.4 * s,
               "H", cx + 0.4 * s, "V", cy - 0.02 * s),
            stroke=color, width=w,
        )
        self.path(
            _d("M", cx - 0.06 * s, cy + 0.06 * s, "L", cx + 0.42 * s, cy - 0.42 * s,
               "M", cx + 0.1 * s, cy - 0.42 * s, "H", cx + 0.42 * s, "V", cy - 0.1 * s),
            stroke=color, width=w,
        )

    def downvote(self, cx, cy, s, color) -> None:
        self.path(
            _d("M", cx - 0.17 * s, cy - 0.42 * s, "H", cx + 0.17 * s, "V", cy, "H", cx + 0.38 * s,
               "L", cx, cy + 0.42 * s, "L", cx - 0.38 * s, cy, "H", cx - 0.17 * s, "Z"),
            fill=WHITE, stroke=color, width=0.08 * s,
        )

    def warning(self, cx, cy, s, color) -> None:
        self.path(
            _d("M", cx, cy - 0.42 * s, "L", cx + 0.47 * s, cy + 0.38 * s, "H", cx - 0.47 * s, "Z"),
            fill=WHITE, stroke=color, width=0.08 * s,
        )
        self.line(cx, cy - 0.12 * s, cx, cy + 0.1 * s, stroke=color, width=0.09 * s)
        self.circle(cx, cy + 0.24 * s, 0.05 * s, fill=color)

    def speech(self, cx, cy, s, color) -> None:
        self.path(
            _d("M", cx - 0.3 * s, cy - 0.36 * s, "H", cx + 0.3 * s,
               "Q", cx + 0.46 * s, cy - 0.36 * s, cx + 0.46 * s, cy - 0.2 * s,
               "V", cy + 0.08 * s, "Q", cx + 0.46 * s, cy + 0.24 * s, cx + 0.3 * s, cy + 0.24 * s,
               "H", cx - 0.02 * s, "L", cx - 0.24 * s, cy + 0.42 * s, "V", cy + 0.24 * s,
               "H", cx - 0.3 * s, "Q", cx - 0.46 * s, cy + 0.24 * s, cx - 0.46 * s, cy + 0.08 * s,
               "V", cy - 0.2 * s, "Q", cx - 0.46 * s, cy - 0.36 * s, cx - 0.3 * s, cy - 0.36 * s, "Z"),
            fill=WHITE, stroke=color, width=0.08 * s,
        )
        self.line(cx - 0.26 * s, cy - 0.14 * s, cx + 0.26 * s, cy - 0.14 * s, stroke=color, width=0.06 * s)
        self.line(cx - 0.26 * s, cy + 0.02 * s, cx + 0.1 * s, cy + 0.02 * s, stroke=color, width=0.06 * s)


# Shared components.

def panel_heading(c: Canvas, x: float, baseline: float, letter: str, title: str) -> None:
    c.text(x, baseline, letter, size=12, bold=True)
    c.text(x + 12, baseline, title, size=11, bold=True)


def card(c: Canvas, x: float, y: float, w: float, h: float) -> None:
    c.rect(x, y, w, h, fill=CARD_FILL, rx=5)


def label_chip(c: Canvas, x: float, y: float, label: str, *, filled: bool) -> float:
    width = text_width(label, 8.8, bold=True) + 9.0
    if filled:
        c.rect(x, y, width, 11.5, fill=CHIP, rx=2.8)
    else:
        c.rect(x + 0.4, y + 0.4, width - 0.8, 10.7, fill=WHITE, stroke=CHIP, stroke_width=0.8, rx=2.6)
    c.text(x + width / 2.0, y + 8.3, label, size=8.8, bold=True, fill=WHITE if filled else CHIP, anchor="middle")
    return width


def topic_chip(c: Canvas, x: float, y: float, label: str) -> float:
    width = text_width(label, 9.0) + 8.0
    c.rect(x, y, width, 12, fill=WHITE, stroke=CARD_STROKE, stroke_width=0.8, rx=6)
    c.text(x + 4.0, y + 8.7, label, size=9.0)
    return width


# Panel a: setup.

def draw_setup(c: Canvas, x: float, y: float, w: float, height: float) -> list[float]:
    """Draw panel a and return the heights where connectors leave its cards."""

    inner = w - 16.0
    gap = 6.0
    card(c, x, y, w, 58)
    c.rss(x + 12, y + 10.5, 9, MUTED)
    c.text(x + 21, y + 14, "External RSS feeds", size=9.5, bold=True, max_width=w - 29)
    cursor = x + 8
    for label in ("Music", "Film", "Television", "\u2026"):
        if label == "\u2026":
            cursor += c.text(cursor + 2, y + 26.7, label, size=9.0, fill=MUTED) + 2
        else:
            cursor += topic_chip(c, cursor, y + 18, label) + 3
    if cursor > x + w - 8:
        raise ValueError("topic chips overflow panel a")
    c.text(x + 8, y + 41, "Publisher and", size=8.8, fill=MUTED, max_width=inner)
    c.text(x + 8, y + 50, "subreddit sources", size=8.8, fill=MUTED, max_width=inner)

    ai_y = y + 58 + gap
    card(c, x, ai_y, w, 38)
    c.robot(x + 12, ai_y + 10.5, 10, AI)
    c.text(x + 21, ai_y + 14, "AI accounts", size=9.5, bold=True, max_width=w - 29)
    c.id_card(x + 14, ai_y + 25.5, 10, MUTED)
    c.text(x + 23, ai_y + 28.7, "Personas", size=9)
    c.database(x + 72, ai_y + 25.5, 10, MUTED)
    c.text(x + 80, ai_y + 28.7, "Memory", size=9, max_width=w - 88)

    small_h = (y + height - (ai_y + 38 + 2 * gap)) / 2.0
    human_y = ai_y + 38 + gap
    card(c, x, human_y, w, small_h)
    for dx in (10, 19, 28):
        c.person(x + dx, human_y + small_h / 2.0, 8.5, HUMAN)
    c.text(x + 37, human_y + small_h / 2.0 + 3.4, "Humans", size=9.5, bold=True)

    days_y = human_y + small_h + gap
    card(c, x, days_y, w, small_h)
    c.calendar(x + 13, days_y + small_h / 2.0, 11, MUTED)
    c.text(x + 23, days_y + small_h / 2.0 + 3.4, "21 experiment days", size=9.5, bold=True, max_width=w - 31)
    return [y + 29, ai_y + 19, human_y + small_h / 2.0]


# Panel b: interaction, drawn after the YSocial thread layout.

def small_arrow(c: Canvas, cx: float, cy: float, *, up: bool) -> None:
    d = -1.0 if up else 1.0
    c.path(
        _d("M", cx - 1.7, cy + d * 0.3, "L", cx, cy + d * 2.0, "L", cx + 1.7, cy + d * 0.3,
           "M", cx, cy + d * 2.0, "V", cy - d * 2.0),
        stroke=MUTED, width=0.75,
    )


def gear(c: Canvas, cx: float, cy: float, s: float, color: str) -> None:
    for k in range(8):
        angle = k * math.pi / 4.0
        c.line(
            cx + 0.3 * s * math.cos(angle), cy + 0.3 * s * math.sin(angle),
            cx + 0.46 * s * math.cos(angle), cy + 0.46 * s * math.sin(angle),
            stroke=color, width=0.16 * s,
        )
    c.circle(cx, cy, 0.3 * s, fill="none", stroke=color, stroke_width=0.14 * s)


def kebab(c: Canvas, x: float, y: float) -> None:
    c.rect(x, y, 13, 10.5, fill=WHITE, stroke=CARD_STROKE, stroke_width=0.7, rx=3)
    for dy in (2.9, 5.25, 7.6):
        c.circle(x + 6.5, y + dy, 0.75, fill=INK)


def reputation_pill(c: Canvas, x: float, y: float) -> float:
    width = 26.0
    c.rect(x, y, width, 9, fill=PAGE_BG, rx=4.5)
    gear(c, x + 6, y + 4.5, 6.5, MUTED)
    c.rect(x + 11, y + 3.4, 11, 2.2, fill=PILL_BAR, rx=1.1)
    return width


VOTE_BOX_H = 25.0


def vote_box(c: Canvas, x: float, y: float, votes: int) -> None:
    """Vertical vote control: up arrow, count, down arrow."""

    c.rect(x, y, 12, VOTE_BOX_H, fill=PAGE_BG, rx=3)
    small_arrow(c, x + 6, y + 5.4, up=True)
    c.text(x + 6, y + 15.65, str(votes), size=8.8, bold=True, anchor="middle")
    small_arrow(c, x + 6, y + 19.6, up=False)


def contribution_height(*, title: bool, body_lines: int) -> float:
    return 21.5 + (7.5 if title else 0.0) + 5.5 * body_lines + 1.5 + VOTE_BOX_H + 4


def contribution(
    c: Canvas, x: float, y: float, w: float, user: str, *, ai_label: bool,
    title: bool, body: Sequence[float], reply_icon: bool, votes: int,
) -> float:
    """One YSocial post or comment card; returns its height."""

    height = contribution_height(title=title, body_lines=len(body))
    c.rect(x, y, w, height, fill=WHITE, rx=4)
    c.avatar(x + 10, y + 10, 5)
    user_w = c.text(x + 18, y + 9.5, user, size=9, bold=True)
    pill_x = x + 18 + user_w + 4
    pill_w = reputation_pill(c, pill_x, y + 2.5)
    if ai_label:
        c.parts.append('<g class="ai-label">')
        c.robot(pill_x + pill_w + 7, y + 7, 9, AI)
        c.parts.append("</g>")
    c.rect(x + 18, y + 13.3, 30, 2.2, fill=SKELETON, rx=1.1)
    kebab(c, x + w - 19, y + 3)
    row = y + 21.5
    if title:
        c.rect(x + 7, row, w * 0.62, 3.4, fill=TITLE_BAR, rx=1.7)
        row += 7.5
    for fraction in body:
        c.rect(x + 7, row, (w - 14) * fraction, 2.6, fill=SKELETON, rx=1.3)
        row += 5.5
    vote_y = row + 1.5
    vote_box(c, x + 7, vote_y, votes)
    if reply_icon:
        c.speech(x + 27, vote_y + 12.6, 8, MUTED)
        reply_x = x + 33
    else:
        reply_x = x + 25
    c.text(reply_x, vote_y + 15.65, "Reply", size=8.8, bold=True, fill=MUTED)
    return height


def forum_mock(c: Canvas, x: float, y: float, w: float, color: str, *, ai_label: bool) -> float:
    """Illustrative YSocial thread: a participant's post and an AI account's comment.

    Avatars are identical for every author; only the robot icon after the
    reputation pill marks the AI account, as in the visible-label interface.
    """

    pad = 6.0
    post_h = contribution_height(title=True, body_lines=2)
    comment_h = contribution_height(title=False, body_lines=1)
    height = pad + post_h + 5 + comment_h + pad
    c.rect(x, y, w, height, fill=PAGE_BG, stroke=color, stroke_width=1.1, rx=5)
    post_votes, comment_votes = THREAD_VOTES
    contribution(c, x + pad, y + pad, w - 2 * pad, "user_a", ai_label=False,
                 title=True, body=(0.96, 0.58), reply_icon=True, votes=post_votes)
    contribution(c, x + pad + 8, y + pad + post_h + 5, w - 2 * pad - 8, "user_b", ai_label=ai_label,
                 title=False, body=(0.86,), reply_icon=False, votes=comment_votes)
    return height


# Illustrative vote counts for the post and the comment. Both communities show
# the same values so the example implies no difference in voting.
THREAD_VOTES = (4, 1)

COMMUNITIES = (
    ("Visible labels", VISIBLE, "20 humans", "40 AI accounts", True),
    ("Hidden labels", HIDDEN, "16 humans", "32 AI accounts", False),
)


def draw_communities(c: Canvas, x: float, y: float, w: float) -> tuple[list[float], float]:
    """Draw the two communities side by side; return column centres and bottom edge."""

    column_w = (w - COLUMN_GAP) / 2.0
    centres = []
    bottom = y
    for index, (title, color, humans, accounts, ai_label) in enumerate(COMMUNITIES):
        left = x + index * (column_w + COLUMN_GAP)
        title_w = c.text(left, y + 9.5, title, size=10, bold=True, fill=color)
        note_w = c.text(left + column_w, y + 9.5, "Illustrative UI", size=8.8, italic=True, fill=MUTED, anchor="end")
        if title_w + note_w + 8 > column_w:
            raise ValueError("community header overflows")
        c.person(left + 4.5, y + 18.5, 9, HUMAN)
        humans_w = c.text(left + 12, y + 21.8, humans, size=9)
        robot_x = left + 12 + humans_w + 10
        c.robot(robot_x + 4.5, y + 18.5, 9, AI)
        c.text(robot_x + 12, y + 21.8, accounts, size=9, max_width=left + column_w - robot_x - 12)
        height = forum_mock(c, left, y + 27, column_w, color, ai_label=ai_label)
        centres.append(left + column_w / 2.0)
        bottom = max(bottom, y + 27 + height)
    return centres, bottom


# Panel c: data.

def draw_data(c: Canvas, x: float, y: float, w: float, bottom: float) -> None:
    card(c, x, y, w, 28)
    c.doc_stack(x + 13, y + 14, 13, MUTED)
    c.text(x + 25, y + 12, "Posts \u00b7 replies \u00b7 votes", size=9.5, bold=True, max_width=w - 33)
    c.text(x + 25, y + 23, "Text + timestamps", size=9, fill=MUTED, max_width=w - 33)
    c.arrow(x + w / 2.0, y + 29.2, x + w / 2.0, y + 37.8)

    notes_y = y + 39.5
    card(c, x, notes_y, w, 55)
    c.text(x + 8, notes_y + 12, "Annotations", size=9.5, bold=True)
    for index, label in enumerate(("Toxicity (Detoxify)", "Criticism", "AI-identity references")):
        chip_y = notes_y + 15.5 + index * 12.8
        width = text_width(label, 9.0)
        if width + 30 > w:
            raise ValueError(f"{label!r} overflows panel c")
        c.rect(x + 8, chip_y, width + 22, 11.5, fill=WHITE, stroke=CARD_STROKE, stroke_width=0.8, rx=5.75)
        c.tag(x + 16, chip_y + 5.75, 8, MUTED)
        c.text(x + 24, chip_y + 8.4, label, size=9)

    telemetry_y = notes_y + 59
    height = bottom - telemetry_y
    card(c, x, telemetry_y, w, height)
    middle = telemetry_y + height / 2.0
    c.monitor(x + 13, middle, 13, MUTED)
    c.text(x + 25, middle - 2, "Human browser telemetry", size=9.5, bold=True, max_width=w - 33)
    c.text(x + 25, middle + 9, "Page views \u00b7 feed-card renders", size=9, fill=MUTED, max_width=w - 33)


# Panel d: research questions.

def draw_choice(c: Canvas, x: float, top: float, h: float) -> None:
    """RQ1: a participant replies to one of several earlier contributions."""

    cy = top + h / 2.0
    c.person(x + 6, cy, 11, HUMAN)
    for index, row_y in enumerate((cy - 8.5, cy, cy + 8.5)):
        chosen = index == 1
        c.arrow(
            x + 13, cy, x + 39, row_y, color=INK if chosen else CONNECTOR,
            width=1.1 if chosen else 0.8, head=3.4,
        )
        c.rect(
            x + 40, row_y - 3.5, 58, 7, fill=WHITE, stroke=INK if chosen else CARD_STROKE,
            stroke_width=0.9 if chosen else 0.8, rx=2,
        )
        if chosen:
            c.robot(x + 46, row_y, 6, AI)
        else:
            c.person(x + 46, row_y, 6, HUMAN)
        c.rect(x + 52, row_y - 1.1, 40, 2.2, fill=SKELETON, rx=1.1)


def draw_entry(c: Canvas, x: float, top: float, h: float) -> None:
    """Thread entry: a recorded feed card, then the opened thread."""

    cy = top + h / 2.0
    c.rect(x + 2, cy - 9, 38, 18, fill=WHITE, stroke=CARD_STROKE, stroke_width=0.8, rx=2)
    c.rect(x + 6, cy - 4.5, 9, 9, fill=THUMB, rx=1)
    c.rect(x + 18, cy - 3.5, 18, 2.4, fill=SKELETON, rx=1.2)
    c.rect(x + 18, cy + 1.5, 12, 2.4, fill=SKELETON, rx=1.2)
    c.arrow(x + 44, cy, x + 60, cy, color=INK, width=1.1, head=3.6)
    page_x = x + 64
    c.rect(page_x, cy - 10, 38, 20, fill=WHITE, stroke=CARD_STROKE, stroke_width=0.8, rx=2)
    c.rect(page_x + 4, cy - 6.5, 22, 2.6, fill=MUTED, rx=1.3)
    for indent, row_y, length in ((4, cy - 1, 28), (10, cy + 3.5, 22), (10, cy + 7.5, 16)):
        c.rect(page_x + indent, row_y - 1.1, length, 2.2, fill=SKELETON, rx=1.1)


def draw_contact(c: Canvas, x: float, top: float, h: float) -> None:
    """RQ2: earlier replies in each direction, then a later reply to that account."""

    c.text(x + 2, top + 8.5, "Earlier", size=8.8, fill=MUTED)
    c.text(x + 64, top + 8.5, "Later", size=8.8, fill=MUTED)
    for row_y, from_ai in ((top + 17.5, True), (top + 29.0, False)):
        c.person(x + 8, row_y, 9, HUMAN)
        c.robot(x + 46, row_y, 9, AI)
        if from_ai:
            c.arrow(x + 39, row_y, x + 15, row_y, color=INK, width=1.0, head=3.6)
        else:
            c.arrow(x + 15, row_y, x + 39, row_y, color=INK, width=1.0, head=3.6)
    c.line(x + 58, top + 4, x + 58, top + h - 4, stroke=CARD_STROKE, width=0.8, dash="2 2")
    later_y = top + 23.5
    c.person(x + 68, later_y, 9.5, HUMAN)
    c.arrow(x + 75, later_y, x + 85, later_y, color=INK, width=1.1, head=3.6)
    c.rect(x + 86, later_y - 7, 18, 14, fill=WHITE, stroke=INK, stroke_width=0.9, rx=2)
    c.robot(x + 95, later_y, 9.5, AI)


def draw_reception(c: Canvas, x: float, top: float, h: float) -> None:
    cy = top + h / 2.0
    c.downvote(x + 16, cy, 13, INK)
    c.warning(x + 50, cy, 13, INK)
    c.speech(x + 84, cy, 13, INK)


QUESTIONS: tuple[tuple[str, str, tuple[str, ...], float, Callable[[Canvas, float, float, float], None]], ...] = (
    ("RQ1", "Reply-target choice", ("Conditional on an observed reply",), 28.0, draw_choice),
    ("Additional", "RQ1 thread entry", ("Navigation after a recorded card render",), 27.0, draw_entry),
    ("RQ2", "Prior contact and selection",
     ("Earlier replies in either direction,", "then later choice of the same account"), 36.0, draw_contact),
    ("RQ3", "Reception", ("Downvotes, toxicity and criticism",), 27.0, draw_reception),
)
QUESTION_GAP = 2.0


def draw_questions(c: Canvas, x: float, y: float, w: float) -> float:
    """Draw panel d as four separate cards and return its bottom edge."""

    text_w = 182.0
    top = y
    for chip, title, lines, h, illustrate in QUESTIONS:
        card(c, x, top, w, h)
        chip_w = label_chip(c, x + 8, top + 4, chip, filled=chip != "Additional")
        c.text(x + 8 + chip_w + 5, top + 12.7, title, size=9.5, bold=True, max_width=text_w - chip_w - 5)
        for index, line in enumerate(lines):
            c.text(x + 8, top + 23.2 + 9 * index, line, size=9, fill=MUTED, max_width=text_w)
        illustrate(c, x + 200, top, h)
        top += h + QUESTION_GAP
    return top - QUESTION_GAP


def build_canvas() -> tuple[Canvas, float]:
    """Lay out all four panels and return the canvas and figure height."""

    c = Canvas()
    row1_top = 3.0
    heading = row1_top + 11.0
    link_y = row1_top + 18.0
    top = row1_top + 23.0

    panel_heading(c, B_X, heading, "b", "Interaction")
    centres, row1_bottom = draw_communities(c, B_X, top, B_W)
    panel_heading(c, A_X, heading, "a", "Setup")
    exits = draw_setup(c, A_X, top, A_W, row1_bottom - top)

    # Setup feeds both communities: stubs into a bus that branches over panel b.
    for exit_y in exits:
        c.line(A_X + A_W, exit_y, BUS_X, exit_y, stroke=CONNECTOR, width=0.9)
    c.line(BUS_X, max(exits), BUS_X, link_y, stroke=CONNECTOR, width=0.9)
    c.line(BUS_X, link_y, max(centres), link_y, stroke=CONNECTOR, width=0.9)
    for centre in centres:
        c.arrow(centre, link_y, centre, top + 0.8)

    row2_top = row1_bottom + 10.0
    data_top = row2_top + 16.5
    panel_heading(c, C_X, row2_top + 11.0, "c", "Data")
    panel_heading(c, D_X, row2_top + 11.0, "d", "Research questions")
    bottom = draw_questions(c, D_X, data_top, D_W)
    draw_data(c, C_X, data_top, C_W, bottom)

    # Interactions in both communities produce the recorded data in panel c.
    link2_y = row1_bottom + 5.0
    data_x = C_X + C_W / 2.0
    for centre in centres:
        c.line(centre, row1_bottom, centre, link2_y, stroke=CONNECTOR, width=0.9)
    c.line(max(centres), link2_y, data_x, link2_y, stroke=CONNECTOR, width=0.9)
    c.arrow(data_x, link2_y, data_x, data_top - 0.8)
    c.line((C_X + C_W + D_X) / 2.0, row2_top, (C_X + C_W + D_X) / 2.0, bottom)

    for box in c.texts:
        if box.x0 < 0 or box.x1 > WIDTH:
            raise ValueError(f"{box.text!r} runs outside the figure")
    return c, bottom + 3.0


def build_svg() -> str:
    """Return the complete SVG document for Figure 1."""

    canvas, height = build_canvas()
    height_mm = FIGURE_WIDTH_MM * height / WIDTH
    body = "\n".join(canvas.parts)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_fmt(FIGURE_WIDTH_MM)}mm" '
        f'height="{_fmt(height_mm)}mm" viewBox="0 0 {_fmt(WIDTH)} {_fmt(height)}">\n'
        "<title>Figure 1. Study design</title>\n"
        f'<rect x="0" y="0" width="{_fmt(WIDTH)}" height="{_fmt(height)}" fill="{WHITE}"/>\n'
        f'<g font-family="{FONT_FAMILY}">\n{body}\n</g>\n</svg>\n'
    )


def convert(svg_path: Path, pdf_path: Path, png_path: Path) -> None:
    """Convert the SVG to a vector PDF and a 300 dpi PNG with rsvg-convert."""

    tool = shutil.which("rsvg-convert")
    if tool is None:
        raise RuntimeError("rsvg-convert (librsvg) is required for PDF and PNG output; use --svg-only")
    subprocess.run([tool, "--format", "pdf", "--output", str(pdf_path), str(svg_path)], check=True)
    subprocess.run(
        [tool, "--format", "png", "--dpi-x", str(PNG_DPI), "--dpi-y", str(PNG_DPI),
         "--background-color", WHITE, "--output", str(png_path), str(svg_path)],
        check=True,
    )


def make_figure(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    stem_name: str = DEFAULT_STEM,
    svg_only: bool = False,
) -> list[Path]:
    """Write the SVG and, unless ``svg_only``, the PDF and PNG."""

    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / f"{stem_name}.svg"
    svg_path.write_text(build_svg(), encoding="utf-8")
    if svg_only:
        return [svg_path]
    pdf_path = output_dir / f"{stem_name}.pdf"
    png_path = output_dir / f"{stem_name}.png"
    convert(svg_path, pdf_path, png_path)
    return [svg_path, pdf_path, png_path]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", "--output-dir", dest="output_dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"directory for the SVG, PDF and PNG (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--stem", default=DEFAULT_STEM, help="output filename stem")
    parser.add_argument("--svg-only", action="store_true", help="write the SVG without converting it")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        paths = make_figure(args.output_dir, stem_name=args.stem, svg_only=args.svg_only)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
