"""Short status PDF: goal, RQs (1+2 only), architecture diagram, implementation,
baseline results, next steps. Roughly 3-4 pages.

Output: status.pdf in the repo root.
"""
from __future__ import annotations

from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUTPUT = Path(__file__).resolve().parents[1] / "status.pdf"

# ------------------------------ styles ------------------------------

styles = getSampleStyleSheet()
title_style = ParagraphStyle(
    name="T", parent=styles["Title"], fontSize=20, leading=24, spaceAfter=4
)
subtitle_style = ParagraphStyle(
    name="St", parent=styles["Normal"], fontSize=12, leading=15,
    textColor=colors.HexColor("#555"), spaceAfter=14,
)
h1 = ParagraphStyle(
    name="H1", parent=styles["Heading1"], fontSize=14, leading=18,
    spaceBefore=12, spaceAfter=6, textColor=colors.HexColor("#1f3a64"),
)
h2 = ParagraphStyle(
    name="H2", parent=styles["Heading2"], fontSize=11.5, leading=15,
    spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#2c4d80"),
)
body = ParagraphStyle(
    name="B", parent=styles["BodyText"], fontSize=10.5, leading=14.5,
    spaceAfter=6,
)
bullet = ParagraphStyle(
    name="Bu", parent=body, leftIndent=18, bulletIndent=6, spaceAfter=1,
)
caption = ParagraphStyle(
    name="Cap", parent=body, fontSize=9, leading=12,
    textColor=colors.HexColor("#555"), alignment=1,  # center
)
cell = ParagraphStyle(
    name="Ce", parent=body, fontSize=9, leading=11, spaceAfter=0,
)
cell_h = ParagraphStyle(
    name="CeH", parent=cell, textColor=colors.white, fontName="Helvetica-Bold",
)


def P(text, style=body):
    return Paragraph(text, style)


def B(items):
    return [Paragraph(f"&ndash;&nbsp;&nbsp;{t}", bullet) for t in items]


def _wrap(data):
    out = []
    for i, row in enumerate(data):
        out.append([
            Paragraph(c, cell_h if i == 0 else cell) if isinstance(c, str) else c
            for c in row
        ])
    return out


def styled_table(data, col_widths=None):
    t = Table(_wrap(data), colWidths=col_widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a64")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("TOPPADDING", (0, 0), (-1, 0), 6),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
        ("TOPPADDING", (0, 1), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f4f6fa")]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#bbb")),
    ]))
    return t


# ------------------------------ diagram ------------------------------

CARD_FILL = colors.HexColor("#cfe2f3")     # light blue
CARD_STROKE = colors.HexColor("#3d85c6")
NEW_FILL = colors.HexColor("#fce5cd")      # light orange
NEW_STROKE = colors.HexColor("#e69138")
NEUTRAL_FILL = colors.HexColor("#e6e6e6")
NEUTRAL_STROKE = colors.HexColor("#666666")


def _box(d, cx, cy, w, h, lines, fill, stroke):
    """Draw a box centred at (cx, cy) with the given text lines."""
    d.add(Rect(cx - w / 2, cy - h / 2, w, h, fillColor=fill,
               strokeColor=stroke, strokeWidth=1.2, rx=5, ry=5))
    n = len(lines)
    # Compute first-line y so the text block is vertically centred.
    line_h = 11
    block_h = n * line_h
    top = cy + block_h / 2 - 9
    for i, line in enumerate(lines):
        d.add(String(cx, top - i * line_h, line, textAnchor="middle",
                     fontName="Helvetica", fontSize=9.5))


def _arrow(d, x1, y1, x2, y2, label=None, label_dx=8, label_dy=0):
    d.add(Line(x1, y1, x2, y2, strokeColor=colors.black, strokeWidth=1))
    # Arrowhead. Compute direction.
    dx = x2 - x1
    dy = y2 - y1
    length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux  # perpendicular
    size = 5
    base_x = x2 - ux * size
    base_y = y2 - uy * size
    d.add(Polygon([
        x2, y2,
        base_x + px * size * 0.6, base_y + py * size * 0.6,
        base_x - px * size * 0.6, base_y - py * size * 0.6,
    ], fillColor=colors.black, strokeColor=colors.black))
    if label:
        d.add(String((x1 + x2) / 2 + label_dx, (y1 + y2) / 2 + label_dy,
                     label, fontName="Helvetica-Oblique", fontSize=8,
                     fillColor=colors.HexColor("#444")))


def make_cascade_diagram():
    """Color-coded architecture: blue = CARD components, orange = our addition."""
    W, H = 460, 360
    d = Drawing(W, H)

    # Layout (cx, cy):
    # Top-center: zero-shot Generator (CARD)
    # Below: CARD gate (CARD)
    # Branch:
    #   - LOW confidence (ŝ < T_RAG) → left → Retrieve+regen (CARD)
    #   - HIGH confidence → down → Static analyzer (NEW)
    # Static analyzer branches:
    #   - fires → right → Retrieve+regen (NEW colour)
    #   - no fire → down → return ŷ₀
    # All paths converge → final output box

    gen_cx, gen_cy = 230, 322
    card_cx, card_cy = 230, 270
    rag_card_cx, rag_card_cy = 75, 220
    static_cx, static_cy = 230, 192
    rag_static_cx, rag_static_cy = 380, 142
    skip_cx, skip_cy = 230, 100
    out_cx, out_cy = 230, 36

    bw, bh = 170, 32  # standard box

    # --- boxes ---
    _box(d, gen_cx, gen_cy, bw, bh,
         ["Generator (zero-shot)",
          "-> y_hat_0, per-token logits"],
         CARD_FILL, CARD_STROKE)

    _box(d, card_cx, card_cy, bw, bh,
         ["CARD's gate: is s_hat_0 < T_RAG ?"],
         CARD_FILL, CARD_STROKE)

    _box(d, rag_card_cx, rag_card_cy, 130, bh,
         ["BM25 + regenerate",
          "(CARD's RAG path)"],
         CARD_FILL, CARD_STROKE)

    _box(d, static_cx, static_cy, bw, bh + 6,
         ["Static Analyzer on y_hat_0    [NEW]",
          "unresolved call / receiver?"],
         NEW_FILL, NEW_STROKE)

    _box(d, rag_static_cx, rag_static_cy, 130, bh,
         ["BM25 + regenerate",
          "(static-triggered)"],
         NEW_FILL, NEW_STROKE)

    _box(d, skip_cx, skip_cy, 130, bh,
         ["return y_hat_0",
          "(no retrieval)"],
         NEUTRAL_FILL, NEUTRAL_STROKE)

    _box(d, out_cx, out_cy, 110, bh,
         ["final output y_hat"],
         NEUTRAL_FILL, NEUTRAL_STROKE)

    # --- arrows ---
    # gen → card
    _arrow(d, gen_cx, gen_cy - bh / 2, card_cx, card_cy + bh / 2)
    # card → rag_card (low conf branch, left)
    _arrow(d, card_cx - bw / 2, card_cy, rag_card_cx + 65, rag_card_cy + bh / 2,
           label="yes (low conf.)", label_dx=-30, label_dy=4)
    # card → static (high conf branch, down)
    _arrow(d, card_cx, card_cy - bh / 2, static_cx, static_cy + (bh + 6) / 2,
           label="no", label_dx=10, label_dy=0)
    # static → rag_static (fires branch, right)
    _arrow(d, static_cx + bw / 2, static_cy, rag_static_cx - 65, rag_static_cy + bh / 2,
           label="fires", label_dx=10, label_dy=6)
    # static → skip (no fire, down)
    _arrow(d, static_cx, static_cy - (bh + 6) / 2, skip_cx, skip_cy + bh / 2,
           label="no fire", label_dx=10)
    # rag_card → out
    _arrow(d, rag_card_cx, rag_card_cy - bh / 2, out_cx - 35, out_cy + bh / 2)
    # rag_static → out
    _arrow(d, rag_static_cx, rag_static_cy - bh / 2, out_cx + 35, out_cy + bh / 2)
    # skip → out
    _arrow(d, skip_cx, skip_cy - bh / 2, out_cx, out_cy + bh / 2)

    # --- legend ---
    lx, ly = 12, 350
    d.add(Rect(lx, ly, 14, 10, fillColor=CARD_FILL, strokeColor=CARD_STROKE, strokeWidth=1))
    d.add(String(lx + 18, ly + 1, "CARD (Zhang et al. 2024)",
                 fontName="Helvetica", fontSize=9))
    d.add(Rect(lx, ly - 16, 14, 10, fillColor=NEW_FILL, strokeColor=NEW_STROKE, strokeWidth=1))
    d.add(String(lx + 18, ly - 15, "Our cascade stage (NEW)",
                 fontName="Helvetica", fontSize=9))
    d.add(Rect(lx, ly - 32, 14, 10, fillColor=NEUTRAL_FILL, strokeColor=NEUTRAL_STROKE, strokeWidth=1))
    d.add(String(lx + 18, ly - 31, "Output", fontName="Helvetica", fontSize=9))

    return d


# ------------------------------ content ------------------------------

def build_story():
    s = []

    # Header
    s += [
        P("Adaptive Retrieval for Code Completion", title_style),
        P("Project status — RQ1 baselines complete; cascade pipeline ready",
          subtitle_style),
    ]

    # 1. Goal
    s += [
        P("1.&nbsp;&nbsp;Goal", h1),
        P(
            "We study whether <b>adaptive retrieval</b> &mdash; selectively triggering RAG "
            "only on instances where it would help &mdash; outperforms always-retrieve "
            "and never-retrieve baselines for repository-level code completion. "
            "Our novel contribution is adding a <b>static-analysis cascade stage</b> "
            "to the CARD framework (Zhang et&nbsp;al. 2024). The hypothesis: CARD&rsquo;s "
            "logit-based uncertainty signal misses a class of failures &mdash; confidently-"
            "generated identifiers that don&rsquo;t actually exist in the repository &mdash; "
            "and static analysis of the prediction&rsquo;s AST can detect these directly. "
            "The cascade is asymmetric by design: static analysis can only <i>add</i> "
            "retrievals to CARD&rsquo;s decisions, never remove them.",
            body,
        ),
    ]

    # 2. Research questions (RQ1 + RQ2 only per user request)
    s += [
        P("2.&nbsp;&nbsp;Research questions", h1),
        styled_table(
            [
                ["RQ", "Question", "Hypothesis"],
                [
                    "<b>RQ1</b>",
                    "How does the proposed cascade perform against "
                    "always-retrieve and never-retrieve baselines on "
                    "CrossCodeEval-Python, by accuracy (EM, ES, Identifier-F1) "
                    "and efficiency (% retrieval, latency)?",
                    "<b>H1.</b> The cascade matches or beats always-retrieve on "
                    "accuracy while performing fewer retrievals.",
                ],
                [
                    "<b>RQ2</b>",
                    "Does the static-analysis stage reduce the rate of "
                    "identifier hallucinations versus vanilla CARD on the same "
                    "instances?",
                    "<b>H2.</b> The cascade cuts per-instance hallucination "
                    "rate by at least 20% relative to vanilla CARD, "
                    "significant under McNemar&rsquo;s exact test (p&nbsp;&lt; 0.05).",
                ],
            ],
            col_widths=[0.5 * inch, 3.0 * inch, 2.9 * inch],
        ),
    ]

    # 3. Architecture
    s += [
        P("3.&nbsp;&nbsp;Architecture", h1),
        P(
            "Three asymmetric stages: zero-shot generation, CARD&rsquo;s uncertainty "
            "gate, and a static-analysis gate that runs only when CARD says skip. "
            "Boxes in blue are reimplemented from CARD; the orange box is "
            "our contribution.",
            body,
        ),
        KeepTogether([
            make_cascade_diagram(),
            Spacer(1, 4),
            P(
                "Figure 1. The cascade pipeline. CARD gate fires on low-confidence "
                "predictions; static-analysis gate fires when y_hat_0 references "
                "a call target or attribute receiver that doesn&rsquo;t resolve in the "
                "current scope, builtins, or the repository symbol table. Either "
                "trigger leads to a BM25 + regenerate step; otherwise y_hat_0 is "
                "returned without retrieval.",
                caption,
            ),
        ]),
        PageBreak(),
    ]

    # 4. Implementation
    s += [
        P("4.&nbsp;&nbsp;Implementation", h1),
        P(
            "All five infrastructure phases are complete (~3000 lines of Python "
            "across 14 modules, 194 passing tests, ruff clean). The Generator is "
            "abstracted behind a single interface with three real backends "
            "(HuggingFace transformers, vLLM, and mlx-lm for Apple Silicon) plus "
            "a Mock for tests, all returning the same per-token probability and "
            "entropy data CARD&rsquo;s features need.",
            body,
        ),
        *B([
            "<b>CARD components</b> (blue in Figure&nbsp;1) &mdash; the 13-D feature "
            "vector from per-token logits, a LightGBM Estimator that regresses "
            "edit similarity on those features, and the gate that fires retrieval "
            "when s_hat_0 &lt; T_RAG.",
            "<b>Cascade components</b> (orange) &mdash; a tree-sitter-based static "
            "analyzer that parses each prediction&rsquo;s AST, classifies every "
            "structurally-significant identifier (call targets, attribute "
            "receivers, subscript values) as in-file / cross-file / unresolved, "
            "and fires when any such identifier fails to resolve.",
            "<b>Shared infrastructure</b> &mdash; BM25 retriever, FIM prompt assembly, "
            "metrics (EM, ES, Identifier-F1, repo-symbol-precision, "
            "hallucination flag, McNemar test, paired bootstrap), per-instance "
            "JSONL writer, and a disk-backed generation cache that means the "
            "same zero-shot prompt is computed once and reused across configs.",
            "<b>Evaluation</b> &mdash; six configurations (C1 no-retrieve, "
            "C2 always-retrieve, C3 CARD, C4 cascade, C5 static-only ablation, "
            "C6 oracle) and the orchestrator that runs them on "
            "CrossCodeEval-Python (~2,665 instances).",
        ]),
    ]

    # 5. Results
    s += [
        P("5.&nbsp;&nbsp;Baseline results", h1),
        P(
            "Both baselines have been run end-to-end on the full "
            "CrossCodeEval-Python (2,665 instances) with "
            "Qwen2.5-Coder-1.5B via mlx-lm on M4&nbsp;Pro.",
            body,
        ),
        styled_table(
            [
                ["Config", "Retr.%", "EM", "ES", "IdF1", "RSP", "Hall.% (strict)", "Lat. (ms)"],
                ["C1 no-retrieve", "0.0", "0.08", "0.27", "0.46", "0.54", "0.69", "1,255"],
                ["<b>C2 always-retrieve</b>",
                 "<b>100.0</b>", "<b>0.34</b>", "<b>0.47</b>", "<b>0.62</b>",
                 "<b>0.54</b>", "<b>0.77</b>", "<b>2,127</b>"],
            ],
            col_widths=[1.7 * inch, 0.55 * inch, 0.45 * inch, 0.45 * inch,
                        0.55 * inch, 0.45 * inch, 1.05 * inch, 0.7 * inch],
        ),
        Spacer(1, 4),
        P(
            "RSP = repository symbol precision (fraction of identifiers in a "
            "prediction that resolve). Hall.% = at least one structurally "
            "significant identifier doesn&rsquo;t resolve, computed against the "
            "per-repository chunk union.",
            caption,
        ),
        P("Key observations", h2),
        *B([
            "<b>Retrieval is consequential on CCE-Python</b> &mdash; EM jumps 4.25× "
            "(0.08 to 0.34), ES rises 74% (0.27 to 0.47). The benchmark was "
            "curated for cross-file dependence, and that shows.",
            "<b>Cost is a 70% latency penalty</b> on every instance (1.25 s to "
            "2.13 s). This is the budget the cascade has to trade against.",
            "<b>Hallucination rates are high in both configs</b>. Even with "
            "retrieval, 77% of predictions contain at least one unresolved "
            "identifier in a structurally significant position. Retrieval "
            "fixes the <i>targeted</i> identifier (the one that drives "
            "EM/ES) but doesn&rsquo;t systematically reduce the rate of "
            "<i>peripheral</i> hallucinated names.",
        ]),
    ]

    # 6. Next steps
    s += [
        P("6.&nbsp;&nbsp;Next steps", h1),
        *B([
            "<b>Decide how to measure hallucinations.</b> The current "
            "hallucination rate (~77%) is high because our symbol table per "
            "instance is small (chunks selected by BM25 from a real repo) "
            "and even good predictions naturally reference names outside "
            "those chunks. Several refinements are under consideration: "
            "import-following resolution, confidence-weighted filtering, "
            "or replacing the binary flag with a continuous metric "
            "(mean unresolved-identifier count per prediction). The choice "
            "affects both the absolute number reportable in the paper and "
            "the cascade&rsquo;s trigger rate.",
            "<b>Calibrate the Estimator on the 7B model.</b> CARD&rsquo;s Estimator "
            "is generator-specific; once we scale up the underlying code LM, "
            "the (features, ES) training-data step must be repeated. With a "
            "7B model on GPU this is ~hours on an A100; alternatives are "
            "running it overnight locally via mlx-lm or scoping a smaller "
            "Estimator training set with a documented tradeoff in MSE.",
            "Once both are settled, C3 (CARD) and C4 (cascade) can be run "
            "to close out RQ1 and answer RQ2.",
        ]),
    ]

    return s


def build_pdf(path):
    doc = SimpleDocTemplate(
        str(path),
        pagesize=LETTER,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title="Adaptive Retrieval — Project Status",
        author="project-group-17",
    )
    doc.build(build_story())


if __name__ == "__main__":
    build_pdf(OUTPUT)
    print(f"Wrote {OUTPUT.relative_to(Path.cwd())}")
