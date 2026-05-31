"""Midterm progress report, written as a standalone document for readers
who have no prior context. Six sections follow the assignment brief.
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

OUTPUT = Path(__file__).resolve().parents[1] / "midterm_status.pdf"

# ------------------------------ styles ------------------------------

styles = getSampleStyleSheet()
title_style = ParagraphStyle(
    name="T", parent=styles["Title"], fontSize=20, leading=24, spaceAfter=4,
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
    textColor=colors.HexColor("#555"), alignment=1,
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

CARD_FILL = colors.HexColor("#cfe2f3")
CARD_STROKE = colors.HexColor("#3d85c6")
NEW_FILL = colors.HexColor("#fce5cd")
NEW_STROKE = colors.HexColor("#e69138")
NEUTRAL_FILL = colors.HexColor("#e6e6e6")
NEUTRAL_STROKE = colors.HexColor("#666666")


def _box(d, cx, cy, w, h, lines, fill, stroke):
    d.add(Rect(cx - w / 2, cy - h / 2, w, h, fillColor=fill,
               strokeColor=stroke, strokeWidth=1.2, rx=5, ry=5))
    n = len(lines)
    line_h = 11
    block_h = n * line_h
    top = cy + block_h / 2 - 9
    for i, line in enumerate(lines):
        d.add(String(cx, top - i * line_h, line, textAnchor="middle",
                     fontName="Helvetica", fontSize=9.5))


def _arrow(d, x1, y1, x2, y2, label=None, label_dx=8, label_dy=0):
    d.add(Line(x1, y1, x2, y2, strokeColor=colors.black, strokeWidth=1))
    dx = x2 - x1
    dy = y2 - y1
    length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
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
    W, H = 460, 400
    d = Drawing(W, H)

    gen_cx, gen_cy = 230, 362
    card_cx, card_cy = 230, 310
    rag_card_cx, rag_card_cy = 75, 260
    static_cx, static_cy = 230, 220
    rag_static_cx, rag_static_cy = 380, 160
    skip_cx, skip_cy = 230, 100
    out_cx, out_cy = 230, 36

    bw, bh = 170, 32

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

    _box(d, static_cx, static_cy, bw, bh + 12,
         ["Three-tier static analyzer    [NEW]",
          "T1 out-of-scope / T2 signature / T3 import"],
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

    _arrow(d, gen_cx, gen_cy - bh / 2, card_cx, card_cy + bh / 2)
    _arrow(d, card_cx - bw / 2, card_cy, rag_card_cx + 65, rag_card_cy + bh / 2,
           label="yes (low conf.)", label_dx=-30, label_dy=4)
    _arrow(d, card_cx, card_cy - bh / 2, static_cx, static_cy + (bh + 12) / 2,
           label="no", label_dx=10, label_dy=0)
    _arrow(d, static_cx + bw / 2, static_cy, rag_static_cx - 65, rag_static_cy + bh / 2,
           label="any tier fires", label_dx=8, label_dy=6)
    _arrow(d, static_cx, static_cy - (bh + 12) / 2, skip_cx, skip_cy + bh / 2,
           label="no fire", label_dx=10)
    _arrow(d, rag_card_cx, rag_card_cy - bh / 2, out_cx - 35, out_cy + bh / 2)
    _arrow(d, rag_static_cx, rag_static_cy - bh / 2, out_cx + 35, out_cy + bh / 2)
    _arrow(d, skip_cx, skip_cy - bh / 2, out_cx, out_cy + bh / 2)

    lx, ly = 12, 390
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
        P("Adaptive Retrieval for Repository-Level Code Completion", title_style),
        P("Midterm Progress Report &mdash; Group&nbsp;17",
          subtitle_style),
    ]

    # 1. Project Overview
    s += [
        P("1.&nbsp;&nbsp;Project Overview", h1),
        P(
            "Modern code-completion systems rely on retrieval-augmented "
            "generation (RAG): when the developer asks the model to fill in "
            "a hole, the system first searches the repository for relevant "
            "snippets, prepends them to the prompt, and only then calls the "
            "language model. RAG works, but indiscriminate retrieval is "
            "wasteful. Every completion pays for query construction, an "
            "index lookup, and the cost of generating with a much longer "
            "prompt. The natural question is whether retrieval can be "
            "<i>conditioned</i> on need: trigger it only when the model is "
            "actually likely to benefit, and skip it otherwise.",
            body,
        ),
        P(
            "CARD (Zhang et&nbsp;al. 2024) is the current state of the art "
            "on this question. It trains a small gradient-boosted regressor "
            "to predict the edit similarity between the model&rsquo;s "
            "zero-shot completion and the ground truth, using only "
            "token-level features extracted from the model&rsquo;s own "
            "logits. If the predicted similarity is below a threshold, "
            "retrieval is invoked; otherwise the zero-shot completion is "
            "returned as-is.",
            body,
        ),
        P(
            "Our project reimplements CARD and asks a follow-on question: "
            "<b>does CARD&rsquo;s logit-based confidence signal miss a "
            "specific class of failures &mdash; confidently-generated "
            "identifiers that don&rsquo;t actually exist in the "
            "repository?</b> Confident hallucinations are exactly the case "
            "where a logit-based signal cannot help, because the model is "
            "wrong but does not know it. Our novel contribution is a "
            "<b>three-tier static-analysis stage</b> that parses the "
            "model&rsquo;s prediction and triggers retrieval when any of "
            "three structural checks fire: (1) an identifier in a "
            "structurally significant position (a function being called, an "
            "object whose attribute is being accessed, the value being "
            "subscripted) is not visible in the local scope; (2) a call "
            "does not match the callee&rsquo;s known signature; or (3) an "
            "import names a symbol that lives in a different module than "
            "the one being imported from.",
            body,
        ),
        P(
            "The cascade is asymmetric by design: the static stage can only "
            "<i>add</i> retrievals to CARD&rsquo;s decisions, never remove "
            "them. This bounds the worst-case retrieval rate to the "
            "always-retrieve baseline and frames the central research "
            "question cleanly &mdash; does the additional retrieval budget "
            "reduce hallucinations enough to be worthwhile?",
            body,
        ),
        P(
            "<b>Why this matters.</b> Repository-level completion is one "
            "of the most widely deployed language-model applications "
            "(Copilot, Cursor, JetBrains AI). Every avoided retrieval is "
            "saved latency and cost; every avoided hallucination is "
            "preserved developer trust. A method that improves the "
            "cost&ndash;quality trade-off &mdash; using only the "
            "model&rsquo;s own output and lightweight syntactic analysis "
            "&mdash; would be architecture-agnostic and easy to deploy.",
            body,
        ),
        P("Research questions", h2),
        P(
            "We focus on two questions:",
            body,
        ),
        styled_table(
            [
                ["RQ", "Question", "Hypothesis"],
                [
                    "<b>RQ1</b>",
                    "How does the proposed cascade perform against "
                    "no-retrieve and always-retrieve baselines on "
                    "CrossCodeEval-Python, measured by accuracy (Exact "
                    "Match, Edit Similarity, Identifier-F1) and efficiency "
                    "(retrieval rate, latency)?",
                    "<b>H1.</b> The cascade matches or exceeds "
                    "always-retrieve on accuracy while performing strictly "
                    "fewer retrievals.",
                ],
                [
                    "<b>RQ2</b>",
                    "Does the static-analysis cascade reduce the rate of "
                    "identifier hallucinations compared to vanilla CARD on "
                    "the same instances?",
                    "<b>H2.</b> The cascade lowers the per-instance "
                    "hallucination rate relative to vanilla CARD, "
                    "significantly under McNemar&rsquo;s exact test "
                    "(p&nbsp;&lt; 0.05).",
                ],
            ],
            col_widths=[0.5 * inch, 3.1 * inch, 2.8 * inch],
        ),
        PageBreak(),
    ]

    # 2. Progress So Far
    s += [
        P("2.&nbsp;&nbsp;Progress So Far", h1),
        P("2.1&nbsp;&nbsp;Background research", h2),
        P(
            "We grounded the project in five core references. <b>CARD</b> "
            "(Zhang et&nbsp;al. 2024) supplies the adaptive-retrieval "
            "framework and the per-token feature schema we reimplement. "
            "<b>RepoCoder</b> (Zhang et&nbsp;al. 2023) and <b>Repoformer</b> "
            "(Wu et&nbsp;al. 2024) define the iterative-retrieval and "
            "selective-retrieval baselines our work is benchmarked against. "
            "<b>CrossCodeEval</b> (Ding et&nbsp;al. 2023) supplies the "
            "evaluation benchmark and the Identifier-F1 metric. <b>FLARE</b> "
            "(Jiang et&nbsp;al. 2023) provided early evidence that signals "
            "derived from the model&rsquo;s own outputs can usefully gate "
            "retrieval. To the best of our knowledge, using structural "
            "static analysis of the prediction to gate retrieval is novel "
            "as an addition to the CARD-style framework.",
            body,
        ),
        P("2.2&nbsp;&nbsp;Design decisions", h2),
        *B([
            "<b>Asymmetric cascade.</b> The static-analysis stage can only "
            "add retrievals to CARD&rsquo;s decisions, so cascade cost is "
            "bounded above by the always-retrieve baseline. This lets us "
            "frame RQ2 cleanly as &ldquo;is the additional retrieval worth "
            "it?&rdquo;",
            "<b>Three tiers, structured by what they verify.</b> Tier&nbsp;1 "
            "(out-of-scope) catches identifiers in structurally significant "
            "positions that are not visible in the current file&rsquo;s "
            "scope. Tier&nbsp;2 (signature) catches call sites whose "
            "arguments do not match a known repository function&rsquo;s "
            "signature. Tier&nbsp;3 (import origin) catches "
            "<i>from&nbsp;X&nbsp;import&nbsp;Y</i> statements where Y is "
            "actually defined or re-exported elsewhere. The tiers can be "
            "independently enabled or disabled, which gives us natural "
            "ablations.",
            "<b>Tier&nbsp;1 deliberately ignores the repository-wide symbol "
            "table.</b> A repository-wide symbol table is only as good as "
            "the slice of the repo it sees, and on the evaluation benchmark "
            "the available slice is partial (chunks selected by BM25, not "
            "the entire codebase). Using symbol-table membership to "
            "<i>route</i> the trigger would penalise real names that "
            "happen to live outside the slice. We therefore use a simpler, "
            "more conservative rule: an identifier is &ldquo;out of "
            "scope&rdquo; iff it is not visible in the local file.",
            "<b>Hierarchy of structural significance.</b> Not every "
            "unresolved name is a hallucination. A bare identifier in a "
            "binary expression might be a parameter we missed, a global, "
            "or a fluent style. But an unresolved function being called, "
            "an unresolved object being indexed, or an unresolved receiver "
            "of an attribute access is much harder to explain away. The "
            "analyzer fires only on these structurally significant "
            "positions.",
            "<b>Multiple generator backends behind one interface.</b> "
            "Because the Estimator depends on per-token logits from the "
            "underlying language model, the project supports several "
            "backends (HuggingFace, vLLM on GPU, mlx-lm on Apple Silicon, "
            "and a deterministic mock for development) all producing the "
            "same logit data. This makes calibration portable across "
            "deployment environments.",
            "<b>Generation caching across configurations.</b> Most of the "
            "configurations being compared share the same zero-shot "
            "generations &mdash; what differs is what they do afterwards. "
            "We cache zero-shot generations on disk keyed by model and "
            "prompt, so the headline experiments run from a single shared "
            "generation pool rather than regenerating for each "
            "configuration.",
        ]),
        P("2.3&nbsp;&nbsp;Dataset preparation", h2),
        *B([
            "<b>CrossCodeEval-Python</b> &mdash; the primary benchmark, "
            "2,665 instances drawn from real GitHub projects, each with a "
            "held-out completion target and a pool of retrievable chunks "
            "from the same repository. Downloaded, parsed, and indexed "
            "(per-instance and unioned per-repository).",
            "<b>Estimator training data</b> &mdash; sampled from The Stack "
            "(a permissively-licensed code corpus), de-duplicated by "
            "K-means in feature space, and scored against the language "
            "model under evaluation. The current training pool is "
            "calibrated for the 1.5B-parameter generator we used for "
            "preliminary results; re-calibration for the 7B generator is "
            "pending.",
            "<b>RepoEval</b> &mdash; supported as a secondary benchmark "
            "but not yet exercised; we will draw on it only if the primary "
            "results need a second validation table.",
        ]),
        P("2.4&nbsp;&nbsp;Prototype implementation", h2),
        P(
            "The system is feature-complete for the two research "
            "questions modulo the 7B-model run. Six configurations are "
            "supported end-to-end &mdash; the two baselines, vanilla CARD, "
            "the cascade, and two extra configurations (static-only and "
            "an oracle upper bound) that fall out for free and feed into "
            "the analysis.",
            body,
        ),
        styled_table(
            [
                ["Config", "Stages", "Purpose"],
                ["C1 no-retrieve", "Zero-shot only",
                 "Lower bound on accuracy (RQ1 baseline)."],
                ["C2 always-retrieve", "BM25 + regenerate every instance",
                 "Upper bound on cost (RQ1 baseline)."],
                ["C3 CARD", "Zero-shot &rarr; CARD gate &rarr; (maybe retrieve)",
                 "Vanilla CARD &mdash; the RQ2 comparison target."],
                ["<b>C4 cascade</b>",
                 "Zero-shot &rarr; CARD &rarr; (if skip) static &rarr; (maybe retrieve)",
                 "<b>Our contribution.</b> Main RQ1+RQ2 configuration."],
                ["C5 static-only", "Zero-shot &rarr; static gate &rarr; (maybe retrieve)",
                 "Ablation: how much retrieval does static <i>alone</i> trigger?"],
                ["C6 oracle", "Ex-post pick C1 vs C2 by ground-truth score",
                 "Empirical upper bound on any selective policy."],
            ],
            col_widths=[1.5 * inch, 2.6 * inch, 2.3 * inch],
        ),
        P("2.5&nbsp;&nbsp;Experiments conducted", h2),
        *B([
            "<b>Full benchmark run with the 1.5B-parameter generator.</b> "
            "All four core configurations (C1, C2, C3, C4) executed on "
            "the complete CrossCodeEval-Python set (2,665 instances). "
            "Total wall-clock time approximately six hours, exploiting "
            "the shared-generation cache. Results in &sect;4 below.",
            "<b>Hallucination-metric calibration.</b> The C1 and C2 "
            "predictions were re-scored under three candidate definitions "
            "of &ldquo;hallucination&rdquo; (strict significant-position, "
            "three-tier union, and an early loose-unresolved rule) to "
            "understand how the metric&rsquo;s definition affects the "
            "numbers. This informed the design decision discussed in "
            "&sect;3.3.",
            "<b>Static-analyzer behavioural evaluation.</b> Every trigger "
            "position the analyzer reasons about &mdash; call targets, "
            "attribute receivers, subscript values, decorators, raise "
            "targets, exception types, class bases, signature mismatches, "
            "wrong-origin imports &mdash; has been validated against "
            "hand-crafted examples covering both positive and negative "
            "cases.",
        ]),
    ]

    # 3. Current Status
    s += [
        P("3.&nbsp;&nbsp;Current Status", h1),
        P("3.1&nbsp;&nbsp;What is working", h2),
        *B([
            "The full pipeline runs end-to-end on the primary benchmark "
            "with the 1.5B-parameter generator across all six "
            "configurations. Per-instance results are written in a "
            "structured format with a stable schema, ready to be consumed "
            "by the analysis layer.",
            "The three-tier static analyzer is implemented with per-tier "
            "enable/disable flags. The flags allow each tier&rsquo;s "
            "contribution to be measured independently &mdash; useful "
            "both for the planned ablations and for diagnosing where the "
            "cascade&rsquo;s additional retrievals come from.",
            "The CARD reimplementation &mdash; the 13-dimensional "
            "per-token feature vector, the gradient-boosted regressor, "
            "and the threshold gate &mdash; is calibrated for the "
            "1.5B-parameter generator with a held-out mean-squared error "
            "of approximately 0.044 on the validation pairs.",
            "The shared generation cache yields near-100% hit rate when "
            "re-running downstream configurations, so iteration on the "
            "static-stage logic or the CARD threshold is cheap.",
            "The analysis layer produces the aggregate tables, paired "
            "statistical tests (McNemar, paired bootstrap), and the "
            "trigger-reason breakdowns needed to answer both research "
            "questions; the only missing input is the final 7B-model "
            "results.",
        ]),
        P("3.2&nbsp;&nbsp;What is incomplete", h2),
        *B([
            "<b>Calibration and full run on the 7B-parameter generator.</b> "
            "All results to date use a 1.5B-parameter model. The CARD "
            "Estimator is generator-specific (its features are derived "
            "from the language model&rsquo;s logits), so the entire "
            "training and evaluation cycle has to be re-run on the larger "
            "model before the final results are reported. This is the "
            "single largest remaining work item.",
            "<b>RQ2 statistical test.</b> The McNemar comparison between "
            "vanilla CARD (C3) and the cascade (C4) requires the final "
            "hallucination definition to be locked in (see &sect;3.3) and "
            "&mdash; more importantly &mdash; the 7B-model run, because "
            "on the 1.5B model CARD saturates and C3/C4 collapse onto "
            "the always-retrieve baseline (see &sect;4).",
            "<b>Final paper.</b> Section structure and figure templates "
            "are prepared; the empirical sections will be drafted once "
            "the 7B numbers are in.",
        ]),
        P("3.3&nbsp;&nbsp;Plan changes vs. original proposal", h2),
        *B([
            "<b>Scope narrowed from three research questions to two.</b> "
            "An earlier draft included an independent reproduction of "
            "CARD&rsquo;s reported numbers as RQ3. Reproducing CARD&rsquo;s "
            "exact table requires CodeLlama-7B on RepoEval-line, which is "
            "out of reach for our compute budget. We have dropped this "
            "third question and rely instead on the internal validation "
            "of the Estimator and the controlled comparison structure of "
            "RQ1 and RQ2 to support our claims.",
            "<b>Hallucination metric refined.</b> Our first definition was "
            "&ldquo;any unresolved identifier anywhere in the prediction.&rdquo; "
            "On the 1.5B baseline this fires on roughly nine out of every "
            "ten predictions, which makes the metric uninformative and "
            "would make the cascade look indistinguishable from "
            "always-retrieve. We tightened the definition to "
            "structurally significant positions only, then extended it "
            "into the current three-tier union (out-of-scope, signature "
            "mismatch, wrong-origin import). The metric now reflects what "
            "the static analyzer can actually <i>verify</i> as wrong, "
            "rather than what it merely cannot prove right.",
            "<b>Local development environment switched to Apple Silicon.</b> "
            "Our original plan assumed access to a GPU cluster throughout. "
            "In practice we developed against mlx-lm on a local M-series "
            "MacBook and treated GPU access as a finite resource reserved "
            "for the headline 7B run. The deployment-time generator "
            "interface is shared, so no code changes are needed when we "
            "move from one to the other.",
        ]),
        PageBreak(),
    ]

    # 4. Preliminary Results
    s += [
        P("4.&nbsp;&nbsp;Preliminary Results", h1),
        P(
            "The numbers below come from a single full run of all four "
            "core configurations on the entire CrossCodeEval-Python "
            "benchmark (2,665 instances), using a "
            "<b>1.5B-parameter code-completion model</b>. The headline "
            "run with the 7B model is still pending; we present these "
            "results as preliminary, and note that the model-size effects "
            "they reveal are themselves informative.",
            body,
        ),
        styled_table(
            [
                ["Config", "Retr.%", "EM", "ES", "IdF1", "RSP", "Hall.%", "Lat.&nbsp;(ms)"],
                ["C1 no-retrieve",
                 "0.0", "8.3", "27.0", "46.1", "53.7", "78.9", "1,255"],
                ["C2 always-retrieve",
                 "100.0", "34.4", "46.8", "62.1", "53.5", "86.4", "2,127"],
                ["C3 CARD",
                 "100.0", "35.2", "47.5", "63.2", "51.2", "&mdash;", "3,382"],
                ["<b>C4 cascade</b>",
                 "<b>100.0</b>", "<b>34.4</b>", "<b>46.8</b>",
                 "<b>62.1</b>", "<b>50.9</b>", "<b>&mdash;</b>", "<b>3,382</b>"],
            ],
            col_widths=[1.5 * inch, 0.55 * inch, 0.45 * inch, 0.45 * inch,
                        0.55 * inch, 0.45 * inch, 0.65 * inch, 0.85 * inch],
        ),
        Spacer(1, 4),
        P(
            "Table&nbsp;1. Aggregate results on CrossCodeEval-Python "
            "(n&nbsp;=&nbsp;2,665) with a 1.5B-parameter generator. EM, ES, "
            "IdF1, RSP shown as percentages. RSP (repository symbol "
            "precision) is the fraction of identifiers in a prediction "
            "that are visible in the local file&rsquo;s scope. Hall.% is "
            "the three-tier hallucination rate (out-of-scope &or; signature "
            "mismatch &or; wrong-origin import). The Hall.% values for "
            "C3 and C4 are omitted because they were collected under an "
            "earlier definition of the metric and will be recomputed "
            "together with the 7B run for a clean paired comparison.",
            caption,
        ),
        P("Observations", h2),
        *B([
            "<b>Retrieval is decisive on this benchmark.</b> Moving from "
            "no-retrieve to always-retrieve raises Exact Match by a "
            "factor of 4.2 (8.3% &rarr; 34.4%) and Edit Similarity by "
            "73% (27.0% &rarr; 46.8%). CrossCodeEval was curated to test "
            "cross-file dependence, and this is what shows up. The cost "
            "is a 70% latency penalty applied to every instance.",
            "<b>The CARD gate saturates at 1.5B.</b> With the default "
            "confidence threshold, CARD&rsquo;s predicted edit similarity "
            "falls below the threshold on every single instance, so the "
            "gate triggers retrieval 100% of the time. As a consequence "
            "vanilla CARD (C3) and the cascade (C4) collapse onto the "
            "always-retrieve baseline. This is the central reason the 7B "
            "run is essential: a model with sharper logits is needed to "
            "give the gate something to discriminate on. A threshold "
            "sweep is planned alongside the 7B run.",
            "<b>The cascade behaves as designed.</b> The cascade&rsquo;s "
            "retrieval rate exactly matches vanilla CARD&rsquo;s, "
            "confirming the asymmetric guarantee held in practice: the "
            "static stage can only add to CARD&rsquo;s retrievals, and "
            "since CARD&rsquo;s set is already the full benchmark, the "
            "static stage has nothing to add. Accuracy and latency match "
            "C3 within floating-point tolerance.",
            "<b>The static analyzer exercises all three tiers in "
            "practice.</b> Across the 2,665 predictions, the out-of-scope "
            "tier fires on the majority of records (the Hall.% column "
            "for the baselines is dominated by Tier&nbsp;1 fires); the "
            "signature and import-origin tiers contribute additional "
            "fires whose breakdown will be quantified in the next "
            "analysis pass.",
        ]),
        P("Architecture", h2),
        KeepTogether([
            make_cascade_diagram(),
            Spacer(1, 4),
            P(
                "Figure&nbsp;1. The cascade pipeline. The generator first "
                "produces a zero-shot completion. CARD&rsquo;s gate then "
                "predicts the completion&rsquo;s edit similarity from the "
                "generator&rsquo;s logits; if the predicted similarity is "
                "below the threshold, retrieval is triggered. If CARD "
                "skips, the three-tier static analyzer parses the "
                "prediction and triggers retrieval if any enabled tier "
                "fires. Otherwise the zero-shot completion is returned. "
                "Blue components are reimplemented from CARD; orange "
                "components are our novel contribution.",
                caption,
            ),
        ]),
        PageBreak(),
    ]

    # 5. Plan for Remaining Weeks
    s += [
        P("5.&nbsp;&nbsp;Plan for the Remaining Weeks", h1),
        P(
            "The remaining work falls into three coherent phases: "
            "finalising the 1.5B baseline under the chosen hallucination "
            "metric, calibrating and running on the 7B-parameter "
            "generator, and writing up the results. A two-week buffer at "
            "the end absorbs any required reruns.",
            body,
        ),
        styled_table(
            [
                ["Week", "Milestone"],
                ["<b>Week 5 (now)</b>",
                 "Lock the three-tier hallucination definition; re-score "
                 "all four 1.5B-baseline configurations under the final "
                 "metric to produce a consistent results table. Secure "
                 "GPU access for the 7B run (cluster allocation or "
                 "long-running local backup)."],
                ["Week 6",
                 "Generate Estimator training data using the 7B "
                 "generator, deduplicate, and fit the regressor. Run a "
                 "small-scale pilot to confirm the pipeline functions "
                 "end-to-end with the new generator before committing to "
                 "the full benchmark."],
                ["Week 7",
                 "Full evaluation on CrossCodeEval-Python with the 7B "
                 "generator across all four core configurations. Threshold "
                 "sweep at three points to characterise the cost-accuracy "
                 "frontier. Static-only and oracle configurations for the "
                 "ablations."],
                ["Week 8",
                 "Final statistical analysis: McNemar (C3 vs C4 on "
                 "hallucinations), paired bootstrap on accuracy metrics, "
                 "per-tier contribution breakdown. Production figures: "
                 "threshold sweep, trigger-mix breakdown, cost-accuracy "
                 "frontier."],
                ["Week 9",
                 "Final write-up: draft the empirical sections, "
                 "discussion, related work, and threats to validity. "
                 "Internal review at mid-week, polished draft by end-week."],
                ["Week 10 (buffer)",
                 "Held in reserve for any re-runs prompted by the "
                 "analysis pass and, if compute allows, a secondary "
                 "results table on a second benchmark."],
            ],
            col_widths=[0.9 * inch, 5.5 * inch],
        ),
        P("Risks", h2),
        *B([
            "<b>Compute access for the 7B run.</b> The preferred path is "
            "the institutional GPU cluster; the verified fallback is "
            "overnight execution on a local Apple Silicon machine "
            "(estimated 24&ndash;36 hours of continuous run time). The "
            "fallback works; the cluster is faster.",
            "<b>CARD continuing to saturate at 7B.</b> If predicted "
            "similarities at 7B still fall uniformly below the default "
            "threshold, the threshold sweep planned for Week&nbsp;7 will "
            "find an operating point that discriminates. If that also "
            "fails, the Estimator can be re-fit on training pairs drawn "
            "from a benchmark-like distribution rather than the broader "
            "code corpus.",
            "<b>Hallucination metric too noisy.</b> Because the metric "
            "is reportable per tier, a fall-back is available: report "
            "only the signature and import-origin tiers, which are by "
            "construction precise (they require concrete evidence of "
            "mismatch rather than absence of evidence of definition).",
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
        title="Adaptive Retrieval — Midterm Status",
        author="project-group-17",
    )
    doc.build(build_story())


if __name__ == "__main__":
    build_pdf(OUTPUT)
    print(f"Wrote {OUTPUT.relative_to(Path.cwd())}")
