"""Compare two static checkers as the cascade's Stage-3 trigger, post-hoc, on
the cached C1/C2/C3 predictions (CPU only — ŝ₀ and the generations are frozen).

Two architectures for the static gate:
  * ast      : the in-house tree-sitter analyzer, **Tier 1 ONLY** (no symbol
               table), run on the **last --context-lines (50) lines** of x_left
               + the zero-shot line + empty right — matching CARD's calibration
               window.
  * pyflakes : pyflakes F821 'undefined name' run on the **full file**
               (x_left + zero-shot line + x_right), so its in-file imports
               resolve. Tree-sitter Tier-1 is the fallback on parse failure.

For each architecture the cascade retrieves iff (ŝ₀ < t) OR (the checker fires
on the zero-shot completion); the chosen output is C2 (retrieved) or C1
(zero-shot). Accuracy = edit similarity; hallucination = a FIXED, checker-
independent gold-grounded metric (a stored static out-of-scope identifier the
ground-truth line never uses) so the only thing that differs between the two
columns is *which instances the trigger retrieves*.

    python scripts/10_cascade_compare.py --checker both
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import click  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

from adaptive_retrieval.eval.datasets import load_crosscodeeval_python  # noqa: E402
from adaptive_retrieval.static_analysis.analyzer import PredictionAnalyzer  # noqa: E402
from adaptive_retrieval.static_analysis.pyflakes_checker import PyflakesChecker  # noqa: E402
from adaptive_retrieval.static_analysis.scope import InFileScopeAnalyzer  # noqa: E402
from adaptive_retrieval.static_analysis.symbol_table import RepositorySymbolTable  # noqa: E402

_IDENT = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


def _last_lines(text: str, n: int) -> str:
    return "\n".join(text.split("\n")[-n:])


def _mcnemar(card: list[int], casc: list[int]) -> tuple[float, int, int]:
    b = sum(1 for a, x in zip(card, casc) if a == 0 and x == 1)  # cascade worse
    c = sum(1 for a, x in zip(card, casc) if a == 1 and x == 0)  # cascade better
    p = binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) else 1.0
    return p, b, c


@click.command()
@click.option("--results-dir", default="hf_artifacts/results/codellama_7b_line",
              type=click.Path(exists=True, file_okay=False))
@click.option("--checker", type=click.Choice(["ast", "pyflakes", "both"]), default="both",
              help="Which cascade static-checker architecture(s) to run.")
@click.option("--context-lines", default=50,
              help="Left-context window (lines) for the AST checker — CARD's calibration window.")
@click.option("--t-grid", default="0.05,0.10,0.15,0.20,0.25,0.28,0.30,0.35,0.40,0.45")
def main(results_dir: str, checker: str, context_lines: int, t_grid: str) -> None:
    D = Path(results_dir)
    load = lambda n: {r["instance_id"]: r for r in (
        json.loads(l) for l in open(D / n, encoding="utf-8") if l.strip())}
    c1, c2, c3 = load("C1_no_retrieve.jsonl"), load("C2_always_retrieve.jsonl"), load("C3_card.jsonl")
    insts = {i.instance_id: i for i in load_crosscodeeval_python()}
    ids = [i for i in sorted(set(c1) & set(c2) & set(c3) & set(insts))
           if c3[i].get("s_hat_0") is not None]
    n = len(ids)
    do_ast = checker in ("ast", "both")
    do_pf = checker in ("pyflakes", "both")

    # AST Tier-1 ONLY, empty symbol table (Tier 1 never consults it anyway).
    ast_an = PredictionAnalyzer(
        InFileScopeAnalyzer(), RepositorySymbolTable.from_files({}),
        fire_on_out_of_scope=True, fire_on_signature=False, fire_on_import=False,
    )
    pf = PyflakesChecker() if do_pf else None
    predof = lambda r: r.get("prediction_truncated", r["prediction"])
    ES = lambda r: float(r["metrics"]["edit_similarity"])

    def gold_hall(rec, gold: set) -> int:
        # fixed, checker-independent: a stored out-of-scope identifier not in gold
        oos = [x for x in rec.get("static_out_of_scope", []) if x not in gold]
        return 1 if (oos or rec.get("signature_issues") or rec.get("import_issues")) else 0

    shat, es1, es2, h1, h2 = {}, {}, {}, {}, {}
    trig_ast, trig_pf = {}, {}
    print(f"[setup] {n} instances; running checkers on zero-shot (C1) ...")
    for i in ids:
        inst = insts[i]
        gold = set(_IDENT.findall(inst.ground_truth))
        shat[i] = float(c3[i]["s_hat_0"])
        es1[i], es2[i] = ES(c1[i]), ES(c2[i])
        h1[i], h2[i] = gold_hall(c1[i], gold), gold_hall(c2[i], gold)
        p0 = predof(c1[i])  # the zero-shot completion the cascade analyses
        if do_ast:
            trig_ast[i] = ast_an.analyze(p0, _last_lines(inst.x_left, context_lines), "").fires
        if do_pf:
            trig_pf[i] = pf.analyze(p0, inst.x_left, inst.x_right).fires

    print(f"        gold-grounded hallucination floor: C1={np.mean([h1[i] for i in ids]):.3f}  "
          f"C2={np.mean([h2[i] for i in ids]):.3f}   ES: C1={np.mean([es1[i] for i in ids]):.3f}  "
          f"C2={np.mean([es2[i] for i in ids]):.3f}")
    print("\n=== static trigger fire-rate on zero-shot ===")
    if do_ast:
        print(f"  AST Tier-1  (last-{context_lines}-line ctx): {np.mean([trig_ast[i] for i in ids]):.1%}")
    if do_pf:
        print(f"  pyflakes    (full file)               : {np.mean([trig_pf[i] for i in ids]):.1%}"
              f"   (parse-failures: {pf.parse_failures}/{n} = {pf.parse_failures/n:.1%})")
    if do_ast and do_pf:
        agree = np.mean([trig_ast[i] == trig_pf[i] for i in ids])
        both = np.mean([trig_ast[i] and trig_pf[i] for i in ids])
        pf_only = np.mean([trig_pf[i] and not trig_ast[i] for i in ids])
        ast_only = np.mean([trig_ast[i] and not trig_pf[i] for i in ids])
        print(f"  agreement: {agree:.1%}   both-fire: {both:.1%}   pyflakes-only: {pf_only:.1%}   AST-only: {ast_only:.1%}")

    thresholds = [float(x) for x in t_grid.split(",")]
    archs = []
    if do_ast:
        archs.append(("Cascade-AST(T1)", trig_ast))
    if do_pf:
        archs.append(("Cascade-pyflakes", trig_pf))

    print("\n=== cascade comparison (outcome: gold-grounded hallucination) ===")
    hdr = f"{'T':>5} | {'CARD retr':>9} {'ES':>6} {'hall':>6}"
    for name, _ in archs:
        hdr += f" | {name+' retr':>20} {'ES':>6} {'hall':>6} {'Δhall':>7} {'p':>8}"
    print(hdr)
    for t in thresholds:
        card_es = np.mean([es2[i] if shat[i] < t else es1[i] for i in ids])
        card_h = [h2[i] if shat[i] < t else h1[i] for i in ids]
        card_retr = np.mean([shat[i] < t for i in ids])
        row = f"{t:>5.2f} | {card_retr:>8.1%} {card_es:>6.3f} {np.mean(card_h):>6.3f}"
        for name, trig in archs:
            retr = [shat[i] < t or trig[i] for i in ids]
            es = np.mean([es2[i] if r else es1[i] for i, r in zip(ids, retr)])
            hh = [h2[i] if r else h1[i] for i, r in zip(ids, retr)]
            p, b, c = _mcnemar(card_h, hh)
            row += (f" | {np.mean(retr):>19.1%} {es:>6.3f} {np.mean(hh):>6.3f} "
                    f"{np.mean(hh)-np.mean(card_h):>+7.4f} {p:>8.1g}")
        print(row)

    if do_ast and do_pf:
        print("\n=== head-to-head: Cascade-pyflakes vs Cascade-AST (same outcome metric) ===")
        for t in (0.20, 0.25, 0.28):
            ra = [shat[i] < t or trig_ast[i] for i in ids]
            rp = [shat[i] < t or trig_pf[i] for i in ids]
            ha = [h2[i] if r else h1[i] for i, r in zip(ids, ra)]
            hp = [h2[i] if r else h1[i] for i, r in zip(ids, rp)]
            p, b, c = _mcnemar(ha, hp)
            print(f"  T={t}: retr AST={np.mean(ra):.1%} pyflakes={np.mean(rp):.1%}  "
                  f"hall AST={np.mean(ha):.3f} pyflakes={np.mean(hp):.3f}  "
                  f"Δ={np.mean(hp)-np.mean(ha):+.4f}  McNemar p={p:.2g} (pf-better={c}, pf-worse={b})")


if __name__ == "__main__":
    main()
