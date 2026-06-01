"""Selectivity check: does CARD's confidence gate target the instances that
actually benefit from retrieval?

Motivation
----------
RQ1/RQ2 are only meaningful if CARD's gate is *selective* — i.e. it retrieves
preferentially for the instances that gain the most from cross-file context. If
it instead fired ~uniformly (or, worse, on everything), the cascade would have
nothing to improve and the experiment would be degenerate.

For every instance we measure the **retrieval ES gain**:

    gain = ES(always-retrieve) - ES(no-retrieve)
         = ES(C2)             - ES(C1)

i.e. how much edit-similarity that instance gains when fed BM25 cross-file
context. CARD's gate fires when ŝ₀ (predicted zero-shot ES) is *low*, so a
genuinely selective gate implies a **negative** association between ŝ₀ and gain:
low-confidence instances should be the ones retrieval helps.

This script matches C1/C2/C3 by instance_id (ŝ₀ is read from the C3 records),
then reports:
  * ES gain bucketed by ŝ₀ quintile,
  * Pearson + Spearman corr(ŝ₀, gain)  — negative ⇒ selective,
  * targeting efficiency: mean gain captured by retrieving the lowest-ŝ k%
    vs. a random k% (= overall mean) vs. the highest-ŝ k%.

It reads frozen JSONL only — no model, no GPU — so it runs on a laptop and can
be re-run verbatim on the real 7B results once the pod finishes.

Usage:
    python scripts/08_selectivity_check.py --results-dir results/cce_py_1.5b
"""
from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np

try:  # faithful to src/adaptive_retrieval/metrics.edit_similarity
    from Levenshtein import distance as _lev
except ImportError:  # laptop fallback — same formula, pure Python
    def _lev(a: str, b: str) -> int:
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]


# FIM / sentinel markers that leak into raw generations when the model is not
# stopped at the completion boundary (CodeLlama: ▁<PRE>/<SUF>/<MID>/<EOT>;
# Qwen: <|fim_*|>/<|file_sep|>). Cutting at the first one removes the
# over-generation tail even when it shares the completion's line.
_SPECIAL_MARKERS = (
    "<|", "▁<", "<fim", "<PRE>", "<SUF>", "<MID>", "<EOT>", "<MID", "</s>", "<｜",
)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise click.ClickException(f"Missing required file: {path}")
    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _edit_similarity(reference: str, prediction: str) -> float:
    """Mirrors src/adaptive_retrieval/metrics.edit_similarity exactly."""
    if not reference and not prediction:
        return 1.0
    denom = max(len(reference), len(prediction))
    if denom == 0:
        return 1.0
    return 1.0 - _lev(reference, prediction) / denom


def _truncate(text: str, n_lines: int) -> str:
    """Keep the first ``n_lines`` lines, cut at the first sentinel marker."""
    out = "\n".join(text.split("\n")[:n_lines])
    for m in _SPECIAL_MARKERS:
        i = out.find(m)
        if i != -1:
            out = out[:i]
    return out


def _es(rec: dict, truncate_lines: int) -> float:
    """Stored ES when ``truncate_lines==0``; otherwise recompute on the
    first-``truncate_lines``-line prediction (the CCE line-completion view)."""
    if truncate_lines <= 0:
        return float(rec["metrics"]["edit_similarity"])
    return _edit_similarity(
        rec["ground_truth"], _truncate(rec["prediction"], truncate_lines)
    )


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho = Pearson on ranks (ties broken by argsort order)."""
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


@click.command()
@click.option(
    "--results-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory holding C1_no_retrieve.jsonl, C2_always_retrieve.jsonl, "
    "C3_card.jsonl.",
)
@click.option("--c1", default="C1_no_retrieve.jsonl")
@click.option("--c2", default="C2_always_retrieve.jsonl")
@click.option("--c3", default="C3_card.jsonl")
@click.option("--n-buckets", default=5, type=int, help="ŝ₀ quantile buckets.")
@click.option(
    "--truncate-lines",
    default=0,
    type=int,
    help="0 = use the ES stored in the JSONL as-is. N>=1 = recompute ES on the "
    "first-N-line prediction (use 1 for CCE line completion, which strips the "
    "FIM over-generation tail the harness does not currently truncate).",
)
def main(
    results_dir: str, c1: str, c2: str, c3: str, n_buckets: int, truncate_lines: int
) -> None:
    rd = Path(results_dir)
    recs1 = {r["instance_id"]: r for r in _read_jsonl(rd / c1)}
    recs2 = {r["instance_id"]: r for r in _read_jsonl(rd / c2)}
    recs3 = {r["instance_id"]: r for r in _read_jsonl(rd / c3)}

    if truncate_lines > 0:
        print(
            f"[note] recomputing ES on first-{truncate_lines}-line predictions "
            "(CCE line-completion view; strips FIM over-generation)."
        )
    else:
        print("[note] using ES stored in the JSONL (raw, un-truncated predictions).")

    # Match across all three; ŝ₀ comes from C3.
    ids = sorted(set(recs1) & set(recs2) & set(recs3))
    s_hat, gain, es_c1, es_c2 = [], [], [], []
    for i in ids:
        s0 = recs3[i].get("s_hat_0")
        if s0 is None:
            continue
        e1, e2 = _es(recs1[i], truncate_lines), _es(recs2[i], truncate_lines)
        s_hat.append(float(s0))
        es_c1.append(e1)
        es_c2.append(e2)
        gain.append(e2 - e1)

    if len(s_hat) < n_buckets:
        raise click.ClickException(
            f"Only {len(s_hat)} matched instances with ŝ₀ — too few to analyse."
        )

    s_hat = np.asarray(s_hat)
    gain = np.asarray(gain)
    es_c1 = np.asarray(es_c1)
    es_c2 = np.asarray(es_c2)
    n = len(s_hat)

    print(f"[setup] results-dir={rd}")
    print(f"        matched instances: {n}")
    print(
        f"        ŝ₀:   min={s_hat.min():.3f}  median={np.median(s_hat):.3f}  "
        f"max={s_hat.max():.3f}"
    )
    print(
        f"        ES:   no-retrieve(C1)={es_c1.mean():.3f}  "
        f"always-retrieve(C2)={es_c2.mean():.3f}  "
        f"mean gain={gain.mean():+.3f}"
    )
    pos = float(np.mean(gain > 0))
    print(
        f"        retrieval helps on {pos:.1%} of instances, "
        f"hurts on {float(np.mean(gain < 0)):.1%}, neutral on "
        f"{float(np.mean(gain == 0)):.1%}"
    )

    # ---- ŝ₀ quantile buckets ----------------------------------------------
    print(f"\n[buckets] ES gain by ŝ₀ quantile (low ŝ₀ = CARD wants to retrieve)")
    print(f"  {'ŝ₀ range':>16}  {'n':>5}  {'mean ŝ₀':>8}  {'mean gain':>10}  "
          f"{'ES C1':>7}  {'ES C2':>7}")
    order = np.argsort(s_hat)
    for b in range(n_buckets):
        lo = b * n // n_buckets
        hi = (b + 1) * n // n_buckets if b < n_buckets - 1 else n
        sl = order[lo:hi]
        rng = f"[{s_hat[sl].min():.3f},{s_hat[sl].max():.3f}]"
        print(
            f"  {rng:>16}  {len(sl):>5}  {s_hat[sl].mean():>8.3f}  "
            f"{gain[sl].mean():>+10.3f}  {es_c1[sl].mean():>7.3f}  "
            f"{es_c2[sl].mean():>7.3f}"
        )

    # ---- correlations ------------------------------------------------------
    pear = float(np.corrcoef(s_hat, gain)[0, 1])
    spear = _spearman(s_hat, gain)
    print(f"\n[corr] corr(ŝ₀, ES gain):  Pearson={pear:+.3f}  Spearman={spear:+.3f}")
    print("       negative ⇒ selective (low confidence ⇒ larger retrieval gain)")

    # ---- targeting efficiency ---------------------------------------------
    # If we could retrieve for only k% of instances, does picking the lowest-ŝ
    # k% beat random (= overall mean) and beat the highest-ŝ k%?
    print(f"\n[targeting] mean ES gain captured by retrieving k% of instances")
    print(f"  {'k%':>5}  {'lowest-ŝ':>9}  {'random':>8}  {'highest-ŝ':>10}  "
          f"{'low/rand':>9}")
    overall = gain.mean()
    for k in (10, 20, 30, 50):
        m = max(1, int(round(n * k / 100)))
        low = gain[order[:m]].mean()       # lowest ŝ₀  (CARD retrieves these)
        high = gain[order[-m:]].mean()     # highest ŝ₀ (CARD skips these)
        ratio = low / overall if overall != 0 else float("nan")
        print(
            f"  {k:>4}%  {low:>+9.3f}  {overall:>+8.3f}  {high:>+10.3f}  "
            f"{ratio:>8.2f}x"
        )

    # ---- verdict -----------------------------------------------------------
    print("\n[verdict]")
    selective = (pear < -0.05) and (spear < -0.05)
    if selective:
        print(
            "  PASS — ŝ₀ is negatively associated with retrieval gain: CARD's "
            "gate preferentially targets the instances retrieval actually helps. "
            "A T_RAG below the ŝ₀ range will retrieve a high-value SUBSET, "
            "leaving room for the cascade's Stage 3 to act on the rest."
        )
    else:
        print(
            "  WEAK/FAIL — ŝ₀ shows little/no negative association with gain. "
            "CARD's gate is close to non-selective on this model+benchmark; a "
            "partial-retrieval operating point would behave near-randomly. "
            "Investigate estimator calibration before trusting C3/C4 contrasts."
        )


if __name__ == "__main__":
    main()
