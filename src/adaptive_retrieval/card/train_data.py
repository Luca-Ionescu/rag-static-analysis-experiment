"""Build (features, ES-score) training pairs for the CARD Estimator.

CARD §3.4 calibrates the Estimator on (X, y) holes sampled from Python files
in The Stack: it generates each y **without retrieval** and regresses the
model's intrinsic token-confidence features against ES(y, ŷ). Because no
cross-file context is used at calibration time (see ``construct_training_data``,
which passes ``retrieved=None``), pairs are sampled per *file* — the paper's
"repos with 50–100 files" framing is about corpus coverage, not about feeding
sibling files into the calibration prompt. So per-file sampling is faithful;
what matters is that files are drawn from real packages and that there are
enough of them.

Faithful recipe:
  * Source: stream ``bigcode/the-stack-dedup`` (the corpus CARD used), Python
    subset. ``scripts/01_construct_training_data.py`` filters while streaming.
  * File filter (``is_valid_file``): ≥3 local (relative) imports AND >20
    non-empty lines — files embedded in a real package, exact to the paper.
  * Per file: sample up to ``PAIRS_PER_FILE`` holes; y length ~ Poisson(λ=2),
    X = the 50 lines before the hole.
  * Deduplicate near-identical pairs via K-means on TF-IDF(x+y).

The generation step (``Generator.generate_batch`` over tens of thousands of
prompts) is the multi-hour GPU job. Everything else is pure-Python and
unit-testable on a laptop; ``construct_training_data`` is the orchestrator.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

POISSON_LAMBDA = 2
LINES_X = 50
# CARD §3.4: keep files with ≥3 local (relative) imports — the paper's proxy
# for "this file lives inside a real multi-file package", which is the
# distribution the Estimator must calibrate on. Exact to the paper.
# NOTE: on the small the-stack-smol sample this rejects ~97% of files (mostly
# standalone scripts), which is why calibration MUST stream the full
# the-stack-dedup corpus (see scripts/01_construct_training_data.py), where
# real package files with relative imports are plentiful. Relaxing this
# threshold to recover volume from the-stack-smol trades data quality for
# quantity and biases the Estimator — do not do it; fix the source instead.
MIN_LOCAL_IMPORTS = 3
MIN_NONEMPTY_LINES = 20
TARGET_PAIRS = 250_000
CLUSTER_RATIO = 0.2          # Repoformer Appendix D
PAIRS_PER_FILE = 25          # adapted; see module docstring


_LOCAL_IMPORT_RE = re.compile(r"^from\s+\.\S*\s+import", re.MULTILINE)


@dataclass
class Pair:
    x: str   # input context (50 lines before the hole)
    y: str   # ground-truth completion (Poisson-sampled block)


def count_local_imports(content: str) -> int:
    """Count ``from .module import X`` style relative imports."""
    return len(_LOCAL_IMPORT_RE.findall(content))


def is_valid_file(content: str) -> bool:
    """Filter rule: at least 3 local imports and >20 non-empty lines."""
    nonempty = sum(1 for line in content.splitlines() if line.strip())
    return (
        count_local_imports(content) >= MIN_LOCAL_IMPORTS
        and nonempty > MIN_NONEMPTY_LINES
    )


def sample_pair(file_content: str, rng: np.random.Generator) -> Pair | None:
    """Sample one (X, y) pair from a file. Returns None if the file is too short."""
    lines = file_content.splitlines()
    k = max(1, int(rng.poisson(POISSON_LAMBDA)))
    if len(lines) < LINES_X + k + 5:
        return None
    y_start = int(rng.integers(LINES_X, len(lines) - k))
    y = "\n".join(lines[y_start : y_start + k])
    x = "\n".join(lines[max(0, y_start - LINES_X) : y_start])
    return Pair(x=x, y=y)


def sample_pairs_per_file(
    files: Iterable[str],
    per_file: int = PAIRS_PER_FILE,
    seed: int = 42,
) -> list[Pair]:
    """Generate up to ``per_file`` pairs from each (filtered) file."""
    rng = np.random.default_rng(seed)
    pairs: list[Pair] = []
    for content in files:
        if not is_valid_file(content):
            continue
        for _ in range(per_file):
            p = sample_pair(content, rng)
            if p is not None:
                pairs.append(p)
    return pairs


def kmeans_deduplicate(
    pairs: list[Pair],
    cluster_ratio: float = CLUSTER_RATIO,
    max_features: int = 2000,
    batch_size: int = 4096,
    random_state: int = 42,
    max_clusters: int = 50_000,
) -> list[Pair]:
    """K-means on TF-IDF(x+y), keep one representative per cluster.

    Cluster count = ``cluster_ratio * len(pairs)``, following Repoformer
    Appendix D (cluster_ratio=0.2 targets ~5x deduplication), capped at
    ``max_clusters``. The cap keeps MiniBatchKMeans tractable: cluster counts
    in the hundreds of thousands are computationally infeasible and
    unnecessary for a 13-feature Estimator (tens of thousands of deduplicated
    pairs already saturate it).
    """
    if not pairs:
        return []
    texts = [p.x + "\n" + p.y for p in pairs]
    vec = TfidfVectorizer(max_features=max_features)
    matrix = vec.fit_transform(texts)
    n_clusters = max(1, int(len(pairs) * cluster_ratio))
    n_clusters = min(n_clusters, matrix.shape[0], max_clusters)
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=batch_size,
        n_init=3,
    )
    labels = km.fit_predict(matrix)
    seen: set[int] = set()
    dedup: list[Pair] = []
    for pair, label in zip(pairs, labels):
        if label not in seen:
            seen.add(int(label))
            dedup.append(pair)
    return dedup


def construct_training_data(
    generator,
    files: Iterable[str],
    n_target_pairs: int = TARGET_PAIRS,
    per_file: int = PAIRS_PER_FILE,
    batch_size: int = 32,
    model_family: str = "qwen",
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Orchestrator: sample pairs, dedup, generate, return (features, scores).

    Heavy GPU step is the ``generator.generate_batch`` call. The rest is CPU.
    Use the GPU node for the full ``n_target_pairs=250_000`` run.

    Args:
        files: iterable of Python source strings (e.g. ``the-stack-smol`` content).
        n_target_pairs: cap on dedup output.

    Returns:
        (features, scores): both arrays of length M ≤ n_target_pairs.
    """
    from ..metrics import edit_similarity
    from ..prompt import build_fim_prompt
    from .features import extract_features

    raw = sample_pairs_per_file(files, per_file=per_file, seed=seed)
    pairs = kmeans_deduplicate(raw, random_state=seed)[:n_target_pairs]

    features_list: list[np.ndarray] = []
    scores_list: list[float] = []
    for batch_start in range(0, len(pairs), batch_size):
        batch = pairs[batch_start : batch_start + batch_size]
        # Bare prompts (no FIM tokens) match the CARD paper's setup.
        # For ablation, callers can wrap in build_fim_prompt themselves.
        prompts = [build_fim_prompt(p.x, "", retrieved=None, model_family=model_family) for p in batch]
        gens = generator.generate_batch(prompts)
        for p, g in zip(batch, gens):
            features_list.append(extract_features(g.token_probs, g.token_entropies))
            scores_list.append(edit_similarity(p.y, g.prediction))

    if not features_list:
        return np.zeros((0, 13), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return (
        np.stack(features_list).astype(np.float32),
        np.asarray(scores_list, dtype=np.float32),
    )
