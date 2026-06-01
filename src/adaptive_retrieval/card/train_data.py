"""Build (features, ES-score) training pairs for the CARD Estimator.

Adapted recipe (IMPLEMENTATION_GUIDE Appendix D.3). The CARD paper §3.4
specifies 11k Python repos with 50–100 files each; ``the-stack-smol`` ships
~10k random Python files without repo grouping, so we sample 25 (X, y)
pairs per file instead of ~22 per repo, targeting the same ~250k pairs.

The actual generation step (Generator.generate_batch on hundreds of
thousands of prompts) is a multi-hour GPU job. The helper functions below
are pure-Python and unit-testable on a laptop; ``construct_training_data``
is the orchestrator that calls the Generator.
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
# CARD §3.4 specifies ≥3 local imports — a filter designed for the full
# Stack where files live inside multi-file repos. On the-stack-smol sample
# (~10k flat Python files), the same threshold rejects ~97% of files
# (mostly standalone scripts / notebooks), yielding ~1.6k pairs instead
# of the targeted ~250k. Relaxed to ≥1 as a defensible deviation: files
# with at least one relative import still belong to a package context.
MIN_LOCAL_IMPORTS = 1
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
) -> list[Pair]:
    """K-means on TF-IDF(x+y), keep one representative per cluster.

    Cluster count = ``cluster_ratio * len(pairs)``, following Repoformer
    Appendix D. With cluster_ratio=0.2 this targets ~5x deduplication.
    """
    if not pairs:
        return []
    texts = [p.x + "\n" + p.y for p in pairs]
    vec = TfidfVectorizer(max_features=max_features)
    matrix = vec.fit_transform(texts)
    n_clusters = max(1, int(len(pairs) * cluster_ratio))
    n_clusters = min(n_clusters, matrix.shape[0])
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
