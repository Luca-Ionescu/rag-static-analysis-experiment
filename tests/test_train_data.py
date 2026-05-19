"""Pure-Python helpers in train_data.py. The full GPU pipeline runs on the
GPU node — these tests only exercise sampling, filtering, and dedup.
"""
from __future__ import annotations

import numpy as np

from adaptive_retrieval.card.train_data import (
    Pair,
    count_local_imports,
    is_valid_file,
    kmeans_deduplicate,
    sample_pair,
    sample_pairs_per_file,
)


# ---------- count_local_imports ----------

def test_count_local_imports_basic():
    src = (
        "from .module1 import a\n"
        "from .module2 import b, c\n"
        "from .pkg.module3 import d\n"
        "import os\n"
        "from numpy import array\n"  # not a relative import
    )
    assert count_local_imports(src) == 3


def test_count_local_imports_zero():
    assert count_local_imports("import os\nimport sys\n") == 0


def test_count_local_imports_empty():
    assert count_local_imports("") == 0


# ---------- is_valid_file ----------

def test_is_valid_file_true_when_imports_and_lines_sufficient():
    src = (
        "from .a import x\n"
        "from .b import y\n"
        "from .c import z\n"
        + "\n".join(f"x = {i}" for i in range(25))
        + "\n"
    )
    assert is_valid_file(src)


def test_is_valid_file_false_when_too_few_imports():
    src = "from .a import x\n" + "\n".join(f"x = {i}" for i in range(50)) + "\n"
    assert not is_valid_file(src)


def test_is_valid_file_false_when_too_few_lines():
    src = (
        "from .a import x\n"
        "from .b import y\n"
        "from .c import z\n"
        "x = 1\n"
    )
    assert not is_valid_file(src)


# ---------- sample_pair ----------

def test_sample_pair_returns_none_for_short_file():
    rng = np.random.default_rng(0)
    short = "\n".join(f"x = {i}" for i in range(20))  # 20 lines: less than LINES_X + k + 5
    assert sample_pair(short, rng) is None


def test_sample_pair_returns_pair_with_correct_x_length():
    rng = np.random.default_rng(0)
    lines = "\n".join(f"x = {i}" for i in range(200))
    p = sample_pair(lines, rng)
    assert p is not None
    # x should have exactly 50 non-empty lines (LINES_X)
    assert len(p.x.split("\n")) == 50
    # y is a Poisson-sampled small block, so 1+ lines.
    assert len(p.y.split("\n")) >= 1


def test_sample_pair_x_precedes_y_in_file():
    rng = np.random.default_rng(42)
    lines = "\n".join(f"line_{i}" for i in range(150))
    p = sample_pair(lines, rng)
    assert p is not None
    # The last line of x should appear in the source, and the first line of y
    # should be the line immediately after it.
    src_lines = lines.split("\n")
    x_last = p.x.split("\n")[-1]
    y_first = p.y.split("\n")[0]
    x_idx = src_lines.index(x_last)
    y_idx = src_lines.index(y_first)
    assert y_idx == x_idx + 1


# ---------- sample_pairs_per_file ----------

def test_sample_pairs_per_file_only_valid_files_contribute():
    valid = (
        "from .a import x\n"
        "from .b import y\n"
        "from .c import z\n"
        + "\n".join(f"line_{i}" for i in range(100))
        + "\n"
    )
    invalid_short = "from .a import x\nx = 1\n"
    invalid_no_imports = "\n".join(f"x = {i}" for i in range(100)) + "\n"
    pairs = sample_pairs_per_file([valid, invalid_short, invalid_no_imports], per_file=5)
    # Only the valid file produced pairs.
    assert 1 <= len(pairs) <= 5


# ---------- kmeans_deduplicate ----------

def test_kmeans_deduplicate_reduces_duplicates():
    # 50 near-duplicate pairs
    base_x = "import os\ndef foo():\n    return 1\n"
    base_y = "def bar():\n    return 2\n"
    pairs = [Pair(x=base_x, y=base_y) for _ in range(50)]
    deduped = kmeans_deduplicate(pairs, cluster_ratio=0.1)
    # With 50 → 5 clusters, we expect roughly 5 representatives.
    assert 1 <= len(deduped) <= 10
    assert len(deduped) < len(pairs)


def test_kmeans_deduplicate_handles_empty_input():
    assert kmeans_deduplicate([]) == []


def test_kmeans_deduplicate_preserves_distinct_pairs():
    # Topically-distinct pairs (different keywords/libraries) — should keep most.
    pairs = [
        Pair(x="import os\ndef list_dir(p):\n    return os.listdir(p)\n", y="path = '/tmp'\n"),
        Pair(x="from typing import List\nclass Node:\n    next: 'Node'\n", y="self.next = None\n"),
        Pair(x="import numpy as np\narr = np.array([1, 2])\n", y="result = np.sum(arr)\n"),
        Pair(x="def sql_query(query):\n    return db.execute(query)\n", y="rows = []\n"),
        Pair(x="async def fetch(url):\n    return await client.get(url)\n", y="payload = resp.json()\n"),
    ]
    deduped = kmeans_deduplicate(pairs, cluster_ratio=1.0)
    # K-means doesn't guarantee one-per-cluster even with k=n, but topically
    # distinct inputs should keep most.
    assert len(deduped) >= 3
