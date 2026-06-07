"""Dataset loaders. Yields canonical ``Instance`` objects.

IMPORTANT DEVIATION from IMPLEMENTATION_GUIDE Appendix D.1:
  The guide says to use ``line_completion.jsonl`` (raw) as the primary file
  because it contains the cross-file context "but no pre-retrieved chunks."
  In the actual shipped data, the raw file has ``crossfile_context: None`` —
  the chunks live ONLY in the rg1/oracle variants. The ``prompt`` field is
  byte-identical across all variants, so swapping in ``rg1_bm25`` just gives
  us a chunk list to retrieve over; it does NOT inject retrieval into the
  prompt. This matches what CARD and Repoformer evidently did.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import jsonlines


@dataclass
class Instance:
    """Canonical per-instance shape used by every config."""

    x_left: str
    x_right: str
    ground_truth: str
    repo_files: dict[str, str] = field(default_factory=dict)  # filename -> content
    instance_id: str = ""
    target_file: str = ""
    repository: str | None = None


# Default to the rg1_bm25 variant — it ships the chunk corpus we treat as
# the per-instance "repository". See the module docstring for why.
DEFAULT_CCE_PYTHON_PATH = (
    "data/crosscodeeval/crosscodeeval_data/python/line_completion_rg1_bm25.jsonl"
)


def load_crosscodeeval_python(
    path: str | Path = DEFAULT_CCE_PYTHON_PATH,
    include_target_file: bool = True,
) -> Iterator[Instance]:
    """Stream Python CrossCodeEval instances.

    Args:
        path: JSONL with ``prompt``, ``right_context``, ``groundtruth``,
            ``metadata``, and (in rg1/oracle variants) ``crossfile_context.list``.
        include_target_file: If True, also include the current file
            (synthesised from x_left + ground_truth + x_right) in repo_files
            so the symbol table picks up in-file names.

    Yields:
        Instance objects.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CrossCodeEval JSONL not found: {p}")

    with jsonlines.open(p) as reader:
        for rec in reader:
            meta = rec.get("metadata", {})
            crossfile = rec.get("crossfile_context")
            if isinstance(crossfile, dict):
                chunks = crossfile.get("list", [])
            elif isinstance(crossfile, list):  # tolerate the guide-shape too
                chunks = crossfile
            else:
                chunks = []

            repo_files: dict[str, str] = {}
            for ch in chunks:
                fname = ch.get("filename")
                text = ch.get("retrieved_chunk")
                if fname and text:
                    # Multiple chunks may share a filename (different windows
                    # of the same source file). Concatenate so we keep all
                    # symbols visible in the symbol table.
                    if fname in repo_files:
                        repo_files[fname] += "\n" + text
                    else:
                        repo_files[fname] = text

            target_file = meta.get("file", "current_file.py")
            if include_target_file:
                synthetic = (
                    rec["prompt"] + rec["groundtruth"] + rec.get("right_context", "")
                )
                # If target_file already exists as a chunk source, prefer the
                # fuller synthetic version (it has the surrounding lines too).
                repo_files[target_file] = synthetic

            yield Instance(
                x_left=rec["prompt"],
                x_right=rec.get("right_context", ""),
                ground_truth=rec["groundtruth"],
                repo_files=repo_files,
                instance_id=meta.get("task_id", ""),
                target_file=target_file,
                repository=meta.get("repository"),
            )


# ---------- Cross-instance repository index ----------

def build_repo_chunks_index(
    instances: Iterable[Instance],
) -> dict[str, dict[str, str]]:
    """Group cross-file chunks across all instances of the same repository.

    Per-instance, CCE ships only ~5 chunks selected by BM25 — a tiny slice of
    the real repo's name space. When multiple instances belong to the same
    repository, their chunk sets typically overlap partially. Unioning the
    chunks per repo gives the analyzer a much larger symbol table to check
    identifier resolution against, without changing what gets retrieved at
    inference time (BM25 retrieval still operates per-instance).

    The synthesised target-file content (x_left + ground_truth + x_right) is
    excluded from the index — it's instance-specific.

    Returns ``{repo_name: {filename: content}}`` where the value is built from
    every cross-file chunk that any instance of this repo shipped. If the same
    filename appears in multiple instances, contents are concatenated so that
    no chunk's symbols are lost.
    """
    index: dict[str, dict[str, str]] = {}
    for inst in instances:
        if not inst.repository:
            continue
        bucket = index.setdefault(inst.repository, {})
        for path, content in inst.repo_files.items():
            if path == inst.target_file:
                continue  # synthesised current-file content, instance-local
            if path in bucket:
                bucket[path] += "\n" + content
            else:
                bucket[path] = content
    return index


# ---------- RepoEval ----------

DEFAULT_REPOEVAL_BASE = Path("data/repoeval")


def load_repoeval(
    task: str = "line",
    base_path: str | Path = DEFAULT_REPOEVAL_BASE,
) -> Iterator[Instance]:
    """Stream RepoEval Python instances.

    Expects the RepoCoder data layout under ``base_path``:
        <base_path>/datasets/<task>_level_completion_2k_context_codex.test.jsonl
        <base_path>/repositories/<repo>/...

    Per IMPLEMENTATION_GUIDE Appendix D.2, ``_2k_context_codex`` matches the
    CARD paper's setup. ``task`` is one of {``line``, ``api``, ``function``}.

    The JSONL ``prompt`` field is left-context-only and is reconstructed from
    the on-disk file (not used as-is — some variants ship retrieval baked in).
    Cross-file context is built by globbing every .py file in the target
    repository's directory tree.
    """
    if task not in ("line", "api", "function"):
        raise ValueError(f"task must be one of line/api/function, got {task!r}")

    base = Path(base_path)
    jsonl = base / "datasets" / f"{task}_level_completion_2k_context_codex.test.jsonl"
    if not jsonl.exists():
        raise FileNotFoundError(
            f"RepoEval data not found: {jsonl}. "
            f"Download from https://github.com/microsoft/CodeT/tree/main/RepoCoder "
            f"and extract to {base}/"
        )
    repos_root = base / "repositories"
    if not repos_root.exists():
        raise FileNotFoundError(
            f"RepoEval repositories not found: {repos_root}. "
            f"Download the repositories archive from the same RepoCoder repo."
        )

    # One repo serves hundreds of instances; read each repo's .py files ONCE and
    # share the same dict across all its instances (the consumers — BM25Retriever,
    # the symbol-union builder — only read it). Without this every instance held a
    # private full copy of its repo, blowing system RAM into the tens of GB.
    repo_files_cache: dict[str, dict[str, str]] = {}

    with jsonlines.open(jsonl) as reader:
        for row_idx, rec in enumerate(reader):
            meta = rec["metadata"]
            fpath_tuple = meta["fpath_tuple"]
            fpath = repos_root.joinpath(*fpath_tuple)
            try:
                full_content = fpath.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            # RepoCoder's function-level metadata uses ``lineno`` (the hole's
            # start line); older/other variants used ``line_no``. Accept both.
            line_no = meta.get("lineno", meta.get("line_no"))
            if line_no is None:
                continue
            gt = meta["ground_truth"]

            # Locate the hole by the literal ground-truth text rather than by
            # counting its newlines. The old ``gt.count("\n")+1`` heuristic
            # mis-reconstructs ~89% of real function-level files: the stored
            # ground_truth omits a trailing newline that the source line has,
            # so ``line_no + gt_line_count`` lands one line off and x_right
            # drops the separating newline. ``ground_truth`` appears verbatim
            # at ``line_no`` (verified on all 455 function instances), so we
            # anchor the search there and take everything after it as x_right.
            lines = full_content.splitlines(keepends=True)
            search_from = len("".join(lines[:line_no]))
            idx = full_content.find(gt, search_from)
            if idx == -1:
                # Fall back to an unanchored search; skip if gt is truly absent
                # (a handful of instances have whitespace-normalised gt).
                idx = full_content.find(gt)
                if idx == -1:
                    continue

            x_left = full_content[:idx]
            x_right = full_content[idx + len(gt):]
            # Invariant: the split is lossless — x_left + gt + x_right rebuilds
            # the file exactly. Cheap guard against future metadata drift.
            assert x_left + gt + x_right == full_content

            # Per-instance "repository" = every .py under the target repo's root,
            # cached and shared across all instances of the same repo.
            repo_name = fpath_tuple[0]
            repo_files = repo_files_cache.get(repo_name)
            if repo_files is None:
                repo_dir = repos_root / repo_name
                repo_files = {}
                for f in repo_dir.rglob("*.py"):
                    try:
                        repo_files[str(f.relative_to(repo_dir))] = f.read_text(encoding="utf-8")
                    except (OSError, UnicodeDecodeError):
                        continue
                repo_files_cache[repo_name] = repo_files

            # RepoCoder's metadata task_id is a non-unique placeholder
            # ("<repo>/idx" — the index is never filled), so all instances of a
            # repo share it. Build a unique instance_id from the row index plus
            # the file/line/function so per-instance joins (selectivity check,
            # cascade compare, McNemar pairing) match correctly.
            base_id = meta.get("task_id", repo_name) or repo_name
            fn = meta.get("function_name", "")
            instance_id = f"{base_id}#{row_idx}:{'/'.join(fpath_tuple)}:{line_no}:{fn}"

            yield Instance(
                x_left=x_left,
                x_right=x_right,
                ground_truth=gt,
                repo_files=repo_files,
                instance_id=instance_id,
                target_file=str(Path(*fpath_tuple)),
                repository=repo_name,
            )


# ---------- central dataset registry (single source of truth for --dataset) ----------
# Lets every script toggle the benchmark with one CLI flag. CrossCodeEval is the
# default; the RepoEval entries need the RepoCoder data provisioned under
# data/repoeval/ (see load_repoeval). 04_run_experiment.py keeps its own copy of
# this map for back-compat; the post-hoc scripts import this one.
DATASET_LOADERS = {
    "crosscodeeval_py": lambda: load_crosscodeeval_python(),
    "repoeval_line": lambda: load_repoeval(task="line"),
    "repoeval_api": lambda: load_repoeval(task="api"),
    "repoeval_function": lambda: load_repoeval(task="function"),
}

# Tasks whose completion is a multi-line body (no first-line truncation at
# scoring time). CrossCodeEval line-completion truncates to one line; RepoEval
# function-completion does not.
MULTILINE_DATASETS = {"repoeval_function"}
