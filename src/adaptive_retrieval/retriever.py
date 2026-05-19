"""BM25 retrieval over a repository's files.

Chunking and query construction follow the RepoCoder convention
(IMPLEMENTATION_GUIDE §8): 20-line windows with stride 10, query = last
20 lines of x_left.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi


@dataclass
class RetrievedChunk:
    text: str
    file_path: str
    start_line: int
    end_line: int
    score: float


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def make_query(x_left: str, n_lines: int = 20) -> str:
    """Last ``n_lines`` lines of the left context (raw, includes blanks).

    Matching the convention used by ``card.pipeline`` and ``cascade`` so
    every retrieval call in the system sees the same query.
    """
    lines = x_left.splitlines()
    return "\n".join(lines[-n_lines:])


class BM25Retriever:
    def __init__(
        self,
        repo_files: dict[str, str],
        chunk_size: int = 20,
        stride: int = 10,
    ):
        if chunk_size <= 0 or stride <= 0:
            raise ValueError("chunk_size and stride must be positive")
        self.chunk_size = chunk_size
        self.stride = stride
        self.chunks: list[dict] = []
        for path, content in repo_files.items():
            lines = content.splitlines()
            if not lines:
                continue
            n = len(lines)
            # Always emit at least one chunk per non-empty file.
            stop = max(1, n - chunk_size + 1)
            for i in range(0, stop, stride):
                chunk_text = "\n".join(lines[i : i + chunk_size])
                self.chunks.append(
                    {
                        "text": chunk_text,
                        "file_path": path,
                        "start_line": i,
                        "end_line": min(i + chunk_size, n),
                    }
                )
        tokenized = [_tokenize(c["text"]) for c in self.chunks]
        # BM25Okapi crashes on an empty corpus or all-empty docs; guard both.
        if self.chunks and any(t for t in tokenized):
            self.bm25 = BM25Okapi(tokenized)
        else:
            self.bm25 = None

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        if self.bm25 is None:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top_idx = scores.argsort()[::-1][:top_k]
        return [
            RetrievedChunk(score=float(scores[i]), **self.chunks[i]) for i in top_idx
        ]

    def __len__(self) -> int:
        return len(self.chunks)
