"""Manual smoke test for the Generator's HF backend.

Run with a small public code model so this works on a laptop:

    python scripts/02_smoke_generator.py --model Qwen/Qwen2.5-Coder-0.5B

This first-run will download the model weights to your HF cache (~1 GB for
0.5B, ~4 GB for 1.5B). Subsequent runs use the cache.

Checks (per IMPLEMENTATION_GUIDE §16 Phase 2 validation):
  - Generator returns a non-empty prediction.
  - token_probs is shape (N,) with values in [0, 1].
  - token_entropies is shape (N,) with values >= 0.
  - Latency is recorded.
  - Both baselines run end-to-end through the FIM pipeline.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import click  # noqa: E402

from adaptive_retrieval.baselines import (  # noqa: E402
    always_retrieve_baseline,
    no_retrieve_baseline,
)
from adaptive_retrieval.generator import HFGenerator  # noqa: E402
from adaptive_retrieval.retriever import BM25Retriever  # noqa: E402


# Synthetic instance from Appendix F.
SYNTHETIC_REPO = {
    "server.py": (
        "class Server:\n"
        "    def __init__(self, port=8080):\n"
        "        self.port = port\n"
        "        self.running = False\n"
        "\n"
        "    async def start(self):\n"
        "        self.running = True\n"
        "\n"
        "    async def stop(self):\n"
        "        self.running = False\n"
    ),
}

X_LEFT = (
    "import asyncio\n"
    "\n"
    "from server import Server\n"
    "\n"
    "async def main():\n"
    "    s = Server()\n"
    "    "
)
X_RIGHT = "\n    await asyncio.sleep(1)\n    await s.stop()\n"


@click.command()
@click.option("--model", default="Qwen/Qwen2.5-Coder-0.5B", help="HF model name")
@click.option("--max-tokens", default=20, type=int)
@click.option(
    "--model-family",
    default="qwen",
    type=click.Choice(["qwen", "codellama", "starcoder"]),
)
def main(model: str, max_tokens: int, model_family: str) -> None:
    print(f"Loading {model} via HFGenerator ...")
    t0 = time.time()
    gen = HFGenerator(model, max_tokens=max_tokens)
    print(f"  loaded on device={gen.device} in {time.time() - t0:.1f}s\n")

    # --- raw Generator check ---
    print("[1/3] Raw Generator.generate()")
    out = gen.generate("def hello():\n    return ")
    print(f"  prediction: {out.prediction!r}")
    print(f"  token_ids: {out.token_ids[:8]}{'...' if len(out.token_ids) > 8 else ''}")
    print(f"  token_probs shape: {out.token_probs.shape}, min/max: "
          f"{out.token_probs.min():.3f}/{out.token_probs.max():.3f}")
    print(f"  token_entropies shape: {out.token_entropies.shape}, "
          f"min/max: {out.token_entropies.min():.3f}/{out.token_entropies.max():.3f}")
    print(f"  latency_ms: {out.latency_ms:.1f}")
    assert len(out.prediction) > 0, "Empty prediction"
    assert out.token_probs.shape == out.token_entropies.shape
    assert (out.token_probs >= 0).all() and (out.token_probs <= 1).all()
    assert (out.token_entropies >= 0).all()

    # --- no-retrieve baseline ---
    print("\n[2/3] no_retrieve_baseline")
    out_nr = no_retrieve_baseline(gen, X_LEFT, X_RIGHT, model_family=model_family)
    print(f"  prediction: {out_nr.prediction!r}")
    print(f"  tokens: {len(out_nr.token_ids)}, latency_ms: {out_nr.latency_ms:.1f}")
    assert len(out_nr.token_ids) > 0

    # --- always-retrieve baseline ---
    print("\n[3/3] always_retrieve_baseline")
    retriever = BM25Retriever(SYNTHETIC_REPO)
    out_ar = always_retrieve_baseline(
        gen, retriever, X_LEFT, X_RIGHT, model_family=model_family
    )
    print(f"  prediction: {out_ar.prediction!r}")
    print(f"  tokens: {len(out_ar.token_ids)}, latency_ms: {out_ar.latency_ms:.1f}")
    assert len(out_ar.token_ids) > 0

    print("\nOK — Generator and both baselines produce well-shaped output.")


if __name__ == "__main__":
    main()
