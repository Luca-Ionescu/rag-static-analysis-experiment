"""Fill-in-the-middle prompt assembly for code completion models."""
from __future__ import annotations

from .retriever import RetrievedChunk

FIM_TOKENS: dict[str, tuple[str, str, str]] = {
    "qwen": ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"),
    "codellama": ("▁<PRE>", "▁<SUF>", "▁<MID>"),
    "starcoder": ("<fim_prefix>", "<fim_suffix>", "<fim_middle>"),
}


def _format_retrieved(chunks: list[RetrievedChunk]) -> str:
    out = "# Here are some relevant code fragments from other files of the repo:\n"
    for chunk in chunks:
        out += f"# the below code fragment can be found in:\n# {chunk.file_path}\n"
        out += "\n".join("# " + line for line in chunk.text.splitlines())
        out += "\n\n"
    return out


def build_fim_prompt(
    x_left: str,
    x_right: str,
    retrieved: list[RetrievedChunk] | None = None,
    model_family: str = "qwen",
) -> str:
    """Assemble a FIM prompt. Retrieved chunks (if any) go above x_left as
    commented context, matching RepoCoder's prompt template.
    """
    try:
        fim_prefix, fim_suffix, fim_middle = FIM_TOKENS[model_family]
    except KeyError as e:
        raise ValueError(f"Unknown model family: {model_family!r}") from e

    if retrieved:
        x_left = _format_retrieved(retrieved) + x_left

    return f"{fim_prefix}{x_left}{fim_suffix}{x_right}{fim_middle}"
