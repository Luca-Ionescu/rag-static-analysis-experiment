"""CachedGenerator (disk-backed generation cache) tests."""
from __future__ import annotations

from adaptive_retrieval.generator import CachedGenerator, MockGenerator


def test_first_call_is_a_miss(tmp_path):
    inner = MockGenerator(default_prediction="OUT")
    cached = CachedGenerator(inner, cache_dir=tmp_path, model_name="m", max_tokens=10)
    cached.generate("prompt-A")
    assert cached.misses == 1
    assert cached.hits == 0
    assert len(inner.call_log) == 1


def test_second_call_same_prompt_is_a_hit(tmp_path):
    inner = MockGenerator(default_prediction="OUT")
    cached = CachedGenerator(inner, cache_dir=tmp_path, model_name="m", max_tokens=10)
    cached.generate("prompt-A")
    cached.generate("prompt-A")
    assert cached.misses == 1
    assert cached.hits == 1
    # Inner generator only called once.
    assert len(inner.call_log) == 1


def test_different_prompts_are_different_misses(tmp_path):
    inner = MockGenerator(default_prediction="OUT")
    cached = CachedGenerator(inner, cache_dir=tmp_path, model_name="m", max_tokens=10)
    cached.generate("A")
    cached.generate("B")
    cached.generate("C")
    assert cached.misses == 3
    assert cached.hits == 0


def test_different_model_names_have_separate_caches(tmp_path):
    inner1 = MockGenerator(default_prediction="OUT")
    inner2 = MockGenerator(default_prediction="OUT")
    c1 = CachedGenerator(inner1, cache_dir=tmp_path, model_name="m1", max_tokens=10)
    c2 = CachedGenerator(inner2, cache_dir=tmp_path, model_name="m2", max_tokens=10)
    c1.generate("prompt")
    c2.generate("prompt")
    # Both should miss (different model names → different keys)
    assert c1.misses == 1 and c2.misses == 1


def test_different_max_tokens_have_separate_caches(tmp_path):
    inner1 = MockGenerator(default_prediction="OUT")
    inner2 = MockGenerator(default_prediction="OUT")
    c1 = CachedGenerator(inner1, cache_dir=tmp_path, model_name="m", max_tokens=10)
    c2 = CachedGenerator(inner2, cache_dir=tmp_path, model_name="m", max_tokens=20)
    c1.generate("prompt")
    c2.generate("prompt")
    assert c1.misses == 1 and c2.misses == 1


def test_cache_persists_across_instances(tmp_path):
    inner1 = MockGenerator(default_prediction="OUT")
    c1 = CachedGenerator(inner1, cache_dir=tmp_path, model_name="m", max_tokens=10)
    c1.generate("prompt")
    # Create a fresh cache instance pointing at the same directory.
    inner2 = MockGenerator(default_prediction="DIFFERENT")
    c2 = CachedGenerator(inner2, cache_dir=tmp_path, model_name="m", max_tokens=10)
    out = c2.generate("prompt")
    # Hit the disk cache (not the new inner generator).
    assert out.prediction == "OUT"
    assert c2.hits == 1
    assert c2.misses == 0
    assert len(inner2.call_log) == 0


def test_generate_batch_mixes_hits_and_misses(tmp_path):
    inner = MockGenerator(default_prediction="OUT")
    cached = CachedGenerator(inner, cache_dir=tmp_path, model_name="m", max_tokens=10)
    cached.generate("A")  # warm cache for A
    cached.generate("B")  # warm cache for B
    inner.call_log.clear()

    outs = cached.generate_batch(["A", "B", "C"])
    assert len(outs) == 3
    assert cached.hits == 2  # A and B hit
    assert cached.misses == 3  # initial A, B, plus new C
    # Inner generator only saw "C" this round.
    assert inner.call_log == ["C"]


def test_corrupt_cache_falls_back_to_regenerate(tmp_path):
    inner = MockGenerator(default_prediction="OUT")
    cached = CachedGenerator(inner, cache_dir=tmp_path, model_name="m", max_tokens=10)
    # Place a corrupt file at the expected location.
    key = cached._key("prompt")
    path = cached._path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a pickle file")

    out = cached.generate("prompt")
    # Fell through to inner.generate; result is still OUT.
    assert out.prediction == "OUT"
    assert cached.misses == 1
    # The corrupt file should now be overwritten with a valid pickle.
    out2 = cached.generate("prompt")
    assert cached.hits == 1
    assert out2.prediction == "OUT"
