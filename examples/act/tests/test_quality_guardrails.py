"""CPU tests for quality guardrails (S11.4)."""

from __future__ import annotations

import json

import pytest

from evaluation.quality_guardrails import (
    build_mock_scorer,
    compute_quality,
    load_responses,
    main,
    repetition_rate,
)


def test_repetition_rate_repeated_text() -> None:
    repeated = "the cat sat the cat sat the cat sat"
    varied = "the cat sat on a mat near the river bank today"
    assert repetition_rate(repeated) > repetition_rate(varied)
    assert 0.0 <= repetition_rate(varied) < 1.0


def test_repetition_rate_short_text() -> None:
    assert repetition_rate("two words", 3) == 0.0
    assert repetition_rate("", 3) == 0.0


def test_repetition_rate_invalid_n() -> None:
    with pytest.raises(ValueError, match="n must be >= 2"):
        repetition_rate("some text here", 1)


def test_repetition_rate_perfectly_repeated() -> None:
    # 12 words, n=3 -> 10 trigrams, all identical -> 1 - 1/10
    assert repetition_rate("x " * 12, 3) == pytest.approx(0.9)


def test_compute_quality_without_scorer() -> None:
    responses = ["the cat sat on the mat", "a totally different sentence here now"]
    out = compute_quality(responses)
    assert out["n_responses"] == 2.0
    assert "perplexity" not in out
    assert 0.0 <= out["repetition_rate"]["mean"] <= 1.0
    assert out["n_words"]["mean"] == pytest.approx(6.0)  # 6 words each


def test_compute_quality_with_mock_scorer() -> None:
    responses = [f"response number {i} about topic {i}" for i in range(5)]
    out = compute_quality(responses, scorer=build_mock_scorer())
    assert out["perplexity_failed"] == 0.0
    assert out["perplexity"]["mean"] > 1.0  # exp(CE) with CE in [1, 5)
    assert out["perplexity"]["n"] == 5.0


def test_compute_quality_counts_scorer_failures() -> None:
    def flaky(text: str) -> float:
        if "bad" in text:
            raise RuntimeError("ref model down")
        return 2.0

    out = compute_quality(["good text here", "bad text here", "also fine here"], scorer=flaky)
    assert out["perplexity_failed"] == 1.0
    assert out["perplexity"]["n"] == 2.0
    assert out["perplexity"]["mean"] == pytest.approx(7.389056, rel=1e-4)


def test_compute_quality_all_scorer_failures() -> None:
    def dead(text: str) -> float:
        raise RuntimeError("no ref model")

    out = compute_quality(["x y z"], scorer=dead)
    assert out["perplexity"] is None
    assert out["perplexity_failed"] == 1.0


def test_compute_quality_empty() -> None:
    out = compute_quality([])
    assert out["n_responses"] == 0.0
    assert out["repetition_rate"]["n"] == 0.0


def test_load_responses_list_file(tmp_path) -> None:
    p = tmp_path / "r.json"
    p.write_text(json.dumps(["a b c", "d e f"]), encoding="utf-8")
    assert load_responses(p) == ["a b c", "d e f"]


def test_load_responses_trace_log(tmp_path) -> None:
    p = tmp_path / "t.json"
    trace = {"turns": [
        {"turn": 0, "response": "first response here"},
        {"turn": 1, "response": None},
    ]}
    p.write_text(json.dumps(trace), encoding="utf-8")
    assert load_responses(p) == ["first response here"]


def test_load_responses_rejects_unknown(tmp_path) -> None:
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"foo": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="expected a JSON list"):
        load_responses(p)


def test_cli_quick_smoke(tmp_path) -> None:
    out = tmp_path / "qg.json"
    rc = main(["--quick", "--out", str(out)])
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "unsteered" in data["conditions"]
    assert "best_of_k" in data["conditions"]
    assert data["scorer"] == "mock"
    assert data["n_gram"] == 3.0
    for m in data["conditions"].values():
        assert m["n_responses"] >= 1.0
        assert "perplexity" in m


def test_cli_scorer_model_wired(tmp_path, monkeypatch) -> None:
    import torch

    import act_steering.directions as d

    class _Tok:
        def __call__(self, text, return_tensors, truncation, max_length):
            n = 1 if not text else 4
            return {
                "input_ids": torch.arange(n).unsqueeze(0),
                "attention_mask": torch.ones(1, n),
            }

    class _Model:
        def __call__(self, input_ids=None, attention_mask=None, output_hidden_states=False):
            n = int(input_ids.shape[1]) if input_ids is not None else 1
            logits = torch.zeros(1, n, 7)
            logits[:, :, 1] = 5.0

            class _Out:
                pass

            out = _Out()
            out.logits = logits
            return out

    monkeypatch.setattr(
        d, "load_model_and_pipeline",
        lambda model, device, dtype: (_Model(), _Tok(), None),
    )
    out = tmp_path / "x.json"
    rc = main(["--scorer", "model", "--quick", "--out", str(out)])
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["scorer"] == "model"
    assert data["conditions"]["unsteered"]["perplexity"]["mean"] > 0.0


def test_cli_with_input_files(tmp_path) -> None:
    pa = tmp_path / "a.json"
    pa.write_text(json.dumps(["one two three four", "five six seven eight"]), encoding="utf-8")
    out = tmp_path / "r.json"
    rc = main(["--input", f"cond_a={pa}", "--scorer", "none", "--out", str(out)])
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["conditions"]["cond_a"]["n_responses"] == 2.0
    assert "perplexity" not in data["conditions"]["cond_a"]
    assert data["scorer"] == "none"


def test_build_model_scorer_returns_nats() -> None:
    from evaluation.quality_guardrails import build_model_scorer

    class _Tok:
        def __call__(self, text, return_tensors, truncation, max_length):
            import torch

            n = 1 if not text else 4
            return {
                "input_ids": torch.arange(n).unsqueeze(0),
                "attention_mask": torch.ones(1, n),
            }

    class _Model:
        def __call__(self, input_ids=None, attention_mask=None, output_hidden_states=False):
            import torch

            n = int(input_ids.shape[1]) if input_ids is not None else 1
            logits = torch.zeros(1, n, 7)
            logits[:, :, 1] = 5.0

            class _Out:
                pass

            out = _Out()
            out.logits = logits
            return out

    scorer = build_model_scorer(_Model(), _Tok())
    value = scorer("hello world")
    assert value > 0.0
    assert scorer("") == 0.0
