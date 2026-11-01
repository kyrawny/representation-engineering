"""CPU tests for the dose-response sweep (S11.7)."""

from __future__ import annotations

import json

import pytest

from evaluation.dose_response import (
    ProbeResult,
    analyze_sweep,
    build_mock_probe,
    main,
    sweep,
)


def test_sweep_requires_baseline() -> None:
    with pytest.raises(ValueError, match="alpha=0"):
        sweep(build_mock_probe(), [0.5, 1.0])


def test_sweep_mock_curve() -> None:
    rows = sweep(build_mock_probe(), [1.0, 0.0, 0.5])
    assert [r["alpha"] for r in rows] == [0.0, 0.5, 1.0]  # sorted
    assert all(r["ok"] for r in rows)
    shifts = [r["epa_shift"][0] for r in rows]
    assert shifts[0] == pytest.approx(0.0)
    assert shifts[1] < shifts[2]  # E shift grows with alpha
    assert rows[0]["perplexity"] == pytest.approx(10.0)


def test_sweep_records_probe_failures() -> None:
    def flaky(alpha: float) -> ProbeResult:
        if alpha == 2.0:
            raise RuntimeError("probe crashed")
        return ProbeResult((0.0, 0.0, 0.0), 10.0, 0.0)

    rows = sweep(flaky, [0.0, 1.0, 2.0])
    assert rows[0]["ok"] is True
    assert rows[1]["ok"] is True
    assert rows[2]["ok"] is False
    assert "probe crashed" in rows[2]["error"]


def test_analyze_sweep_degeneration() -> None:
    rows = sweep(build_mock_probe(), [0.0, 0.25, 0.5, 1.0, 1.5, 2.0])
    report = analyze_sweep(rows)
    deg = report["degeneration"]
    assert deg["degenerated_at"] == pytest.approx(1.5)
    assert deg["safe_alpha_max"] == pytest.approx(1.0)
    assert deg["thresholds"]["ppl_threshold"] == pytest.approx(15.0)
    by_alpha = {r["alpha"]: r for r in report["rows"]}
    assert by_alpha[1.0]["degenerated"] is False
    assert by_alpha[1.5]["degenerated"] is True
    assert by_alpha[0.0]["degenerated"] is False


def test_analyze_sweep_no_degeneration() -> None:
    rows = sweep(build_mock_probe(), [0.0, 0.25, 0.5])
    report = analyze_sweep(rows)
    assert report["degeneration"]["degenerated_at"] is None
    assert report["degeneration"]["safe_alpha_max"] == pytest.approx(0.5)


def test_analyze_sweep_requires_baseline() -> None:
    with pytest.raises(ValueError, match="alpha=0 baseline"):
        analyze_sweep([{"alpha": 0.5, "ok": False, "error": "x"}])


def test_analyze_sweep_slopes_positive() -> None:
    rows = sweep(build_mock_probe(), [0.0, 0.25, 0.5, 1.0])
    report = analyze_sweep(rows)
    slopes = report["per_dimension_slopes"]
    for dim in ("E", "P", "A"):
        assert slopes[dim] is not None
        assert slopes[dim] > 0.0  # type: ignore[operator]
    assert slopes["A"] > slopes["P"]  # mock: A gain 1.1 > P gain 0.6


def test_analyze_sweep_single_point_slopes_none() -> None:
    rows = sweep(build_mock_probe(), [0.0])
    report = analyze_sweep(rows)
    assert report["per_dimension_slopes"] == {"E": None, "P": None, "A": None}
    assert report["degeneration"]["degenerated_at"] is None
    assert report["degeneration"]["safe_alpha_max"] == 0.0


def test_analyze_sweep_failed_rows_ignored_for_slopes() -> None:
    def flaky(alpha: float) -> ProbeResult:
        if alpha >= 1.0:
            raise RuntimeError("down")
        return ProbeResult((alpha, alpha, alpha), 10.0, 0.0)

    rows = sweep(flaky, [0.0, 0.5, 1.0])
    report = analyze_sweep(rows)
    slopes = report["per_dimension_slopes"]
    assert slopes["E"] == pytest.approx(1.0)
    assert report["degeneration"]["degenerated_at"] is None


def test_cli_quick_smoke(tmp_path) -> None:
    out = tmp_path / "dr.json"
    rc = main(["--quick", "--out", str(out)])
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data["rows"]) == 7  # 0, 0.25, 0.5, 1, 1.5, 2, 3
    assert data["degeneration"]["degenerated_at"] == pytest.approx(1.5)
    assert data["degeneration"]["safe_alpha_max"] == pytest.approx(1.0)
    assert data["meta"]["config"]["quick"] is True


def test_cli_custom_alphas(tmp_path) -> None:
    out = tmp_path / "dr.json"
    rc = main(["--probe", "mock", "--alphas", "0", "0.5", "1.0", "--out", str(out)])
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data["rows"]) == 3
    assert data["degeneration"]["degenerated_at"] is None


def test_cli_probe_model_requires_gpu(tmp_path) -> None:
    with pytest.raises(SystemExit):
        main(["--probe", "model", "--out", str(tmp_path / "x.json")])


def test_build_model_probe_sweeps_alpha() -> None:
    import numpy as np

    from act_steering import directions as d
    from evaluation.dose_response import build_model_probe

    class _Tok:
        def __call__(self, text, return_tensors, truncation, max_length):
            import torch

            n = 1 if not text else 3
            return {
                "input_ids": torch.arange(n).unsqueeze(0),
                "attention_mask": torch.ones(1, n),
            }

    class _Model:
        """Hidden state grows with each forward pass (deterministic drift)."""

        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, input_ids=None, attention_mask=None, output_hidden_states=False):
            import torch

            self.calls += 1
            n = int(input_ids.shape[1]) if input_ids is not None else 1
            hidden = torch.zeros(1, n, 4)
            hidden[:, :, 0] = float(self.calls)

            class _Out:
                hidden_states = (hidden,) * 6
                logits = torch.zeros(1, n, 4)

            return _Out()

    class _Steerer:
        def __init__(self, model, tokenizer, bundle, config) -> None:
            self.config = config

        def generate(self, prompt, u_pre=None, **kwargs):
            return [prompt + f" r{self.config.alpha}"]

    directions = {
        "epa.evaluation": {
            2: np.array([1.0, 0.0, 0.0, 0.0]),
            3: np.array([0.0, 1.0, 0.0, 0.0]),
        },
        "epa.potency": {
            2: np.array([0.0, 0.0, 1.0, 0.0]),
            3: np.array([0.0, 0.0, 0.0, 1.0]),
        },
        "epa.activity": {
            2: np.array([0.6, 0.0, 0.0, 0.0]),
            3: np.array([0.0, 0.8, 0.0, 0.0]),
        },
        "deflection_contrast": {
            2: np.array([1.0, 0.0, 0.0, 0.0]),
            3: np.array([0.0, 1.0, 0.0, 0.0]),
        },
    }
    bundle = d.DirectionsBundle(
        model_name="test/dr-fake", hidden_size=4, hidden_layers=[2, 3],
        directions=directions, seed=42,
    )
    model = _Model()
    scorer = lambda text: 2.0  # noqa: E731 - constant nats
    probe = build_model_probe(
        model, _Tok(), bundle, scorer, prompts=["p1"], steerer_cls=_Steerer,
    )
    zero = probe(0.0)
    assert zero.epa_shift == (0.0, 0.0, 0.0)
    assert zero.perplexity == pytest.approx(np.exp(2.0))
    one = probe(1.0)
    # One extra reader pass at alpha=1: E mean = 2/2 = 1.0 vs baseline 0.5;
    # A mean = 0.6*2/2 = 0.6 vs baseline 0.3 (P stays 0).
    assert one.epa_shift[0] == pytest.approx(0.5)
    assert one.epa_shift[1] == pytest.approx(0.0)
    assert one.epa_shift[2] == pytest.approx(0.3)


def test_build_model_probe_requires_prompts() -> None:
    from act_steering import directions as d
    from evaluation.dose_response import build_model_probe

    bundle = d.DirectionsBundle(
        model_name="x", hidden_size=4, hidden_layers=[2], directions={}, seed=0,
    )
    with pytest.raises(ValueError, match="prompts"):
        build_model_probe(None, None, bundle, lambda t: 0.0, prompts=[])
