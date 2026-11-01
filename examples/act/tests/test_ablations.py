"""CPU tests for the ablation condition matrix (S11.6)."""

from __future__ import annotations

import json

import pytest

from act_steering.config import DataPaths, PipelineConfig
from act_steering.dictionary import ACTDictionary
from act_steering.pipeline import QUICK_SCRIPT, ACTCoefficients
from evaluation.ablations import (
    CONDITIONS,
    ConditionSpec,
    battery_condition_summary,
    build_condition_components,
    contrast_vs_baseline,
    load_battery,
    main,
    run_battery_condition,
    run_condition,
    steer_condition_for,
    summarize_trace,
)

EXPECTED_LABELS = {
    "unsteered",
    "monitor_only",
    "steer",
    "random_dir",
    "constant_vec",
    "identity_only",
    "shuffled",
    "situation_dependent",
    "best_of_k",
    "best_of_k_steer",
}


def _small_battery(n: int = 6) -> list[dict]:
    """Build n battery scenarios from real dictionary terms (CPU)."""
    data = DataPaths()
    dictionary = ACTDictionary.load(data.data_dir)
    user_terms = [t.term for t in dictionary.identities[:3]]
    agent_terms = [t.term for t in dictionary.identities[3:6]]
    msg_terms = [t.term for t in dictionary.behaviors[:n]]
    return [
        {
            "index": i,
            "user_term": user_terms[i % 3],
            "agent_term": agent_terms[i % 3],
            "message_term": msg_terms[i % n],
        }
        for i in range(n)
    ]


def test_condition_matrix_labels() -> None:
    names = [spec.name for spec in CONDITIONS]
    assert set(names) == EXPECTED_LABELS
    assert len(names) == len(set(names))


def test_steer_condition_for_mapping() -> None:
    for spec in CONDITIONS:
        cond = steer_condition_for(spec, scenario_index=7, seed=123)
        assert cond.name == spec.name
        assert cond.scenario_index == 7
        assert cond.seed == 123
        if spec.name == "constant_vec":
            assert cond.target_kind == "constant"
            assert cond.constant_target is not None
        else:
            assert cond.constant_target is None
    assert steer_condition_for(CONDITIONS[2]).axis == "deflection"  # steer
    assert steer_condition_for(CONDITIONS[5]).axis == "epa_target"  # identity_only
    assert steer_condition_for(CONDITIONS[6]).target_kind == "shuffled"  # B7
    assert steer_condition_for(CONDITIONS[0]).axis == "none"  # unsteered


def test_load_battery_rejects_empty(tmp_path) -> None:
    p = tmp_path / "battery.json"
    p.write_text(json.dumps({"scenarios": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_battery(p)


def _load_coeffs_and_dictionary():
    data = DataPaths()
    return ACTCoefficients.load(data.coefficients_file), ACTDictionary.load(data.data_dir)


def test_run_battery_condition_summary_fields() -> None:
    scenarios = _small_battery()
    coeffs, dictionary = _load_coeffs_and_dictionary()
    spec = [s for s in CONDITIONS if s.name == "situation_dependent"][0]
    summary = run_battery_condition(
        spec,
        scenarios,
        coeffs,
        dictionary,
        components=build_condition_components(spec),
        mode="mock",
        seed=123,
    )
    assert summary["n_scenarios"] == float(len(scenarios))
    up = summary["u_post"]
    assert up["ci_low"] <= up["mean"] <= up["ci_high"]
    assert up["n"] == float(len(scenarios))
    assert up["n_clusters"] == float(len(scenarios))


def test_contrast_vs_baseline_paired() -> None:
    scenarios = _small_battery()
    coeffs, dictionary = _load_coeffs_and_dictionary()
    results: dict[str, list[dict]] = {}
    from evaluation.ablations import _run_one_scenario

    for name in ("unsteered", "situation_dependent"):
        spec = [s for s in CONDITIONS if s.name == name][0]
        cfg = PipelineConfig(k=spec.k, mode="mock")
        components = build_condition_components(spec)
        results[name] = [
            _run_one_scenario(s, coeffs, dictionary, components, cfg) for s in scenarios
        ]
    c = contrast_vs_baseline(results["situation_dependent"], results["unsteered"])
    assert c is not None
    assert 0.0 <= c["p_value"] <= 1.0
    assert c["n"] == float(len(scenarios))
    # fewer than 2 overlapping scenarios -> None
    assert contrast_vs_baseline(results["unsteered"][:1], results["unsteered"]) is None


def test_battery_condition_summary_needs_two_scenarios() -> None:
    spec = CONDITIONS[0]
    with pytest.raises(ValueError):
        battery_condition_summary(spec, [{"scenario_index": 0, "u_post": 1.0}])


def test_cli_battery_smoke(tmp_path) -> None:
    battery_path = tmp_path / "battery.json"
    battery_path.write_text(
        json.dumps({"seed": 1, "n_scenarios": 6, "scenarios": _small_battery()}),
        encoding="utf-8",
    )
    out = tmp_path / "abl_battery.json"
    rc = main(
        [
            "--battery",
            str(battery_path),
            "--conditions",
            "unsteered",
            "situation_dependent",
            "shuffled",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["mode"] == "mock"
    assert set(data["conditions"].keys()) == {"unsteered", "situation_dependent", "shuffled"}
    for s in data["conditions"].values():
        assert s["u_post"]["ci_low"] <= s["u_post"]["mean"] <= s["u_post"]["ci_high"]
    assert set(data["contrasts_vs_unsteered"].keys()) == {"situation_dependent", "shuffled"}
    assert data["battery"]["n_scenarios"] == 6.0


def test_cli_battery_max_scenarios(tmp_path) -> None:
    battery_path = tmp_path / "battery.json"
    battery_path.write_text(
        json.dumps({"seed": 1, "n_scenarios": 6, "scenarios": _small_battery(6)}),
        encoding="utf-8",
    )
    out = tmp_path / "abl_battery.json"
    rc = main(
        [
            "--battery",
            str(battery_path),
            "--conditions",
            "unsteered",
            "--max-scenarios",
            "4",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["battery"]["n_scenarios"] == 4.0


def test_build_condition_components() -> None:
    plain = build_condition_components(CONDITIONS[0])  # unsteered
    assert plain.monitor is None
    gen = plain.candidate_generator(None)  # type: ignore[arg-type]
    assert list(gen) == ["unsteered:candidate-0"]

    mon = build_condition_components(CONDITIONS[1])  # monitor_only
    assert mon.monitor is not None
    trace = mon.monitor("hello world there")
    assert trace.n_tokens == 3

    best = build_condition_components(CONDITIONS[8])  # best_of_k
    assert len(list(best.candidate_generator(None))) == 4  # type: ignore[arg-type]


def test_run_condition_deterministic() -> None:
    spec = CONDITIONS[0]
    a = run_condition(spec, QUICK_SCRIPT, seed=1)
    b = run_condition(spec, QUICK_SCRIPT, seed=1)
    assert a["turns"] == b["turns"]
    assert len(a["turns"]) == 2


def test_summarize_trace_fields() -> None:
    spec = CONDITIONS[0]
    trace = run_condition(spec, QUICK_SCRIPT)
    s = summarize_trace(trace, spec)
    assert s["n_turns"] == 2.0
    assert s["k"] == 1.0
    assert s["u_post_mean"] >= 0.0
    assert s["n_monitor_fires"] == 0.0
    assert s["description"]


def test_monitor_only_condition_fires_monitor() -> None:
    spec = CONDITIONS[1]
    trace = run_condition(spec, QUICK_SCRIPT)
    s = summarize_trace(trace, spec)
    # every turn must carry a monitor trace
    for turn in trace["turns"]:
        assert turn["monitor"] is not None
    assert s["n_monitor_fires"] >= 0.0


def test_selector_picks_min_deflection() -> None:
    # with repair off, the chosen u_post equals the min candidate deflection
    for spec in (CONDITIONS[0], CONDITIONS[7]):
        trace = run_condition(spec, QUICK_SCRIPT)
        for turn in trace["turns"]:
            defs = [c["deflection"] for c in turn["candidates"] if c["deflection"] is not None]
            assert turn["u_post"] == pytest.approx(min(defs))


def test_cli_quick_smoke(tmp_path) -> None:
    out = tmp_path / "abl.json"
    rc = main(["--quick", "--out", str(out)])
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert set(data["order"]) == EXPECTED_LABELS
    assert set(data["conditions"].keys()) == EXPECTED_LABELS
    for s in data["conditions"].values():
        assert s["n_turns"] == 2.0
    assert data["script"]["n_messages"] == 2
    assert data["meta"]["model"] is None


def test_cli_conditions_subset(tmp_path) -> None:
    out = tmp_path / "abl.json"
    rc = main(["--quick", "--conditions", "unsteered", "best_of_k", "--out", str(out)])
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["order"] == ["unsteered", "best_of_k"]
    assert set(data["conditions"].keys()) == {"unsteered", "best_of_k"}


def test_cli_unknown_condition_rejected(tmp_path) -> None:
    with pytest.raises(SystemExit):
        main(["--quick", "--conditions", "bogus", "--out", str(tmp_path / "x.json")])


def test_cli_requires_script_or_quick(tmp_path) -> None:
    with pytest.raises(SystemExit):
        main(["--out", str(tmp_path / "x.json")])


def test_cli_custom_script(tmp_path) -> None:
    script = {
        "user": "consultant",
        "agent": "client",
        "messages": ["Please review this proposal."],
    }
    sp = tmp_path / "script.json"
    sp.write_text(json.dumps(script), encoding="utf-8")
    out = tmp_path / "abl.json"
    rc = main(["--script", str(sp), "--conditions", "unsteered", "--out", str(out)])
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["script"]["n_messages"] == 1
    assert data["conditions"]["unsteered"]["n_turns"] == 1.0


def test_run_condition_invalid_script() -> None:
    spec = ConditionSpec("x")
    with pytest.raises((KeyError, ValueError)):
        run_condition(spec, {"user": "student"})
