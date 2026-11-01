"""Tests for the LoRRA triplet builder (S13.1, CPU).

Run from ``examples/act``: ``python -m pytest tests/test_lorra.py -q``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from act_steering.lorra import (
    DEFAULT_CONTROL_TEMPLATE,
    DEFAULT_NEG_TYPE,
    DEFAULT_POS_TYPE,
    DEFAULT_REPO_ROOT,
    LorraArgs,
    build_train_command,
    build_triplet,
    build_triplets,
    load_contrast_set,
    write_triplets_jsonl,
)
from act_steering.stimuli import ContrastItem, ContrastPair

#: ``examples/act`` (parent of this tests directory).
ACT_DIR = Path(__file__).resolve().parents[1]


def _pair(delta_u: float = 2.0) -> ContrastPair:
    low = ContrastItem(
        context="You are a tutor. An upset client says: I feel ripped off.",
        user_term="upset client",
        agent_term="patient tutor",
        message_term="criticize",
        behaviour_term="apologize",
        behaviour_epa=(0.5, 0.1, 0.2),
        deflection=0.1,
    )
    high = ContrastItem(
        context=low.context,
        user_term=low.user_term,
        agent_term=low.agent_term,
        message_term=low.message_term,
        behaviour_term="attack",
        behaviour_epa=(-1.0, 1.5, 2.0),
        deflection=0.1 + delta_u,
    )
    return ContrastPair(low=low, high=high, delta_u=delta_u)


class TestBuildTriplet:
    def test_uses_shared_context_and_low_response(self):
        args = LorraArgs()
        t = build_triplet(_pair(), args)
        assert t["instruction"] == _pair().low.context
        assert t["response"] == "You respond by apologize."

    def test_templates_match_upstream_format(self):
        args = LorraArgs(
            user_tag="[INST]",
            assistant_tag="[/INST]",
            pos_type="a low-deflection",
            neg_type="a high-deflection",
            control_template="Give a {type} response.",
        )
        ctx = "You are a tutor. An upset client says: I feel ripped off."
        resp = "You respond by apologize."
        t = build_triplet(_pair(), args)
        assert t["orig"] == f"[INST] {ctx} [/INST] {resp}"
        assert t["pos"] == f"[INST] {ctx} Give a a low-deflection response. [/INST] {resp}"
        assert t["neg"] == f"[INST] {ctx} Give a a high-deflection response. [/INST] {resp}"

    def test_control_statements_come_from_control_template(self):
        args = LorraArgs(control_template="Act {type} here.")
        t = build_triplet(_pair(), args)
        assert "Act a low-deflection here." in t["pos"]
        assert "Act a high-deflection here." in t["neg"]

    def test_pos_neg_share_everything_but_the_control(self):
        t = build_triplet(_pair(), LorraArgs())
        pos_wo = t["pos"].replace("Give a a low-deflection response.", "CTRL")
        neg_wo = t["neg"].replace("Give a a high-deflection response.", "CTRL")
        assert pos_wo == neg_wo

    def test_provenance_fields(self):
        t = build_triplet(_pair(delta_u=3.5), LorraArgs())
        assert t["user_term"] == "upset client"
        assert t["agent_term"] == "patient tutor"
        assert t["low_behaviour"] == "apologize"
        assert t["high_behaviour"] == "attack"
        assert t["delta_u"] == pytest.approx(3.5)

    def test_default_args_match_documented_defaults(self):
        args = LorraArgs()
        assert args.pos_type == DEFAULT_POS_TYPE
        assert args.neg_type == DEFAULT_NEG_TYPE
        assert args.control_template == DEFAULT_CONTROL_TEMPLATE


class TestBuildTriplets:
    def test_one_triplet_per_pair_in_order(self):
        pairs = [_pair(delta_u=float(i)) for i in range(3)]
        out = build_triplets(pairs, LorraArgs())
        assert len(out) == 3
        assert [t["delta_u"] for t in out] == pytest.approx([0.0, 1.0, 2.0])

    def test_min_delta_u_filters(self):
        pairs = [_pair(delta_u=0.1), _pair(delta_u=1.0), _pair(delta_u=5.0)]
        out = build_triplets(pairs, LorraArgs(), min_delta_u=1.0)
        assert [t["delta_u"] for t in out] == pytest.approx([1.0, 5.0])

    def test_max_triplets_subsamples_seeded(self):
        pairs = [_pair(delta_u=float(i)) for i in range(10)]
        a = build_triplets(pairs, LorraArgs(), max_triplets=4, seed=7)
        b = build_triplets(pairs, LorraArgs(), max_triplets=4, seed=7)
        c = build_triplets(pairs, LorraArgs(), max_triplets=4, seed=8)
        assert len(a) == 4
        assert a == b
        assert a != c

    def test_max_triplets_larger_than_kept_is_identity(self):
        pairs = [_pair() for _ in range(3)]
        out = build_triplets(pairs, LorraArgs(), max_triplets=100)
        assert len(out) == 3

    def test_invalid_args_raise(self):
        pairs = [_pair()]
        with pytest.raises(ValueError, match="max_triplets"):
            build_triplets(pairs, LorraArgs(), max_triplets=0)
        with pytest.raises(ValueError, match="min_delta_u"):
            build_triplets(pairs, LorraArgs(), min_delta_u=-0.1)


class TestJsonlIO:
    def test_write_then_roundtrip(self, tmp_path: object):
        triplets = build_triplets([_pair(), _pair(delta_u=4.0)], LorraArgs())
        path = tmp_path / "lorra_triplets.jsonl"  # type: ignore[operator]
        write_triplets_jsonl(triplets, path)  # type: ignore[arg-type]
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        loaded = [json.loads(line) for line in lines]
        assert loaded == triplets

    def test_creates_parent_directories(self, tmp_path: object):
        path = tmp_path / "nested" / "deep" / "t.jsonl"  # type: ignore[operator]
        out = write_triplets_jsonl([build_triplet(_pair(), LorraArgs())], path)  # type: ignore[arg-type]
        assert out.exists()

    def test_load_contrast_set_roundtrips(self, tmp_path: object):
        pairs = [_pair(), _pair(delta_u=2.5)]
        data = {"pairs": [asdict(p) for p in pairs]}
        path = tmp_path / "contrast.json"  # type: ignore[operator]
        path.write_text(json.dumps(data), encoding="utf-8")
        loaded = load_contrast_set(path)  # type: ignore[arg-type]
        assert len(loaded) == 2
        assert loaded[1].delta_u == pytest.approx(2.5)
        # and the loader output feeds the builder
        out = build_triplets(loaded, LorraArgs())
        assert len(out) == 2


class TestTrainCommand:
    def test_command_structure(self, tmp_path: object):
        cmd = build_train_command(
            model="meta-llama/Llama-3.1-8B-Instruct",
            repo_root=DEFAULT_REPO_ROOT,
            lorra_args=LorraArgs(),
            output_dir=tmp_path / "run",  # type: ignore[operator]
        )
        assert Path(cmd[1]).as_posix().endswith("lorra_finetune/src/llama2_lorra.py")
        assert Path(cmd[1]).exists()
        flags = dict(zip(cmd[2::2], cmd[3::2]))
        assert flags["--model_name_or_path"] == "meta-llama/Llama-3.1-8B-Instruct"
        assert flags["--pos_type"] == "a low-deflection"
        assert flags["--neg_type"] == "a high-deflection"
        assert flags["--control_template"] == "Give a {type} response."
        assert flags["--target_layers"] == "20,22,24"
        assert flags["--lorra_alpha"] == "5.0"
        assert flags["--lorra_beta"] == "0.0"
        assert flags["--max_res_len"] == "64"
        assert flags["--num_train_epochs"] == "1"

    def test_custom_args_forwarded(self, tmp_path: object):
        args = LorraArgs(
            pos_type="X", neg_type="Y",
            lorra_alpha=7.0, lorra_beta=1.0, max_res_len=32,
        )
        cmd = build_train_command(
            model="m", repo_root=DEFAULT_REPO_ROOT, lorra_args=args,
            output_dir=tmp_path,  # type: ignore[arg-type]
            target_layers="5,7", epochs=3, batch_size=2, learning_rate=2e-5,
        )
        flags = dict(zip(cmd[2::2], cmd[3::2]))
        assert flags["--pos_type"] == "X"
        assert flags["--neg_type"] == "Y"
        assert flags["--lorra_alpha"] == "7.0"
        assert flags["--target_layers"] == "5,7"
        assert flags["--num_train_epochs"] == "3"
        assert flags["--per_device_train_batch_size"] == "2"
        assert flags["--learning_rate"] == "2e-05"


class TestCli:
    def _run(self, *extra: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "act_steering.lorra", *extra],
            capture_output=True, text=True, cwd=ACT_DIR, timeout=300,
        )

    def test_quick_writes_triplets(self, tmp_path: object):
        out = tmp_path / "t.jsonl"  # type: ignore[operator]
        proc = self._run("--quick", "--out", str(out))
        assert proc.returncode == 0, proc.stderr
        lines = out.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 20
        first = json.loads(lines[0])
        assert {"orig", "pos", "neg", "instruction", "response"} <= set(first)

    def test_train_dry_run_prints_command(self, tmp_path: object):
        out = tmp_path / "t.jsonl"  # type: ignore[operator]
        proc = self._run("--quick", "--out", str(out), "--train")
        assert proc.returncode == 0, proc.stderr
        assert "llama2_lorra.py" in proc.stdout
        assert "--pos_type" in proc.stdout
        assert "low-deflection" in proc.stdout
        assert "tatsu-lab/alpaca" in proc.stdout
        # dry-run must not create the HF output directory
        assert not (out.parent / "lorra_run").exists()

    def test_execute_requires_train(self, tmp_path: object):
        proc = self._run("--quick", "--out", str(tmp_path / "t.jsonl"), "--execute")
        assert proc.returncode == 2
        assert "--train" in proc.stderr
