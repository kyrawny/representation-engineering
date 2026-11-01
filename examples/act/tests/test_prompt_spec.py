"""Tests for chat-template prompt construction (T3-S1).

Covers the :class:`PromptSpec` system-role variants, the pure message
builder, the chat-template rendering, and the backward-compatible raw
prompt path.
"""

from __future__ import annotations

import pytest

from act_steering.act_core import ACTCoefficients
from act_steering.dictionary import ACTDictionary
from act_steering.pipeline import (
    ANTI_SYCOPHANCY_LINE,
    PromptSpec,
    build_messages,
    build_prompt,
    descriptors_for,
    render_prompt,
    role_system_prompt,
    system_prompt_for,
)
from act_steering.state import ConversationState
from act_steering.stimuli import CONTEXT_TEMPLATE

DATA_DIR = r"C:\Users\Kyra\Documents\Repos\qwen\representation-engineering\data\act"


@pytest.fixture(scope="module")
def dictionary() -> ACTDictionary:
    return ACTDictionary.load(DATA_DIR)


@pytest.fixture(scope="module")
def coeffs() -> ACTCoefficients:
    from tests.test_pipeline import _synthetic_coeffs

    return _synthetic_coeffs()


class _FakeTokenizer:
    """A minimal stand-in exposing only ``apply_chat_template``."""

    def apply_chat_template(
        self, messages, tokenize: bool = False, add_generation_prompt: bool = False
    ) -> str:
        assert tokenize is False
        parts = [f"<{m['role']}>{m['content']}</{m['role']}> " for m in messages]
        if add_generation_prompt:
            parts.append("<assistant> ")
        return "".join(parts)


# --- system prompt variants ---------------------------------------------------


def test_role_system_prompt_contains_roles() -> None:
    text = role_system_prompt("student", "teacher")
    assert "teacher" in text
    assert "student" in text
    assert text == role_system_prompt("student", "teacher")  # deterministic


def test_descriptors_for_buckets() -> None:
    assert (
        descriptors_for((3.0, 2.0, 1.5))
        == "warm and kind, assertive and confident, lively and energetic"
    )
    assert (
        descriptors_for((-3.0, -2.0, -1.5))
        == "cool and blunt, soft and deferential, slow and quiet"
    )
    assert descriptors_for((0.0, 0.0, 0.0)) == "even-tempered, calm and measured, steady-paced"
    # boundary: exactly +/-1.0 is the neutral bucket
    assert descriptors_for((1.0, 0.0, 0.0)) == "even-tempered, calm and measured, steady-paced"


def test_system_prompt_for_variants() -> None:
    base = role_system_prompt("student", "teacher")
    assert system_prompt_for(PromptSpec(), "student", "teacher") == base

    desc = system_prompt_for(
        PromptSpec(variant="act_descriptors", target=(3.0, 0.0, 0.0)),
        "student",
        "teacher",
    )
    assert desc.startswith(base)
    assert "warm and kind" in desc

    # None target falls back to the neutral EPA descriptors
    neutral = system_prompt_for(PromptSpec(variant="act_descriptors"), "student", "teacher")
    assert "even-tempered" in neutral

    anti = system_prompt_for(PromptSpec(variant="anti_sycophancy"), "student", "teacher")
    assert anti.startswith(base)
    assert ANTI_SYCOPHANCY_LINE in anti

    # explicit override wins
    override = system_prompt_for(
        PromptSpec(system="Custom system.", variant="anti_sycophancy"),
        "student",
        "teacher",
    )
    assert override == "Custom system."

    # raw style has no system prompt
    assert system_prompt_for(PromptSpec(style="raw"), "student", "teacher") is None


# --- message building ---------------------------------------------------------


def test_build_messages_first_turn(dictionary: ACTDictionary, coeffs: ACTCoefficients) -> None:
    state = ConversationState.from_ids(dictionary, "student", "teacher")
    state.record_user_message(coeffs, (1.0, 0.0, 0.0), label="hello")
    messages = build_messages(state)
    assert messages[0]["role"] == "system"
    assert messages[1:] == [{"role": "user", "content": "hello"}]


def test_build_messages_multi_turn(dictionary: ACTDictionary, coeffs: ACTCoefficients) -> None:
    state = ConversationState.from_ids(dictionary, "student", "teacher")
    state.record_user_message(coeffs, (1.0, 0.0, 0.0), label="hello")
    state.record_agent_response(coeffs, (0.5, 0.0, 0.0), label="hi there")
    state.record_user_message(coeffs, (0.5, 0.0, 0.0), label="again")
    messages = build_messages(state)
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user", "assistant", "user"]
    assert messages[1]["content"] == "hello"
    assert messages[2]["content"] == "hi there"
    assert messages[3]["content"] == "again"


def test_build_messages_raw_style_has_no_system(
    dictionary: ACTDictionary, coeffs: ACTCoefficients
) -> None:
    state = ConversationState.from_ids(dictionary, "student", "teacher")
    state.record_user_message(coeffs, (1.0, 0.0, 0.0), label="hello")
    messages = build_messages(state, PromptSpec(style="raw"))
    assert all(m["role"] != "system" for m in messages)


def test_build_messages_raises_without_pending(dictionary: ACTDictionary) -> None:
    state = ConversationState.from_ids(dictionary, "student", "teacher")
    with pytest.raises(RuntimeError, match="pending"):
        build_messages(state)


# --- rendering + build_prompt --------------------------------------------------


def test_render_prompt_uses_chat_template() -> None:
    messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "hello"},
    ]
    out = render_prompt(messages, _FakeTokenizer())
    assert out == "<system>S</system> <user>hello</user> <assistant> "


def test_build_prompt_chat_style(dictionary: ACTDictionary, coeffs: ACTCoefficients) -> None:
    state = ConversationState.from_ids(dictionary, "student", "teacher")
    state.record_user_message(coeffs, (1.0, 0.0, 0.0), label="hello")
    out = build_prompt(state, _FakeTokenizer(), PromptSpec())
    assert "<system>" in out
    assert "teacher" in out
    assert "<user>hello</user>" in out
    assert out.endswith("<assistant> ")


def test_build_prompt_default_is_raw_unchanged(
    dictionary: ACTDictionary, coeffs: ACTCoefficients
) -> None:
    state = ConversationState.from_ids(dictionary, "student", "teacher")
    state.record_user_message(coeffs, (1.0, 0.0, 0.0), label="hello")
    assert build_prompt(state) == CONTEXT_TEMPLATE.format(
        user="student", agent="teacher", message="hello"
    )
    # a raw-style spec is also the old flat template (system ignored)
    assert build_prompt(
        state, _FakeTokenizer(), PromptSpec(style="raw", variant="anti_sycophancy")
    ).startswith("You are teacher.")


def test_build_prompt_chat_requires_tokenizer(
    dictionary: ACTDictionary, coeffs: ACTCoefficients
) -> None:
    state = ConversationState.from_ids(dictionary, "student", "teacher")
    state.record_user_message(coeffs, (1.0, 0.0, 0.0), label="hello")
    with pytest.raises(ValueError, match="tokenizer"):
        build_prompt(state, None, PromptSpec())


def test_build_prompt_raises_without_pending(dictionary: ACTDictionary) -> None:
    state = ConversationState.from_ids(dictionary, "student", "teacher")
    with pytest.raises(RuntimeError, match="pending"):
        build_prompt(state, _FakeTokenizer(), PromptSpec())
