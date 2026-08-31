# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for exact custom-LM replay across optimizer interruption."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gepa.lm import LM, LMProviderError, NativeToolCall, ProviderIdentityMismatchError, ToolCompletion
from gepa.response_journal import ResponseJournalError, response_journal_scope


def completion_response(
    content: str,
    *,
    model: str = "deepseek-runtime",
    fingerprint: str = "fp_launch",
) -> MagicMock:
    """Build one LiteLLM-shaped plain response.

    Args:
        content: Assistant content returned by the fake provider.
        model: Provider-returned model identity.
        fingerprint: Provider-returned system fingerprint.

    Returns:
        Mock completion carrying content, identity, and no usage charge.
    """
    response = MagicMock()
    response.model = model
    response.system_fingerprint = fingerprint
    response.usage = None
    response.choices = [MagicMock()]
    response.choices[0].finish_reason = "stop"
    response.choices[0].message.content = content
    response.choices[0].message.reasoning_content = ""
    response.choices[0].message.tool_calls = []
    return response


def journal_lm(path: Path, namespace: str = "reflection-proposer", **kwargs: object) -> LM:
    """Build an identity-pinned LM attached to one response journal.

    Args:
        path: SQLite response-journal path.
        namespace: Stable LM role stored in journal keys.
        **kwargs: Additional LM constructor settings.

    Returns:
        Configured test LM.
    """
    return LM(
        "deepseek/deepseek-v4-flash",
        expected_response_model="deepseek-runtime",
        expected_system_fingerprint="fp_launch",
        response_journal_path=str(path),
        response_journal_namespace=namespace,
        **kwargs,
    )


def test_replays_by_logical_occurrence_without_collapsing_duplicate_prompts(tmp_path: Path) -> None:
    """Keep repeated live calls distinct while replaying both after restart."""
    journal_path = tmp_path / "private" / "responses.sqlite3"
    first_lm = journal_lm(journal_path)
    with patch(
        "litellm.completion",
        side_effect=[completion_response("first"), completion_response("second")],
    ) as provider:
        with response_journal_scope("optimizer-iteration-7"):
            first = first_lm("same prompt")
            second = first_lm("same prompt")

    assert (first, second) == ("first", "second")
    assert provider.call_count == 2

    resumed_lm = journal_lm(journal_path)
    with patch("litellm.completion") as resumed_provider:
        with response_journal_scope("optimizer-iteration-7"):
            replayed = (resumed_lm("same prompt"), resumed_lm("same prompt"))

    assert replayed == ("first", "second")
    resumed_provider.assert_not_called()


def test_request_mismatch_fails_closed_without_calling_the_provider(tmp_path: Path) -> None:
    """Reject changed inputs at an already assigned logical call slot."""
    journal_path = tmp_path / "private" / "responses.sqlite3"
    with patch("litellm.completion", return_value=completion_response("recorded")):
        with response_journal_scope("optimizer-iteration-2"):
            journal_lm(journal_path)("original")

    with patch("litellm.completion") as provider:
        with pytest.raises(ResponseJournalError, match="request mismatch"):
            with response_journal_scope("optimizer-iteration-2"):
                journal_lm(journal_path)("changed")
    provider.assert_not_called()


def test_replay_ignores_rotated_authentication_and_ephemeral_loopback_port(tmp_path: Path) -> None:
    """Resume the same local model after transport credentials and port change."""
    journal_path = tmp_path / "private" / "responses.sqlite3"
    first_lm = journal_lm(
        journal_path,
        api_key="first-secret",
        api_base="http://127.0.0.1:31001/v1",
        extra_headers={"Authorization": "Bearer first", "X-Experiment": "fixed"},
    )
    with patch("litellm.completion", return_value=completion_response("recorded")):
        with response_journal_scope("optimizer-iteration-9"):
            assert first_lm("question") == "recorded"

    resumed_lm = journal_lm(
        journal_path,
        api_key="second-secret",
        api_base="http://127.0.0.1:41999/v1",
        extra_headers={"Authorization": "Bearer second", "X-Experiment": "fixed"},
    )
    with patch("litellm.completion") as provider:
        with response_journal_scope("optimizer-iteration-9"):
            assert resumed_lm("question") == "recorded"
    provider.assert_not_called()


@pytest.mark.parametrize(
    ("first_kwargs", "changed_kwargs"),
    [
        ({"temperature": 0.7}, {"temperature": 0.8}),
        ({"api_base": "https://provider-a.test/v1"}, {"api_base": "https://provider-b.test/v1"}),
        (
            {"extra_headers": {"X-Experiment": "first"}},
            {"extra_headers": {"X-Experiment": "second"}},
        ),
    ],
)
def test_replay_rejects_changed_scientific_request_semantics(
    tmp_path: Path,
    first_kwargs: dict[str, object],
    changed_kwargs: dict[str, object],
) -> None:
    """Fail closed when decoding, provider, or non-auth request metadata changes."""
    journal_path = tmp_path / "private" / "responses.sqlite3"
    with patch("litellm.completion", return_value=completion_response("recorded")):
        with response_journal_scope("optimizer-iteration-10"):
            journal_lm(journal_path, **first_kwargs)("question")

    with patch("litellm.completion") as provider:
        with pytest.raises(ResponseJournalError, match="request mismatch"):
            with response_journal_scope("optimizer-iteration-10"):
                journal_lm(journal_path, **changed_kwargs)("question")
    provider.assert_not_called()


def test_provider_failure_does_not_consume_the_logical_slot(tmp_path: Path) -> None:
    """Retry the same occurrence after a provider failure and then replay it."""
    journal_path = tmp_path / "private" / "responses.sqlite3"
    lm = journal_lm(journal_path)
    with patch(
        "litellm.completion",
        side_effect=[RuntimeError("temporary provider failure"), completion_response("recovered")],
    ) as provider:
        with response_journal_scope("optimizer-iteration-3"):
            with pytest.raises(LMProviderError, match="Completion provider failed"):
                lm("question")
            assert lm("question") == "recovered"
    assert provider.call_count == 2

    with patch("litellm.completion") as resumed_provider:
        with response_journal_scope("optimizer-iteration-3"):
            assert journal_lm(journal_path)("question") == "recovered"
    resumed_provider.assert_not_called()


def test_native_tools_replay_reasoning_ids_and_arguments_exactly(tmp_path: Path) -> None:
    """Preserve all assistant state needed for a resumed native-tool turn."""
    journal_path = tmp_path / "private" / "responses.sqlite3"
    response = completion_response("")
    response.choices[0].finish_reason = "tool_calls"
    response.choices[0].message.reasoning_content = "private chain state"
    call = MagicMock()
    call.id = "call-exact-7"
    call.function.name = "REPLACE_TEXT"
    call.function.arguments = '{"target":"old","text":"new"}'
    response.choices[0].message.tool_calls = [call]
    messages = [{"role": "user", "content": "edit"}]
    tools = [{"type": "function", "function": {"name": "REPLACE_TEXT", "parameters": {}}}]

    with patch("litellm.completion", return_value=response):
        with response_journal_scope("optimizer-iteration-4"):
            recorded = journal_lm(journal_path, "controller-proposer").complete_with_tools(messages, tools)

    with patch("litellm.completion") as provider:
        with response_journal_scope("optimizer-iteration-4"):
            replayed = journal_lm(journal_path, "controller-proposer").complete_with_tools(messages, tools)

    assert recorded == replayed == ToolCompletion(
        content="",
        tool_calls=(
            NativeToolCall(
                id="call-exact-7",
                name="REPLACE_TEXT",
                arguments='{"target":"old","text":"new"}',
            ),
        ),
        reasoning_content="private chain state",
    )
    provider.assert_not_called()


def test_batch_completion_replays_order_and_provider_identity(tmp_path: Path) -> None:
    """Replay an ordered reflection batch without recontacting the provider."""
    journal_path = tmp_path / "private" / "responses.sqlite3"
    messages = [
        [{"role": "user", "content": "one"}],
        [{"role": "user", "content": "two"}],
    ]
    with patch(
        "litellm.batch_completion",
        return_value=[completion_response(" first "), completion_response(" second ")],
    ):
        with response_journal_scope("optimizer-iteration-5"):
            recorded = journal_lm(journal_path).batch_complete(messages, max_workers=2)

    resumed = journal_lm(journal_path)
    with patch("litellm.batch_completion") as provider:
        with response_journal_scope("optimizer-iteration-5"):
            replayed = resumed.batch_complete(messages, max_workers=2)

    assert recorded == replayed == ["first", "second"]
    assert resumed.last_response_identity.model == "deepseek-runtime"
    provider.assert_not_called()


def test_partial_batch_failure_retries_only_missing_slot_and_counts_usage_once(tmp_path: Path) -> None:
    """Commit successful batch items before retrying one failed provider slot.

    Args:
        tmp_path: Temporary response-journal directory supplied by pytest.
    """
    journal_path = tmp_path / "private" / "responses.sqlite3"
    messages = [
        [{"role": "user", "content": "one"}],
        [{"role": "user", "content": "two"}],
        [{"role": "user", "content": "three"}],
    ]
    first = completion_response(" first ")
    first.usage = MagicMock(prompt_tokens=10, completion_tokens=1)
    second = completion_response(" second ")
    second.usage = MagicMock(prompt_tokens=20, completion_tokens=2)
    third = completion_response(" third ")
    third.usage = MagicMock(prompt_tokens=30, completion_tokens=3)
    lm = journal_lm(journal_path)

    with (
        patch(
            "litellm.batch_completion",
            side_effect=[[first, RuntimeError("middle failed"), third], [second]],
        ) as provider,
        patch("litellm.completion_cost", return_value=0.25),
        response_journal_scope("optimizer-iteration-partial"),
    ):
        with pytest.raises(LMProviderError, match="Batch completion provider failed"):
            lm.batch_complete(messages, max_workers=3)
        outputs = lm.batch_complete(messages, max_workers=3)

    assert outputs == ["first", "second", "third"]
    assert provider.call_count == 2
    assert provider.call_args_list[1].kwargs["messages"] == [messages[1]]
    assert (lm.total_cost, lm.total_tokens_in, lm.total_tokens_out) == (0.75, 60, 6)

    resumed = journal_lm(journal_path)
    with patch("litellm.batch_completion") as resumed_provider:
        with response_journal_scope("optimizer-iteration-partial"):
            assert resumed.batch_complete(messages, max_workers=3) == outputs

    resumed_provider.assert_not_called()
    assert (resumed.total_cost, resumed.total_tokens_in, resumed.total_tokens_out) == (0.75, 60, 6)


def test_batch_slots_replay_individual_fallback_calls_after_restart(tmp_path: Path) -> None:
    """Address batch items like their per-task fallback completions."""
    journal_path = tmp_path / "private" / "responses.sqlite3"
    messages = [
        [{"role": "user", "content": "one"}],
        [{"role": "user", "content": "two"}],
    ]
    fallback_lm = journal_lm(journal_path)
    cursor = fallback_lm.response_journal_cursor_state()
    with patch("litellm.batch_completion", side_effect=RuntimeError("batch failed")):
        with pytest.raises(LMProviderError, match="Batch completion provider failed"):
            with response_journal_scope("optimizer-iteration-12"):
                fallback_lm.batch_complete(messages)
    fallback_lm.restore_response_journal_cursor_state(cursor)

    with patch(
        "litellm.completion",
        side_effect=[completion_response(" first "), completion_response(" second ")],
    ):
        with response_journal_scope("optimizer-iteration-12"):
            fallback_outputs = [fallback_lm(message) for message in messages]

    resumed_lm = journal_lm(journal_path)
    with patch("litellm.batch_completion") as provider:
        with response_journal_scope("optimizer-iteration-12"):
            replayed = resumed_lm.batch_complete(messages)

    assert replayed == fallback_outputs == [" first ", " second "]
    provider.assert_not_called()


def test_usage_totals_survive_replay_cursor_rewind_and_process_restart(tmp_path: Path) -> None:
    """Count each paid response once across every supported journaled interface.

    Args:
        tmp_path: Temporary response-journal directory supplied by pytest.
    """
    journal_path = tmp_path / "private" / "responses.sqlite3"
    plain = completion_response("plain")
    plain.usage = MagicMock(prompt_tokens=11, completion_tokens=7)
    native = completion_response("native")
    native.usage = MagicMock(prompt_tokens=13, completion_tokens=5)
    first_batch = completion_response(" first ")
    first_batch.usage = MagicMock(prompt_tokens=17, completion_tokens=3)
    second_batch = completion_response(" second ")
    second_batch.usage = MagicMock(prompt_tokens=19, completion_tokens=2)
    messages = [[{"role": "user", "content": "one"}], [{"role": "user", "content": "two"}]]
    tools = [{"type": "function", "function": {"name": "noop", "parameters": {}}}]
    lm = journal_lm(journal_path)

    with (
        patch("litellm.completion", side_effect=[plain, native]) as completion_provider,
        patch("litellm.batch_completion", return_value=[first_batch, second_batch]) as batch_provider,
        patch("litellm.completion_cost", return_value=0.25),
    ):
        with response_journal_scope("plain-scope"):
            assert lm("plain request") == "plain"
        with response_journal_scope("tool-scope"):
            assert lm.complete_with_tools([{"role": "user", "content": "native request"}], tools) == ToolCompletion(
                content="native",
                tool_calls=(),
                reasoning_content="",
            )
        with response_journal_scope("batch-scope"):
            assert lm.batch_complete(messages, max_workers=2) == ["first", "second"]

    assert completion_provider.call_count == 2
    batch_provider.assert_called_once()
    assert lm.total_cost == 1.0
    assert lm.total_tokens_in == 60
    assert lm.total_tokens_out == 17

    lm.restore_response_journal_cursor_state({})
    with (
        patch("litellm.completion") as rewound_completion_provider,
        patch("litellm.batch_completion") as rewound_batch_provider,
    ):
        with response_journal_scope("plain-scope"):
            assert lm("plain request") == "plain"
        with response_journal_scope("tool-scope"):
            assert lm.complete_with_tools([{"role": "user", "content": "native request"}], tools).content == "native"
        with response_journal_scope("batch-scope"):
            assert lm.batch_complete(messages, max_workers=2) == ["first", "second"]

    rewound_completion_provider.assert_not_called()
    rewound_batch_provider.assert_not_called()
    assert (lm.total_cost, lm.total_tokens_in, lm.total_tokens_out) == (1.0, 60, 17)

    resumed = journal_lm(journal_path)
    assert (resumed.total_cost, resumed.total_tokens_in, resumed.total_tokens_out) == (1.0, 60, 17)
    with (
        patch("litellm.completion") as resumed_completion_provider,
        patch("litellm.batch_completion") as resumed_batch_provider,
    ):
        with response_journal_scope("plain-scope"):
            assert resumed("plain request") == "plain"
        with response_journal_scope("tool-scope"):
            assert resumed.complete_with_tools(
                [{"role": "user", "content": "native request"}], tools
            ).content == "native"
        with response_journal_scope("batch-scope"):
            assert resumed.batch_complete(messages, max_workers=2) == ["first", "second"]

    resumed_completion_provider.assert_not_called()
    resumed_batch_provider.assert_not_called()
    assert (resumed.total_cost, resumed.total_tokens_in, resumed.total_tokens_out) == (1.0, 60, 17)


def test_cached_provider_identity_drift_and_corruption_fail_closed(tmp_path: Path) -> None:
    """Reject a changed launch identity and a damaged persisted response."""
    journal_path = tmp_path / "private" / "responses.sqlite3"
    with patch("litellm.completion", return_value=completion_response("answer")):
        with response_journal_scope("optimizer-iteration-6"):
            journal_lm(journal_path)("question")

    changed_identity_lm = LM(
        "deepseek/deepseek-v4-flash",
        expected_response_model="different-runtime",
        expected_system_fingerprint="fp_launch",
        response_journal_path=str(journal_path),
        response_journal_namespace="reflection-proposer",
    )
    with patch("litellm.completion") as provider:
        with pytest.raises(ProviderIdentityMismatchError, match="Cached provider response identity"):
            with response_journal_scope("optimizer-iteration-6"):
                changed_identity_lm("question")
    provider.assert_not_called()

    with sqlite3.connect(journal_path) as connection:
        connection.execute("UPDATE responses SET response_json = '{}' WHERE ordinal = 0")
    with pytest.raises(ResponseJournalError, match="checksum mismatch"):
        with response_journal_scope("optimizer-iteration-6"):
            journal_lm(journal_path)("question")


def test_journal_is_private_and_does_not_persist_request_secrets(tmp_path: Path) -> None:
    """Store only request hashes while keeping output-bearing files private."""
    journal_path = tmp_path / "private" / "responses.sqlite3"
    secret_key = "sk-test-never-persist-this"
    secret_url = "https://user:password@example.test/v1"
    lm = journal_lm(journal_path, api_key=secret_key, api_base=secret_url)
    with patch("litellm.completion", return_value=completion_response("safe output")):
        with response_journal_scope("optimizer-iteration-8"):
            assert lm("confidential prompt text") == "safe output"

    journal_bytes = journal_path.read_bytes()
    assert secret_key.encode() not in journal_bytes
    assert secret_url.encode() not in journal_bytes
    assert b"confidential prompt text" not in journal_bytes
    assert os.stat(journal_path.parent).st_mode & 0o777 == 0o700
    assert os.stat(journal_path).st_mode & 0o777 == 0o600
