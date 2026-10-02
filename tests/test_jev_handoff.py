"""Keep offline stages identical to native typed requests without duplicate work."""

import json
import random
import shutil

import httpx2
import pytest
from test_three_role import PROMPT
from typesafe_sdk import RetryPolicy, TypeSafeClient

from gepa.response_journal import ResponseJournalError, response_journal_scope
from gepa.strategies.document_template import TEMPLATES
from gepa.strategies.edit_tools import EDIT_TOOL_SETS
from gepa.strategies.intervention import build_controller_menu
from gepa.strategies.jev_controller import JEV_MODEL, JevController
from gepa.strategies.jev_handoff import HANDOFF_ENV, load, resolve


@pytest.fixture
def setup(tmp_path, monkeypatch):
    """Prepare isolated handoff clients and an inspectable mocked provider.

    Args:
        tmp_path: Directory for mailbox artifacts, ledgers and response journals.
        monkeypatch: Fixture for isolating allocation, credential and timeout state.

    Returns:
        Controller factory, captured request list and mutable failure switches.
    """
    monkeypatch.setattr("gepa.strategies.jev_handoff.HANDOFF_WAIT_SECONDS", 0.0)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setenv(HANDOFF_ENV, str(tmp_path / "remote"))
    calls = []
    failures = []

    def handler(request):
        """Record an SDK request and simulate a typed response or connection failure.

        Args:
            request: Outgoing HTTP request from the mocked external Controller.

        Returns:
            Valid response with all probability mass on the first choice.

        Raises:
            httpx2.ConnectError: A test has enabled the shared failure switch.
        """
        data = json.loads(request.content)
        calls.append(data)
        if failures:
            raise httpx2.ConnectError("test-only connection failure")
        choices = data["questions"]["edit"]["criteria"]
        selected = next(iter(choices))
        return httpx2.Response(
            200,
            json={
                "model": JEV_MODEL,
                "usage": {"input_tokens": 100, "output_tokens": 0},
                "answers": {
                    "edit": {
                        "type": "choice",
                        "choice": selected,
                        "confidence": 1.0,
                        "probabilities": {key: float(key == selected) for key in choices},
                    }
                },
            },
        )

    def controller(live=False):
        """Create a journaled allocation client or a mocked direct-API resolver.

        Args:
            live: Whether to attach mocked HTTP transport instead of local journals.

        Returns:
            Controller configured for the requested side of the file handoff.
        """
        client = JevController(
            api_key="test-private-key" if live else None,
            response_journal_path=None if live else tmp_path / "responses.sqlite3",
            attempt_log_path=None if live else tmp_path / "attempts.jsonl",
        )
        if live:
            client._client = TypeSafeClient(
                api_key="test-private-key",
                retry=RetryPolicy(max_retries=0),
                http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
            )
        return client

    return controller, calls, failures


def choose(controller):
    """Select the standard test menu under a stable logical request scope.

    Args:
        controller: Allocation or replay Controller to exercise.

    Returns:
        Audit metadata for the selected action/section choice.
    """
    template = TEMPLATES["system_prompt"]
    menu = build_controller_menu(template, "sys", EDIT_TOOL_SETS["broad"], 2, rng=random.Random(0))
    with response_journal_scope("pilot/opportunity-3"):
        _, record = controller.select(
            menu,
            sections=template.parse(PROMPT),
            section_descriptions=template.sections,
            traces="unchanged",
            rng=random.Random(0),
        )
        return record


def export_request(tmp_path, controller, calls):
    """Capture a timed-out allocation request and copy it to the resolver directory.

    Args:
        tmp_path: Shared test root containing the allocation mailbox.
        controller: Journaled Controller configured for an immediate handoff timeout.
        calls: Captured provider calls, which must remain empty during export.

    Returns:
        Original allocation request path and its independent external copy.
    """
    with pytest.raises(SystemExit) as exc:
        choose(controller)
    assert exc.value.code == 75
    assert not calls
    request = next((tmp_path / "remote").glob("*/request.json"))
    target = tmp_path / "external" / request.parent.name
    target.mkdir(parents=True)
    shutil.copyfile(request, target / request.name)
    return request, target / request.name


def test_pause_external_request_resume_and_journal_replay(setup, tmp_path, monkeypatch):
    """Resolve and replay one external request without duplicate calls or charges."""
    factory, calls, _ = setup
    remote, external = export_request(tmp_path, factory(), calls)
    assert not (tmp_path / "attempts.jsonl").exists()
    monkeypatch.delenv(HANDOFF_ENV)
    response = resolve(external, factory(live=True))
    assert resolve(external, factory(live=True)) == response
    assert len(calls) == 1
    shutil.copyfile(response, remote.with_name("response.json"))
    monkeypatch.setenv(HANDOFF_ENV, str(tmp_path / "remote"))
    resumed = factory()
    result = choose(resumed)
    assert sum(result["probs"].values()) == max(result["probs"].values()) == 1.0
    assert resumed.total_tokens_in == 100
    replay = factory()
    assert choose(replay)["replayed"]
    assert replay.total_tokens_in == 100
    assert len(calls) == 1
    assert len((tmp_path / "attempts.jsonl").read_text().splitlines()) == 2
    assert "test-private-key" not in response.read_text()


def test_response_arrives_without_exiting_or_reloading_controller(setup, tmp_path, monkeypatch):
    """Import a response arriving during polling while retaining the resident Controller."""
    factory, calls, _ = setup
    remote, external = export_request(tmp_path, factory(), calls)
    monkeypatch.delenv(HANDOFF_ENV)
    response = resolve(external, factory(live=True))
    monkeypatch.setenv(HANDOFF_ENV, str(tmp_path / "remote"))
    monkeypatch.setattr("gepa.strategies.jev_handoff.HANDOFF_WAIT_SECONDS", 300.0)
    waits = []

    def deliver(seconds):
        """Deliver the saved response during a mocked polling interval.

        Args:
            seconds: Requested sleep duration to record before delivering the file.
        """
        waits.append(seconds)
        shutil.copyfile(response, remote.with_name("response.json"))

    monkeypatch.setattr("gepa.strategies.jev_handoff.time.sleep", deliver)
    resident = factory()
    assert choose(resident)["probs"]
    assert resident.total_tokens_in == 100
    assert waits == [1.0]
    assert len(calls) == 1


def test_external_failure_is_retained_and_not_retried_by_resume(setup, tmp_path, monkeypatch):
    """Preserve an exhausted external request and import its attempts only once."""
    factory, calls, failures = setup
    remote, external = export_request(tmp_path, factory(), calls)
    failures.append(True)
    monkeypatch.setattr("gepa.strategies.jev_controller.time.sleep", lambda _: None)
    monkeypatch.delenv(HANDOFF_ENV)
    response = resolve(external, factory(live=True))
    assert len(calls) == 4
    assert load(response)["error_type"] == "JevControllerError"
    resolve(external, factory(live=True))
    assert len(calls) == 4
    shutil.copyfile(response, remote.with_name("response.json"))
    monkeypatch.setenv(HANDOFF_ENV, str(tmp_path / "remote"))
    for _ in range(2):
        with pytest.raises(ResponseJournalError, match="External Jev request failed"):
            choose(factory())
    assert len((tmp_path / "attempts.jsonl").read_text().splitlines()) == 8
    assert len(calls) == 4


def test_interrupted_external_attempt_requires_review(setup, tmp_path, monkeypatch):
    """Refuse another external call when a started marker has no completed response."""
    factory, calls, _ = setup
    _, external = export_request(tmp_path, factory(), calls)
    external.with_name("started.json").write_text("{}")
    monkeypatch.delenv(HANDOFF_ENV)
    with pytest.raises(FileExistsError):
        resolve(external, factory(live=True))
    assert not calls


def test_import_after_journal_store_interruption_charges_once(setup, tmp_path, monkeypatch):
    """Avoid duplicate calls and charges after interruption between import and journaling."""
    factory, calls, _ = setup
    remote, external = export_request(tmp_path, factory(), calls)
    monkeypatch.delenv(HANDOFF_ENV)
    response = resolve(external, factory(live=True))
    shutil.copyfile(response, remote.with_name("response.json"))
    monkeypatch.setenv(HANDOFF_ENV, str(tmp_path / "remote"))
    first = factory()
    monkeypatch.setattr(first._journal, "store", lambda *args: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        choose(first)
    resumed = factory()
    choose(resumed)
    assert resumed.total_tokens_in == 100
    assert len((tmp_path / "attempts.jsonl").read_text().splitlines()) == 2
    assert len(calls) == 1


@pytest.mark.parametrize("kind", ["request", "response"])
def test_tampering_is_rejected_before_reuse(setup, tmp_path, monkeypatch, kind):
    """Reject modified sealed requests or responses before they can be reused."""
    factory, calls, _ = setup
    remote, external = export_request(tmp_path, factory(), calls)
    monkeypatch.delenv(HANDOFF_ENV)
    if kind == "request":
        external.write_text(external.read_text().replace("unchanged", "changed"))
        with pytest.raises(ResponseJournalError, match="checksum"):
            resolve(external, factory(live=True))
        assert not calls
    else:
        response = resolve(external, factory(live=True))
        remote.with_name("response.json").write_text(
            response.read_text().replace('"confidence": 1.0', '"confidence": 0.0')
        )
        monkeypatch.setenv(HANDOFF_ENV, str(tmp_path / "remote"))
        with pytest.raises(ResponseJournalError, match="checksum"):
            choose(factory())
        assert not (tmp_path / "attempts.jsonl").exists()


def test_external_resolver_refuses_compute_allocation(setup, tmp_path, monkeypatch):
    """Prevent direct provider resolution from a compute allocation."""
    factory, calls, _ = setup
    _, external = export_request(tmp_path, factory(), calls)
    monkeypatch.delenv(HANDOFF_ENV)
    monkeypatch.setenv("SLURM_JOB_ID", "allocation")
    with pytest.raises(ValueError, match="outside"):
        resolve(external, factory(live=True))
    assert not calls
