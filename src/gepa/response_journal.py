# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Durably replay language-model responses after optimizer interruption."""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import math
import os
import re
import sqlite3
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

RESPONSE_JOURNAL_SCHEMA_VERSION = 2
RESPONSE_JOURNAL_SCOPE_POLICY = "optimizer-state-iteration"
ACTIVE_RESPONSE_JOURNAL_SCOPE: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "gepa_response_journal_scope",
    default=None,
)
_NAMESPACE_RE = re.compile(r"[A-Za-z0-9._-]+")
_AUTH_REQUEST_FIELDS = frozenset(
    {
        "api_key",
        "api_token",
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "azure_ad_token",
    }
)
_AUTH_HEADER_NAMES = frozenset({"authorization", "proxy-authorization", "x-api-key"})
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


class ResponseJournalError(RuntimeError):
    """Signal an invalid, corrupt, or incompatible response-journal record."""


def stable_api_base_identity(api_base: str) -> str:
    """Remove transport-only identity from one completion endpoint.

    A new Slurm allocation starts the same pinned local model on a new
    ephemeral loopback port. That port cannot be part of interruption-replay
    identity. Hostnames, ports, and paths for non-loopback providers remain
    material.

    Args:
        api_base: Provider or local inference endpoint.

    Returns:
        Endpoint identity with credentials removed and a loopback port omitted.
    """
    parsed = urlsplit(api_base)
    hostname = parsed.hostname
    if hostname is None:
        return api_base
    host = hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    port = None if hostname.lower() in _LOOPBACK_HOSTS else parsed.port
    netloc = host if port is None else f"{host}:{port}"
    return urlunsplit((parsed.scheme.lower(), netloc, parsed.path, parsed.query, parsed.fragment))


def _semantic_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Project a provider request onto restart-stable scientific semantics.

    Authentication rotates independently of experiment behavior. Likewise,
    the port of the pinned loopback vLLM service changes between Slurm jobs.
    All other request fields remain exact, including external endpoints,
    prompts, tools, decoding settings, and batch order.

    Args:
        request: Effective provider request.

    Returns:
        JSON-ready request identity without authentication-only values.
    """
    semantic = {key: value for key, value in request.items() if key not in _AUTH_REQUEST_FIELDS}
    api_base = semantic.get("api_base")
    if isinstance(api_base, str):
        semantic["api_base"] = stable_api_base_identity(api_base)
    extra_headers = semantic.get("extra_headers")
    if isinstance(extra_headers, Mapping):
        semantic["extra_headers"] = {
            key: value for key, value in extra_headers.items() if str(key).lower() not in _AUTH_HEADER_NAMES
        }
    return semantic


def canonical_request_digest(request: Mapping[str, Any]) -> str:
    """Hash restart-stable request semantics without persisting request content.

    Args:
        request: JSON-serializable effective model request. Authentication and
            an ephemeral loopback port do not affect its replay identity.

    Returns:
        Lowercase SHA-256 digest of the canonical request bytes.

    Raises:
        TypeError: The request contains a value without a stable JSON form.
    """
    rendered = json.dumps(
        _semantic_request(request),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


@contextlib.contextmanager
def response_journal_scope(scope: str) -> Iterator[None]:
    """Bind custom LM calls to one restart-stable logical scope.

    Args:
        scope: Stable scope repeated when the same optimizer work is resumed.

    Yields:
        Control to the scoped reflection operation.

    Raises:
        ValueError: The scope is empty.
    """
    if not scope:
        raise ValueError("Response-journal scope must be non-empty.")
    token = ACTIVE_RESPONSE_JOURNAL_SCOPE.set(scope)
    try:
        yield
    finally:
        ACTIVE_RESPONSE_JOURNAL_SCOPE.reset(token)


class ResumeResponseJournal:
    """Persist normalized LM responses by scope, namespace, and call ordinal."""

    def __init__(self, path: str | Path, namespace: str):
        """Create or open one private SQLite response journal.

        Args:
            path: SQLite database path inside a condition-specific run directory.
            namespace: Stable logical identity for one LM role.

        Raises:
            ValueError: The namespace is empty or contains unsafe characters.
            ResponseJournalError: The journal cannot be initialized securely.
        """
        if _NAMESPACE_RE.fullmatch(namespace) is None:
            raise ValueError(
                "Response-journal namespace may contain only letters, numbers, dots, underscores, and hyphens."
            )
        self.path = Path(path).expanduser().resolve()
        self.namespace = namespace
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        """Open a transaction-capable connection with durable writes enabled.

        Returns:
            SQLite connection configured for full synchronous commits.
        """
        connection = sqlite3.connect(self.path, timeout=60.0)
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=60000")
        return connection

    def _initialize(self) -> None:
        """Create the private journal directory, database, and schema.

        Raises:
            ResponseJournalError: Filesystem or SQLite setup fails.
        """
        try:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(self.path.parent, 0o700)
            with self._connect() as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS responses (
                        scope TEXT NOT NULL,
                        namespace TEXT NOT NULL,
                        ordinal INTEGER NOT NULL,
                        request_sha256 TEXT NOT NULL,
                        response_json TEXT NOT NULL,
                        response_sha256 TEXT NOT NULL,
                        PRIMARY KEY (scope, namespace, ordinal)
                    )
                    """
                )
            os.chmod(self.path, 0o600)
        except (OSError, sqlite3.Error) as exc:
            raise ResponseJournalError(f"Could not initialize response journal at {self.path}.") from exc

    def load(self, scope: str, ordinal: int, request_sha256: str) -> dict[str, Any] | None:
        """Load and validate one previously committed normalized response.

        Args:
            scope: Restart-stable logical optimizer scope.
            ordinal: Zero-based call occurrence within this LM and scope.
            request_sha256: Digest of the exact request being replayed.

        Returns:
            Validated response payload, or ``None`` when the slot is unused.

        Raises:
            ResponseJournalError: The slot belongs to a different request or
                contains corrupt or unsupported response data.
        """
        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT request_sha256, response_json, response_sha256
                    FROM responses
                    WHERE scope = ? AND namespace = ? AND ordinal = ?
                    """,
                    (scope, self.namespace, ordinal),
                ).fetchone()
        except sqlite3.Error as exc:
            raise ResponseJournalError("Could not read the response journal.") from exc
        if row is None:
            return None
        recorded_request, response_json, recorded_response = row
        if recorded_request != request_sha256:
            raise ResponseJournalError(
                f"Response-journal request mismatch at {self.namespace}/{scope}/{ordinal}: "
                f"expected {recorded_request}, received {request_sha256}."
            )
        actual_response = hashlib.sha256(response_json.encode("utf-8")).hexdigest()
        if actual_response != recorded_response:
            raise ResponseJournalError(
                f"Response-journal checksum mismatch at {self.namespace}/{scope}/{ordinal}."
            )
        try:
            payload = json.loads(response_json)
        except json.JSONDecodeError as exc:
            raise ResponseJournalError(
                f"Response journal contains invalid JSON at {self.namespace}/{scope}/{ordinal}."
            ) from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != RESPONSE_JOURNAL_SCHEMA_VERSION:
            raise ResponseJournalError(
                f"Response journal contains an unsupported record at {self.namespace}/{scope}/{ordinal}."
            )
        return payload

    def usage_totals(self) -> tuple[float, int, int]:
        """Sum durable provider usage for this logical LM namespace.

        A fresh process initializes its cost and token counters from these
        records before resuming optimization. Because live calls are stored once
        per logical occurrence, cursor rewinds within one process cannot count a
        replay twice.

        Returns:
            Cumulative cost, input tokens, and output tokens.

        Raises:
            ResponseJournalError: A stored record is corrupt, uses an
                unsupported schema, or contains invalid usage values.
        """
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT scope, ordinal, response_json, response_sha256
                    FROM responses
                    WHERE namespace = ?
                    ORDER BY scope, ordinal
                    """,
                    (self.namespace,),
                ).fetchall()
        except sqlite3.Error as exc:
            raise ResponseJournalError("Could not read response-journal usage.") from exc
        total_cost = 0.0
        total_tokens_in = 0
        total_tokens_out = 0
        for scope, ordinal, response_json, recorded_response in rows:
            actual_response = hashlib.sha256(response_json.encode("utf-8")).hexdigest()
            if actual_response != recorded_response:
                raise ResponseJournalError(
                    f"Response-journal checksum mismatch at {self.namespace}/{scope}/{ordinal}."
                )
            try:
                payload = json.loads(response_json)
            except json.JSONDecodeError as exc:
                raise ResponseJournalError(
                    f"Response journal contains invalid JSON at {self.namespace}/{scope}/{ordinal}."
                ) from exc
            if not isinstance(payload, dict) or payload.get("schema_version") != RESPONSE_JOURNAL_SCHEMA_VERSION:
                raise ResponseJournalError(
                    f"Response journal contains an unsupported record at {self.namespace}/{scope}/{ordinal}."
                )
            usage = payload.get("usage")
            if not isinstance(usage, Mapping):
                raise ResponseJournalError(
                    f"Response journal contains invalid usage at {self.namespace}/{scope}/{ordinal}."
                )
            cost = usage.get("cost")
            tokens_in = usage.get("tokens_in")
            tokens_out = usage.get("tokens_out")
            valid_cost = (
                isinstance(cost, int | float)
                and not isinstance(cost, bool)
                and math.isfinite(float(cost))
                and float(cost) >= 0
            )
            valid_tokens = all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in (tokens_in, tokens_out)
            )
            if not valid_cost or not valid_tokens:
                raise ResponseJournalError(
                    f"Response journal contains invalid usage at {self.namespace}/{scope}/{ordinal}."
                )
            total_cost += float(cost)
            total_tokens_in += tokens_in
            total_tokens_out += tokens_out
        return total_cost, total_tokens_in, total_tokens_out

    def store(self, scope: str, ordinal: int, request_sha256: str, response: Mapping[str, Any]) -> None:
        """Commit one normalized response before exposing it to the optimizer.

        Args:
            scope: Restart-stable logical optimizer scope.
            ordinal: Zero-based call occurrence within this LM and scope.
            request_sha256: Digest of the exact request that produced the response.
            response: JSON-serializable normalized response payload.

        Raises:
            ResponseJournalError: Serialization or the durable transaction fails,
                or a concurrent writer already assigned the slot differently.
        """
        payload = dict(response)
        payload["schema_version"] = RESPONSE_JOURNAL_SCHEMA_VERSION
        try:
            response_json = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError) as exc:
            raise ResponseJournalError("Normalized LM response is not JSON serializable.") from exc
        response_sha256 = hashlib.sha256(response_json.encode("utf-8")).hexdigest()
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                existing = connection.execute(
                    """
                    SELECT request_sha256, response_sha256
                    FROM responses
                    WHERE scope = ? AND namespace = ? AND ordinal = ?
                    """,
                    (scope, self.namespace, ordinal),
                ).fetchone()
                if existing is not None:
                    if existing != (request_sha256, response_sha256):
                        raise ResponseJournalError(
                            f"Response-journal slot collision at {self.namespace}/{scope}/{ordinal}."
                        )
                    return
                connection.execute(
                    """
                    INSERT INTO responses (
                        scope,
                        namespace,
                        ordinal,
                        request_sha256,
                        response_json,
                        response_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        scope,
                        self.namespace,
                        ordinal,
                        request_sha256,
                        response_json,
                        response_sha256,
                    ),
                )
        except ResponseJournalError:
            raise
        except sqlite3.Error as exc:
            raise ResponseJournalError("Could not commit the normalized LM response.") from exc
