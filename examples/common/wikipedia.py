"""Cached MediaWiki retrieval for Wikipedia-backed benchmark examples."""

import json
import os
import sqlite3
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

DEFAULT_WIKIPEDIA_ENDPOINT = "https://en.wikipedia.org/w/api.php"


@dataclass(frozen=True)
class WikipediaPassage:
    """Represent a retrieved Wikipedia page and its plain-text introduction."""

    title: str
    text: str

    def render(self) -> str:
        """Render the passage in the format used by the GEPA artifact."""
        return f"{self.title} | {self.text}"


class WikipediaRetriever(Protocol):
    """Describe the retrieval dependency used by the benchmark programs."""

    def search(self, query: str, limit: int) -> list[WikipediaPassage]:
        """Return ranked Wikipedia passages for a query."""
        raise NotImplementedError


class MediaWikiTransport(Protocol):
    """Describe an injectable MediaWiki request transport."""

    def __call__(
        self,
        endpoint: str,
        params: Mapping[str, str],
        timeout: float,
        headers: Mapping[str, str],
    ) -> Mapping[str, Any]:
        """Execute one MediaWiki API request."""
        raise NotImplementedError


class WikipediaRetrievalError(RuntimeError):
    """Signal that Wikipedia retrieval failed or returned malformed data."""


def _default_transport(
    endpoint: str,
    params: Mapping[str, str],
    timeout: float,
    headers: Mapping[str, str],
) -> Mapping[str, Any]:
    """Execute a MediaWiki request with the Python standard library."""
    url = f"{endpoint}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers=dict(headers))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except Exception as exc:
        raise WikipediaRetrievalError(f"MediaWiki request failed for query {params.get('gsrsearch')!r}: {exc}") from exc
    if not isinstance(payload, dict):
        raise WikipediaRetrievalError("MediaWiki returned a non-object response")
    return payload


class WikipediaClient:
    """Search Wikipedia through MediaWiki with an optional persistent cache."""

    def __init__(
        self,
        endpoint: str = DEFAULT_WIKIPEDIA_ENDPOINT,
        cache_path: str | os.PathLike[str] | None = None,
        timeout: float = 20.0,
        user_agent: str = "gepa-benchmark-harness/0.1 (https://github.com/gepa-ai/gepa)",
        transport: MediaWikiTransport | None = None,
    ) -> None:
        """Configure the endpoint, cache, timeout, and injectable transport."""
        self.endpoint = endpoint
        self.timeout = timeout
        self.user_agent = user_agent
        self.transport = transport or _default_transport
        self.cache_path = Path(cache_path).expanduser() if cache_path is not None else _default_cache_path()
        if self.cache_path is not None:
            self._initialize_cache()

    def search(self, query: str, limit: int) -> list[WikipediaPassage]:
        """Return ranked page introductions for a MediaWiki full-text search."""
        query = " ".join(query.split())
        if not query or limit <= 0:
            return []

        cached = self._read_cache(query, limit)
        if cached is not None:
            return cached

        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "generator": "search",
            "gsrnamespace": "0",
            "gsrsearch": query,
            "gsrlimit": str(limit),
            "prop": "extracts",
            "exintro": "1",
            "explaintext": "1",
            "redirects": "1",
        }
        payload = self.transport(
            self.endpoint,
            params,
            self.timeout,
            {"Accept": "application/json", "User-Agent": self.user_agent},
        )
        passages = self._parse_response(payload, limit)
        self._write_cache(query, limit, passages)
        return passages

    def _parse_response(self, payload: Mapping[str, Any], limit: int) -> list[WikipediaPassage]:
        """Parse generator search results while preserving MediaWiki rank."""
        query_payload = payload.get("query", {})
        if not isinstance(query_payload, dict):
            raise WikipediaRetrievalError("MediaWiki response has an invalid query field")
        pages = query_payload.get("pages", [])
        if pages is None:
            return []
        if not isinstance(pages, list):
            raise WikipediaRetrievalError("MediaWiki response has an invalid pages field")

        ranked_pages = sorted(
            (page for page in pages if isinstance(page, dict)),
            key=lambda page: (page.get("index", limit + 1), page.get("pageid", 0)),
        )
        passages: list[WikipediaPassage] = []
        for page in ranked_pages[:limit]:
            title = str(page.get("title", "")).strip()
            if not title:
                continue
            passages.append(WikipediaPassage(title=title, text=str(page.get("extract", "")).strip()))
        return passages

    def _initialize_cache(self) -> None:
        """Create the SQLite cache table when persistent caching is enabled."""
        assert self.cache_path is not None
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.cache_path, timeout=30) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS searches "
                "(endpoint TEXT NOT NULL, query TEXT NOT NULL, result_limit INTEGER NOT NULL, payload TEXT NOT NULL, "
                "PRIMARY KEY(endpoint, query, result_limit))"
            )

    def _read_cache(self, query: str, limit: int) -> list[WikipediaPassage] | None:
        """Read a cached result, returning None for a cache miss."""
        if self.cache_path is None:
            return None
        with sqlite3.connect(self.cache_path, timeout=30) as connection:
            row = connection.execute(
                "SELECT payload FROM searches WHERE endpoint = ? AND query = ? AND result_limit = ?",
                (self.endpoint, query, limit),
            ).fetchone()
        if row is None:
            return None
        data = json.loads(row[0])
        return [WikipediaPassage(title=item["title"], text=item["text"]) for item in data]

    def _write_cache(self, query: str, limit: int, passages: Sequence[WikipediaPassage]) -> None:
        """Persist a successful search response."""
        if self.cache_path is None:
            return
        payload = json.dumps([asdict(passage) for passage in passages], ensure_ascii=False)
        with sqlite3.connect(self.cache_path, timeout=30) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO searches(endpoint, query, result_limit, payload) VALUES (?, ?, ?, ?)",
                (self.endpoint, query, limit, payload),
            )


def _default_cache_path() -> Path:
    """Resolve the shared cache path without writing inside the repository."""
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_root / "gepa" / "wikipedia.sqlite3"
