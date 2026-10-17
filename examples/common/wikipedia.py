"""Define the passage and retrieval interface used by the frozen Wikipedia index."""

from dataclasses import dataclass
from typing import Callable, Protocol


@dataclass(frozen=True)
class WikipediaPassage:
    """Represent a retrieved Wikipedia page and its plain-text introduction."""

    title: str
    text: str


class WikipediaRetriever(Protocol):
    """Expose a ranked ``search(query, limit)`` Wikipedia retrieval callable."""

    search: Callable[[str, int], list[WikipediaPassage]]
