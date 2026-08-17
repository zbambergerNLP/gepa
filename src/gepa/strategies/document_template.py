# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Document templates: the canonical section format of a component's text.

The 3-role architecture never inspects an arbitrary text to guess its
structure. Every component is *declared* to be a document of a known kind
(``"prompt"`` or ``"skill"``, see :data:`TEMPLATES`) and is guaranteed to be
written in that kind's canonical format: one ``## <Section>`` markdown header
per template section, all sections present exactly once, in template order,
nothing before the first header. Text that predates this convention is brought
into the format once, up front, by :func:`migrate_document`, which asks an LM to
restructure it (content preserved, structure imposed) and verifies the reply
parses.

A :class:`DocumentTemplate` is a kind plus its ordered sections. It parses and
renders documents in the format, and lists the :class:`EditTarget` s the
Controller can address; the RLM edits one section body in isolation and
re-renders the document around it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from gepa.proposer.reflective_mutation.base import LanguageModel
from gepa.utils.text import strip_think_tags

# A section header line: exactly two hashes, so "###" sub-headers inside a body
# do not count as sections.
_HEADER_RE = re.compile(r"^## (.+?)[ \t]*$", re.MULTILINE)

MIGRATION_PROMPT = """You are restructuring a {kind} document into a fixed section format so that it can later be edited one section at a time.

Rewrite the document below so that it consists of exactly these sections, in this order, each introduced by a markdown level-2 header line (`## <Section>`):

{section_guide}

Rules:
- Preserve the original content and intent. Sort every sentence into the section it belongs to; do not invent new requirements or drop existing ones.
- Keep every section header even when nothing in the original belongs there (leave its body empty).
- Put nothing before the first header: no title, preamble, or closing remarks.
- Do not use `## ` header lines inside a section body (`###` sub-headers are fine).

Return only the restructured document, inside <document></document> tags.

<original>
{text}
</original>
"""


@dataclass(frozen=True)
class EditTarget:
    """An addressable region of the current candidate that an edit may modify.

    ``section`` names a template section; ``None`` addresses the whole document,
    which is what cross-section operations (e.g. moving text between sections)
    act on.

    Attributes:
        component_name: Name of the candidate component the region belongs to.
        section: The template section addressed, or ``None`` for the whole
            document.
    """

    component_name: str
    section: str | None

    @property
    def name(self) -> str:
        """Short region name used in menu ids and prompts.

        Returns:
            The section name, or ``"whole"`` when the target is the whole
            document.
        """
        return self.section or "whole"

    @property
    def label(self) -> str:
        """Human-readable label for menus and logs.

        Returns:
            ``"<component_name>:<name>"``, e.g. ``"system_prompt:Rules"``.
        """
        return f"{self.component_name}:{self.name}"


class MalformedDocumentError(ValueError):
    """The text is not in its template's canonical section format.

    Raised by :meth:`DocumentTemplate.parse` when a document lacks a section,
    has sections out of order, or carries content before the first header, and
    by :func:`migrate_document` when the LM could not produce a conforming
    document within its retry budget.
    """


@dataclass
class DocumentTemplate:
    """A document kind and its ordered sections (name -> what belongs there).

    The section descriptions steer :func:`migrate_document` when it sorts
    free-form content into sections; the names are the ``## <Section>`` headers
    of the canonical format.

    Attributes:
        kind: Short identifier of the document kind (e.g. ``"prompt"``); used to
            look the template up in :data:`TEMPLATES` and in error messages.
        sections: Ordered mapping from section name to a one-line description
            of what belongs in it.
    """

    kind: str
    sections: dict[str, str]

    def edit_targets(self, component_name: str) -> list[EditTarget]:
        """List the regions the Controller may target: every section, then the whole document.

        No text is needed because a document of this kind is guaranteed to
        contain every section (see :meth:`parse`).

        Args:
            component_name: Name of the candidate component the targets refer to.

        Returns:
            One :class:`EditTarget` per section in template order, followed by
            the whole-document target (``section=None``).
        """
        targets = [EditTarget(component_name, section) for section in self.sections]
        targets.append(EditTarget(component_name, None))
        return targets

    def parse(self, text: str) -> dict[str, str]:
        """Split a canonical document into its section bodies.

        The document must consist of exactly this template's sections, in
        order, each introduced by a ``## <Section>`` line, with nothing but
        whitespace before the first header. Bodies are returned stripped of
        surrounding whitespace, so ``render(parse(text))`` is the canonical
        spelling of ``text``.

        Args:
            text: The full component text.

        Returns:
            Section name -> body text, in template order.

        Raises:
            MalformedDocumentError: The headers found are not exactly the
                template's sections in order, or content precedes the first
                header. The message names what was found so a caller (or the
                RLM, via ``<error>`` feedback) can correct it.
        """
        headers = list(_HEADER_RE.finditer(text))
        found = [match.group(1) for match in headers]
        expected = list(self.sections)
        if found != expected:
            raise MalformedDocumentError(
                f"{self.kind} document must have exactly the sections {expected} as '## <Section>' headers, "
                f"in that order; found {found}."
            )
        if text[: headers[0].start()].strip():
            raise MalformedDocumentError(f"{self.kind} document has content before the first '## ' header.")
        bodies: dict[str, str] = {}
        for i, match in enumerate(headers):
            end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
            bodies[match.group(1)] = text[match.end() : end].strip()
        return bodies

    def render(self, bodies: dict[str, str]) -> str:
        """Write section bodies out in the canonical format.

        Sections missing from ``bodies`` are rendered with an empty body, so a
        migrated document always has every header even when the source had
        nothing to put under one.

        Args:
            bodies: Section name -> body text; keys outside the template are
                ignored.

        Returns:
            The document: ``## <Section>`` header lines in template order, each
            followed by its stripped body and a blank line.
        """
        blocks = [f"## {section}\n{bodies.get(section, '').strip()}\n" for section in self.sections]
        return "\n".join(blocks)


# Section catalogs, aligned with STRUCTURED_SECTIONS in action_space.py (the
# references-grounded 7-component taxonomy: persona, directive, context,
# constraints, reasoning, exemplars, output format).
TEMPLATES: dict[str, DocumentTemplate] = {
    "prompt": DocumentTemplate(
        "prompt",
        {
            "Role": "Who the model is: persona, expertise, and stance.",
            "Task": "What the model must accomplish, including the procedure to follow.",
            "Context": "Background facts, domain knowledge, and inputs the task depends on.",
            "Rules": "Constraints and requirements the output must satisfy.",
            "Reasoning": "How the model should think before answering (steps, checks, decomposition).",
            "Examples": "Worked input/output examples that demonstrate the expected behavior.",
            "Output Format": "The exact shape of the final answer (structure, fields, length, style).",
        },
    ),
    "skill": DocumentTemplate(
        "skill",
        {
            "Name": "The skill's short identifier.",
            "Description": "What the skill does and when an agent should use it.",
            "Instructions": "Step-by-step procedure the agent follows when applying the skill.",
            "Examples": "Worked examples of the skill being applied.",
        },
    ),
}


def migrate_document(text: str, template: DocumentTemplate, lm: LanguageModel) -> str:
    """Bring a free-form document into ``template``'s canonical section format.

    This is the one-time migration path for prompts and skills written before
    the format existed. Text that already parses is returned unchanged.
    Otherwise the LM is asked (:data:`MIGRATION_PROMPT`) to sort the content
    into the template's sections; its reply is parsed and re-rendered so
    whitespace is canonical. A non-conforming reply is sent back once with the
    parse error so the LM can correct it.

    Args:
        text: The document to migrate.
        template: The declared kind of the document.
        lm: The language model that performs the restructuring.

    Returns:
        The document in canonical format.

    Raises:
        MalformedDocumentError: The LM's reply still did not conform after the
            retry; the message carries the last parse error.
    """
    try:
        template.parse(text)
        return text
    except MalformedDocumentError:
        pass

    section_guide = "\n".join(f"- ## {name}: {what}" for name, what in template.sections.items())
    prompt = MIGRATION_PROMPT.format(kind=template.kind, section_guide=section_guide, text=text)
    feedback = ""
    last_error = ""
    for _attempt in range(2):
        raw = strip_think_tags(lm(prompt + feedback))
        match = re.search(r"<document>(.*?)</document>", raw, re.DOTALL)
        reply = match.group(1) if match is not None else raw
        try:
            return template.render(template.parse(reply))
        except MalformedDocumentError as exc:
            last_error = str(exc)
            feedback = f"\n\nYour previous reply was rejected: {last_error} Reply again with the corrected document."
    raise MalformedDocumentError(f"Could not migrate {template.kind} document into canonical format: {last_error}")
