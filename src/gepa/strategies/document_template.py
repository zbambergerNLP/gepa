# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Define section templates for system prompts, user prompts, and skills.

Each template specifies an ordered set of optional ``## <Section>`` headings.
:meth:`DocumentTemplate.parse` validates that format, while
:func:`migrate_document` converts free-form text.
Provider families follow their published section ordering and use ``generic``
when no provider schema applies. PromptPrism (arXiv:2505.12592) motivates
treating section order as part of the schema.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from gepa.proposer.reflective_mutation.base import LanguageModel

# A section header line: exactly two hashes, so "###" sub-headers inside a body
# do not count as sections.
_HEADER_RE = re.compile(r"^## (.+?)[ \t]*$", re.MULTILINE)

MIGRATION_PROMPT = """You are restructuring a {kind} document into a section format so that it can later be edited one section at a time.

Rewrite the document using the applicable sections below, in their listed order. Introduce each section you use with a markdown level-2 header line (`## <Section>`):

{section_guide}

Rules:
- Preserve the original content and intent. Sort every sentence into the section it belongs to; do not invent new requirements or drop existing ones.
- Omit a section entirely when nothing in the original belongs there. Never emit an empty section header.
- Put nothing before the first header: no title, preamble, or closing remarks.
- Do not use `## ` header lines inside a section body (`###` sub-headers are fine).

Return only the restructured document, inside <document></document> tags.

<original>
{text}
</original>
"""


@dataclass(frozen=True)
class EditTarget:
    """One named section of a candidate component that an edit may modify.

    Attributes:
        component_name: Name of the candidate component the region belongs to.
        section: Non-empty template section addressed by the edit.
        label: Component-scoped section label used in menus and history.

    Raises:
        ValueError: ``section`` is not a non-empty string.
    """

    component_name: str
    section: str
    label: str = field(init=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the section name and store its component-scoped label.

        The derived label is materialized once because edit targets are frozen
        values used repeatedly as menu and branch-history identifiers.

        Raises:
            ValueError: ``section`` is not a non-empty string.
        """
        if not isinstance(self.section, str) or not self.section:
            raise ValueError("EditTarget requires a non-empty section name.")
        object.__setattr__(self, "label", f"{self.component_name}:{self.section}")


class MalformedDocumentError(ValueError):
    """The text is not in its template's canonical section format.

    Raised by :meth:`DocumentTemplate.parse` when a document uses an unknown,
    duplicate, or out-of-order section or carries content before the first
    header, and by :func:`migrate_document` when the LM could not produce a
    conforming document within its retry budget.
    """


@dataclass
class DocumentTemplate:
    """A document kind and its ordered sections (name -> what belongs there).

    The section descriptions steer :func:`migrate_document` when it sorts
    free-form content into sections; the names are the ``## <Section>`` headers
    of the canonical format.

    Attributes:
        kind: Semantic document kind (``"prompt"`` or ``"skill"``). System and
            user prompt templates both use ``"prompt"`` so they share the same
            action catalog while remaining separate entries in :data:`TEMPLATES`.
        sections: Ordered mapping from section name to a one-line description
            of what belongs in it.
    """

    kind: str
    sections: dict[str, str]

    def parse(self, text: str) -> dict[str, str]:
        """Split a canonical document into its section bodies.

        The document may contain any subset of this template's sections. Used
        sections must appear in schema order, each introduced by a
        ``## <Section>`` line, with nothing but whitespace before the first
        header. Missing sections are returned with empty bodies. An empty
        document therefore represents a template with no populated sections.

        Args:
            text: The full component text.

        Returns:
            Section name -> body text, in template order.

        Raises:
            MalformedDocumentError: A header is unknown, duplicated, or out of
                order, or content appears outside a section.
        """
        headers = list(_HEADER_RE.finditer(text))
        found = [match.group(1) for match in headers]
        expected = list(self.sections)
        unknown = [section for section in found if section not in self.sections]
        duplicate = next((section for section in found if found.count(section) > 1), None)
        positions = [expected.index(section) for section in found if section in self.sections]
        if unknown or duplicate is not None or positions != sorted(positions):
            raise MalformedDocumentError(
                f"{self.kind} document may use the sections {expected} as '## <Section>' headers, at most once "
                f"each and in that order; found {found}."
            )
        if not headers:
            if text.strip():
                raise MalformedDocumentError(
                    f"{self.kind} document contains text but no recognized '## <Section>' header."
                )
            return dict.fromkeys(self.sections, "")
        if text[: headers[0].start()].strip():
            raise MalformedDocumentError(f"{self.kind} document has content before the first '## ' header.")
        bodies = dict.fromkeys(self.sections, "")
        for i, match in enumerate(headers):
            end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
            bodies[match.group(1)] = text[match.end() : end].strip()
        return bodies

    def section_body_span(self, text: str, section: str) -> tuple[int, int]:
        """Locate a section body without including its surrounding whitespace.

        The returned offsets select exactly the same body text as
        ``parse(text)[section]``. Splicing at these offsets therefore preserves
        every byte outside the selected section body, including header spacing
        and blank lines between sections.

        Args:
            text: Canonical structured document.
            section: Section whose body should be located.

        Returns:
            Start and end character offsets into ``text``.

        Raises:
            KeyError: ``section`` is not part of this template.
            MalformedDocumentError: ``text`` does not conform to the template,
                or the section is omitted because it is empty.
        """
        self.parse(text)
        if section not in self.sections:
            raise KeyError(section)
        headers = list(_HEADER_RE.finditer(text))
        rendered_sections = [match.group(1) for match in headers]
        if section not in rendered_sections:
            raise MalformedDocumentError(
                f"Section {section!r} is omitted from this {self.kind} document because its body is empty."
            )
        index = rendered_sections.index(section)
        raw_start = headers[index].end()
        raw_end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        raw_body = text[raw_start:raw_end]
        if not raw_body.strip():
            if raw_body.startswith("\r\n"):
                insertion = raw_start + 2
            elif raw_body.startswith(("\n", "\r")):
                insertion = raw_start + 1
            else:
                insertion = raw_start
            return insertion, insertion
        leading = len(raw_body) - len(raw_body.lstrip())
        trailing = len(raw_body) - len(raw_body.rstrip())
        body_start = raw_start + leading
        body_end = raw_end - trailing if trailing else raw_end
        return body_start, max(body_start, body_end)

    def render(self, bodies: dict[str, str]) -> str:
        """Write section bodies out in the canonical format.

        Missing and empty bodies are omitted. Their sections still belong to
        the template and remain addressable by name.

        Args:
            bodies: Section name -> body text; keys outside the template are
                ignored.

        Returns:
            Populated ``## <Section>`` blocks in template order, or an empty
            string when no section has content.
        """
        blocks = [f"## {section}\n{body}\n" for section in self.sections if (body := bodies.get(section, "").strip())]
        return "\n".join(blocks)

    def replace_section_body(self, text: str, section: str, body: str) -> str:
        """Replace one body and render the document in canonical sparse form.

        This method also handles sections that are currently omitted. Adding
        text inserts the header in schema order; clearing the body removes it.

        Args:
            text: Current structured document.
            section: Template section to update.
            body: New body text.

        Returns:
            The updated canonical document.

        Raises:
            KeyError: ``section`` is not part of this template.
            MalformedDocumentError: ``text`` does not conform to the template,
                or ``body`` contains another ``## <Section>`` header.
        """
        if section not in self.sections:
            raise KeyError(section)
        if _HEADER_RE.search(body):
            raise MalformedDocumentError(
                "A section body cannot contain a '## <Section>' header; use '###' for a nested heading."
            )
        bodies = self.parse(text)
        bodies[section] = body
        return self.render(bodies)


# Generic guidance does not distinguish the two message roles, so the registry
# deliberately points both names at the same template object.
_GENERIC_PROMPT = DocumentTemplate(
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
)

_SKILL_TEMPLATE = DocumentTemplate(
    "skill",
    {
        "Name": "The skill's short identifier.",
        "Description": "What the skill does and when an agent should use it.",
        "Instructions": "Step-by-step procedure the agent follows when applying the skill.",
        "Examples": "Worked examples of the skill being applied.",
    },
)

TEMPLATES: dict[str, DocumentTemplate] = {
    "system_prompt": _GENERIC_PROMPT,
    "user_prompt": _GENERIC_PROMPT,
    "skill": _SKILL_TEMPLATE,
}

# OpenAI's provider-wide prompt-engineering guide ("Message formatting with
# Markdown and XML"): a developer message "will contain the following sections,
# usually in this order (though the exact optimal content and order may vary by
# which model you are using)" -- Identity / Instructions / Examples / Context,
# with Context last because it varies per request. This provider-wide template
# applies to every OpenAI model.
# Source: https://developers.openai.com/api/docs/guides/prompt-engineering
_OPENAI_SYSTEM_PROMPT = DocumentTemplate(
    "prompt",
    {
        "Identity": "Who the assistant is: its purpose, communication style, and high-level goals.",
        "Instructions": "Guidance and rules the model must follow when generating the response.",
        "Examples": "Example inputs paired with the desired output from the model.",
        "Context": "Additional information the task depends on, placed near the end because it varies per request.",
    },
)

_OPENAI_USER_PROMPT = DocumentTemplate(
    "prompt",
    {"Input": "The end user's request or input for this turn."},
)

# Anthropic does not define a standard set of sections for Claude prompts.
#
# For long-context prompts, it recommends putting documents and other source
# material near the top and leaving the actual query or task near the end.
# The system message carries the role and any standing instructions. The user
# message follows the recommended long-context ordering, with source material
# near the top and the turn's task last. Constraints stay with Instructions.
#
# Reasoning is available for tasks that need explicit thinking guidance. It is
# omitted from the rendered message until a proposer gives it content.
#
# Source:
# https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices
_ANTHROPIC_SYSTEM_PROMPT = DocumentTemplate(
    "prompt",
    {
        "Role": "The role or perspective Claude should take.",
        "Instructions": "Standing task guidance, requirements, and constraints.",
    },
)

_ANTHROPIC_USER_PROMPT = DocumentTemplate(
    "prompt",
    {
        "Context": "Background information, documents, and other material needed for the task.",
        "Examples": "Examples of inputs and the kind of output expected.",
        "Output Format": "How the answer should be structured or formatted.",
        "Reasoning": "Task-specific guidance for how Claude should reason when the task calls for it.",
        "Instructions": "The task itself, along with any requirements and constraints.",
    },
)

# The Gemini "example template combining best practices" in Google's prompt
# design strategies doc: role -> instructions -> constraints -> output_format
# in the system instruction, then context -> task -> final_instruction in the
# user prompt, with a closing reminder at the very end. The doc treats
# XML-style tags and markdown headers as interchangeable delimiters, so the
# family keeps the shared markdown format.
# Source: https://ai.google.dev/gemini-api/docs/prompting-strategies
_GOOGLE_SYSTEM_PROMPT = DocumentTemplate(
    "prompt",
    {
        "Role": "Who the model is: persona, identity, and qualities.",
        "Instructions": "The numbered workflow the model should follow (plan, execute, validate).",
        "Constraints": "Behavioral limits and requirements the output must satisfy (verbosity, tone, what to avoid).",
        "Output Format": "The exact shape of the final answer (structure, fields, length, style).",
    },
)

_GOOGLE_USER_PROMPT = DocumentTemplate(
    "prompt",
    {
        "Context": "Documents, code, and background data, marked as data rather than instructions.",
        "Task": "The specific request the model must act on.",
        "Final Instruction": "Closing reminder placed at the very end (e.g. to think step by step).",
    },
)

# Alibaba Cloud Model Studio's prompt-engineering guide prescribes Context /
# Objective / Style / Tone / Audience / Response. Its API guidance places
# standing task constraints and response behavior in the system message, while
# the user's context and request stay in the user message.
# Source: https://www.alibabacloud.com/help/en/model-studio/prompt-engineering-guide
_ALIBABA_SYSTEM_PROMPT = DocumentTemplate(
    "prompt",
    {
        "Objective": "The standing task or behavioral objective the model must follow.",
        "Style": "The writing style the output should follow.",
        "Tone": "The tone the output should carry (e.g. formal, humorous, warm).",
        "Audience": "The target readers of the output.",
        "Response": "The exact form and format of the output.",
    },
)

_ALIBABA_USER_PROMPT = DocumentTemplate(
    "prompt",
    {
        "Context": "Background information closely related to the task.",
        "Objective": "The specific task the model must complete.",
    },
)

# Template families expose separate system and user message roles. The generic
# family shares one prompt shape across both roles. Skills are provider-neutral,
# so every family shares one skill template. Providers without a documented
# prompt structure use the generic family.
TEMPLATE_FAMILIES: dict[str, dict[str, DocumentTemplate]] = {
    "generic": TEMPLATES,
    "openai": {
        "system_prompt": _OPENAI_SYSTEM_PROMPT,
        "user_prompt": _OPENAI_USER_PROMPT,
        "skill": _SKILL_TEMPLATE,
    },
    "anthropic": {
        "system_prompt": _ANTHROPIC_SYSTEM_PROMPT,
        "user_prompt": _ANTHROPIC_USER_PROMPT,
        "skill": _SKILL_TEMPLATE,
    },
    "google": {
        "system_prompt": _GOOGLE_SYSTEM_PROMPT,
        "user_prompt": _GOOGLE_USER_PROMPT,
        "skill": _SKILL_TEMPLATE,
    },
    "alibaba": {
        "system_prompt": _ALIBABA_SYSTEM_PROMPT,
        "user_prompt": _ALIBABA_USER_PROMPT,
        "skill": _SKILL_TEMPLATE,
    },
}


def infer_template_family(model: str | None) -> str:
    """Infer the template family whose prompting guidance covers ``model``.

    The match is a substring test on the (LiteLLM-style) model identifier:
    Claude models map to ``"anthropic"``, Gemini/Gemma to ``"google"``, GPT and
    o-series models to ``"openai"``, and Qwen/QwQ to ``"alibaba"``. Everything
    else maps to ``"generic"`` -- the providers behind Muse, Llama, Grok,
    DeepSeek, Mistral, Kimi, and GLM models prescribe no prompt structure to
    mirror.

    Args:
        model: Model identifier such as ``"openai/gpt-5"`` or
            ``"anthropic/claude-opus-4"``; ``None`` when no task model name is
            available.

    Returns:
        A key of :data:`TEMPLATE_FAMILIES`.
    """
    if not model:
        return "generic"
    lowered = model.lower()
    if "claude" in lowered:
        return "anthropic"
    if "gemini" in lowered or "gemma" in lowered:
        return "google"
    if "gpt" in lowered or lowered.startswith("openai/") or re.search(r"(?:^|/)o\d+(?:$|[-.])", lowered):
        return "openai"
    if "qwen" in lowered or "qwq" in lowered:
        return "alibaba"
    return "generic"


def migrate_document(text: str, template: DocumentTemplate, lm: LanguageModel) -> str:
    """Bring a free-form document into ``template``'s canonical section format.

    This is the one-time migration path for prompts and skills written before
    the format existed. Text that already parses is re-rendered canonically.
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
        return template.render(template.parse(text))
    except MalformedDocumentError:
        pass

    section_guide = "\n".join(f"- ## {name}: {what}" for name, what in template.sections.items())
    prompt = MIGRATION_PROMPT.format(kind=template.kind, section_guide=section_guide, text=text)
    feedback = ""
    last_error = ""
    for _attempt in range(2):
        raw = lm(prompt + feedback)
        match = re.search(r"<document>(.*?)</document>", raw, re.DOTALL)
        reply = match.group(1) if match is not None else raw
        try:
            migrated = template.render(template.parse(reply))
            if text.strip() and not migrated:
                raise MalformedDocumentError("Migration removed all content from a non-empty document.")
            return migrated
        except MalformedDocumentError as exc:
            last_error = str(exc)
            feedback = f"\n\nYour previous reply was rejected: {last_error} Reply again with the corrected document."
    raise MalformedDocumentError(f"Could not migrate {template.kind} document into canonical format: {last_error}")
