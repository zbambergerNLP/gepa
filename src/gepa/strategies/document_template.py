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
Controller can address; ReAct V2 edits one section body in isolation and
splices it back without changing the surrounding document bytes.

Templates come in *families* (:data:`TEMPLATE_FAMILIES`): the ``"generic"``
family is grounded in the convergent prompt-component taxonomies of the
academic literature, while ``"openai"``, ``"anthropic"``, ``"google"``, and
``"alibaba"`` rename and reorder the prompt sections to match the
corresponding provider's own prompting guidance -- the closest public proxy to
the prompt structure the provider's post-training rewarded. A provider gets a
family only while its official guidance prescribes prompt structure, whether a
named section skeleton (OpenAI's prompt-engineering guide, Google's Gemini
template, Alibaba's six-part prompt framework) or explicit placement rules
(Anthropic); providers whose guidance is technique-only -- Meta (Muse, and
Llama before it), xAI, DeepSeek, Mistral, Moonshot, and Zhipu among them --
map to ``"generic"``. When a provider additionally publishes a
*model-specific* skeleton, that model line gets its own family
(``"openai-gpt-5.6"``), preferred over the provider family for the models it
names. Section *order* is the axis that varies: PromptPrism
(arXiv:2505.12592) finds semantic component ordering statistically significant
while delimiter changes are not, so every family keeps the same
``## <Section>`` markdown format (one parser, one renderer) and differs only in
which sections exist, what they are called, and where they sit.
:func:`infer_template_family` maps a model name to its family.
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
                ReAct V2, via an error observation) can correct it.
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
            MalformedDocumentError: ``text`` does not conform to the template.
        """
        self.parse(text)
        if section not in self.sections:
            raise KeyError(section)
        headers = list(_HEADER_RE.finditer(text))
        index = list(self.sections).index(section)
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

# OpenAI's provider-wide prompt-engineering guide ("Message formatting with
# Markdown and XML"): a developer message "will contain the following sections,
# usually in this order (though the exact optimal content and order may vary by
# which model you are using)" -- Identity / Instructions / Examples / Context,
# with Context last because it varies per request. That per-model caveat is the
# hook for _OPENAI_GPT56_PROMPT below; for GPT models without a model-specific
# skeleton this provider-wide one is the operative prescription (the older
# GPT-4.1 seven-section skeleton is retired).
# Source: https://developers.openai.com/api/docs/guides/prompt-engineering
_OPENAI_PROMPT = DocumentTemplate(
    "prompt",
    {
        "Identity": "Who the assistant is: its purpose, communication style, and high-level goals.",
        "Instructions": "Guidance and rules the model must follow when generating the response.",
        "Examples": "Example inputs paired with the desired output from the model.",
        "Context": "Additional information the task depends on, placed near the end because it varies per request.",
    },
)

# OpenAI's GPT-5.6 family guide ("Suggested prompt structure", applying to
# "GPT-5.6 Sol or the GPT-5.6 family") prescribes its own eight-section
# skeleton, offered as "a starting point for complex prompts". It supersedes
# the provider-wide skeleton for GPT-5.6 models only; other GPT and o-series
# models stay on _OPENAI_PROMPT.
# Source: https://developers.openai.com/api/docs/guides/prompt-guidance-gpt-5p6
_OPENAI_GPT56_PROMPT = DocumentTemplate(
    "prompt",
    {
        "Role": "The model's function and the context it operates in.",
        "Personality": "The tone and collaboration style the model should adopt.",
        "Goal": "The user-visible outcome the model must deliver.",
        "Success Criteria": "What must be true before the final answer is given.",
        "Constraints": "Policy, safety, business, evidence, and side-effect limits on the work.",
        "Tools": "Which tools to use, when to use them, and what not to use.",
        "Output": "The sections, length, format, and tone of the final answer.",
        "Stop Rules": "When to retry, fall back, abstain, ask, or stop.",
    },
)

# Claude prompting best practices: longform data and context go at the top,
# "above your query, instructions, and examples", and the query/instructions go
# last. Constraints are folded into Instructions (Anthropic's docs never split
# them out, and warn against over-structured rule piles). Instruction-last is
# also PromptPrism's strongest per-model ordering effect for Claude. Anthropic
# prescribes these placement rules rather than a named section skeleton.
# Source: https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices
_ANTHROPIC_PROMPT = DocumentTemplate(
    "prompt",
    {
        "Role": "Who the model is: persona, expertise, and stance.",
        "Context": "Background facts, documents, and longform data, placed before the instructions.",
        "Examples": "Worked input/output examples that demonstrate the expected behavior.",
        "Reasoning": "How the model should think before answering (steps, checks, decomposition).",
        "Output Format": "The exact shape of the final answer (structure, fields, length, style).",
        "Instructions": "What the model must accomplish, including its rules and constraints, stated last.",
    },
)

# The Gemini "example template combining best practices" in Google's prompt
# design strategies doc: role -> instructions -> constraints -> output_format
# in the system instruction, then context -> task -> final_instruction in the
# user prompt, with a closing reminder at the very end. The doc treats
# XML-style tags and markdown headers as interchangeable delimiters, so the
# family keeps the shared markdown format.
# Source: https://ai.google.dev/gemini-api/docs/prompting-strategies
_GOOGLE_PROMPT = DocumentTemplate(
    "prompt",
    {
        "Role": "Who the model is: persona, identity, and qualities.",
        "Instructions": "The numbered workflow the model should follow (plan, execute, validate).",
        "Constraints": "Behavioral limits and requirements the output must satisfy (verbosity, tone, what to avoid).",
        "Output Format": "The exact shape of the final answer (structure, fields, length, style).",
        "Context": "Documents, code, and background data, marked as data rather than instructions.",
        "Task": "The specific request the model must act on.",
        "Final Instruction": "Closing reminder placed at the very end (e.g. to think step by step).",
    },
)

# Alibaba Cloud Model Studio's prompt-engineering guide prescribes a six-part
# prompt framework for Qwen models: Context / Objective / Style / Tone /
# Audience / Response -- task background first, output format last. The guide
# leaves the framework unnamed; the identical structure is known in the
# community as CO-STAR.
# Source: https://www.alibabacloud.com/help/en/model-studio/prompt-engineering-guide
_ALIBABA_PROMPT = DocumentTemplate(
    "prompt",
    {
        "Context": "Background information closely related to the task.",
        "Objective": "The specific task the model must complete.",
        "Style": "The writing style the output should follow.",
        "Tone": "The tone the output should carry (e.g. formal, humorous, warm).",
        "Audience": "The target readers of the output.",
        "Response": "The exact form and format of the output.",
    },
)

# Template families: kind -> template, per target-model family. The "generic"
# family is the papers-grounded schema above; the provider families rename and
# reorder the prompt sections to the provider's own prompting guidance. The
# skill kind mirrors the agent-skill document shape and is provider-invariant,
# so every family shares it. Providers whose guidance prescribes no prompt
# structure (Meta, xAI, DeepSeek, Mistral, Moonshot, and Zhipu among them) get
# no family; their models use "generic".
TEMPLATE_FAMILIES: dict[str, dict[str, DocumentTemplate]] = {
    "generic": TEMPLATES,
    "openai": {"prompt": _OPENAI_PROMPT, "skill": TEMPLATES["skill"]},
    "openai-gpt-5.6": {"prompt": _OPENAI_GPT56_PROMPT, "skill": TEMPLATES["skill"]},
    "anthropic": {"prompt": _ANTHROPIC_PROMPT, "skill": TEMPLATES["skill"]},
    "google": {"prompt": _GOOGLE_PROMPT, "skill": TEMPLATES["skill"]},
    "alibaba": {"prompt": _ALIBABA_PROMPT, "skill": TEMPLATES["skill"]},
}


def infer_template_family(model: str | None) -> str:
    """Infer the template family whose prompting guidance covers ``model``.

    The match is a substring test on the (LiteLLM-style) model identifier:
    Claude models map to ``"anthropic"``, Gemini/Gemma to ``"google"``, GPT-5.6
    models to their model-specific ``"openai-gpt-5.6"``, other GPT and o-series
    models to ``"openai"``, Qwen/QwQ to ``"alibaba"``. Everything else maps to
    ``"generic"`` -- the providers behind Muse, Llama, Grok, DeepSeek, Mistral,
    Kimi, and GLM models prescribe no prompt structure to mirror.

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
    if re.search(r"gpt[-_]?5\.6", lowered):
        return "openai-gpt-5.6"
    if "gpt" in lowered or lowered.startswith("openai/") or re.search(r"(?:^|/)o\d+(?:$|[-.])", lowered):
        return "openai"
    if "qwen" in lowered or "qwq" in lowered:
        return "alibaba"
    return "generic"


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
