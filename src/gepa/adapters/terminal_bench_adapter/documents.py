"""Define and render the reusable documents optimized for Terminus.

Candidate text is data. Tool instructions are editable, while runtime fields,
the parser, and actual command behavior remain outside the optimization target.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path

from gepa.strategies.document_template import TEMPLATE_FAMILIES

BUNDLE_VERSION = 2
SKILLS_PATH = "/opt/gepa-skills"
COMMAND_FORMAT_SEED = r"""Return a JSON object with required fields "analysis" (string), "plan" (string), and "commands" (array).
Each command has "keystrokes" (string) and optional "duration" (seconds, default 1.0, capped at 60).
Keystrokes are sent verbatim to tmux. End shell commands with \n; use C-c for Ctrl+C and C-d for Ctrl+D.
An empty commands array is allowed. Optional "task_complete" (boolean, default false) requests completion; a second confirmation ends the task.
Example: {"analysis": "...", "plan": "...", "commands": [{"keystrokes": "ls -la\n", "duration": 0.1}], "task_complete": false}
"""
PROMPT_SEEDS = {
    "instruction_prompt": "Complete the assigned command-line task in the provided Linux terminal. Inspect the environment, make the required changes, and verify the result.",
    "terminal_tool": "Use the tmux terminal to inspect files, execute commands, and read their output. Choose short waits for quick commands and poll when work is still running. Explain what you learned and what you will do next.",
    "command_format": COMMAND_FORMAT_SEED,
    "skill_discovery": "Read the descriptions of the available skills. When a skill is relevant, read its SKILL.md with the terminal before applying it. Skills provide reusable procedures; the task description specifies the required result.",
    "summary": "Summarize the work so another agent can continue. Include completed actions, evidence, important paths, errors and attempted fixes, current state, and remaining work. Preserve information needed to avoid repeating failed approaches.",
    "summary_questions": "Read the handoff summary and ask questions about missing information needed to finish the task. The previous agent can answer now, but will not be available after the handoff.",
    "summary_answers": "Answer the next agent's questions using the execution history. Distinguish observed facts from assumptions and identify anything still unknown.",
    "handoff": "Continue the task using the summary and answers. You can no longer question the previous agent. Use the terminal interface to inspect the current state and finish the work.",
    "short_summary": "State the next steps in two or three sentences using the task and current terminal state.",
    "context_recovery": "Continue working from the available task, summary, and terminal state. Recheck missing details in the environment before relying on them.",
    "completion": 'Re-read the task requirements and verify the delivered files or behavior before confirming completion. Check for unintended changes and unresolved errors. Confirm with "task_complete": true to end the task and run the verifier. No further corrections are possible after confirmation.',
    "timeout": "Check whether the previous command is still running or waiting for interactive input. Wait or send the appropriate keystrokes based on the current terminal state.",
    "parse_error": "Use the parser feedback to correct the response. Return a valid command response in the required format.",
    "output_limit": "Split the requested work into smaller responses that fit the output limit. Reissue the commands that were not executed.",
}
SKILL_SEEDS = {
    "skill_debugging": {
        "Name": "terminal-debugging",
        "Description": "Investigate a failing command, program, or service before changing it.",
        "Instructions": "Reproduce the failure with a targeted command. Inspect relevant files, logs, and configuration. Make a change supported by the evidence, then repeat the failing check. Keep track of unsuccessful attempts.",
    },
    "skill_verification": {
        "Name": "terminal-verification",
        "Description": "Verify that the requested artifact or behavior has been delivered.",
        "Instructions": "Translate the task into observable checks. Run the relevant tests or exercise the requested workflow. Inspect the outputs and confirm that the final result satisfies the task requirements.",
    },
}
COMPONENT_KINDS = {**dict.fromkeys(PROMPT_SEEDS, "user_prompt"), **dict.fromkeys(SKILL_SEEDS, "skill")}
INITIAL_COMPONENTS = ("instruction_prompt", "terminal_tool", "skill_discovery", "command_format")

TASK_FIELDS = """Task Description:
{instruction}

Current terminal state:
{terminal_state}
"""
CONTEXT_FIELDS = {
    "summary": "Original task:\n{original_instruction}",
    "summary_questions": "Original task:\n{original_instruction}\n\nSummary:\n{summary}\n\nCurrent terminal state:\n{terminal_state}",
    "summary_answers": "Questions:\n{questions}",
    "handoff": "Answers from the previous agent:\n{answers}",
    "short_summary": "Original task:\n{original_instruction}\n\nCurrent terminal state:\n{terminal_state}",
    "context_recovery": "Original task:\n{original_instruction}\n\nSummary:\n{summary}\n\nCurrent terminal state:\n{terminal_state}",
    "completion": "Current terminal state:\n{terminal_state}",
    "timeout": "Previous command:\n{command}\n\nThe command timed out after {timeout_sec} seconds.\n\nCurrent terminal state:\n{terminal_state}",
    "parse_error": "",
    "output_limit": "No requested actions were performed because the response exceeded {limit_str}.\n{warnings_text}",
}


def seed_documents(template_family: str) -> dict[str, str]:
    """Create the same document bundle for every optimization condition.

    Args:
        template_family: Resolved provider family for prompt section names.

    Returns:
        Structured prompts and skill documents with stable component names.
    """
    template = TEMPLATE_FAMILIES[template_family]["user_prompt"]
    task_section = {
        "generic": "Task",
        "openai": "Input",
        "anthropic": "Instructions",
        "google": "Task",
        "alibaba": "Objective",
    }[template_family]
    candidate = {name: template.render({task_section: text}) for name, text in PROMPT_SEEDS.items()}
    skill_template = TEMPLATE_FAMILIES[template_family]["skill"]
    candidate.update({name: skill_template.render(sections) for name, sections in SKILL_SEEDS.items()})
    return candidate


def validate_documents(candidate: Mapping[str, str]) -> None:
    """Require one text value for every runtime document.

    Args:
        candidate: Proposed bundle to validate.

    Raises:
        ValueError: Components are missing, unknown, or not strings.
    """
    if set(candidate) != set(COMPONENT_KINDS):
        raise ValueError(
            f"Terminal Bench requires the complete document bundle; missing={sorted(set(COMPONENT_KINDS) - set(candidate))}, unknown={sorted(set(candidate) - set(COMPONENT_KINDS))}"
        )
    if any(not isinstance(text, str) for text in candidate.values()):
        raise ValueError("Terminal Bench document values must be strings")


def document_digest(candidate: Mapping[str, str]) -> str:
    """Hash every candidate document independently of mapping order.

    Args:
        candidate: Complete candidate bundle.

    Returns:
        SHA-256 digest of its canonical JSON representation.
    """
    validate_documents(candidate)
    return hashlib.sha256(json.dumps(dict(candidate), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def escape_document(text: str) -> str:
    """Keep candidate braces literal during runtime field substitution.

    Args:
        text: Candidate-authored text.

    Returns:
        Text safe to concatenate with a fixed Python format template.
    """
    return text.replace("{", "{{").replace("}", "}}")


def render_initial_instructions(candidate: Mapping[str, str]) -> str:
    """Compose one initial instruction block shared by both optimization scopes.

    Guidance documents use nested headings so their repeated provider section
    names do not become duplicate sections in the unified editable prompt.
    Only heading depth changes; component bodies and literal braces are retained.
    """
    validate_documents(candidate)
    return "\n\n".join(
        candidate[name]
        if name == "instruction_prompt"
        else re.sub(r"^## ", "### ", candidate[name], flags=re.MULTILINE)
        for name in INITIAL_COMPONENTS
        if candidate[name].strip()
    )


def render_instruction(candidate: Mapping[str, str]) -> str:
    """Assemble editable instructions and tool documentation with runtime inputs.

    Args:
        candidate: Complete reusable document bundle.

    Returns:
        Terminus template with task and terminal-state fields preserved.
    """
    initial = render_initial_instructions(candidate)
    return "\n\n".join(part for part in (escape_document(initial), TASK_FIELDS) if part)


def write_document_bundle(directory: Path, candidate: Mapping[str, str]) -> Path:
    """Materialize an isolated candidate for Harbor's separate interpreter.

    Args:
        directory: New evaluation directory.
        candidate: Complete documents from GEPA.

    Returns:
        Bundle JSON path containing all prompts and skill metadata.
    """
    validate_documents(candidate)
    (directory / "terminus-prompt.txt").write_text(render_instruction(candidate), encoding="utf-8")
    prompts = {
        name: "\n\n".join(part for part in (escape_document(candidate[name]), context) if part)
        for name, context in CONTEXT_FIELDS.items()
    }
    (directory / "timeout.txt").write_text(prompts["timeout"], encoding="utf-8")
    skills = []
    for component in SKILL_SEEDS:
        # Vanilla may produce free-form text; both methods use the same renderer.
        text = candidate[component]
        sections = dict.fromkeys(("Name", "Description", "Instructions", "Examples"), "")
        active = "Instructions"
        for line in text.splitlines(keepends=True):
            heading = line.rstrip().removeprefix("## ") if line.startswith("## ") else None
            if heading in sections:
                active = heading
            else:
                sections[active] += line
        name = sections["Name"].strip() or component
        description = sections["Description"].strip() or f"Reusable guidance in {component}."
        skill_dir = directory / "skills" / component
        skill_dir.mkdir(parents=True)
        frontmatter = f"---\nname: {json.dumps(name, ensure_ascii=False)}\ndescription: {json.dumps(description, ensure_ascii=False)}\n---\n\n"
        (skill_dir / "SKILL.md").write_text(frontmatter + text, encoding="utf-8")
        skills.append(
            {
                "component": component,
                "name": name,
                "description": description,
                "path": f"{SKILLS_PATH}/{component}/SKILL.md",
            }
        )
    bundle = {
        "version": BUNDLE_VERSION,
        "digest": document_digest(candidate),
        "documents": dict(candidate),
        "prompts": prompts,
        "skills": skills,
    }
    path = directory / "document-bundle.json"
    path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
