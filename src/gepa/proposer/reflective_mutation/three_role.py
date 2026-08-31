# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Route reflection through the Controller, Manifestor, and proposer.

The strategy changes only reflective mutation. GEPA's evaluator, Pareto search,
acceptance, and merge behavior remain unchanged. Reflection level 0 delegates
to vanilla GEPA. Level 1 selects a document region and lets ReAct V2 operate
over the configured edit basis. Level 2 also selects a semantic action and uses
the Manifestor to steer ReAct V2.
"""

from __future__ import annotations

import json
import math
import os
import random
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from gepa.proposer.reflective_mutation.base import LanguageModel
from gepa.proposer.reflective_mutation.manifestor import (
    MAX_TRACES_CHARS,
    ManifestationError,
    Manifestor,
)
from gepa.proposer.reflective_mutation.react_v2_proposer import ReActV2Proposer
from gepa.proposer.reflective_mutation.reflection_lm import (
    ReflectionJob,
    ReflectionProposal,
    StatelessReflectionLM,
)
from gepa.response_journal import stable_api_base_identity
from gepa.strategies.action_space import MAX_PROPOSAL_CHARS, IncompleteActionDistributionError
from gepa.strategies.document_template import TEMPLATE_FAMILIES, DocumentTemplate, MalformedDocumentError
from gepa.strategies.edit_tools import EDIT_TOOL_SETS
from gepa.strategies.intervention import (
    CONTROLLER_POLICY_CONTRACT,
    SEMANTIC_ACTION_CATALOGS,
    UNIFORM_RANDOM_CONTROLLER_POLICY_CONTRACT,
    Controller,
    ControllerChoice,
    build_controller_menu,
    summarize_feedback,
)

MAX_HISTORY_TEXT_CHARS = 2000
MAX_HISTORY_STEPS = 16
MAX_HISTORY_EDIT_ENTRIES = 32
REFLECTION_RUN_CONTRACT_FILENAME = "reflection-run-contract.json"
_CONTROLLER_SELECTIONS = ("verbalized", "uniform_random")
_SENSITIVE_CONFIG_KEYS = {
    "access_token",
    "api_key",
    "apikey",
    "api_token",
    "authorization",
    "auth_token",
    "azure_ad_token",
    "bearer_token",
    "credential",
    "credentials",
    "password",
    "private_key",
    "secret",
    "secret_key",
    "token",
}


def _is_sensitive_config_key(key: str) -> bool:
    """Classify a configuration key as authentication material.

    Args:
        key: Configuration key, matched case-insensitively.

    Returns:
        Whether the key is sensitive itself or ends in a sensitive suffix.
    """
    lowered = key.lower()
    return lowered in _SENSITIVE_CONFIG_KEYS or any(lowered.endswith(f"_{suffix}") for suffix in _SENSITIVE_CONFIG_KEYS)


def _public_run_identity_value(value: Any) -> Any:
    """Convert configuration to stable public JSON data.

    Mappings and sequences are normalized recursively, credential values are
    redacted, and unsupported runtime objects are represented by type rather
    than potentially secret or unstable string content.

    Args:
        value: Configuration value to normalize.

    Returns:
        JSON-compatible public representation of ``value``.
    """
    if isinstance(value, Mapping):
        public: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            if _is_sensitive_config_key(key):
                public[key] = "<redacted>"
            elif key == "api_base" and isinstance(item, str):
                public[key] = stable_api_base_identity(item)
            else:
                public[key] = _public_run_identity_value(item)
        return public
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_public_run_identity_value(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    value_type = type(value)
    return f"<{value_type.__module__}.{value_type.__qualname__}>"


def _language_model_run_identity(lm: LanguageModel, explicit: Mapping[str, Any] | None) -> dict[str, Any]:
    """Describe one role LM without credentials or runtime counters.

    Args:
        lm: Controller, Manifestor, or proposer language model.
        explicit: Stable caller-supplied configuration, or ``None`` to infer
            conventional model fields from ``lm``.

    Returns:
        Public type and configuration identity with a source label indicating
        whether it was explicit, inferred, partial, or opaque.
    """
    lm_type = type(lm)
    identity: dict[str, Any] = {"type": f"{lm_type.__module__}.{lm_type.__qualname__}"}
    if explicit is not None:
        identity["configuration"] = _public_run_identity_value(explicit)
        identity["configuration_source"] = "explicit"
        return identity

    model = getattr(lm, "model", None)
    if isinstance(model, str):
        identity["model"] = model
    completion_kwargs = getattr(lm, "completion_kwargs", None)
    if isinstance(completion_kwargs, Mapping):
        identity["completion_kwargs"] = _public_run_identity_value(completion_kwargs)
    num_retries = getattr(lm, "num_retries", None)
    if isinstance(num_retries, int) and not isinstance(num_retries, bool):
        identity["num_retries"] = num_retries
    if isinstance(model, str) and isinstance(completion_kwargs, Mapping):
        identity["configuration_source"] = "inferred"
    elif len(identity) > 1:
        identity["configuration_source"] = "partial"
    else:
        identity["configuration_source"] = "opaque"
    return identity


def ensure_reflection_run_contract(run_dir: str, contract: Mapping[str, Any]) -> str:
    """Persist a reflection contract and reject incompatible resume state.

    Args:
        run_dir: GEPA state directory.
        contract: JSON-serializable reflection strategy identity.

    Returns:
        Path to the validated contract file.

    Raises:
        ValueError: The directory contains a different contract or legacy state
            without a reflection contract.
    """
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, REFLECTION_RUN_CONTRACT_FILENAME)
    normalized = json.loads(json.dumps(dict(contract), sort_keys=True, default=str))
    if os.path.exists(path):
        with open(path) as file:
            existing = json.load(file)
        if existing != normalized:
            raise ValueError(f"Run directory {run_dir} contains a different reflection strategy contract.")
        return path
    if os.path.exists(os.path.join(run_dir, "gepa_state.bin")):
        raise ValueError(
            f"Run directory {run_dir} has GEPA state but no {REFLECTION_RUN_CONTRACT_FILENAME}; "
            "choose a clean directory."
        )
    with open(path, "w") as file:
        json.dump(normalized, file, indent=2, sort_keys=True)
        file.write("\n")
    return path


def _bounded_history_text(value: Any) -> str | None:
    """Render one optional history field within its persistent text bound.

    Args:
        value: Field value to stringify, or ``None`` when absent.

    Returns:
        Original string representation, a length-marked prefix, or ``None``.
    """
    if value is None:
        return None
    text = str(value)
    if len(text) <= MAX_HISTORY_TEXT_CHARS:
        return text
    return text[:MAX_HISTORY_TEXT_CHARS] + f"...(+{len(text) - MAX_HISTORY_TEXT_CHARS} chars)"


def _react_chat_messages(steps: Sequence[Any]) -> list[dict[str, str]]:
    """Convert actual ReAct turns into persistent chat messages.

    Args:
        steps: ReAct steps carrying assistant output, action, and observation.

    Returns:
        Assistant messages and non-finish user observations in turn order.
    """
    messages: list[dict[str, str]] = []
    for step in steps:
        assistant = str(step.assistant) if step.assistant is not None else None
        observation = str(step.observation) if step.observation is not None else None
        if assistant:
            messages.append({"role": "assistant", "content": assistant})
        if observation and step.action != "FINISH":
            messages.append({"role": "user", "content": observation})
    return messages


def _controller_sampling_record(history: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize Controller sampling provenance as JSON primitives.

    Args:
        history: Raw selector-history record for one Controller call.

    Returns:
        Distribution, sampled propensities, fallback flags, and policy metrics
        with stable primitive types.
    """
    probabilities = history.get("probs", {})
    sampling_probabilities = history.get("sampling_probs", {})
    sampled = history.get("sampled", [])
    return {
        "probs": {str(name): float(probability) for name, probability in dict(probabilities).items()},
        "sampling_probs": {str(name): float(probability) for name, probability in dict(sampling_probabilities).items()},
        "sampled": [str(name) for name in sampled],
        "sampled_probabilities": [float(value) for value in history.get("sampled_probabilities", [])],
        "fallback": bool(history.get("fallback", False)),
        "n_parsed_entries": int(history.get("n_parsed_entries", 0)),
        "tail_mass": float(history.get("tail_mass", 0.0)),
        "tau": float(history.get("tau", 0.0)),
        "sampling_policy": str(history.get("sampling_policy", "tail")),
        "exploration_epsilon": float(history.get("exploration_epsilon", 0.0)),
        "used_full_fallback": bool(history.get("used_full_fallback", False)),
        "entropy_bits": float(history.get("entropy_bits", 0.0)),
    }


def _joint_controller_sampling_record(history: Mapping[str, Any]) -> dict[str, Any]:
    """Persist one joint region/action decision and its propensity.

    Args:
        history: Raw selector-history record for a level-2 Controller call.

    Returns:
        Normalized Controller record labeled with the joint policy and the
        selected pair's sampling probability.
    """
    record = _controller_sampling_record(history)
    return {
        **record,
        "policy": "joint_region_action_v4",
        "joint_sampling_probability": record["sampled_probabilities"][0],
    }


def _uniform_controller_sampling_record(
    menu: Sequence[ControllerChoice],
    action: ControllerChoice,
    level: int,
) -> dict[str, Any]:
    """Persist a uniform Controller draw over the complete visible menu.

    Args:
        menu: Controller choices available for this component.
        action: Choice drawn from ``menu`` by the strategy RNG.
        level: Reflection level that determines the policy label.

    Returns:
        Full uniform distribution, sampled propensity, and policy identity.

    Raises:
        ValueError: ``menu`` is empty or ``action`` is not one of its choices.
    """
    if not menu:
        raise ValueError("Uniform Controller selection requires a non-empty menu.")
    if action not in menu:
        raise ValueError("The sampled Controller action must belong to the visible menu.")
    probability = 1.0 / len(menu)
    probabilities = {choice.menu_id: probability for choice in menu}
    record = {
        "probs": probabilities,
        "sampling_probs": dict(probabilities),
        "sampled": [action.menu_id],
        "sampled_probabilities": [probability],
        "fallback": False,
        "n_parsed_entries": 0,
        "tail_mass": 0.0,
        "tau": 0.0,
        "sampling_policy": "uniform",
        "exploration_epsilon": 0.0,
        "used_full_fallback": False,
        "entropy_bits": math.log2(len(menu)),
        "policy": "joint_region_action_uniform_v1" if level >= 2 else "region_uniform_v1",
    }
    if level >= 2:
        record["joint_sampling_probability"] = probability
    return record


def _summarize_traces(entries: Sequence[Mapping[str, Any]]) -> str:
    """Flatten reflective rows into the execution evidence shown to the roles.

    Args:
        entries: Reflective-dataset rows with inputs, outputs, and feedback.

    Returns:
        One labeled block per example, or a no-traces marker.
    """
    blocks: list[str] = []
    for index, entry in enumerate(entries):
        inputs = entry.get("Inputs")
        outputs = entry.get("Generated Outputs", entry.get("Generated Output"))
        feedback = entry.get("Feedback") or entry.get("execution_feedback")
        blocks.append(f"[example {index + 1}]\nInputs: {inputs}\nOutput: {outputs}\nFeedback: {feedback}")
    return "\n\n".join(blocks) or "(no traces available)"


def _tracking_id(action: ControllerChoice) -> str:
    """Return the action-diversity bucket for one Controller choice.

    Args:
        action: Selected region and optional semantic action.

    Returns:
        Semantic action name at level 2, or ``"edit:<region>"`` at level 1.
    """
    if action.semantic_action is not None:
        return action.semantic_action.name
    return f"edit:{action.edit_target.section}"


def _branch_history(metadata: Mapping[str, Any] | None, edit_target: str) -> list[dict[str, str]]:
    """Return chat history recorded for one exact component and section.

    Args:
        metadata: Per-job context generated from the selected parent candidate.
        edit_target: ``"<component>:<section>"`` label selected by the Controller.

    Returns:
        User/assistant transcript for accepted, rejected, and dropped attempts
        on this target in the parent branch. Unscoped legacy messages and
        sibling-section records are not replayed.

    Raises:
        TypeError: The history or message content has the wrong type.
        ValueError: A message has extra fields or a non-chat role.
    """
    if metadata is None:
        return []
    value = metadata.get("branch_edit_history", [])
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TypeError("branch_edit_history must be a sequence of revision mappings.")
    history: list[dict[str, str]] = []
    pending: list[dict[str, str]] = []
    target_marker = f"Edit target: {edit_target}."
    for message in value:
        if not isinstance(message, Mapping):
            raise TypeError("Every branch_edit_history entry must be a mapping.")
        if set(message) != {"role", "content"}:
            raise ValueError("Every branch_edit_history entry must contain only 'role' and 'content'.")
        role = message["role"]
        content = message["content"]
        if role not in {"user", "assistant"}:
            raise ValueError("Every branch_edit_history role must be 'user' or 'assistant'.")
        if not isinstance(content, str):
            raise TypeError("Every branch_edit_history content value must be a string.")
        pending.append({"role": role, "content": content})
        if role == "user" and content.startswith("Optimizer result: "):
            if target_marker in content:
                history.extend(pending)
            pending = []
    return history


class ThreeRoleReflectionLM:
    """Controller/Manifestor reflection with a ReAct V2 proposer.

    Args:
        base_lm: Reflection model used by ReAct V2 and by verbalized Controller
            selection.
        level: ``0`` vanilla GEPA, ``1`` region plus edit basis, or ``2`` region
            plus semantic action and Manifestor steering.
        edit_tool_set: ``"minimal"`` for insert/delete or ``"broad"`` for all
            insert/delete/replace/move tools.
        component_kinds: Component-to-template mapping using ``system_prompt``,
            ``user_prompt``, ``skill``, or a custom template key. Conventional
            component names resolve to their matching key; all other unlisted
            components default to ``system_prompt``.
        template_family: Canonical provider template family.
        templates: Optional per-component-type template overrides.
        k: Verbalized-sampling distribution size at level 1. Level 2 scores
            every joint region/action option in one Controller call.
        tau: Tail-sampling threshold.
        controller_selection: ``"verbalized"`` for LM-ranked selection or
            ``"uniform_random"`` for the clean random-Controller ablation.
        rng: Seeded random stream. When omitted, GEPA binds the engine RNG.
            An explicit RNG remains independent of engine sampling.
        logger: Optional run logger shared by all roles.
        reflection_prompt_template: Vanilla level-0 prompt template.
        max_menu: Optional level-1 region bound. Level 2 requires it to retain
            every cataloged region/action pair; semantic choices are never subsampled.
        max_chars: Maximum completed component size.
        manifestor_lm: Deterministic LM used to manifest level-2 actions.
        base_lm_run_identity: Optional stable, non-secret configuration identity
            for a custom Controller/ReAct callable.
        manifestor_lm_run_identity: Optional stable, non-secret configuration
            identity for a custom Manifestor callable.
        manifestor_traces_chars: Trace budget for the Manifestor.
        proposer_model: Provider/model identifier recorded in the run contract.
            When omitted, ``base_lm.model`` is inspected. Manifestor steering is
            delivered as a user message for every ReAct provider.
        react_max_iterations: Maximum ReAct assistant turns per component.
        react_max_tool_calls: Maximum valid calls in an atomic-basis proposal.

    Raises:
        ValueError: Configuration names or reflection level are invalid.
    """

    def __init__(
        self,
        base_lm: LanguageModel,
        level: int,
        *,
        edit_tool_set: str = "broad",
        component_kinds: dict[str, str] | None = None,
        template_family: str = "generic",
        templates: Mapping[str, DocumentTemplate] | None = None,
        k: int = 5,
        tau: float | None = None,
        controller_selection: str = "verbalized",
        rng: random.Random | None = None,
        logger: Any | None = None,
        reflection_prompt_template: str | dict[str, str] | None = None,
        max_menu: int | None = None,
        max_chars: int = MAX_PROPOSAL_CHARS,
        manifestor_lm: LanguageModel | None = None,
        base_lm_run_identity: Mapping[str, Any] | None = None,
        manifestor_lm_run_identity: Mapping[str, Any] | None = None,
        manifestor_traces_chars: int | None = MAX_TRACES_CHARS,
        proposer_model: str | None = None,
        react_max_iterations: int = 8,
        react_max_tool_calls: int = 4,
    ):
        """Validate and store the complete three-role strategy configuration.

        Args:
            base_lm: ReAct V2 model, also used for verbalized Controller selection.
            level: Reflection level: vanilla, region-only, or region/action.
            edit_tool_set: Named atomic or broad execution basis.
            component_kinds: Optional component-to-template-kind overrides.
            template_family: Provider family supplying default templates.
            templates: Template-kind overrides merged into the family defaults.
            k: Number of Controller samples below level 2.
            tau: Optional verbalized-sampling tail-mass threshold.
            controller_selection: Controller selection policy. Uniform random
                selection draws once from the same section/action menu and does
                not call the Controller LM.
            rng: Seeded strategy RNG. When ``None``, GEPA replaces the
                deterministic default with the engine RNG at wiring time.
            logger: Optional run logger shared by all roles.
            reflection_prompt_template: Vanilla level-0 reflection template.
            max_menu: Optional level-1 region-menu bound.
            max_chars: Maximum reconstructed component length.
            manifestor_lm: Separate Manifestor model, or ``None`` to share the
                base model.
            base_lm_run_identity: Stable public identity for a custom base model.
            manifestor_lm_run_identity: Stable public identity for a custom
                Manifestor model.
            manifestor_traces_chars: Maximum trace characters shown to the
                Manifestor.
            proposer_model: Model identifier persisted in the run contract.
            react_max_iterations: Maximum ReAct assistant turns per proposal.
            react_max_tool_calls: Maximum valid calls in an atomic ReAct path.

        Raises:
            ValueError: A level, tool set, Controller selection, template
                family, or component kind is invalid.
        """
        if level not in (0, 1, 2):
            raise ValueError(f"reflection level must be 0, 1, or 2; got {level}")
        if edit_tool_set not in EDIT_TOOL_SETS:
            raise ValueError(f"edit_tool_set must be one of {sorted(EDIT_TOOL_SETS)}; got {edit_tool_set!r}")
        if controller_selection not in _CONTROLLER_SELECTIONS:
            raise ValueError(
                f"controller_selection must be one of {list(_CONTROLLER_SELECTIONS)}; got {controller_selection!r}"
            )
        if level == 0 and controller_selection != "verbalized":
            raise ValueError("controller_selection must be 'verbalized' when reflection level is 0")
        if template_family not in TEMPLATE_FAMILIES:
            raise ValueError(f"template_family must be one of {sorted(TEMPLATE_FAMILIES)}; got {template_family!r}")
        self.templates: dict[str, DocumentTemplate] = {**TEMPLATE_FAMILIES[template_family], **(templates or {})}
        for name, kind in (component_kinds or {}).items():
            if kind not in self.templates:
                raise ValueError(f"component_kinds[{name!r}] must be one of {sorted(self.templates)}; got {kind!r}")

        self.base_lm = base_lm
        self.level = level
        self.edit_tool_set = edit_tool_set
        self.edit_tools = EDIT_TOOL_SETS[edit_tool_set]
        self.component_kinds = component_kinds or {}
        self.template_family = template_family
        self.k = k
        self.tau = tau
        self.controller_selection = controller_selection
        self._rng_explicit = rng is not None
        self.rng = rng if rng is not None else random.Random(0)
        self.logger = logger
        self.reflection_prompt_template = reflection_prompt_template
        self.max_menu = max_menu
        self.max_chars = max_chars
        self.manifestor_lm = manifestor_lm if manifestor_lm is not None else base_lm
        self.base_lm_run_identity = base_lm_run_identity
        self.manifestor_lm_run_identity = (
            base_lm_run_identity
            if manifestor_lm is None and manifestor_lm_run_identity is None
            else manifestor_lm_run_identity
        )
        self.manifestor_traces_chars = manifestor_traces_chars
        inferred_model = proposer_model
        if inferred_model is None:
            model_attribute = getattr(base_lm, "model", None)
            inferred_model = model_attribute if isinstance(model_attribute, str) else None
        self.proposer_model = inferred_model
        self.react_max_iterations = react_max_iterations
        self.react_max_tool_calls = react_max_tool_calls
        self._stateless: StatelessReflectionLM | None = (
            StatelessReflectionLM(base_lm, reflection_prompt_template, logger, rng=self.rng) if level == 0 else None
        )

    def _component_kind(self, name: str) -> str:
        """Resolve a candidate component to its role-specific template key.

        Args:
            name: Candidate component name.

        Returns:
            Explicit mapping, matching registered key, or ``system_prompt``.
        """
        if name in self.component_kinds:
            return self.component_kinds[name]
        if name in self.templates:
            return name
        return "system_prompt"

    def run_contract(self, candidate: Mapping[str, str]) -> dict[str, Any]:
        """Return the complete JSON-serializable three-role strategy identity.

        Args:
            candidate: Seed component mapping used to resolve default document
                kinds alongside explicit ``component_kinds``.

        Returns:
            Contract whose drift must prevent state resumption.

        Raises:
            MalformedDocumentError: A candidate component is not in its
                canonical template format.
            ValueError: A component kind lacks a level-2 catalog or a custom LM
                has no stable explicit or inferable identity.
        """
        self.validate_candidate(dict(candidate))
        component_kinds = {name: self._component_kind(name) for name in candidate}
        active_kinds = sorted(set(component_kinds.values()))
        templates = {
            kind: {
                "document_kind": self.templates[kind].kind,
                "sections": list(self.templates[kind].sections.items()),
            }
            for kind in active_kinds
        }
        controller: dict[str, Any]
        if self.level >= 2 and self.controller_selection == "uniform_random":
            controller = {
                **UNIFORM_RANDOM_CONTROLLER_POLICY_CONTRACT,
                "max_menu": self.max_menu,
            }
        elif self.level >= 2:
            controller = {
                **CONTROLLER_POLICY_CONTRACT,
                "tau": self.tau,
                "max_menu": self.max_menu,
            }
        elif self.controller_selection == "uniform_random":
            controller = {
                "version": 1,
                "factorization": "region_only",
                "selection": "uniform_random",
                "sampling": "uniform over all candidates",
                "context": "none",
                "max_menu": self.max_menu,
            }
        else:
            controller = {
                "version": 1,
                "factorization": "region_only",
                "k": self.k,
                "tau": self.tau,
                "max_menu": self.max_menu,
            }
        controller_lm_identity = _language_model_run_identity(self.base_lm, self.base_lm_run_identity)
        manifestor_lm_identity = (
            _language_model_run_identity(self.manifestor_lm, self.manifestor_lm_run_identity)
            if self.level >= 2
            else None
        )
        unstable_roles = [
            role
            for role, identity in (
                ("Controller/Proposer", controller_lm_identity),
                ("Manifestor", manifestor_lm_identity),
            )
            if identity is not None and identity["configuration_source"] in {"opaque", "partial"}
        ]
        if unstable_roles:
            roles = " and ".join(unstable_roles)
            raise ValueError(
                f"A stable run identity is required for the {roles} LM. Pass base_lm_run_identity and/or "
                "manifestor_lm_run_identity when constructing ThreeRoleReflectionLM with custom callables."
            )
        return {
            "schema_version": 4,
            "strategy": "three_role_reflection",
            "reflection_level": self.level,
            "edit_tool_set": self.edit_tool_set,
            "edit_tools": [tool.value for tool in self.edit_tools],
            "component_kinds": component_kinds,
            "template_family": self.template_family,
            "templates": templates,
            "reflection_prompt_template": self.reflection_prompt_template,
            "controller": controller,
            "semantic_action_spaces": (
                {
                    kind: deepcopy(SEMANTIC_ACTION_CATALOGS[self.templates[kind].kind])
                    for kind in active_kinds
                }
                if self.level >= 2
                else None
            ),
            "max_chars": self.max_chars,
            "manifestor_traces_chars": self.manifestor_traces_chars,
            "manifestor_delivery": "user_message",
            "branch_history": {
                "storage": "target_scoped_user_assistant_messages",
                "direct_deepseek_native_delivery": "quoted_user_context",
                "other_delivery": "provider_chat_messages",
            },
            "proposer_model": self.proposer_model,
            "proposer_backend": "react_v2",
            "controller_react_lm": controller_lm_identity,
            "manifestor_lm": manifestor_lm_identity,
            "max_proposer_model_calls": self.react_max_iterations,
            "react_max_iterations": self.react_max_iterations,
            "react_max_tool_calls": self.react_max_tool_calls,
        }

    def validate_candidate(self, candidate: dict[str, str]) -> None:
        """Validate every component against its declared document template.

        Args:
            candidate: Component mapping to validate.

        Raises:
            MalformedDocumentError: A component is not in canonical section format.
            ValueError: Level 2 has no semantic catalog for a component kind.
        """
        for name, text in candidate.items():
            template = self.templates[self._component_kind(name)]
            if self.level >= 2 and not SEMANTIC_ACTION_CATALOGS.get(template.kind, {}).get("actions"):
                raise ValueError(
                    f"Component {name!r} uses document kind {template.kind!r}, which has no level-2 semantic catalog."
                )
            try:
                parsed = template.parse(text)
                if template.render(parsed) != text:
                    raise MalformedDocumentError(
                        "Populated sections must use canonical spacing, and empty sections must be omitted."
                    )
            except MalformedDocumentError as exc:
                raise MalformedDocumentError(
                    f"Component {name!r} is not in the canonical {template.kind!r} section format required by "
                    f"reflection_level > 0: {exc} Convert it once with "
                    "gepa.strategies.document_template.migrate_document(text, template, lm)."
                ) from exc

    def bind_rng(self, rng: random.Random) -> None:
        """Bind GEPA's run RNG unless the caller supplied one explicitly.

        Sharing the engine stream preserves existing behavior when the strategy
        has no dedicated RNG. An explicit RNG keeps Controller sampling from
        perturbing GEPA's candidate, batch, and Pareto-selection stream.

        Args:
            rng: Run RNG.
        """
        if not self._rng_explicit:
            self.rng = rng
            if self._stateless is not None:
                self._stateless.bind_rng(rng)

    def get_state(self) -> dict[str, Any]:
        """Return the private Controller RNG state for exact resume.

        Returns:
            Serializable RNG snapshot. Branch-local user and assistant history
            remains in :class:`GEPAState` rather than this strategy object.
        """
        state = {"rng_state": self.rng.getstate()}
        return state

    def get_batch_retry_state(self) -> dict[str, Any]:
        """Snapshot role-local state before a batched reflection attempt.

        Returns:
            Controller RNG state and response-journal cursors for the shared
            Controller/ReAct model and the Manifestor model.
        """
        if self._stateless is not None:
            return self._stateless.get_batch_retry_state()
        state: dict[str, Any] = {"rng_state": self.rng.getstate()}
        base_cursor = getattr(self.base_lm, "response_journal_cursor_state", None)
        if callable(base_cursor):
            state["base_lm_cursor"] = base_cursor()
        if self.manifestor_lm is not self.base_lm:
            manifestor_cursor = getattr(self.manifestor_lm, "response_journal_cursor_state", None)
            if callable(manifestor_cursor):
                state["manifestor_lm_cursor"] = manifestor_cursor()
        return state

    def set_batch_retry_state(self, state: Mapping[str, Any]) -> None:
        """Restore role-local state before per-task reflection fallback.

        Args:
            state: Snapshot returned by :meth:`get_batch_retry_state`.

        Raises:
            TypeError: The RNG snapshot is malformed or a role cannot restore
                a recorded journal cursor.
        """
        if self._stateless is not None:
            self._stateless.set_batch_retry_state(state)
            return
        rng_state = state.get("rng_state")
        if not isinstance(rng_state, tuple):
            raise TypeError("ThreeRoleReflectionLM retry rng_state must be a tuple.")
        self.rng.setstate(rng_state)
        base_cursor = state.get("base_lm_cursor")
        if base_cursor is not None:
            restore = getattr(self.base_lm, "restore_response_journal_cursor_state", None)
            if not callable(restore):
                raise TypeError("Controller/ReAct LM cannot restore its response-journal cursor.")
            restore(base_cursor)
        manifestor_cursor = state.get("manifestor_lm_cursor")
        if manifestor_cursor is not None:
            restore = getattr(self.manifestor_lm, "restore_response_journal_cursor_state", None)
            if not callable(restore):
                raise TypeError("Manifestor LM cannot restore its response-journal cursor.")
            restore(manifestor_cursor)

    def set_state(self, state: Mapping[str, Any]) -> None:
        """Restore the Controller RNG from a durable optimizer checkpoint.

        Args:
            state: Snapshot previously returned by :meth:`get_state`.

        Raises:
            TypeError: ``rng_state`` is not a tuple accepted by
                :class:`random.Random`.
        """
        rng_state = state.get("rng_state")
        if not isinstance(rng_state, tuple):
            raise TypeError("Persisted ThreeRoleReflectionLM rng_state must be a tuple")
        self.rng.setstate(rng_state)
        if self._stateless is not None:
            self._stateless.bind_rng(self.rng)

    def bind_logger(self, logger: Any) -> None:
        """Bind the logger shared by every role.

        Args:
            logger: Object exposing ``log(message)``.
        """
        self.logger = logger
        if self._stateless is not None:
            self._stateless.logger = logger

    def bind_reflection_prompt_template(self, template: str | dict[str, str] | None) -> None:
        """Bind the level-0 reflection prompt template.

        Args:
            template: Global or per-component vanilla template.
        """
        self.reflection_prompt_template = template
        if self._stateless is not None:
            self._stateless.reflection_prompt_template = template

    def bind_lm_kwargs(self, _lm_kwargs: dict[str, Any] | None) -> None:
        """Satisfy the GEPA binding hook; model kwargs are already configured.

        Args:
            _lm_kwargs: Ignored model configuration.
        """

    @property
    def total_cost(self) -> float:
        """Return provider spend without double-counting a shared Manifestor LM.

        Returns:
            Combined tracked cost.
        """
        cost = float(getattr(self.base_lm, "total_cost", 0.0))
        if self.manifestor_lm is not self.base_lm:
            cost += float(getattr(self.manifestor_lm, "total_cost", 0.0))
        return cost

    def supports_cost_tracking(self) -> bool:
        """Report whether the base LM exposes provider spend.

        Returns:
            Whether ``base_lm.total_cost`` exists.
        """
        return hasattr(self.base_lm, "total_cost")

    def reflect(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[ReflectionProposal, ThreeRoleReflectionLM]:
        """Propose revisions while keeping context local to the selected branch.

        Args:
            candidate: Parent candidate.
            reflective_dataset: Per-component feedback and execution evidence.
            components_to_update: Components selected for mutation.
            metadata: Per-job context, including the branch-local chat transcript.

        Returns:
            Proposal and this strategy instance.
        """
        if self._stateless is not None:
            proposal, _ = self._stateless.reflect(candidate, reflective_dataset, components_to_update)
            return proposal, self
        return self._reflect_operated(candidate, reflective_dataset, components_to_update, metadata)

    def reflect_many(
        self,
        jobs: list[ReflectionJob],
        *,
        metadatas: Sequence[Mapping[str, Any] | None] | None = None,
    ) -> list[tuple[ReflectionProposal, ThreeRoleReflectionLM]]:
        """Reflect on independent jobs with index-aligned branch histories.

        Args:
            jobs: Candidate, reflective dataset, and component triples.
            metadatas: Per-job branch context. ``None`` supplies empty context.

        Returns:
            Results in job order.

        Raises:
            ValueError: Metadata length does not match job length.
        """
        contexts = list(metadatas) if metadatas is not None else [None] * len(jobs)
        if len(contexts) != len(jobs):
            raise ValueError(f"Expected {len(jobs)} metadata records; got {len(contexts)}")
        return [
            self.reflect(candidate, dataset, components, metadata=context)
            for (candidate, dataset, components), context in zip(jobs, contexts, strict=True)
        ]

    def _reflect_operated(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
        metadata: Mapping[str, Any] | None,
    ) -> tuple[ReflectionProposal, ThreeRoleReflectionLM]:
        """Run Controller, optional Manifestor, and ReAct V2 per component.

        Args:
            candidate: Parent candidate.
            reflective_dataset: Per-component run evidence.
            components_to_update: Components selected for mutation.
            metadata: Parent-specific chat history and iteration anchors.

        Returns:
            Reflection proposal and this strategy.

        """
        proposal = ReflectionProposal(new_texts={}, prompts={}, raw_lm_outputs={}, metadata={})
        records: list[dict[str, Any]] = []
        accepted_revisions: list[dict[str, Any]] = []
        controller_failures: list[dict[str, str]] = []
        dropped: list[str] = []

        for name in components_to_update:
            entries = reflective_dataset.get(name)
            if not entries:
                if self.logger is not None:
                    self.logger.log(f"Component '{name}' is not in reflective dataset. Skipping.")
                continue

            template = self.templates[self._component_kind(name)]
            text = candidate[name]
            feedback = summarize_feedback(entries)
            traces = _summarize_traces(entries)
            section_bodies = template.parse(text)
            # Sparse rendering keeps empty sections out of task-model messages.
            # The Controller still needs their occupancy to judge which semantic
            # actions have the text required by their coupled operators.
            controller_candidate = (
                "Controller-only section inventory. [EMPTY SECTION] is metadata, not document text. "
                "An empty region has no target bytes: assign probability 0 to its DELETE_TEXT, REPLACE_TEXT, and "
                "MOVE_TEXT choices. Judge its INSERT_TEXT choices by their semantic fit.\n\n"
                + "\n\n".join(
                    f"## {section}\n{body if body else '[EMPTY SECTION]'}" for section, body in section_bodies.items()
                )
            )
            menu = build_controller_menu(
                template,
                name,
                self.edit_tools,
                self.level,
                rng=self.rng,
                max_menu=self.max_menu,
            )
            if self.controller_selection == "uniform_random":
                action = self.rng.choice(menu)
                controller_sampling = _uniform_controller_sampling_record(menu, action, self.level)
            else:
                controller = Controller(
                    menu,
                    self.base_lm,
                    k=len(menu) if self.level >= 2 else self.k,
                    tau=self.tau,
                    rng=self.rng,
                    require_full_support=self.level >= 2,
                )
                try:
                    action = controller.select(
                        1,
                        self.rng,
                        candidate=controller_candidate,
                        feedback_summary=feedback,
                    )[0]
                except IncompleteActionDistributionError as exc:
                    error = _bounded_history_text(exc) or "Controller action distribution failed."
                    controller_failures.append({"component": name, "error": error})
                    dropped.append(name)
                    if self.logger is not None:
                        self.logger.log(f"Component {name!r} dropped after Controller failure: {error}")
                    continue
                if self.level >= 2:
                    controller_sampling = _joint_controller_sampling_record(controller.history[-1])
                else:
                    controller_sampling = _controller_sampling_record(controller.history[-1])
            preferred_edit_tool = action.edit_tool.value if action.edit_tool is not None else None
            semantic_action = action.semantic_action.name if action.semantic_action else None

            section = action.edit_target.section
            region_text = section_bodies[section]
            history = _branch_history(metadata, action.edit_target.label)
            steering_message = None
            if self.level >= 2:
                manifestor = Manifestor(
                    self.manifestor_lm,
                    self.logger,
                    self.manifestor_traces_chars,
                )
                try:
                    steering_message = manifestor.manifest(action, region_text, feedback, traces)
                except ManifestationError as exc:
                    error = _bounded_history_text(exc)
                    failed_proposer_record = {
                        "react_iterations": 0,
                        "react_tool_calls": 0,
                        "react_steps": [],
                        "react_steps_truncated": 0,
                    }
                    records.append(
                        {
                            "backend": "react_v2",
                            "component": name,
                            "edit_target": action.edit_target.label,
                            "action_choice": action.menu_id,
                            "action_operator": preferred_edit_tool,
                            "action_target_section": section,
                            "preferred_edit_tool": preferred_edit_tool,
                            "semantic_action": semantic_action,
                            "steering_message": "",
                            "manifestor_delivery": "user_message",
                            "feedback": _bounded_history_text(feedback),
                            "controller_sampling": controller_sampling,
                            "manifestor_error": error,
                            "executed_edit": [],
                            "chat_messages": [
                                {
                                    "role": "user",
                                    "content": f"Manifestor error: {exc}",
                                }
                            ],
                            "dropped_reason": error,
                            "attempt_status": "dropped",
                            "tracking_id": _tracking_id(action),
                            "branch_history_length": len(history),
                            **failed_proposer_record,
                        }
                    )
                    dropped.append(name)
                    if self.logger is not None:
                        self.logger.log(f"Component {name!r} dropped after Manifestor failure: {exc}")
                    continue

            react = ReActV2Proposer(
                self.base_lm,
                template,
                self.edit_tools,
                max_iterations=self.react_max_iterations,
                max_tool_calls=self.react_max_tool_calls,
                logger=self.logger,
            )
            result = react.propose(
                region_text,
                action.edit_target,
                action.edit_tool,
                steering_message,
                feedback,
                traces,
                history,
                self.max_chars,
            )
            proposer_record = {
                "react_iterations": result.iterations,
                "react_tool_calls": result.tool_calls,
                "react_steps": [
                    {
                        "turn": step.turn,
                        "assistant": _bounded_history_text(step.assistant),
                        "action": step.action,
                        "observation": _bounded_history_text(step.observation),
                        "error": _bounded_history_text(step.error),
                        "executed_edit": [
                            _bounded_history_text(value) or ""
                            for value in list(step.executed_edit)[:MAX_HISTORY_EDIT_ENTRIES]
                        ],
                    }
                    for step in result.steps[:MAX_HISTORY_STEPS]
                ],
                "react_steps_truncated": max(0, len(result.steps) - MAX_HISTORY_STEPS),
                "chat_messages": _react_chat_messages(result.steps),
            }

            new_component = None
            if result.changed:
                new_component = template.replace_section_body(text, section, result.new_text)
                if self.max_chars is not None and len(new_component) > self.max_chars:
                    result.changed = False
                    result.dropped_reason = (
                        f"Edited component is {len(new_component)} characters, exceeding max_chars={self.max_chars}."
                    )
                    new_component = None

            record = {
                "backend": "react_v2",
                "component": name,
                "edit_target": action.edit_target.label,
                "action_choice": action.menu_id,
                "action_operator": preferred_edit_tool,
                "action_target_section": section,
                "preferred_edit_tool": preferred_edit_tool,
                "semantic_action": semantic_action,
                "steering_message": _bounded_history_text(steering_message) if steering_message is not None else "",
                "manifestor_delivery": "user_message",
                "feedback": _bounded_history_text(feedback),
                "controller_sampling": controller_sampling,
                "manifestor_error": None,
                "executed_edit": [
                    _bounded_history_text(value) or ""
                    for value in list(result.executed_edit)[:MAX_HISTORY_EDIT_ENTRIES]
                ],
                "dropped_reason": _bounded_history_text(result.dropped_reason),
                "attempt_status": "completed" if result.changed else "dropped",
                "tracking_id": _tracking_id(action),
                "branch_history_length": len(history),
                **proposer_record,
            }
            records.append(record)

            if result.changed:
                assert new_component is not None
                proposal.new_texts[name] = new_component
                proposal.raw_lm_outputs[name] = result.final_output
                proposal.prompts[name] = steering_message or ""
                accepted_revisions.append(record)
            else:
                dropped.append(name)

        if records:
            primary = records[0]
            proposal.metadata.update(
                {
                    "action": primary["tracking_id"],
                    "reflection_level": self.level,
                    "proposer_backend": "react_v2",
                    "edit_target": primary["edit_target"],
                    "action_choice": primary["action_choice"],
                    "action_operator": primary["action_operator"],
                    "action_target_section": primary["action_target_section"],
                    "edit_tool": primary["preferred_edit_tool"],
                    "preferred_edit_tool": primary["preferred_edit_tool"],
                    "semantic_action": primary["semantic_action"],
                    "steering_message": primary["steering_message"],
                    "manifestor_delivery": primary["manifestor_delivery"],
                    "executed_edit": primary["executed_edit"],
                    "controller_sampling": primary["controller_sampling"],
                    "branch_history_length": primary["branch_history_length"],
                    "three_role_actions": records,
                    "attempt_records": records,
                    "revision_records": accepted_revisions,
                }
            )
        if controller_failures:
            proposal.metadata["controller_failures"] = controller_failures
        if dropped:
            proposal.metadata["react_v2_dropped"] = dropped
            proposal.metadata["length_capped_dropped"] = dropped
        return proposal, self
