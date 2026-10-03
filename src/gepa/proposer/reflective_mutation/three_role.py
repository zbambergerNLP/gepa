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
from dataclasses import replace
from typing import Any

from gepa.proposer.reflective_mutation.base import LanguageModel, ReflectionComponentSelector
from gepa.proposer.reflective_mutation.generation_recovery import RECOVERY_POLICY_CONTRACT, GenerationRecoveryPlanner
from gepa.proposer.reflective_mutation.manifestor import (
    ManifestationError,
    Manifestor,
)
from gepa.proposer.reflective_mutation.react_v2_proposer import REACT_V2_EXECUTION_CONTRACT, ReActV2Proposer
from gepa.proposer.reflective_mutation.reflection_lm import (
    ReflectionJob,
    ReflectionProposal,
    StatelessReflectionLM,
)
from gepa.proposer.reflective_mutation.single_call_proposer import SINGLE_CALL_EXECUTION_CONTRACT, SingleCallProposer
from gepa.response_journal import stable_api_base_identity
from gepa.strategies.action_space import DEFAULT_VERBALIZED_ACTION_K, IncompleteActionDistributionError
from gepa.strategies.document_template import TEMPLATE_FAMILIES, DocumentTemplate, MalformedDocumentError
from gepa.strategies.edit_tools import EDIT_TOOL_SETS
from gepa.strategies.forest_constants import (
    BROAD_EDIT_TOOL_SET,
    JEV_SELECTION,
    REACT_EDITOR_MODE,
    SEMANTIC_REFLECTION_LEVEL,
    SINGLE_CALL_EDITOR_MODE,
    UNIFORM_RANDOM_SELECTION,
    VERBALIZED_SELECTION,
)
from gepa.strategies.intervention import (
    CONTROLLER_COMPONENT_SELECTION_CONTRACT,
    CONTROLLER_POLICY_CONTRACT,
    SEMANTIC_ACTION_CATALOGS,
    UNIFORM_RANDOM_CONTROLLER_POLICY_CONTRACT,
    Controller,
    ControllerChoice,
    build_controller_menu,
    summarize_feedback,
)
from gepa.strategies.jev_controller import JevController
from gepa.strategies.reflection_context import (
    CONTROLLER_AUTHORITY_GUIDANCE,
    FOREST_REFLECTION_CONTRACT,
    GENERALIZATION_GUIDANCE,
    REFLECTION_CONTEXT_CONTRACT,
)
from gepa.strategies.text_limits import TextLimitError, TextLimits, clip_text, resolve_text_limits

MAX_HISTORY_STEPS = 16
MAX_HISTORY_EDIT_ENTRIES = 32
REFLECTION_RUN_CONTRACT_FILENAME = "reflection-run-contract.json"
_CONTROLLER_SELECTIONS = (VERBALIZED_SELECTION, UNIFORM_RANDOM_SELECTION, JEV_SELECTION)
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


def _bounded_history_text(value: Any, max_chars: int | None = None) -> str | None:
    """Render one optional history field within its persistent text bound.

    Args:
        value: Field value to stringify, or ``None`` when absent.
        max_chars: Optional source-character limit for this stored field.

    Returns:
        Original string representation, a length-marked prefix, or ``None``.
    """
    if value is None:
        return None
    return clip_text(str(value), max_chars)


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
        "sampled_reasonings": [str(reason) for reason in history.get("sampled_reasonings", [])],
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
        "policy": "joint_region_action_v5",
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
        "policy": "joint_region_action_uniform_v1" if level >= SEMANTIC_REFLECTION_LEVEL else "region_uniform_v1",
    }
    if level >= SEMANTIC_REFLECTION_LEVEL:
        record["joint_sampling_probability"] = probability
    return record


def _summarize_traces(entries: Sequence[Mapping[str, Any]]) -> str:
    """Preserve ordered reflection records and adapter diagnostics for every role.

    Args:
        entries: Reflective-dataset rows with inputs, outputs, and feedback.

    Returns:
        One labeled JSON record per example, or a no-traces marker.
    """
    blocks: list[str] = []
    for index, entry in enumerate(entries):
        blocks.append(f"[example {index + 1}]\n{json.dumps(dict(entry), ensure_ascii=False, default=str)}")
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
        base_lm: Reflection model used by ReAct V2 and, by default, Controller
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
        max_chars: Optional maximum completed component size; no limit by default.
        controller_lm: Optional separate LM for verbalized Controller selection.
        manifestor_lm: LM used to manifest level-2 actions.
        base_lm_run_identity: Optional stable, non-secret configuration identity
            for a custom ReAct callable.
        controller_lm_run_identity: Optional stable, non-secret configuration
            identity for a custom Controller callable.
        manifestor_lm_run_identity: Optional stable, non-secret configuration
            identity for a custom Manifestor callable.
        manifestor_traces_chars: Trace budget for the Manifestor.
        proposer_model: Provider/model identifier recorded in the run contract.
            When omitted, ``base_lm.model`` is inspected. Manifestor steering is
            delivered as a user message for every ReAct provider.
        react_max_iterations: Optional ReAct assistant-turn limit per component.
        react_max_tool_calls: Optional valid tool-call limit per proposal.

    Raises:
        ValueError: Configuration names or reflection level are invalid.
    """

    def __init__(
        self,
        base_lm: LanguageModel,
        level: int,
        *,
        edit_tool_set: str = BROAD_EDIT_TOOL_SET,
        component_kinds: dict[str, str] | None = None,
        template_family: str = "generic",
        templates: Mapping[str, DocumentTemplate] | None = None,
        k: int = DEFAULT_VERBALIZED_ACTION_K,
        tau: float | None = None,
        controller_selection: str = VERBALIZED_SELECTION,
        jev_controller: JevController | None = None,
        rng: random.Random | None = None,
        logger: Any | None = None,
        reflection_prompt_template: str | dict[str, str] | None = None,
        max_menu: int | None = None,
        max_chars: int | None = None,
        controller_lm: LanguageModel | None = None,
        manifestor_lm: LanguageModel | None = None,
        base_lm_run_identity: Mapping[str, Any] | None = None,
        controller_lm_run_identity: Mapping[str, Any] | None = None,
        manifestor_lm_run_identity: Mapping[str, Any] | None = None,
        manifestor_traces_chars: int | None = None,
        proposer_model: str | None = None,
        react_max_iterations: int | None = None,
        react_max_tool_calls: int | None = None,
        editor_mode: str = REACT_EDITOR_MODE,
        proposal_policy: str = "real_edit",
        text_limits: TextLimits | None = None,
    ):
        """Validate and store the complete three-role strategy configuration.

        ``text_limits`` configures optional character budgets for all roles.
        Explicit legacy ``max_chars`` and ``manifestor_traces_chars`` values
        override their corresponding entries.

        Args:
            base_lm: ReAct V2 model, also the default Controller model.
            level: Reflection level: vanilla, region-only, or region/action.
            edit_tool_set: Named atomic or broad execution basis.
            component_kinds: Optional component-to-template-kind overrides.
            template_family: Provider family supplying default templates.
            templates: Template-kind overrides merged into the family defaults.
            k: Number of Controller samples below level 2.
            tau: Optional verbalized-sampling tail-mass threshold.
            controller_selection: Controller selection policy. Uniform random
                selection draws once from the same section/action menu and does
                not call the Controller LM. Jev uses a typed joint-choice API.
            jev_controller: Required typed API client when selecting ``jev``.
            rng: Seeded strategy RNG. When ``None``, GEPA replaces the
                deterministic default with the engine RNG at wiring time.
            logger: Optional run logger shared by all roles.
            reflection_prompt_template: Vanilla level-0 reflection template.
            max_menu: Optional level-1 region-menu bound.
            max_chars: Maximum reconstructed component length, or ``None`` for no limit.
            controller_lm: Separate Controller model, or ``None`` to share the
                base model.
            manifestor_lm: Separate Manifestor model, or ``None`` to share the
                base model.
            base_lm_run_identity: Stable public identity for a custom base model.
            controller_lm_run_identity: Stable public identity for a custom
                Controller model.
            manifestor_lm_run_identity: Stable public identity for a custom
                Manifestor model.
            manifestor_traces_chars: Maximum trace characters shown to the
                Manifestor.
            editor_mode: Legacy multi-turn ``react`` or one-response ``single_call`` editing.
            proposal_policy: ``real_edit`` retries failed generation for level-2
                generative, Jev and random Controllers. ``independent`` retains
                historical behavior; lower-level Controllers are unchanged.
            proposer_model: Model identifier persisted in the run contract.
            react_max_iterations: Maximum ReAct turns, or ``None`` for no limit.
            react_max_tool_calls: Maximum valid calls, or ``None`` for no limit.
            text_limits: Optional document, section and role-context character budgets.

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
        if level == 0 and controller_selection != VERBALIZED_SELECTION:
            raise ValueError("controller_selection must be 'verbalized' when reflection level is 0")
        if controller_selection == JEV_SELECTION and (
            level != SEMANTIC_REFLECTION_LEVEL or jev_controller is None or controller_lm is not None
        ):
            raise ValueError("Jev requires level 2, a jev_controller, and no generative controller_lm.")
        if controller_selection != JEV_SELECTION and jev_controller is not None:
            raise ValueError("jev_controller requires controller_selection='jev'.")
        if template_family not in TEMPLATE_FAMILIES:
            raise ValueError(f"template_family must be one of {sorted(TEMPLATE_FAMILIES)}; got {template_family!r}")
        self.templates: dict[str, DocumentTemplate] = {**TEMPLATE_FAMILIES[template_family], **(templates or {})}
        for name, kind in (component_kinds or {}).items():
            if kind not in self.templates:
                raise ValueError(f"component_kinds[{name!r}] must be one of {sorted(self.templates)}; got {kind!r}")

        if editor_mode not in {REACT_EDITOR_MODE, SINGLE_CALL_EDITOR_MODE}:
            raise ValueError("editor_mode must be react or single_call")
        if proposal_policy not in {"real_edit", "independent"}:
            raise ValueError("proposal_policy must be real_edit or independent")
        self.recovery_planner = (
            GenerationRecoveryPlanner(self)
            if proposal_policy == "real_edit" and level == SEMANTIC_REFLECTION_LEVEL
            else None
        )
        if self.recovery_planner is not None:
            editor_mode = SINGLE_CALL_EDITOR_MODE
        if editor_mode == SINGLE_CALL_EDITOR_MODE and edit_tool_set != BROAD_EDIT_TOOL_SET:
            raise ValueError("Single-call editing requires the broad direct-tool basis")
        self.editor_mode = editor_mode
        self.proposer_backend = SINGLE_CALL_EDITOR_MODE if editor_mode == SINGLE_CALL_EDITOR_MODE else "react_v2"
        self.base_lm = base_lm
        self.level = level
        self.edit_tool_set = edit_tool_set
        self.edit_tools = EDIT_TOOL_SETS[edit_tool_set]
        self.component_kinds = component_kinds or {}
        self.template_family = template_family
        self.k = k
        self.tau = tau
        self.controller_selection = controller_selection
        self.controller_selects_component = False
        self.jev_controller = jev_controller
        self._rng_explicit = rng is not None
        self.rng = rng if rng is not None else random.Random(0)
        self.logger = logger
        self.reflection_prompt_template = reflection_prompt_template
        self.max_menu = max_menu
        limits = resolve_text_limits(text_limits)
        if max_chars is not None:
            limits = replace(limits, max_component_chars=max_chars)
        if manifestor_traces_chars is not None:
            limits = replace(limits, manifestor_trace_chars=manifestor_traces_chars)
        self.text_limits = limits
        self.max_chars = limits.max_component_chars
        self.controller_lm = controller_lm if controller_lm is not None else base_lm
        self.manifestor_lm = manifestor_lm if manifestor_lm is not None else base_lm
        self.base_lm_run_identity = base_lm_run_identity
        self.controller_lm_run_identity = (
            base_lm_run_identity
            if controller_lm is None and controller_lm_run_identity is None
            else controller_lm_run_identity
        )
        self.manifestor_lm_run_identity = (
            base_lm_run_identity
            if manifestor_lm is None and manifestor_lm_run_identity is None
            else manifestor_lm_run_identity
        )
        self.manifestor_traces_chars = limits.manifestor_trace_chars
        inferred_model = proposer_model
        if inferred_model is None:
            model_attribute = getattr(base_lm, "model", None)
            inferred_model = model_attribute if isinstance(model_attribute, str) else None
        self.proposer_model = inferred_model
        self.react_max_iterations = react_max_iterations
        self.react_max_tool_calls = react_max_tool_calls
        self._stateless: StatelessReflectionLM | None = (
            StatelessReflectionLM(base_lm, reflection_prompt_template, logger, rng=self.rng, text_limits=limits)
            if level == 0
            else None
        )

    def bind_module_selector(self, module_selector: ReflectionComponentSelector | str) -> None:
        """Let the Controller choose a component only for the explicit opt-in mode."""
        enabled = isinstance(module_selector, str) and module_selector == "controller"
        if enabled and (
            self.level != SEMANTIC_REFLECTION_LEVEL or self.controller_selection not in {VERBALIZED_SELECTION, "jev"}
        ):
            raise ValueError("Controller component selection requires level 2 and a verbalized or Jev Controller.")
        self.controller_selects_component = enabled
        if self.recovery_planner is not None:
            self.recovery_planner.contract = deepcopy(RECOVERY_POLICY_CONTRACT)
            if enabled:
                self.recovery_planner.contract.update(
                    component_selection=deepcopy(CONTROLLER_COMPONENT_SELECTION_CONTRACT),
                    recovery="remaining_joint_component_section_action_choices_without_replacement",
                    controller_distribution="one_joint_distribution_per_opportunity",
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
        if self.jev_controller is not None:
            controller = self.jev_controller.run_contract()
        elif self.level >= SEMANTIC_REFLECTION_LEVEL and self.controller_selection == UNIFORM_RANDOM_SELECTION:
            controller = {
                **UNIFORM_RANDOM_CONTROLLER_POLICY_CONTRACT,
                "max_menu": self.max_menu,
            }
        elif self.level >= SEMANTIC_REFLECTION_LEVEL:
            controller = {
                **(self.recovery_planner.contract if self.recovery_planner else CONTROLLER_POLICY_CONTRACT),
                "tau": self.tau,
                "max_menu": self.max_menu,
            }
        elif self.controller_selection == UNIFORM_RANDOM_SELECTION:
            controller = {
                "version": 1,
                "factorization": "region_only",
                "selection": UNIFORM_RANDOM_SELECTION,
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
        if self.controller_selects_component:
            controller = {**controller, **deepcopy(CONTROLLER_COMPONENT_SELECTION_CONTRACT)}
        proposer_lm_identity = _language_model_run_identity(self.base_lm, self.base_lm_run_identity)
        controller_lm_identity = (
            _language_model_run_identity(self.controller_lm, self.controller_lm_run_identity)
            if self.level >= 1 and self.controller_selection == VERBALIZED_SELECTION
            else None
        )
        manifestor_lm_identity = (
            _language_model_run_identity(self.manifestor_lm, self.manifestor_lm_run_identity)
            if self.level >= SEMANTIC_REFLECTION_LEVEL
            else None
        )
        unstable_roles = [
            role
            for role, identity in (
                ("Proposer", proposer_lm_identity),
                ("Controller", controller_lm_identity),
                ("Manifestor", manifestor_lm_identity),
            )
            if identity is not None and identity["configuration_source"] in {"opaque", "partial"}
        ]
        if unstable_roles:
            roles = " and ".join(unstable_roles)
            raise ValueError(
                f"A stable run identity is required for the {roles} LM. Pass the corresponding "
                "base_lm_run_identity, controller_lm_run_identity, or manifestor_lm_run_identity "
                "when constructing ThreeRoleReflectionLM with custom callables."
            )
        max_proposer_model_calls = 1 if self.editor_mode == "single_call" else self.react_max_iterations
        execution_contract = dict(
            SINGLE_CALL_EXECUTION_CONTRACT if self.editor_mode == "single_call" else REACT_V2_EXECUTION_CONTRACT
        )
        if self.recovery_planner is not None:
            max_proposer_model_calls = 2 * sum(
                len(self.templates[kind].sections) * len(SEMANTIC_ACTION_CATALOGS[self.templates[kind].kind]["actions"])
                for kind in component_kinds.values()
            )
            execution_contract.update(
                version=2,
                completion="atomic_batch_with_optional_native_protocol_correction",
                unchanged_finish="generation_error_then_replan",
                invalid_batch="atomic_rollback_then_replan_another_pair",
                missing_native_calls="one_same_action_protocol_correction_then_replan",
                max_responses_per_pair=2,
            )
        return {
            "schema_version": 11,
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
                {kind: deepcopy(SEMANTIC_ACTION_CATALOGS[self.templates[kind].kind]) for kind in active_kinds}
                if self.level >= SEMANTIC_REFLECTION_LEVEL
                else None
            ),
            "max_chars": self.max_chars,
            "document_length": self.text_limits.document_contract(),
            "text_limits": self.text_limits.to_dict(),
            "manifestor_traces_chars": self.manifestor_traces_chars,
            "reflection_context": deepcopy(REFLECTION_CONTEXT_CONTRACT),
            "generalization": {
                **deepcopy(self.recovery_planner.contract if self.recovery_planner else FOREST_REFLECTION_CONTRACT),
                **(
                    {
                        "controller_direction": "Jev selects the pair; Manifestor derives the edit direction from evidence"
                    }
                    if self.jev_controller is not None
                    else {}
                ),
            },
            "manifestor_delivery": "user_message",
            "branch_history": {
                "storage": "target_scoped_user_assistant_messages",
                "direct_deepseek_native_delivery": "quoted_user_context",
                "other_delivery": "quoted_user_context"
                if self.editor_mode == SINGLE_CALL_EDITOR_MODE
                else "provider_chat_messages",
            },
            "proposer_model": self.proposer_model,
            "proposer_backend": self.proposer_backend,
            "proposer_lm": proposer_lm_identity,
            "controller_lm": controller_lm_identity,
            "manifestor_lm": manifestor_lm_identity,
            "max_proposer_model_calls": max_proposer_model_calls,
            "react_max_iterations": self.react_max_iterations,
            "react_max_tool_calls": self.react_max_tool_calls,
            "react_execution": {
                **execution_contract,
                "max_iterations": 2
                if self.recovery_planner is not None
                else (1 if self.editor_mode == SINGLE_CALL_EDITOR_MODE else self.react_max_iterations),
                "max_tool_calls": self.react_max_tool_calls,
            },
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
            if self.level >= SEMANTIC_REFLECTION_LEVEL and not SEMANTIC_ACTION_CATALOGS.get(template.kind, {}).get(
                "actions"
            ):
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
        state: dict[str, Any] = {"rng_state": self.rng.getstate()}
        if self.recovery_planner is not None:
            state["generation_recovery"] = self.recovery_planner.get_state()
        return state

    def get_batch_retry_state(self) -> dict[str, Any]:
        """Snapshot role-local state before a batched reflection attempt.

        Returns:
            Controller RNG state and response-journal cursors for each
            distinct role model.
        """
        if self._stateless is not None:
            return self._stateless.get_batch_retry_state()
        state: dict[str, Any] = self.get_state()
        base_cursor = getattr(self.base_lm, "response_journal_cursor_state", None)
        if callable(base_cursor):
            state["base_lm_cursor"] = base_cursor()
        if self.controller_lm is not self.base_lm:
            controller_cursor = getattr(self.controller_lm, "response_journal_cursor_state", None)
            if callable(controller_cursor):
                state["controller_lm_cursor"] = controller_cursor()
        if self.manifestor_lm is not self.base_lm and self.manifestor_lm is not self.controller_lm:
            manifestor_cursor = getattr(self.manifestor_lm, "response_journal_cursor_state", None)
            if callable(manifestor_cursor):
                state["manifestor_lm_cursor"] = manifestor_cursor()
        if self.jev_controller is not None:
            state["jev_controller_cursor"] = self.jev_controller.response_journal_cursor_state()
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
        self.set_state(state)
        jev_cursor = state.get("jev_controller_cursor")
        if jev_cursor is not None:
            if self.jev_controller is None:
                raise TypeError("Jev Controller cannot restore its response-journal cursor.")
            self.jev_controller.restore_response_journal_cursor_state(jev_cursor)
        base_cursor = state.get("base_lm_cursor")
        if base_cursor is not None:
            restore = getattr(self.base_lm, "restore_response_journal_cursor_state", None)
            if not callable(restore):
                raise TypeError("ReAct LM cannot restore its response-journal cursor.")
            restore(base_cursor)
        controller_cursor = state.get("controller_lm_cursor")
        if controller_cursor is not None:
            restore = getattr(self.controller_lm, "restore_response_journal_cursor_state", None)
            if not callable(restore):
                raise TypeError("Controller LM cannot restore its response-journal cursor.")
            restore(controller_cursor)
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
        if self.recovery_planner is not None:
            self.recovery_planner.set_state(state.get("generation_recovery", {}))
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
        """Return provider spend without double-counting shared role models.

        Returns:
            Combined tracked cost.
        """
        cost = float(getattr(self.base_lm, "total_cost", 0.0))
        if self.jev_controller is not None:
            cost += self.jev_controller.total_cost
        if self.controller_lm is not self.base_lm:
            cost += float(getattr(self.controller_lm, "total_cost", 0.0))
        if self.manifestor_lm is not self.base_lm and self.manifestor_lm is not self.controller_lm:
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
        if self.recovery_planner is not None:
            return self.recovery_planner.generate(candidate, reflective_dataset, components_to_update, metadata), self
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
            self.reflect(
                candidate,
                dataset,
                components,
                metadata={**(context or {}), "proposal_slot": index} if self.recovery_planner else context,
            )
            for index, ((candidate, dataset, components), context) in enumerate(zip(jobs, contexts, strict=True))
        ]

    def _select_action(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components: list[str],
    ) -> tuple[ControllerChoice, dict[str, Any], str | None]:
        """Select one edit with the same backend for fixed or Controller-selected modules."""
        contexts = {}
        menu = []
        for name in components:
            template = self.templates[self._component_kind(name)]
            contexts[name] = {
                "sections": template.parse(candidate[name]),
                "section_descriptions": dict(template.sections),
                "training_evidence": _summarize_traces(reflective_dataset[name]),
            }
            menu.extend(
                build_controller_menu(
                    template,
                    name,
                    self.edit_tools,
                    self.level,
                    rng=self.rng,
                    max_menu=self.max_menu,
                    include_component=self.controller_selects_component,
                )
            )
        if self.controller_selects_component and self.max_menu is not None and len(menu) > self.max_menu:
            raise ValueError("max_menu would remove joint component/section/action choices.")
        context = contexts[components[0]]
        inventory = (
            "Controller-only section inventory. [EMPTY SECTION] is metadata, not document text. "
            "An empty region has no target bytes: assign probability 0 to its DELETE_TEXT, REPLACE_TEXT, and "
            "MOVE_TEXT choices. Judge its INSERT_TEXT choices by their semantic fit.\n\n"
        )
        if self.controller_selects_component:
            controller_candidate = inventory + json.dumps(
                {
                    name: {key: value for key, value in context.items() if key != "training_evidence"}
                    for name, context in contexts.items()
                },
                ensure_ascii=False,
            )
            controller_traces = (
                "Choose exactly one component, section and action together. Compare all components' training traces; "
                "locate where relevant information or behavior first became missing or incorrect, identify the "
                "component whose instructions can address it, and state the observable improvement the edit should "
                "produce. A wrong final answer alone does not implicate the final-answer component.\n"
                + json.dumps({name: value["training_evidence"] for name, value in contexts.items()}, ensure_ascii=False)
            )
        else:
            controller_candidate = inventory + "\n\n".join(
                f"## {section}\n{body if body else '[EMPTY SECTION]'}" for section, body in context["sections"].items()
            )
            controller_traces = context["training_evidence"]
        controller_direction = None
        if self.jev_controller is not None and self.controller_selects_component:
            action, controller_sampling = self.jev_controller.select_components(menu, components=contexts, rng=self.rng)
        elif self.jev_controller is not None:
            action, controller_sampling = self.jev_controller.select(
                menu,
                sections=context["sections"],
                section_descriptions=context["section_descriptions"],
                traces=context["training_evidence"],
                rng=self.rng,
            )
        elif self.controller_selection == UNIFORM_RANDOM_SELECTION:
            action = self.rng.choice(menu)
            controller_sampling = _uniform_controller_sampling_record(menu, action, self.level)
        else:
            controller = Controller(
                menu,
                self.controller_lm,
                k=len(menu) if self.level >= SEMANTIC_REFLECTION_LEVEL else self.k,
                tau=self.tau,
                rng=self.rng,
                require_full_support=self.level >= SEMANTIC_REFLECTION_LEVEL,
                text_limits=self.text_limits,
            )
            action = controller.select(
                1,
                self.rng,
                candidate=controller_candidate,
                feedback_summary=(
                    GENERALIZATION_GUIDANCE
                    + "\n"
                    + CONTROLLER_AUTHORITY_GUIDANCE
                    + "\nYou set the direction of the edit. For each action's reasoning, identify an observable "
                    "mismatch, the intended reusable change and its scope, and why this action can express it. "
                    "The sampled option's rationale will be passed verbatim to the Manifestor and Editor; "
                    "make it specific enough for them to realize your direction without choosing a new goal. "
                    "Score semantic fit as well as tool applicability.\n\n"
                    "## Structured training evidence\n"
                    + clip_text(controller_traces, self.text_limits.controller_feedback_chars)
                ),
            )[0]
            if self.level >= SEMANTIC_REFLECTION_LEVEL:
                controller_sampling = _joint_controller_sampling_record(controller.history[-1])
            else:
                controller_sampling = _controller_sampling_record(controller.history[-1])
            controller_direction = controller.history[-1]["sampled_reasonings"][0] or None
        if self.controller_selects_component:
            controller_sampling.update(
                {
                    "component_selection": "controller",
                    "eligible_components": list(components),
                    "selected_component": action.edit_target.component_name,
                    "factorization": "P(component, region, action)",
                }
            )
        return action, controller_sampling, controller_direction

    def _reflect_operated(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
        metadata: Mapping[str, Any] | None,
        *,
        selected: tuple[ControllerChoice, dict[str, Any], str | None] | None = None,
        require_edit: bool = False,
    ) -> tuple[ReflectionProposal, ThreeRoleReflectionLM]:
        """Run Controller, optional Manifestor, and ReAct V2 per component.

        Args:
            candidate: Parent candidate.
            reflective_dataset: Per-component run evidence.
            components_to_update: Components selected for mutation.
            metadata: Parent-specific chat history and iteration anchors.
            selected: Planner-selected action, sampling evidence, and direction.
            require_edit: Return generation errors for invalid or unchanged edits.

        Returns:
            Reflection proposal and this strategy.

        """
        proposal = ReflectionProposal(new_texts={}, prompts={}, raw_lm_outputs={}, metadata={})
        records: list[dict[str, Any]] = []
        accepted_revisions: list[dict[str, Any]] = []
        controller_failures: list[dict[str, str]] = []
        dropped: list[str] = []

        groups = (
            [components_to_update] if self.controller_selects_component else [[name] for name in components_to_update]
        )
        for group in groups:
            names = [name for name in group if reflective_dataset.get(name)]
            if not names:
                if self.logger is not None:
                    self.logger.log(f"Components {group!r} have no reflective evidence. Skipping.")
                continue
            try:
                if selected is not None:
                    action, controller_sampling, controller_direction = selected
                else:
                    action, controller_sampling, controller_direction = self._select_action(
                        candidate,
                        reflective_dataset,
                        names,
                    )
            except IncompleteActionDistributionError as exc:
                error = (
                    _bounded_history_text(exc, self.text_limits.history_text_chars)
                    or "Controller action distribution failed."
                )
                controller_failures.extend({"component": name, "error": error} for name in names)
                dropped.extend(names)
                if self.logger is not None:
                    self.logger.log(f"Components {names!r} dropped after Controller failure: {error}")
                continue
            name = action.edit_target.component_name
            template = self.templates[self._component_kind(name)]
            text = candidate[name]
            feedback = summarize_feedback(reflective_dataset[name], self.text_limits.controller_feedback_chars)
            traces = _summarize_traces(reflective_dataset[name])
            section_bodies = template.parse(text)
            preferred_edit_tool = action.edit_tool.value if action.edit_tool is not None else None
            semantic_action = action.semantic_action.name if action.semantic_action else None

            section = action.edit_target.section
            region_text = section_bodies[section]
            history = _branch_history(metadata, action.edit_target.label)
            steering_message = None
            if self.level >= SEMANTIC_REFLECTION_LEVEL:
                manifestor = Manifestor(
                    self.manifestor_lm,
                    self.logger,
                    self.manifestor_traces_chars,
                    text_limits=self.text_limits,
                )
                try:
                    steering_message = manifestor.manifest(
                        action,
                        region_text,
                        feedback,
                        traces,
                        controller_direction=controller_direction,
                        require_edit=require_edit,
                    )
                except ManifestationError as exc:
                    error = _bounded_history_text(exc, self.text_limits.history_text_chars)
                    failed_proposer_record = {
                        "react_iterations": 0,
                        "react_tool_calls": 0,
                        "react_steps": [],
                        "react_steps_truncated": 0,
                    }
                    records.append(
                        {
                            "backend": self.proposer_backend,
                            "component": name,
                            "edit_target": action.edit_target.label,
                            "action_choice": action.menu_id,
                            "action_operator": preferred_edit_tool,
                            "action_target_section": section,
                            "preferred_edit_tool": preferred_edit_tool,
                            "semantic_action": semantic_action,
                            "steering_message": "",
                            "manifestor_delivery": "user_message",
                            "feedback": _bounded_history_text(feedback, self.text_limits.history_text_chars),
                            "controller_sampling": controller_sampling,
                            "controller_direction": controller_direction,
                            "manifestor_error": error,
                            "executed_edit": [],
                            "chat_messages": [
                                {
                                    "role": "user",
                                    "content": f"Manifestor error: {exc}",
                                }
                            ],
                            "dropped_reason": error,
                            "attempt_status": "generation_error" if require_edit else "dropped",
                            "tracking_id": _tracking_id(action),
                            "branch_history_length": len(history),
                            **failed_proposer_record,
                        }
                    )
                    dropped.append(name)
                    if self.logger is not None:
                        self.logger.log(f"Component {name!r} dropped after Manifestor failure: {exc}")
                    continue

            proposer_class = SingleCallProposer if self.editor_mode == SINGLE_CALL_EDITOR_MODE else ReActV2Proposer
            react = proposer_class(
                self.base_lm,
                template,
                self.edit_tools,
                max_iterations=self.react_max_iterations,
                max_tool_calls=self.react_max_tool_calls,
                logger=self.logger,
                text_limits=self.text_limits,
            )
            editor_steering = steering_message
            if action.semantic_action is not None:
                spec = action.semantic_action
                editor_steering = (
                    f"Selected semantic action: {spec.name}\nDescription: {spec.description}\n"
                    f"Binding action instruction: {spec.instruction or spec.fixed_text}\n\n"
                    f"Manifestor steering:\n{steering_message or ''}"
                )
            result = react.propose(
                region_text,
                action.edit_target,
                action.edit_tool,
                editor_steering,
                feedback,
                traces,
                history,
                self.max_chars,
                controller_direction=controller_direction,
                **({"require_edit": True} if require_edit else {}),
            )
            proposer_record = {
                "react_iterations": result.iterations,
                "react_tool_calls": result.tool_calls,
                "react_steps": [
                    {
                        "turn": step.turn,
                        "assistant": _bounded_history_text(step.assistant, self.text_limits.history_text_chars),
                        "action": step.action,
                        "observation": _bounded_history_text(step.observation, self.text_limits.history_text_chars),
                        "error": _bounded_history_text(step.error, self.text_limits.history_text_chars),
                        "executed_edit": [
                            _bounded_history_text(value, self.text_limits.history_text_chars) or ""
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
                if require_edit and new_component == text:
                    result.changed = False
                    result.dropped_reason = "The batch produced no net change after canonical section rendering."
                    new_component = None
                elif self.max_chars is not None and len(new_component) > self.max_chars:
                    result.changed = False
                    result.dropped_reason = (
                        f"Edited component is {len(new_component)} characters, exceeding max_chars={self.max_chars}."
                    )
                    new_component = None
                elif self.text_limits.max_candidate_chars is not None:
                    try:
                        self.text_limits.check_candidate({**candidate, **proposal.new_texts, name: new_component})
                    except TextLimitError as exc:
                        result.changed = False
                        result.dropped_reason = str(exc)
                        new_component = None

            record = {
                "backend": self.proposer_backend,
                "component": name,
                "edit_target": action.edit_target.label,
                "action_choice": action.menu_id,
                "action_operator": preferred_edit_tool,
                "action_target_section": section,
                "preferred_edit_tool": preferred_edit_tool,
                "semantic_action": semantic_action,
                "steering_message": _bounded_history_text(steering_message, self.text_limits.history_text_chars)
                if steering_message is not None
                else "",
                "manifestor_delivery": "user_message",
                "feedback": _bounded_history_text(feedback, self.text_limits.history_text_chars),
                "controller_sampling": controller_sampling,
                "controller_direction": controller_direction,
                "manifestor_error": None,
                "executed_edit": [
                    _bounded_history_text(value, self.text_limits.history_text_chars) or ""
                    for value in list(result.executed_edit)[:MAX_HISTORY_EDIT_ENTRIES]
                ],
                "dropped_reason": _bounded_history_text(result.dropped_reason, self.text_limits.history_text_chars),
                "attempt_status": "completed"
                if result.changed
                else ("generation_error" if require_edit else "dropped"),
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
                    "proposer_backend": self.proposer_backend,
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
                    "controller_direction": primary["controller_direction"],
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
