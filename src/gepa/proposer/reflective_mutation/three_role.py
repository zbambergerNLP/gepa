# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Three-role reflection strategy: Controller -> Manifestor -> ReAct V2.

The strategy changes only reflective mutation. GEPA's evaluator, Pareto search,
acceptance, and merge behavior remain unchanged. Reflection level 0 delegates
to vanilla GEPA. Level 1 selects a document region and lets ReAct V2 operate
over the configured edit basis. Level 2 also selects a semantic action and uses
the Manifestor to steer ReAct V2 through a provider-appropriate chat role.
"""

from __future__ import annotations

import json
import os
import random
from collections.abc import Mapping, Sequence
from typing import Any

from gepa.proposer.reflective_mutation.base import LanguageModel
from gepa.proposer.reflective_mutation.manifestor import (
    MAX_TRACES_CHARS,
    ManifestationError,
    Manifestor,
    infer_manifestor_injection_site,
)
from gepa.proposer.reflective_mutation.react_v2_proposer import ReActV2Proposer
from gepa.proposer.reflective_mutation.reflection_lm import (
    ReflectionJob,
    ReflectionProposal,
    StatelessReflectionLM,
)
from gepa.strategies.action_space import MAX_PROPOSAL_CHARS
from gepa.strategies.document_template import TEMPLATE_FAMILIES, DocumentTemplate, MalformedDocumentError
from gepa.strategies.edit_tools import EDIT_TOOL_SETS
from gepa.strategies.intervention import (
    Controller,
    ControllerAction,
    InjectionSite,
    build_controller_menu,
    build_semantic_action_menu,
    controller_policy_contract,
    semantic_action_catalog,
    summarize_feedback,
)

MAX_HISTORY_TEXT_CHARS = 2000
MAX_HISTORY_STEPS = 16
MAX_HISTORY_EDIT_ENTRIES = 32
REFLECTION_RUN_CONTRACT_FILENAME = "reflection-run-contract.json"
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
    """Return whether a configuration key names authentication material."""
    lowered = key.lower()
    return lowered in _SENSITIVE_CONFIG_KEYS or any(lowered.endswith(f"_{suffix}") for suffix in _SENSITIVE_CONFIG_KEYS)


def _public_run_identity_value(value: Any) -> Any:
    """Convert configuration to stable JSON data while redacting credentials."""
    if isinstance(value, Mapping):
        public: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            public[key] = "<redacted>" if _is_sensitive_config_key(key) else _public_run_identity_value(item)
        return public
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_public_run_identity_value(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    value_type = type(value)
    return f"<{value_type.__module__}.{value_type.__qualname__}>"


def _language_model_run_identity(lm: LanguageModel, explicit: Mapping[str, Any] | None) -> dict[str, Any]:
    """Describe one role LM without persisting credentials or runtime counters."""
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
    """Render one history field within the persistent text bound."""
    if value is None:
        return None
    text = str(value)
    if len(text) <= MAX_HISTORY_TEXT_CHARS:
        return text
    return text[:MAX_HISTORY_TEXT_CHARS] + f"...(+{len(text) - MAX_HISTORY_TEXT_CHARS} chars)"


def _bounded_history_edits(values: Sequence[Any]) -> list[str]:
    """Keep a bounded, serializable prefix of executed edit descriptions."""
    return [_bounded_history_text(value) or "" for value in list(values)[:MAX_HISTORY_EDIT_ENTRIES]]


def _react_chat_messages(steps: Sequence[Any]) -> list[dict[str, str]]:
    """Convert actual ReAct turns into persistent assistant/user messages."""
    messages: list[dict[str, str]] = []
    for step in steps[:MAX_HISTORY_STEPS]:
        assistant = _bounded_history_text(step.assistant)
        observation = _bounded_history_text(step.observation)
        if assistant:
            messages.append({"role": "assistant", "content": assistant})
        if observation and step.action != "FINISH":
            messages.append({"role": "user", "content": observation})
    return messages


def _controller_sampling_record(history: Mapping[str, Any]) -> dict[str, Any]:
    """Copy Controller distribution and fallback provenance as JSON primitives."""
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


def _factored_controller_sampling_record(
    region_history: Mapping[str, Any],
    action_history: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist both conditional Controller stages and their sampled propensity."""
    region = _controller_sampling_record(region_history)
    action = _controller_sampling_record(action_history)
    region_probability = region["sampled_probabilities"][0]
    action_probability = action["sampled_probabilities"][0]
    return {
        **action,
        "policy": "region_then_action_v1",
        "fallback": region["fallback"] or action["fallback"],
        "region": region,
        "action": action,
        "joint_sampling_probability": region_probability * action_probability,
    }


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


def _tracking_id(action: ControllerAction) -> str:
    """Return the action-diversity bucket for one Controller choice.

    Args:
        action: Selected region and optional semantic action.

    Returns:
        Semantic action name at level 2, or ``"edit:<region>"`` at level 1.
    """
    if action.intervention_spec is not None:
        return action.intervention_spec.name
    return f"edit:{action.edit_target.name}"


def _branch_history(metadata: Mapping[str, Any] | None) -> list[dict[str, str]]:
    """Validate branch history supplied by the reflective proposer.

    Args:
        metadata: Per-job context generated from the selected parent candidate.

    Returns:
        User/assistant transcript for accepted, rejected, and dropped attempts
        on this parent branch.

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
        history.append({"role": role, "content": content})
    return history


class ThreeRoleReflectionLM:
    """Controller/Manifestor/ReAct V2 reflection with explicit ablation levels.

    Args:
        base_lm: Reflection model reused by the Controller and ReAct V2.
        level: ``0`` vanilla GEPA, ``1`` region plus edit basis, or ``2`` region
            plus semantic action and Manifestor steering.
        edit_tool_set: ``"minimal"`` for insert/delete or ``"broad"`` for all
            insert/delete/replace/move tools.
        component_kinds: Component-to-kind mapping. Unlisted components are prompts.
        template_family: Canonical provider template family.
        templates: Optional per-kind template overrides.
        k: Verbalized-sampling distribution size at level 1. Level 2 scores
            every option in each factored Controller stage.
        tau: Tail-sampling threshold.
        rng: Seeded random stream.
        logger: Optional run logger shared by all roles.
        reflection_prompt_template: Vanilla level-0 prompt template.
        max_menu: Optional level-1 region bound. Level 2 requires it to retain
            every applicable region; semantic actions are never subsampled.
        max_chars: Maximum completed component size.
        manifestor_lm: Deterministic LM used to manifest level-2 actions.
        base_lm_run_identity: Optional stable, non-secret configuration identity
            for a custom Controller/ReAct callable.
        manifestor_lm_run_identity: Optional stable, non-secret configuration
            identity for a custom Manifestor callable.
        manifestor_traces_chars: Trace budget for the Manifestor.
        proposer_model: Provider/model identifier used to route Manifestor
            steering. When omitted, ``base_lm.model`` is inspected.
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
        """Validate and store strategy configuration."""
        if level not in (0, 1, 2):
            raise ValueError(f"reflection level must be 0, 1, or 2; got {level}")
        if edit_tool_set not in EDIT_TOOL_SETS:
            raise ValueError(f"edit_tool_set must be one of {sorted(EDIT_TOOL_SETS)}; got {edit_tool_set!r}")
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
        self.manifestor_injection_site: InjectionSite = infer_manifestor_injection_site(inferred_model)
        self.react_max_iterations = react_max_iterations
        self.react_max_tool_calls = react_max_tool_calls
        self._stateless: StatelessReflectionLM | None = (
            StatelessReflectionLM(base_lm, reflection_prompt_template, logger) if level == 0 else None
        )

    def run_contract(self, candidate: Mapping[str, str]) -> dict[str, Any]:
        """Return the complete JSON-serializable three-role strategy identity.

        Args:
            candidate: Seed component mapping used to resolve default document
                kinds alongside explicit ``component_kinds``.

        Returns:
            Contract whose drift must prevent state resumption.
        """
        self.validate_candidate(dict(candidate))
        component_kinds = {name: self.component_kinds.get(name, "prompt") for name in candidate}
        active_kinds = sorted(set(component_kinds.values()))
        templates = {
            kind: {
                "document_kind": self.templates[kind].kind,
                "sections": list(self.templates[kind].sections.items()),
            }
            for kind in active_kinds
        }
        controller: dict[str, Any]
        if self.level >= 2:
            controller = {
                **controller_policy_contract(),
                "tau": self.tau,
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
                ("Controller/ReAct", controller_lm_identity),
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
            "schema_version": 1,
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
                {kind: semantic_action_catalog(self.templates[kind].kind) for kind in active_kinds}
                if self.level >= 2
                else None
            ),
            "max_chars": self.max_chars,
            "manifestor_traces_chars": self.manifestor_traces_chars,
            "manifestor_injection_site": self.manifestor_injection_site,
            "proposer_model": self.proposer_model,
            "controller_react_lm": controller_lm_identity,
            "manifestor_lm": manifestor_lm_identity,
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
            template = self.templates[self.component_kinds.get(name, "prompt")]
            if self.level >= 2 and not semantic_action_catalog(template.kind)["actions"]:
                raise ValueError(
                    f"Component {name!r} uses document kind {template.kind!r}, which has no level-2 semantic catalog."
                )
            try:
                template.parse(text)
            except MalformedDocumentError as exc:
                raise MalformedDocumentError(
                    f"Component {name!r} is not in the canonical {template.kind!r} section format required by "
                    f"reflection_level > 0: {exc} Convert it once with "
                    "gepa.strategies.document_template.migrate_document(text, template, lm)."
                ) from exc

    def bind_rng(self, rng: random.Random) -> None:
        """Bind GEPA's seeded run RNG.

        Args:
            rng: Run RNG.
        """
        self.rng = rng
        if self._stateless is not None:
            self._stateless.bind_rng(rng)

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
        dropped: list[str] = []
        history = _branch_history(metadata)

        for name in components_to_update:
            entries = reflective_dataset.get(name)
            if not entries:
                self._log(f"Component '{name}' is not in reflective dataset. Skipping.")
                continue

            template = self.templates[self.component_kinds.get(name, "prompt")]
            text = candidate[name]
            feedback = summarize_feedback(entries)
            traces = _summarize_traces(entries)
            menu = build_controller_menu(
                template,
                name,
                self.edit_tools,
                self.level,
                rng=self.rng,
                max_menu=self.max_menu,
            )
            region_controller = Controller(
                menu,
                self.base_lm,
                k=len(menu) if self.level >= 2 else self.k,
                tau=self.tau,
                rng=self.rng,
                require_full_support=self.level >= 2,
            )
            region_controller.set_context(text, feedback)
            action = region_controller.select_controller(1, self.rng)[0]
            controller_sampling = _controller_sampling_record(region_controller.history[-1])

            if self.level >= 2:
                semantic_menu = build_semantic_action_menu(template, action.edit_target)
                if semantic_menu:
                    selected_section = action.edit_target.section
                    selected_region = text if selected_section is None else template.parse(text)[selected_section]
                    semantic_controller = Controller(
                        semantic_menu,
                        self.base_lm,
                        k=len(semantic_menu),
                        tau=self.tau,
                        rng=self.rng,
                        require_full_support=True,
                    )
                    semantic_controller.set_context(
                        text,
                        f"Selected region: {action.edit_target.name}\nCurrent region text:\n{selected_region}\n\n{feedback}",
                    )
                    action = semantic_controller.select_controller(1, self.rng)[0]
                    controller_sampling = _factored_controller_sampling_record(
                        region_controller.history[-1],
                        semantic_controller.history[-1],
                    )
            preferred_edit_tool = action.edit_tool.value if action.edit_tool is not None else None
            intervention_spec = action.intervention_spec.name if action.intervention_spec else None

            section = action.edit_target.section
            region_text = text if section is None else template.parse(text)[section]
            intervention = None
            if self.level >= 2:
                manifestor = Manifestor(
                    self.manifestor_lm,
                    self.logger,
                    self.manifestor_traces_chars,
                    inject_as=self.manifestor_injection_site,
                )
                try:
                    intervention = manifestor.manifest(action, region_text, text, feedback, traces)
                except ManifestationError as exc:
                    error = _bounded_history_text(exc)
                    records.append(
                        {
                            "backend": "react_v2",
                            "component": name,
                            "edit_target": action.edit_target.label,
                            "preferred_edit_tool": preferred_edit_tool,
                            "intervention_spec": intervention_spec,
                            "manifested_intervention": "",
                            "inject_as": None,
                            "feedback": _bounded_history_text(feedback),
                            "controller_sampling": controller_sampling,
                            "manifestor_error": error,
                            "executed_edit": [],
                            "react_iterations": 0,
                            "react_tool_calls": 0,
                            "react_steps": [],
                            "react_steps_truncated": 0,
                            "chat_messages": [
                                {
                                    "role": "user",
                                    "content": _bounded_history_text(f"Manifestor error: {exc}") or "",
                                }
                            ],
                            "dropped_reason": error,
                            "attempt_status": "dropped",
                            "tracking_id": _tracking_id(action),
                        }
                    )
                    dropped.append(name)
                    self._log(f"Component {name!r} dropped after Manifestor failure: {exc}")
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
                text,
                action.edit_target,
                action.edit_tool,
                intervention,
                feedback,
                traces,
                history,
                self.max_chars,
            )

            record = {
                "backend": "react_v2",
                "component": name,
                "edit_target": action.edit_target.label,
                "preferred_edit_tool": preferred_edit_tool,
                "intervention_spec": intervention_spec,
                "manifested_intervention": _bounded_history_text(intervention.text) if intervention is not None else "",
                "inject_as": intervention.inject_as if intervention is not None else None,
                "feedback": _bounded_history_text(feedback),
                "controller_sampling": controller_sampling,
                "manifestor_error": None,
                "executed_edit": _bounded_history_edits(result.executed_edit),
                "react_iterations": result.iterations,
                "react_tool_calls": result.tool_calls,
                "react_steps": [
                    {
                        "turn": step.turn,
                        "assistant": _bounded_history_text(step.assistant),
                        "action": step.action,
                        "observation": _bounded_history_text(step.observation),
                        "error": _bounded_history_text(step.error),
                        "executed_edit": _bounded_history_edits(step.executed_edit),
                    }
                    for step in result.steps[:MAX_HISTORY_STEPS]
                ],
                "react_steps_truncated": max(0, len(result.steps) - MAX_HISTORY_STEPS),
                "chat_messages": _react_chat_messages(result.steps),
                "dropped_reason": _bounded_history_text(result.dropped_reason),
                "attempt_status": "completed" if result.changed else "dropped",
                "tracking_id": _tracking_id(action),
            }
            records.append(record)

            if result.changed:
                proposal.new_texts[name] = result.new_text
                proposal.raw_lm_outputs[name] = result.final_output
                proposal.prompts[name] = intervention.text if intervention is not None else ""
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
                    "edit_tool": primary["preferred_edit_tool"],
                    "preferred_edit_tool": primary["preferred_edit_tool"],
                    "intervention_spec": primary["intervention_spec"],
                    "manifested_intervention": primary["manifested_intervention"],
                    "executed_edit": primary["executed_edit"],
                    "controller_sampling": primary["controller_sampling"],
                    "branch_history_length": len(history),
                    "three_role_actions": records,
                    "attempt_records": records,
                    "revision_records": accepted_revisions,
                }
            )
        if dropped:
            proposal.metadata["react_v2_dropped"] = dropped
            proposal.metadata["length_capped_dropped"] = dropped
        return proposal, self

    def _log(self, message: str) -> None:
        """Forward a message to the run logger.

        Args:
            message: Diagnostic message.
        """
        if self.logger is not None:
            self.logger.log(message)
