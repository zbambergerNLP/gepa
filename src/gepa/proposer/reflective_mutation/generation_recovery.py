"""Recover failed edit generation on the same selected component and training evidence."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gepa.proposer.reflective_mutation.reflection_lm import ReflectionProposal
from gepa.response_journal import (
    ResponseJournalError,
    ResumeResponseJournal,
    canonical_request_digest,
    response_journal_scope,
)
from gepa.strategies.action_space import FULL_SUPPORT_EXPLORATION_EPSILON, IncompleteActionDistributionError
from gepa.strategies.edit_tools import EditTool
from gepa.strategies.intervention import (
    Controller,
    ControllerChoice,
    build_controller_menu,
    canonical_action_constraints,
)
from gepa.strategies.reflection_context import REAL_EDIT_GUIDANCE

if TYPE_CHECKING:
    from gepa.proposer.reflective_mutation.three_role import ThreeRoleReflectionLM

RECOVERY_POLICY_ID = "forest-real-edit-recovery-v1"
RECOVERY_POLICY_CONTRACT = {
    "identity": RECOVERY_POLICY_ID,
    "version": 1,
    "recovery": "remaining_positive_pairs_then_executable_zero_pairs_without_replacement",
    "exhaustion": "return_no_candidate_without_changing_parent_or_component_selection",
    "editor": "single_response_atomic_batch_with_net_change",
    "manifestor": "structured_ready_or_incompatible",
    "checkpoint": "rng_plus_hash_verified_recovery_journal",
    "sampling": "positive_support_uniform_mixture_without_replacement_then_uniform_zero",
    "exploration_epsilon": FULL_SUPPORT_EXPLORATION_EPSILON,
    "action_constraints": "same_full_canonical_catalog_in_all_three_roles",
    "generation_guidance": REAL_EDIT_GUIDANCE,
}


class GenerationRecoveryError(ResponseJournalError):
    """Reject a recovery result whose evidence or identity does not match."""


def _restore_rng_tuples(value: Any) -> Any:
    """Restore JSON-encoded RNG tuples without changing scalar values."""
    return tuple(_restore_rng_tuples(item) for item in value) if isinstance(value, list) else value


class GenerationRecoveryPlanner:
    """Retry generation before evaluation, without remembering prior children or scores."""

    def __init__(self, owner: ThreeRoleReflectionLM) -> None:
        self.owner = owner
        self.contract = deepcopy(RECOVERY_POLICY_CONTRACT)
        self.journal: ResumeResponseJournal | None = None
        self.steps: dict[str, dict[str, Any]] = {}

    def bind_run_dir(self, run_dir: str) -> None:
        """Persist completed generation steps before the next engine checkpoint."""
        self.journal = ResumeResponseJournal(Path(run_dir) / "generation-recovery.sqlite3", self.contract["identity"])

    def get_state(self) -> dict[str, Any]:
        """Snapshot the recovery contract and any steps not already on disk."""
        return deepcopy({"contract": self.contract, "steps": self.steps if self.journal is None else {}})

    def set_state(self, state: Mapping[str, Any]) -> None:
        """Restore only a matching recovery contract."""
        if state.get("contract") != self.contract:
            raise GenerationRecoveryError("Generation recovery checkpoint identity mismatch.")
        self.steps = deepcopy(state["steps"])

    def _step(self, scope: str, request: dict[str, Any], run: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        """Replay a completed planner step and its exact post-step RNG state.

        Args:
            scope: Stable scoring or edit identity within a recovery opportunity.
            request: Evidence and action context bound to the saved result.
            run: Role work to execute only when no completed result exists.

        Returns:
            Fresh or replayed role result, with the shared RNG advanced identically.

        Raises:
            ResponseJournalError: Saved evidence, RNG state, or result hashes differ.
        """
        # Match the role prompts' handling of adapter-specific trace values (e.g. prediction objects).
        context = json.loads(
            json.dumps(
                {**request, "policy_contract": self.contract, "rng_before": self.owner.rng.getstate()},
                default=str,
            )
        )
        digest = canonical_request_digest(context)
        cached = self.journal.load(scope, 0, digest) if self.journal else self.steps.get(scope)
        if cached is not None:
            if cached["request_digest"] != digest:
                raise GenerationRecoveryError("Recovery scope was reused with different evidence or RNG state.")
            self.owner.rng.setstate(_restore_rng_tuples(cached["rng_after"]))
            return deepcopy(cached["result"])
        # Role response journals preserve completed provider calls inside an interrupted step.
        with response_journal_scope(scope):
            result = run()
        record = {"request_digest": digest, "rng_after": self.owner.rng.getstate(), "result": result}
        if self.journal:
            self.journal.store(scope, 0, digest, record)
        else:
            self.steps[scope] = deepcopy(record)
        return result

    def _score(self, menu: list[ControllerChoice], text: str, evidence: str) -> dict[str, Any]:
        """Retain all Controller replies, including malformed distributions."""
        selector = Controller(
            menu,
            self.owner.controller_lm,
            k=len(menu),
            rng=self.owner.rng,
            require_full_support=True,
            text_limits=self.owner.text_limits,
            allow_zero_weights=True,
            action_constraints=REAL_EDIT_GUIDANCE + "\n" + canonical_action_constraints(),
        )
        try:
            distribution = selector._generate_distribution(self.owner.rng, text, evidence)
            return {
                "entries": {
                    action.menu_id: {"weight": weight, "direction": direction}
                    for action, weight, direction in distribution.entries
                },
                "raw_outputs": selector.raw_outputs,
            }
        except IncompleteActionDistributionError as exc:
            return {"entries": {}, "error": str(exc), "raw_outputs": selector.raw_outputs}

    def _impossible_reason(self, action: ControllerChoice, body: str) -> str | None:
        """Exclude only directly checkable operator preconditions."""
        if action.edit_tool not in self.owner.edit_tools:
            return "selected_direct_operator_unavailable_in_atomic_batch_basis"
        if not body and action.edit_tool is not EditTool.INSERT_TEXT:
            return "nonempty_target_required"
        if action.edit_tool is EditTool.MOVE_TEXT and len(body) < 2:
            return "no_distinct_move_destination"
        return None

    def generate(
        self,
        candidate: dict[str, str],
        dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components: list[str],
        metadata: Mapping[str, Any] | None,
    ) -> ReflectionProposal:
        """Exhaust finite pairs on the same evidence until one changed candidate exists.

        Args:
            candidate: Canonically formatted parent component texts.
            dataset: Captured training traces for the selected components.
            components: Components already selected by the configured module selector.
            metadata: Parent node, iteration, minibatch, slot, and branch-history context.

        Returns:
            One changed component or explicit exhaustion, retaining every failed attempt.

        Raises:
            GenerationRecoveryError: Parent identity is missing or a success contains no change.
        """
        self.owner.validate_candidate(candidate)
        if (
            metadata is None
            or not isinstance(metadata.get("candidate_idx"), int)
            or isinstance(metadata.get("candidate_idx"), bool)
        ):
            raise GenerationRecoveryError("Generation recovery requires the parent's candidate_idx node identity.")
        parent = metadata["candidate_idx"]
        opportunity = canonical_request_digest(
            {
                "parent": parent,
                "iteration": metadata.get("optimizer_iteration", metadata.get("iteration_id")),
                "minibatch_ids": metadata["minibatch_ids"]
                if "minibatch_ids" in metadata
                else json.dumps(dataset, ensure_ascii=False, default=str),
            }
        )
        scope = f"{self.contract['identity']}/{opportunity}/{metadata.get('proposal_slot', 0)}"
        exclusions: list[dict[str, str]] = []
        menus: dict[str, list[ControllerChoice]] = {}
        plans: dict[str, Any] = {}
        records: list[dict[str, Any]] = []

        def evidence_for(name: str) -> str:
            return json.dumps(dataset[name], ensure_ascii=False, default=str, sort_keys=True)

        def prepare_component(name: str) -> None:
            if name not in candidate or not dataset.get(name):
                exclusions.append({"component": name, "reason": "no_training_trace_for_component"})
                return
            template = self.owner.templates[self.owner._component_kind(name)]
            sections = template.parse(candidate[name])
            menu = build_controller_menu(template, name, self.owner.edit_tools, 2, rng=self.owner.rng)
            available = []
            for action in menu:
                reason = self._impossible_reason(action, sections[action.edit_target.section])
                if reason:
                    exclusions.append({"component": name, "pair": action.menu_id, "reason": reason})
                else:
                    available.append(action)
            if not available:
                return
            menus[name] = available
            evidence = evidence_for(name)
            plans[name] = self._step(
                f"{scope}/score/{name}",
                {"candidate": candidate, "evidence": evidence, "menu": [action.menu_id for action in available]},
                lambda menu=available, text=candidate[name], evidence=evidence: self._score(menu, text, evidence),
            )

        for name in dict.fromkeys(components):
            prepare_component(name)
        remaining = [(name, action) for name, menu in menus.items() for action in menu]

        proposal = ReflectionProposal(new_texts={})
        while remaining:
            weights = [plans[name]["entries"].get(action.menu_id, {}).get("weight", 0.0) for name, action in remaining]
            total = sum(weights)
            positives = sum(weight > 0 for weight in weights)
            epsilon = FULL_SUPPORT_EXPLORATION_EPSILON
            sampling = (
                [((1 - epsilon) * weight / total + epsilon / positives) if weight > 0 else 0.0 for weight in weights]
                if total
                else [1 / len(remaining)] * len(remaining)
            )
            distribution = [
                {
                    "component": component,
                    "pair": choice.menu_id,
                    "model_weight": weight,
                    "probability": probability,
                }
                for (component, choice), weight, probability in zip(remaining, weights, sampling, strict=True)
            ]
            index = self.owner.rng.choices(range(len(remaining)), weights=sampling, k=1)[0]
            name, action = remaining.pop(index)
            direction = plans[name]["entries"].get(action.menu_id, {}).get("direction")
            selection = {
                "policy": self.contract["identity"],
                "sampled": [action.menu_id],
                "sampled_reasonings": [direction],
                "sampled_probabilities": [sampling[index]],
                "phase": "positive" if total else "zero_weight_fallback",
                "parent_node_id": parent,
                "remaining_pair_count": len(remaining),
                "distribution": distribution,
            }

            def execute(
                name: str = name,
                action: ControllerChoice = action,
                direction: str | None = direction,
                selection: dict[str, Any] = selection,
            ) -> dict[str, Any]:
                if not direction:
                    retry = self._score([action], candidate[name], evidence_for(name))
                    direction = retry["entries"].get(action.menu_id, {}).get("direction")
                    selection["direction_recovery"] = retry
                    if not direction:
                        return asdict(
                            ReflectionProposal(
                                new_texts={},
                                metadata={
                                    "attempt_records": [
                                        {
                                            "component": name,
                                            "action_choice": action.menu_id,
                                            "semantic_action": action.semantic_action.name
                                            if action.semantic_action
                                            else None,
                                            "action_target_section": action.edit_target.section,
                                            "attempt_status": "generation_error",
                                            "dropped_reason": "Controller did not provide an executable direction.",
                                            "controller_sampling": selection,
                                        }
                                    ]
                                },
                            )
                        )
                result, _ = self.owner._reflect_operated(
                    candidate,
                    dataset,
                    [name],
                    metadata,
                    selected=(action, selection, direction),
                    require_edit=True,
                )
                return asdict(result)

            payload = self._step(
                f"{scope}/edit/{name}/{action.menu_id}",
                {
                    "candidate": candidate,
                    "dataset": dataset,
                    "direction": direction,
                    "selection": selection,
                    "branch_history": metadata.get("branch_edit_history", []),
                },
                execute,
            )
            current = ReflectionProposal(**payload)
            attempt_id = f"{scope}/edit/{name}/{action.menu_id}"
            current_records = current.metadata.get("attempt_records", [])
            for record in current_records:
                record["attempt_id"] = attempt_id
            records.extend(current_records)
            if current.new_texts:
                if current.new_texts.get(name) == candidate[name]:
                    raise GenerationRecoveryError("Planner received an unchanged candidate as a successful edit.")
                proposal = current
                proposal.metadata["evaluated_attempt_id"] = attempt_id
                break
        exhausted = not proposal.new_texts
        proposal.metadata.update(
            {
                "proposal_policy": self.contract["identity"],
                "parent_node_id": parent,
                "generation_outcome": "generation_exhausted" if exhausted else "changed_candidate",
                "generation_exhausted": exhausted,
                "attempt_records": records,
                "three_role_actions": records,
                "planner_exclusions": exclusions,
                "controller_plans": plans,
                "generation_error_count": sum(record.get("attempt_status") == "generation_error" for record in records)
                + sum(bool(plan.get("error")) for plan in plans.values()),
            }
        )
        return proposal
