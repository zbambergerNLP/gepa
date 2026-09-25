"""Plan real edits with accepted choices scoped to one parent node and component."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gepa.gepa_utils import select_program_candidate_from_pareto_front
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
    from gepa.core.state import GEPAState
    from gepa.proposer.base import CandidateProposal
    from gepa.proposer.reflective_mutation.three_role import ThreeRoleReflectionLM

SIBLING_POLICY_ID = "forest-sibling-real-edits-v1"
SIBLING_POLICY_CONTRACT = {
    "identity": SIBLING_POLICY_ID,
    "version": 7,
    "key": "parent_node_id/component/action/section",
    "consume": "changed_component_in_accepted_child_only",
    "inherit_exclusions": False,
    "recovery": "remaining_positive_pairs_then_executable_zero_pairs_without_replacement",
    "exhaustion": "defer_parent_for_next_draw_when_another_eligible_parent_exists",
    "editor": "single_response_atomic_batch_with_net_change",
    "manifestor": "structured_ready_or_incompatible",
    "checkpoint": "accepted_edges_and_rng_plus_hash_verified_recovery_journal",
    "sampling": "positive_support_uniform_mixture_without_replacement_then_uniform_zero",
    "exploration_epsilon": FULL_SUPPORT_EXPLORATION_EPSILON,
    "action_constraints": "same_full_canonical_catalog_in_all_three_roles",
    "generation_guidance": REAL_EDIT_GUIDANCE,
}


class SiblingInvariantError(ResponseJournalError):
    """Reject a duplicate sibling or an invalid acceptance contract before mutation."""


def _tuples(value: Any) -> Any:
    """Restore JSON-encoded RNG tuples without changing scalar values."""
    return tuple(_tuples(item) for item in value) if isinstance(value, list) else value


class SiblingProposalPlanner:
    """Keep accepted edges permanent and failed attempts local to one recovery opportunity."""

    def __init__(self, owner: ThreeRoleReflectionLM):
        self.owner = owner
        self.accepted: dict[int, dict[str, dict[str, int]]] = {}
        self.edges: list[dict[str, Any]] = []
        self.deferred: set[int] = set()
        self.reserved: set[tuple[int, str, str]] = set()
        self.attempted: dict[str, set[tuple[str, str]]] = {}
        self.batch_active = False
        self.journal: ResumeResponseJournal | None = None
        self.steps: dict[str, dict[str, Any]] = {}

    def bind_run_dir(self, run_dir: str) -> None:
        """Persist recovery separately from iteration-boundary engine checkpoints."""
        self.journal = ResumeResponseJournal(Path(run_dir) / "sibling-recovery.sqlite3", SIBLING_POLICY_ID)

    def get_state(self) -> dict[str, Any]:
        """Snapshot accepted node identities and outstanding in-memory recovery."""
        return deepcopy(
            {
                "policy": SIBLING_POLICY_ID,
                "accepted": self.accepted,
                "edges": self.edges,
                "deferred": sorted(self.deferred),
                "reserved": sorted(self.reserved),
                "attempted": {key: sorted(value) for key, value in self.attempted.items()},
                "batch_active": self.batch_active,
                # On-disk steps are already committed and hash-verified by the response journal.
                "steps": self.steps if self.journal is None else {},
            }
        )

    def set_state(self, state: Mapping[str, Any]) -> None:
        """Restore only an explicitly matching policy identity."""
        if state.get("policy") != SIBLING_POLICY_ID:
            raise SiblingInvariantError("Sibling policy checkpoint identity mismatch.")
        self.accepted = {int(key): deepcopy(value) for key, value in state["accepted"].items()}
        self.edges = deepcopy(state["edges"])
        self.deferred = set(state["deferred"])
        self.reserved = {tuple(item) for item in state["reserved"]}
        self.attempted = {key: {tuple(item) for item in value} for key, value in state["attempted"].items()}
        self.batch_active = bool(state["batch_active"])
        self.steps = deepcopy(state["steps"])

    def begin_batch(self) -> None:
        """Start reservations that span all same-parent proposals in this batch."""
        self.batch_active = True
        self.reserved.clear()
        self.attempted.clear()

    def end_batch(self) -> None:
        """Release temporary choices, including every evaluated tie or loss."""
        self.batch_active = False
        self.reserved.clear()
        self.attempted.clear()

    def _step(self, scope: str, request: dict[str, Any], run: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        """Replay a completed planner step and its exact post-step RNG state."""
        # Match the role prompts' handling of adapter-specific trace values (e.g. prediction objects).
        context = json.loads(json.dumps({**request, "rng_before": self.owner.rng.getstate()}, default=str))
        digest = canonical_request_digest(context)
        cached = self.journal.load(scope, 0, digest) if self.journal else self.steps.get(scope)
        if cached is not None:
            if cached["request_digest"] != digest:
                raise SiblingInvariantError("Recovery scope was reused with different evidence or RNG state.")
            self.owner.rng.setstate(_tuples(cached["rng_after"]))
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
        """Exhaust finite pairs on the same evidence until one changed candidate exists."""
        if (
            metadata is None
            or not isinstance(metadata.get("candidate_idx"), int)
            or isinstance(metadata.get("candidate_idx"), bool)
        ):
            raise SiblingInvariantError("Sibling planning requires the parent's candidate_idx node identity.")
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
        scope = f"{SIBLING_POLICY_ID}/{opportunity}/{metadata.get('proposal_slot', 0)}"
        attempted = self.attempted.setdefault(opportunity, set())
        exclusions: list[dict[str, str]] = []
        menus: dict[str, list[ControllerChoice]] = {}
        plans: dict[str, Any] = {}
        records: list[dict[str, Any]] = []
        fallback_components = [name for name in dataset if name not in components]

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
                if action.menu_id in self.accepted.get(parent, {}).get(name, {}):
                    reason = "accepted_sibling"
                elif (parent, name, action.menu_id) in self.reserved:
                    reason = "reserved_sibling_in_batch"
                elif (name, action.menu_id) in attempted:
                    reason = "already_attempted_in_parent_minibatch"
                if reason:
                    exclusions.append({"component": name, "pair": action.menu_id, "reason": reason})
                else:
                    available.append(action)
            if not available:
                return
            menus[name] = available
            evidence = json.dumps(dataset[name], ensure_ascii=False, default=str)
            plans[name] = self._step(
                f"{scope}/score/{name}",
                {"candidate": candidate, "evidence": evidence, "menu": [action.menu_id for action in available]},
                lambda menu=available, text=candidate[name], evidence=evidence: self._score(menu, text, evidence),
            )

        for name in dict.fromkeys(components):
            prepare_component(name)
        remaining = [(name, action) for name, menu in menus.items() for action in menu]

        def add_fallback_components() -> None:
            for name in fallback_components:
                prepare_component(name)
                remaining.extend((name, action) for action in menus.get(name, []))
            fallback_components.clear()

        proposal = ReflectionProposal({}, {}, {}, {})
        while remaining or fallback_components:
            if not remaining:
                add_fallback_components()
                if not remaining:
                    break
            weights = [plans[name]["entries"].get(action.menu_id, {}).get("weight", 0.0) for name, action in remaining]
            total = sum(weights)
            if not total and fallback_components:
                add_fallback_components()
                continue
            positives = sum(weight > 0 for weight in weights)
            epsilon = FULL_SUPPORT_EXPLORATION_EPSILON
            sampling = (
                [((1 - epsilon) * weight / total + epsilon / positives) if weight > 0 else 0.0 for weight in weights]
                if total
                else [1 / len(remaining)] * len(remaining)
            )
            index = self.owner.rng.choices(range(len(remaining)), weights=sampling, k=1)[0]
            name, action = remaining.pop(index)
            attempted.add((name, action.menu_id))
            direction = plans[name]["entries"].get(action.menu_id, {}).get("direction")
            selection = {
                "policy": SIBLING_POLICY_ID,
                "sampled": [action.menu_id],
                "sampled_reasonings": [direction],
                "sampled_probabilities": [sampling[index]],
                "phase": "positive" if total else "zero_weight_fallback",
                "parent_node_id": parent,
                "remaining_pair_count": len(remaining),
            }

            def execute(
                name: str = name,
                action: ControllerChoice = action,
                direction: str | None = direction,
                selection: dict[str, Any] = selection,
            ) -> dict[str, Any]:
                if not direction:
                    retry = self._score([action], candidate[name], json.dumps(dataset[name], default=str))
                    direction = retry["entries"].get(action.menu_id, {}).get("direction")
                    selection["direction_recovery"] = retry
                    if not direction:
                        return asdict(
                            ReflectionProposal(
                                {},
                                {},
                                {},
                                {
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
            records.extend(current.metadata.get("attempt_records", []))
            if current.new_texts:
                if current.new_texts.get(name) == candidate[name]:
                    raise SiblingInvariantError("Planner received an unchanged candidate as a successful edit.")
                proposal = current
                self.reserved.add((parent, name, action.menu_id))
                proposal.metadata["sibling_choice"] = {
                    "parent": parent,
                    "component": name,
                    "pair": action.menu_id,
                    "action": action.semantic_action.name if action.semantic_action else None,
                    "section": action.edit_target.section,
                }
                break
            add_fallback_components()
        exhausted = not proposal.new_texts
        if exhausted:
            self.deferred.add(parent)
        proposal.metadata.update(
            {
                "proposal_policy": SIBLING_POLICY_ID,
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

    def eligible_parent(self, state: GEPAState, selected: int) -> int:
        """Defer an exhausted parent for one draw when another frontier node exists."""
        if selected not in self.deferred:
            return selected
        mapping = {key: values - self.deferred for key, values in state.get_pareto_front_mapping().items()}
        mapping = {key: values for key, values in mapping.items() if values}
        self.deferred.discard(selected)
        if not mapping:
            return selected
        return select_program_candidate_from_pareto_front(mapping, state.per_program_tracked_scores, self.owner.rng)

    def _choice(self, proposal: CandidateProposal, state: GEPAState) -> dict[str, Any]:
        """Validate the actual changed component and its selected parent before insertion."""
        choice = (proposal.metadata or {}).get("sibling_choice")
        if not isinstance(choice, dict) or proposal.parent_program_ids != [choice.get("parent")]:
            raise SiblingInvariantError("Accepted proposal lacks its exact parent/component/action contract.")
        parent = state.program_candidates[choice["parent"]]
        changed = {name for name in parent if parent[name] != proposal.candidate.get(name)}
        if changed != {choice["component"]} or parent.keys() != proposal.candidate.keys():
            raise SiblingInvariantError("Accepted child must contain exactly its declared changed component.")
        template = self.owner.templates[self.owner._component_kind(choice["component"])]
        old_sections = template.parse(parent[choice["component"]])
        new_sections = template.parse(proposal.candidate[choice["component"]])
        if {section for section in old_sections if old_sections[section] != new_sections[section]} != {
            choice["section"]
        }:
            raise SiblingInvariantError("Child changed sections outside its declared action/section pair.")
        menu = build_controller_menu(template, choice["component"], self.owner.edit_tools, 2, rng=self.owner.rng)
        if not any(
            action.menu_id == choice["pair"]
            and action.edit_target.section == choice["section"]
            and action.semantic_action is not None
            and action.semantic_action.name == choice["action"]
            for action in menu
        ):
            raise SiblingInvariantError("Child declares a noncanonical action/section pair.")
        return choice

    def validate_children(self, proposals: list[CandidateProposal], state: GEPAState) -> None:
        """Enforce sibling uniqueness across permanent edges and the entire accepted batch."""
        used = {
            (parent, component, pair)
            for parent, components in self.accepted.items()
            for component, pairs in components.items()
            for pair in pairs
        }
        for proposal in proposals:
            choice = self._choice(proposal, state)
            key = (choice["parent"], choice["component"], choice["pair"])
            if key in used:
                raise SiblingInvariantError(f"Duplicate accepted sibling choice: {key}")
            used.add(key)

    def accept_child(self, proposal: CandidateProposal, child: int, state: GEPAState) -> None:
        """Consume choices only after insertion; a child's outgoing ledger always starts empty."""
        self.validate_children([proposal], state)
        choice = self._choice(proposal, state)
        self.accepted.setdefault(choice["parent"], {}).setdefault(choice["component"], {})[choice["pair"]] = child
        self.accepted.setdefault(child, {})
        self.edges.append({**choice, "child": child})
