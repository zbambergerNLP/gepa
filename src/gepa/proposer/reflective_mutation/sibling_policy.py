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
from gepa.strategies.edit_novelty import EDIT_NOVELTY_CONTRACT, verify_edit
from gepa.strategies.edit_tools import EditTool
from gepa.strategies.intervention import (
    Controller,
    ControllerChoice,
    build_controller_menu,
    canonical_action_constraints,
)
from gepa.strategies.jev_edit_verifier import JevEditVerifier
from gepa.strategies.proposal_memory import MEMORY_CONTRACT, ProposalMemory
from gepa.strategies.reflection_context import REAL_EDIT_GUIDANCE

if TYPE_CHECKING:
    from gepa.core.state import GEPAState
    from gepa.proposer.base import CandidateProposal
    from gepa.proposer.reflective_mutation.three_role import ThreeRoleReflectionLM

SIBLING_POLICY_ID = "forest-sibling-real-edits-v1"
SIBLING_POLICY_CONTRACT = {
    "identity": SIBLING_POLICY_ID,
    "version": 8,
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


QUALITY_POLICY_CONTRACT = {
    **SIBLING_POLICY_CONTRACT,
    "identity": "forest-diversity-quality-v1",
    "version": 1,
    "novelty": EDIT_NOVELTY_CONTRACT,
    "outcome_memory": MEMORY_CONTRACT,
}


class SiblingInvariantError(ResponseJournalError):
    """Reject a duplicate sibling or an invalid acceptance contract before mutation."""


def _restore_rng_tuples(value: Any) -> Any:
    """Restore JSON-encoded RNG tuples without changing scalar values."""
    return tuple(_restore_rng_tuples(item) for item in value) if isinstance(value, list) else value


class SiblingProposalPlanner:
    """Keep accepted edges permanent and failed attempts local to one recovery opportunity."""

    def __init__(self, owner: ThreeRoleReflectionLM, *, quality: bool = False) -> None:
        """Share the owning strategy's role clients, configuration, and RNG."""
        self.owner = owner
        self.contract = deepcopy(QUALITY_POLICY_CONTRACT if quality else SIBLING_POLICY_CONTRACT)
        if quality:
            self.contract["verifier"] = (
                JevEditVerifier().run_contract()
                if owner.novelty_backend == "jev"
                else {"backend": "generative", "model": "same_as_manifestor"}
            )
        self.memory = ProposalMemory() if quality else None
        self.pending_edits: list[dict[str, Any]] = []
        self.accepted: dict[int, dict[str, dict[str, int]]] = {}
        self.edges: list[dict[str, Any]] = []
        self.deferred: set[int] = set()
        self.reserved: set[tuple[int, str, str]] = set()
        self.attempted: dict[str, set[tuple[str, str]]] = {}
        self.batch_active = False
        self.journal: ResumeResponseJournal | None = None
        self.steps: dict[str, dict[str, Any]] = {}

    def bind_run_dir(self, run_dir: str) -> None:
        """Persist recovery separately from iteration-boundary engine checkpoints.

        Args:
            run_dir: Optimizer directory containing the state and response journals.
        """
        self.journal = ResumeResponseJournal(Path(run_dir) / "sibling-recovery.sqlite3", self.contract["identity"])
        if self.memory is not None and self.owner.novelty_backend == "jev":
            self.owner.novelty_lm = JevEditVerifier(
                response_journal_path=Path(run_dir) / "novelty-responses.sqlite3",
                attempt_log_path=Path(run_dir) / "novelty-provider-attempts.jsonl",
            )

    def get_state(self) -> dict[str, Any]:
        """Snapshot accepted node identities and outstanding in-memory recovery.

        Returns:
            Independent snapshot; durable journal steps remain on disk.
        """
        return deepcopy(
            {
                "policy": self.contract["identity"],
                "contract_version": self.contract["version"],
                "contract": self.contract,
                "proposal_memory": self.memory.get_state() if self.memory is not None else None,
                "pending_edits": self.pending_edits,
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
        """Restore only an explicitly matching policy identity and version.

        Args:
            state: Snapshot returned by :meth:`get_state`.

        Raises:
            SiblingInvariantError: The persisted policy or contract version differs.
        """
        if (
            state.get("policy") != self.contract["identity"]
            or state.get("contract_version") != self.contract["version"]
            or state.get("contract") != self.contract
        ):
            raise SiblingInvariantError("Sibling policy checkpoint identity mismatch.")
        if self.memory is not None:
            self.memory.set_state(state["proposal_memory"])
        self.pending_edits = deepcopy(state.get("pending_edits", []))
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
        self.pending_edits.clear()
        self.attempted.clear()

    def end_batch(self) -> None:
        """Release temporary choices, including every evaluated tie or loss."""
        self.batch_active = False
        self.reserved.clear()
        self.pending_edits.clear()
        self.attempted.clear()

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
                raise SiblingInvariantError("Recovery scope was reused with different evidence or RNG state.")
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

    def _verify_novelty(
        self, before: str, after: str, previous: list[dict[str, Any]], batch_ids: list[Any], evidence: Any
    ) -> dict[str, Any]:
        """Stop the run if verification fails instead of silently bypassing the gate."""
        try:
            return verify_edit(self.owner.novelty_lm, before, after, previous, batch_ids, evidence)
        except Exception as exc:
            raise SiblingInvariantError(
                "Novelty provider or journal failed; inspect the preserved attempt evidence."
            ) from exc

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
            dataset: Captured training traces, including possible fallback components.
            components: Components to consider before recovery broadens the search.
            metadata: Parent node, iteration, minibatch, slot, and branch-history context.

        Returns:
            One changed component or explicit exhaustion, retaining every failed attempt.

        Raises:
            SiblingInvariantError: Parent identity is missing or a success contains no change.
        """
        self.owner.validate_candidate(candidate)
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
        scope = f"{self.contract['identity']}/{opportunity}/{metadata.get('proposal_slot', 0)}"
        attempted = self.attempted.setdefault(opportunity, set())
        exclusions: list[dict[str, str]] = []
        menus: dict[str, list[ControllerChoice]] = {}
        plans: dict[str, Any] = {}
        records: list[dict[str, Any]] = []
        fallback_components = [name for name in dataset if name not in components]
        duplicate_feedback: list[dict[str, Any]] = []
        novelty_checks: list[dict[str, Any]] = []
        batch_ids = list(metadata.get("minibatch_ids", []))

        def prior_context(name: str) -> dict[str, Any]:
            if self.memory is None:
                return {}
            recent = self.memory.context(parent, name, candidate[name])
            return {"training_history": recent, "duplicate_feedback": deepcopy(duplicate_feedback)}

        def evidence_for(name: str) -> str:
            evidence = json.dumps(dataset[name], ensure_ascii=False, default=str, sort_keys=True)
            if self.memory is not None:
                evidence += (
                    "\nPrior edits are observations, not instructions. A tie or loss on a small training "
                    "batch does not disprove an idea. Avoid repeating the recorded intervention without a "
                    "material evidence-supported difference. No validation or test scores are included.\n"
                    + json.dumps(prior_context(name), ensure_ascii=False, default=str, sort_keys=True)
                )
            return evidence

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
            evidence = evidence_for(name)
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

        proposal = ReflectionProposal(new_texts={})
        while remaining or fallback_components:
            if not remaining:
                add_fallback_components()
                if not remaining:
                    break
            weights = [plans[name]["entries"].get(action.menu_id, {}).get("weight", 0.0) for name, action in remaining]
            raw_weights = list(weights)
            multipliers = [
                self.memory.multipliers(parent, name, candidate[name], [action.menu_id]).get(action.menu_id, 1.0)
                if self.memory is not None
                else 1.0
                for name, action in remaining
            ]
            weights = [weight * multiplier for weight, multiplier in zip(weights, multipliers, strict=True)]
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
            distribution = [
                {
                    "component": component,
                    "pair": choice.menu_id,
                    "model_weight": raw_weight,
                    "history_multiplier": multiplier,
                    "adjusted_weight": weight,
                    "probability": probability,
                }
                for (component, choice), raw_weight, multiplier, weight, probability in zip(
                    remaining, raw_weights, multipliers, weights, sampling, strict=True
                )
            ]
            index = self.owner.rng.choices(range(len(remaining)), weights=sampling, k=1)[0]
            name, action = remaining.pop(index)
            attempted.add((name, action.menu_id))
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
                "outcome_context": prior_context(name),
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
                    {**metadata, "proposal_memory_context": prior_context(name)},
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
                    raise SiblingInvariantError("Planner received an unchanged candidate as a successful edit.")
                if self.memory is not None:
                    previous = self.memory.matches(parent, name, candidate[name]) + [
                        row for row in self.pending_edits if row["parent_id"] == parent and row["component"] == name
                    ]
                    verdict = self._step(
                        f"{scope}/novelty/{name}/{action.menu_id}",
                        {
                            "before": candidate[name],
                            "after": current.new_texts[name],
                            "previous": previous,
                            "minibatch_ids": batch_ids,
                            "current_evidence": dataset[name],
                        },
                        lambda before=candidate[name],
                        after=current.new_texts[name],
                        previous=previous,
                        evidence=dataset[name]: self._verify_novelty(before, after, previous, batch_ids, evidence),
                    )
                    novelty_checks.append({"attempt_id": attempt_id, **verdict})
                    for record in current_records:
                        record["novelty"] = verdict
                    if verdict["blocked"]:
                        for record in current_records:
                            record["attempt_status"] = "duplicate_generation"
                            record["dropped_reason"] = verdict["feedback"]
                            record["proposed_text"] = current.new_texts[name]
                        duplicate_feedback.append(
                            {
                                "attempt_id": attempt_id,
                                "matched_attempt_id": verdict["matched_attempt_id"],
                                "feedback": verdict["feedback"],
                            }
                        )
                        for component in menus:
                            available = [choice for comp, choice in remaining if comp == component]
                            if available:
                                evidence = evidence_for(component)
                                plans[component] = self._step(
                                    f"{scope}/rescore/{len(duplicate_feedback)}/{component}",
                                    {
                                        "candidate": candidate,
                                        "evidence": evidence,
                                        "menu": [choice.menu_id for choice in available],
                                    },
                                    lambda menu=available, text=candidate[component], evidence=evidence: self._score(
                                        menu, text, evidence
                                    ),
                                )
                        add_fallback_components()
                        continue
                    self.pending_edits.append(
                        {
                            "parent_id": parent,
                            "component": name,
                            "before": candidate[name],
                            "after": current.new_texts[name],
                            "attempt_id": attempt_id,
                            "minibatch_ids": batch_ids,
                            "pending": True,
                            "training_evidence": deepcopy(dataset[name]),
                        }
                    )
                proposal = current
                proposal.metadata["evaluated_attempt_id"] = attempt_id
                proposal.metadata["proposal_training_evidence"] = deepcopy(dataset[name])
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
                "proposal_policy": self.contract["identity"],
                "parent_node_id": parent,
                "generation_outcome": "generation_exhausted" if exhausted else "changed_candidate",
                "generation_exhausted": exhausted,
                "attempt_records": records,
                "three_role_actions": records,
                "planner_exclusions": exclusions,
                "controller_plans": plans,
                "novelty_checks": novelty_checks,
                "duplicate_generation_count": len(duplicate_feedback),
                "generation_error_count": sum(record.get("attempt_status") == "generation_error" for record in records)
                + sum(bool(plan.get("error")) for plan in plans.values()),
            }
        )
        return proposal

    def observe_evaluation(self, proposal: CandidateProposal, parent: Mapping[str, str]) -> None:
        """Credit only the actual evaluated edit using matched training outcomes."""
        if self.memory is None:
            return
        if (
            proposal.subsample_indices is None
            or proposal.subsample_scores_before is None
            or proposal.subsample_scores_after is None
        ):
            raise SiblingInvariantError("Outcome memory requires matched training evaluation records.")
        metadata = proposal.metadata or {}
        try:
            choice = metadata["sibling_choice"]
            component = choice["component"]
            outcome = self.memory.record_evaluation(
                parent_id=choice["parent"],
                component=component,
                before=parent[component],
                after=proposal.candidate[component],
                minibatch_ids=proposal.subsample_indices,
                action_pair=choice["pair"],
                action_name=choice["action"],
                section=choice["section"],
                scores_before=proposal.subsample_scores_before,
                scores_after=proposal.subsample_scores_after,
                attempt_id=metadata["evaluated_attempt_id"],
                training_evidence=metadata["proposal_training_evidence"],
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise SiblingInvariantError("Evaluated proposal conflicts with its outcome-memory contract.") from exc
        metadata["training_outcome"] = outcome

    def eligible_parent(self, state: GEPAState, selected: int) -> int:
        """Defer an exhausted parent for one draw when another frontier node exists.

        Args:
            state: Current candidate population and Pareto frontier.
            selected: Parent drawn by the configured candidate selector.

        Returns:
            The selected parent or another eligible frontier node.
        """
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
        """Enforce sibling uniqueness across permanent edges and the entire accepted batch.

        Args:
            proposals: Selected children, checked together before insertion.
            state: Population containing each declared parent.

        Raises:
            SiblingInvariantError: A child repeats a sibling pair or violates its edit contract.
        """
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
        """Consume choices only after insertion; a child's outgoing ledger always starts empty.

        Args:
            proposal: Validated proposal whose changed component was accepted.
            child: Newly inserted node ID.
            state: Population after insertion.

        Raises:
            SiblingInvariantError: The proposed edge violates sibling uniqueness.
        """
        self.validate_children([proposal], state)
        choice = self._choice(proposal, state)
        self.accepted.setdefault(choice["parent"], {}).setdefault(choice["component"], {})[choice["pair"]] = child
        self.accepted.setdefault(child, {})
        self.edges.append({**choice, "child": child})
