"""Apply candidate documents inside Harbor 0.22.0's Terminus agent.

The summarization and recovery methods are adapted from harbor-framework/harbor
v0.22.0, src/harbor/agents/terminus_2/terminus_2.py (Apache-2.0). They retain
the document and context flow while the shared provider policy replaces nested
transport retries and propagates exhausted provider requests.
See HARBOR_LICENSE for the upstream license.
This module deliberately has no GEPA imports: Harbor uses its own interpreter.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from types import MethodType
from typing import Any, NoReturn, cast

from harbor.agents.terminus_2 import Terminus2
from harbor.agents.terminus_2.tmux_session import TmuxSession
from harbor.environments.base import BaseEnvironment
from harbor.llms.base import ContextLengthExceededError, LLMResponse, OutputLengthExceededError
from harbor.llms.chat import Chat
from harbor.llms.lite_llm import LiteLLM
from harbor.models.trajectories import Step, SubagentTrajectoryRef
from litellm.exceptions import BadRequestError

from examples.common.provider_retries import (
    PROVIDER_RETRY_KEY,
    ProviderRequestError,
    is_provider_request_error,
    provider_retry_kwargs,
)
from examples.terminalbench.token_usage import observe_harbor


class PromptedTerminus(Terminus2):
    """Run the fixed terminal agent with one candidate's reusable documents."""

    def __init__(
        self,
        logs_dir: Path,
        prompt_template_path: str,
        document_bundle_path: str,
        token_limits: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Load the immutable prompts and skills for one candidate evaluation.

        Args:
            logs_dir: Harbor trial's agent log directory.
            prompt_template_path: Rendered main prompt beside the bundle.
            document_bundle_path: Candidate bundle produced by GEPA.
            token_limits: Campaign limits to record with raw provider usage.
            **kwargs: Standard pinned Terminus model settings.

        Raises:
            ValueError: The bundle version, content digest, or runtime differs.
        """
        if version("harbor") != "0.22.0":
            raise ValueError("Document prompts require Harbor 0.22.0")
        self._bundle_path = Path(document_bundle_path).resolve()
        self._bundle = json.loads(self._bundle_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(
            json.dumps(self._bundle["documents"], sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        if self._bundle["version"] != 2 or digest != self._bundle["digest"]:
            raise ValueError("Invalid Terminal Bench document bundle")
        self._candidate_prompt_template_path = Path(prompt_template_path).resolve()
        if self._candidate_prompt_template_path.parent != self._bundle_path.parent:
            raise ValueError("Prompt and document bundle must belong to the same evaluation")
        # Task-provided skills and MCP servers are not part of the candidate.
        kwargs.pop("skills_dir", None)
        kwargs.pop("mcp_servers", None)
        requested_model = kwargs.get("model_name", "")
        if requested_model.startswith("hosted_vllm/") and requested_model.count("/") > 1:
            # Harbor 0.22 only accepts a short registry alias; LiteLLM accepts the full Hub model ID.
            kwargs["model_name"] = "hosted_vllm/" + requested_model.rsplit("/", 1)[1]
        super().__init__(logs_dir=logs_dir, skills_dir="/opt/gepa-skills", mcp_servers=[], **kwargs)
        if requested_model != kwargs.get("model_name", ""):
            self._model_name = requested_model
            cast(LiteLLM, self._llm)._model_name = requested_model
        llm = cast(LiteLLM, self._llm)
        llm._llm_kwargs.update(provider_retry_kwargs(logs_dir / "provider-attempts.jsonl", "task_agent"))
        # The transport wrapper owns all retries; Harbor's decorator would multiply them.
        llm.call = MethodType(cast(Any, LiteLLM.call).__wrapped__, llm)

        def translate_error(e: Exception) -> NoReturn:
            """Preserve Harbor's recognition of context errors in HTTP 400 bodies."""
            cause = e.__cause__
            if isinstance(e, ProviderRequestError) and isinstance(cause, BadRequestError):
                if llm._is_context_length_error(cause):
                    raise ContextLengthExceededError from cause
            LiteLLM._handle_litellm_error(llm, e)

        llm._handle_litellm_error = translate_error
        logs_dir.mkdir(parents=True, exist_ok=True)
        if token_limits is not None:
            observe_harbor(self._llm, logs_dir / "token-usage.jsonl", token_limits)
        self._llm_call_kwargs.update({
            key: llm._llm_kwargs[key] for key in (PROVIDER_RETRY_KEY, "num_retries", "max_retries")
        })
        (logs_dir / "document-bundle.json").write_text(self._bundle_path.read_text(encoding="utf-8"), encoding="utf-8")

    def _document(self, component: str, **fields: str) -> str:
        """Render candidate guidance with the fixed runtime inputs.

        Args:
            component: Prompt component in the bundle.
            **fields: Observations and task inputs supplied by the agent.

        Returns:
            Model input containing literal candidate text and runtime fields.
        """
        return self._bundle["prompts"][component].format(**fields)

    def _get_prompt_template_path(self) -> Path:
        """Return this candidate's main instruction template."""
        return self._candidate_prompt_template_path

    def _get_timeout_template_path(self) -> Path:
        """Return this candidate's command-timeout guidance."""
        return self._bundle_path.parent / "timeout.txt"

    async def setup(self, environment: BaseEnvironment) -> None:
        """Install candidate skills before starting the standard terminal.

        Args:
            environment: Isolated task environment owned by Harbor.
        """
        await environment.upload_dir(source_dir=self._bundle_path.parent / "skills", target_dir="/opt/gepa-skills")
        await super().setup(environment)

    def _get_completion_confirmation_message(self, terminal_output: str) -> str:
        """Render editable completion guidance around the fixed confirmation gate.

        Args:
            terminal_output: Actual current terminal state.

        Returns:
            Completion prompt with the confirmation protocol intact.
        """
        return self._document("completion", terminal_state=terminal_output)

    async def _handle_llm_interaction(
        self, chat: Chat, prompt: str, original_instruction: str = "", session: TmuxSession | None = None
    ) -> tuple:
        """Attach editable repair instructions to real parser feedback.

        Args:
            chat: Current main-agent conversation.
            prompt: Pending model input.
            original_instruction: Benchmark task instruction.
            session: Running terminal session.

        Returns:
            Standard Terminus interaction result with repair guidance on errors.
        """
        commands, complete, feedback, analysis, plan, response = await super()._handle_llm_interaction(
            chat, prompt, original_instruction, session
        )
        if "ERROR:" in feedback:
            feedback += "\n\n" + self._document("parse_error")
        return commands, complete, feedback, analysis, plan, response

    async def _summarize(
        self, chat: Chat, original_instruction: str, session: TmuxSession
    ) -> tuple[str, list[SubagentTrajectoryRef] | None]:
        """Render candidate context-management prompts through the pinned Harbor flow.

        Args:
            chat: Current main-agent conversation.
            original_instruction: Unmodified benchmark task instruction.
            session: Terminal session supplying observed state.

        Returns:
            Handoff prompt and summarization trajectory references.
        """
        if len(chat.messages) == 0:
            return (original_instruction, None)
        self._summarization_count += 1
        subagent_trajectory_refs = []
        summary_session_id = f"{self._session_id}-summarization-{self._summarization_count}-summary"
        steps_to_include = 1 + (len(chat.messages) - 1) // 2
        summary_steps, step_id_counter = self._prepare_copied_trajectory_steps(steps_to_include)
        summary_prompt = self._document("summary", original_instruction=original_instruction)
        summary_response, summary_trajectory_ref = await self._run_subagent(
            prompt=summary_prompt,
            message_history=chat.messages,
            steps=summary_steps,
            session_id=summary_session_id,
            agent_name="terminus-2-summarization-summary",
            filename_suffix="summary",
            summary_text=f"Context summarization {self._summarization_count}: Step 1 - Summary generation",
            subagent_name_for_logging="summary generation LLM call",
        )
        subagent_trajectory_refs.append(summary_trajectory_ref)
        current_screen = await session.capture_pane(capture_entire=False)
        questions_session_id = f"{self._session_id}-summarization-{self._summarization_count}-questions"
        questions_steps = []
        question_prompt = self._document(
            "summary_questions",
            original_instruction=original_instruction,
            summary=summary_response.content,
            terminal_state=current_screen,
        )
        questions_response, questions_trajectory_ref = await self._run_subagent(
            prompt=question_prompt,
            message_history=[],
            steps=questions_steps,
            session_id=questions_session_id,
            agent_name="terminus-2-summarization-questions",
            filename_suffix="questions",
            summary_text=f"Context summarization {self._summarization_count}: Step 2 - Question asking",
            subagent_name_for_logging="questions subagent",
        )
        model_questions = questions_response.content
        subagent_trajectory_refs.append(questions_trajectory_ref)
        answers_session_id = f"{self._session_id}-summarization-{self._summarization_count}-answers"
        answers_steps, step_id_counter = self._prepare_copied_trajectory_steps(steps_to_include)
        answers_steps.append(
            Step(
                step_id=step_id_counter,
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="user",
                message=summary_prompt,
                is_copied_context=True,
            )
        )
        step_id_counter += 1
        answers_steps.append(
            Step(
                step_id=step_id_counter,
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="agent",
                model_name=summary_response.model_name or self._model_name,
                message=summary_response.content,
                reasoning_content=summary_response.reasoning_content,
                is_copied_context=True,
                extra={"note": "Copied from summary subagent - metrics already recorded there"},
            )
        )
        step_id_counter += 1
        answer_request_prompt = self._document("summary_answers", questions=model_questions)
        answers_message_history = chat.messages + [
            {"role": "user", "content": summary_prompt},
            {"role": "assistant", "content": summary_response.content},
        ]
        answers_response, answers_trajectory_ref = await self._run_subagent(
            prompt=answer_request_prompt,
            message_history=answers_message_history,
            steps=answers_steps,
            session_id=answers_session_id,
            agent_name="terminus-2-summarization-answers",
            filename_suffix="answers",
            summary_text=f"Context summarization {self._summarization_count}: Step 3 - Answer providing",
            subagent_name_for_logging="answers subagent",
        )
        subagent_trajectory_refs.append(answers_trajectory_ref)
        chat._messages = [
            chat.messages[0],
            {"role": "user", "content": question_prompt},
            {"role": "assistant", "content": model_questions},
        ]
        chat.reset_response_chain()
        handoff_prompt = self._document("handoff", answers=answers_response.content)
        return (handoff_prompt, subagent_trajectory_refs)

    async def _check_proactive_summarization(
        self, chat: Chat, original_instruction: str, session: TmuxSession,
    ) -> tuple[str, list[SubagentTrajectoryRef] | None] | None:
        """Keep Harbor's trigger while propagating exhausted provider requests."""
        free_tokens = self._llm.get_model_context_limit() - self._count_total_tokens(chat)
        if free_tokens < self._proactive_summarization_threshold:
            try:
                return await self._summarize(chat, original_instruction, session)
            except Exception as error:
                if is_provider_request_error(error):
                    raise
                self.logger.error(f"Error in proactively summarizing: {error}")
        return None

    async def _query_llm(
        self, chat: Chat, prompt: str, original_instruction: str = "", session: TmuxSession | None = None
    ) -> LLMResponse:
        """Query the model using candidate retry and fallback instructions.

        Args:
            chat: Current main-agent conversation.
            prompt: Next model input.
            original_instruction: Unmodified benchmark task instruction.
            session: Terminal session used during context recovery.

        Returns:
            Model response following the shared provider-attempt policy.

        Raises:
            ContextLengthExceededError: Summarization is disabled.
        """
        try:
            start_time = time.time()
            llm_response = await chat.chat(prompt, **self._llm_call_kwargs)
            end_time = time.time()
            request_time_ms = (end_time - start_time) * 1000
            self._api_request_times.append(request_time_ms)
            return llm_response
        except ContextLengthExceededError:
            if not self._enable_summarize:
                self.logger.debug("Context length exceeded and summarization is OFF.")
                raise
            self.logger.debug("Context length exceeded. Using fallback summarization.")
            if session is None:
                raise RuntimeError("Cannot handle context length error without session")
            self._unwind_messages_to_free_tokens(chat, target_free_tokens=4000)
            summary_prompt = None
            try:
                self.logger.debug("SUMMARIZATION: Attempting full summary")
                summary_prompt, subagent_trajectory_refs = await self._summarize(chat, original_instruction, session)
                self._pending_subagent_refs = subagent_trajectory_refs
                self._pending_handoff_prompt = summary_prompt
                self.logger.debug("SUMMARIZATION: Full summary succeeded")
            except Exception as e:
                if is_provider_request_error(e):
                    raise
                self.logger.debug(f"SUMMARIZATION: Full summary failed: {e}")
            if summary_prompt is None:
                try:
                    self.logger.debug("SUMMARIZATION: Attempting short summary")
                    current_screen = await session.capture_pane(capture_entire=False)
                    limited_screen = current_screen[-1000:] if current_screen else ""
                    short_prompt = self._document(
                        "short_summary", original_instruction=original_instruction, terminal_state=limited_screen
                    )
                    short_llm_response: LLMResponse = await self._llm.call(prompt=short_prompt, **self._llm_call_kwargs)
                    summary_prompt = self._document(
                        "context_recovery",
                        original_instruction=original_instruction,
                        terminal_state=limited_screen,
                        summary=short_llm_response.content,
                    )
                    self.logger.debug("SUMMARIZATION: Short summary succeeded")
                except Exception as e:
                    if is_provider_request_error(e):
                        raise
                    self.logger.error(f"SUMMARIZATION: Short summary failed: {e}")
            if summary_prompt is None:
                self.logger.debug("SUMMARIZATION: Using ultimate fallback")
                current_screen = await session.capture_pane(capture_entire=False)
                limited_screen = current_screen[-1000:] if current_screen else ""
                summary_prompt = self._document(
                    "context_recovery",
                    original_instruction=original_instruction,
                    terminal_state=limited_screen,
                    summary="",
                )
            try:
                start_time = time.time()
                llm_response = await chat.chat(summary_prompt, **self._llm_call_kwargs)
                end_time = time.time()
                request_time_ms = (end_time - start_time) * 1000
                self._api_request_times.append(request_time_ms)
            except Exception as e:
                if is_provider_request_error(e):
                    raise
                self.logger.error(f"Even fallback chat failed: {e}")
                llm_response = LLMResponse(content="Technical difficulties. Please continue with the task.")
            return llm_response
        except OutputLengthExceededError as e:
            self.logger.debug(f"Output length exceeded: {e}")
            truncated_response = getattr(e, "truncated_response", "[TRUNCATED RESPONSE NOT AVAILABLE]")
            salvaged_response = None
            _has_multiple_blocks = False
            if hasattr(self._parser, "salvage_truncated_response"):
                salvaged_response, _has_multiple_blocks = self._parser.salvage_truncated_response(truncated_response)
            if salvaged_response:
                self.logger.debug("Output exceeded length but found valid response. Using truncated version.")
                return salvaged_response
            warnings_text = ""
            try:
                parse_result = self._parser.parse_response(truncated_response)
                if parse_result.warning:
                    warnings_text = f"\n\nParser warnings from your truncated response:\n{parse_result.warning}"
            except Exception as parse_error:
                self.logger.debug(f"Failed to parse truncated response: {parse_error}")
            output_limit = self._llm.get_model_output_limit()
            if output_limit is not None:
                limit_str = f"{output_limit} tokens"
            else:
                limit_str = "the maximum output length"
            error_msg = self._document("output_limit", limit_str=limit_str, warnings_text="")
            if warnings_text:
                error_msg += warnings_text
            chat.messages.append({"role": "user", "content": prompt})
            chat.messages.append({"role": "assistant", "content": truncated_response})
            chat.reset_response_chain()
            return await self._query_llm(
                chat=chat, prompt=error_msg, original_instruction=original_instruction, session=session
            )
        except Exception as e:
            self.logger.error(f"Unknown Error in LLM interaction: {e}")
            raise e
