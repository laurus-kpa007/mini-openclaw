"""Result aggregation strategies for multi-child agent spawning."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class AggregationStrategy(Enum):
    WAIT_ALL = "wait_all"          # Wait for all children, merge results
    FIRST_SUCCESS = "first_success"  # Return as soon as one child succeeds
    MAJORITY_VOTE = "majority_vote"  # Return the most common answer
    PRIORITY_CHAIN = "priority_chain"  # Try children in order, stop at first success


@dataclass
class ChildTask:
    """A single child task to be run in parallel."""
    task_description: str
    tool_allowlist: list[str] | None = None
    system_prompt: str | None = None
    priority: int = 0  # Lower = higher priority for PRIORITY_CHAIN


@dataclass
class AggregatedResult:
    """Combined result from multiple child agents."""
    success: bool
    summary: str
    individual_results: list[dict[str, Any]] = field(default_factory=list)
    strategy_used: str = ""
    children_total: int = 0
    children_succeeded: int = 0
    children_failed: int = 0


class ResultAggregator:
    """
    Manages parallel child agent spawning and result aggregation.
    Supports multiple strategies for combining results from child agents.
    """

    def __init__(self, gateway: Any, parent_agent: Any) -> None:
        self._gateway = gateway
        self._parent = parent_agent

    async def spawn_and_aggregate(
        self,
        tasks: list[ChildTask],
        strategy: AggregationStrategy = AggregationStrategy.WAIT_ALL,
        timeout: float | None = None,
    ) -> AggregatedResult:
        """Spawn multiple children and aggregate their results."""
        if strategy == AggregationStrategy.WAIT_ALL:
            return await self._wait_all(tasks, timeout)
        elif strategy == AggregationStrategy.FIRST_SUCCESS:
            return await self._first_success(tasks, timeout)
        elif strategy == AggregationStrategy.MAJORITY_VOTE:
            return await self._majority_vote(tasks, timeout)
        elif strategy == AggregationStrategy.PRIORITY_CHAIN:
            return await self._priority_chain(tasks, timeout)
        else:
            return AggregatedResult(
                success=False,
                summary=f"Unknown strategy: {strategy.value}",
                strategy_used=strategy.value,
            )

    async def _spawn_child(self, task: ChildTask) -> dict[str, Any]:
        """Spawn a single child agent and return its result as a dict."""
        try:
            child = await self._gateway.spawn_agent(
                session_id=self._parent.session.session_id,
                system_prompt=task.system_prompt,
                tool_allowlist=task.tool_allowlist,
                parent_agent_id=self._parent.agent_id,
                depth=self._parent.depth + 1,
            )
            self._parent.children.append(child.agent_id)

            async with self._gateway._semaphore:
                result = await child.run(task.task_description)

            return {
                "agent_id": child.agent_id,
                "task": task.task_description,
                "success": result.success,
                "content": result.content,
                "tool_calls": result.tool_calls_made,
                "tokens_used": result.tokens_used,
            }
        except Exception as e:
            logger.exception("Child spawn failed for task: %s", task.task_description)
            return {
                "agent_id": None,
                "task": task.task_description,
                "success": False,
                "content": f"Spawn failed: {e}",
                "tool_calls": 0,
                "tokens_used": 0,
            }

    async def _wait_all(
        self, tasks: list[ChildTask], timeout: float | None
    ) -> AggregatedResult:
        """Run all children concurrently, wait for all to complete."""
        coros = [self._spawn_child(t) for t in tasks]
        if timeout:
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*coros, return_exceptions=True), timeout=timeout
                )
            except asyncio.TimeoutError:
                results = []
        else:
            results = await asyncio.gather(*coros, return_exceptions=True)

        individual: list[dict[str, Any]] = []
        succeeded = 0
        for r in results:
            if isinstance(r, Exception):
                individual.append({"success": False, "content": str(r)})
            else:
                individual.append(r)
                if r.get("success"):
                    succeeded += 1

        # Merge all successful outputs
        merged_parts = []
        for i, r in enumerate(individual):
            status = "OK" if r.get("success") else "FAILED"
            merged_parts.append(
                f"[Task {i+1} ({status})]: {r.get('content', 'No output')}"
            )
        summary = "\n\n---\n\n".join(merged_parts)

        return AggregatedResult(
            success=succeeded > 0,
            summary=summary,
            individual_results=individual,
            strategy_used="wait_all",
            children_total=len(tasks),
            children_succeeded=succeeded,
            children_failed=len(tasks) - succeeded,
        )

    async def _first_success(
        self, tasks: list[ChildTask], timeout: float | None
    ) -> AggregatedResult:
        """Return as soon as one child succeeds, cancel the rest."""
        pending_tasks: list[asyncio.Task] = []
        for t in tasks:
            pending_tasks.append(asyncio.create_task(self._spawn_child(t)))

        first_success = None
        completed_results: list[dict[str, Any]] = []

        try:
            done_set, pending_set = await asyncio.wait(
                pending_tasks,
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )

            while done_set:
                for done_task in done_set:
                    result = done_task.result()
                    completed_results.append(result)
                    if result.get("success") and first_success is None:
                        first_success = result
                        # Cancel remaining
                        for p in pending_set:
                            p.cancel()
                        break

                if first_success:
                    break

                if not pending_set:
                    break

                done_set, pending_set = await asyncio.wait(
                    pending_set,
                    timeout=timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )

        except Exception as e:
            logger.exception("first_success aggregation error")

        if first_success:
            return AggregatedResult(
                success=True,
                summary=first_success["content"],
                individual_results=completed_results,
                strategy_used="first_success",
                children_total=len(tasks),
                children_succeeded=1,
                children_failed=len(completed_results) - 1,
            )

        return AggregatedResult(
            success=False,
            summary="All children failed",
            individual_results=completed_results,
            strategy_used="first_success",
            children_total=len(tasks),
            children_succeeded=0,
            children_failed=len(completed_results),
        )

    async def _majority_vote(
        self, tasks: list[ChildTask], timeout: float | None
    ) -> AggregatedResult:
        """Run all, return the most common answer among successes."""
        base = await self._wait_all(tasks, timeout)

        # Count successful content occurrences
        content_counts: dict[str, int] = {}
        for r in base.individual_results:
            if r.get("success"):
                content = r.get("content", "").strip()
                content_counts[content] = content_counts.get(content, 0) + 1

        if not content_counts:
            return AggregatedResult(
                success=False,
                summary="No successful results to vote on",
                individual_results=base.individual_results,
                strategy_used="majority_vote",
                children_total=base.children_total,
                children_succeeded=0,
                children_failed=base.children_total,
            )

        winner = max(content_counts, key=content_counts.get)  # type: ignore[arg-type]
        vote_count = content_counts[winner]

        return AggregatedResult(
            success=True,
            summary=f"[Majority vote: {vote_count}/{base.children_succeeded} agents agreed]\n\n{winner}",
            individual_results=base.individual_results,
            strategy_used="majority_vote",
            children_total=base.children_total,
            children_succeeded=base.children_succeeded,
            children_failed=base.children_failed,
        )

    async def _priority_chain(
        self, tasks: list[ChildTask], timeout: float | None
    ) -> AggregatedResult:
        """Try children in priority order, return first success."""
        sorted_tasks = sorted(tasks, key=lambda t: t.priority)
        individual: list[dict[str, Any]] = []

        for task in sorted_tasks:
            result = await self._spawn_child(task)
            individual.append(result)
            if result.get("success"):
                return AggregatedResult(
                    success=True,
                    summary=result["content"],
                    individual_results=individual,
                    strategy_used="priority_chain",
                    children_total=len(tasks),
                    children_succeeded=1,
                    children_failed=len(individual) - 1,
                )

        return AggregatedResult(
            success=False,
            summary="All priority chain children failed",
            individual_results=individual,
            strategy_used="priority_chain",
            children_total=len(tasks),
            children_succeeded=0,
            children_failed=len(individual),
        )
