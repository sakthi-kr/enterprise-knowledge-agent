"""Shared types for agent planning and execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from enterprise_knowledge_agent.grounded_answer import GroundedAnswer, TokenUsage


class AgentStrategy(str, Enum):
    """Retrieval strategies the planner may select."""

    DENSE_ONLY = "dense_only"
    DENSE_PLUS_GRAPH = "dense_plus_graph"


@dataclass(frozen=True)
class AgentPlan:
    """Validated planner decision for one enterprise question."""

    strategy: AgentStrategy
    reason: str
    model_name: str
    usage: TokenUsage


@dataclass(frozen=True)
class ToolExecution:
    """One observable tool execution in the agent workflow."""

    tool_name: str
    status: str
    result_count: int
    detail: str = ""


@dataclass(frozen=True)
class AgentResult:
    """Final agent result with planner and tool-execution metadata."""

    answer: GroundedAnswer
    plan: AgentPlan
    tool_trace: tuple[ToolExecution, ...]
    planner_fallback: bool
    tool_call_count: int
