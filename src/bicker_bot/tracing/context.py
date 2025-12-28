"""Trace context and step data models."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4


@dataclass
class TraceStep:
    """A single step in the pipeline trace."""

    stage: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    decision: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    # LLM-specific fields (None for non-LLM stages)
    model: str | None = None
    prompt: str | None = None
    raw_response: str | None = None
    thinking: str | None = None
    thought_signatures: list[str] | None = None
    token_usage: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "stage": self.stage,
            "timestamp": self.timestamp.isoformat(),
            "inputs": self.inputs,
            "outputs": self.outputs,
            "decision": self.decision,
            "details": self.details,
            "model": self.model,
            "prompt": self.prompt,
            "raw_response": self.raw_response,
            "thinking": self.thinking,
            "thought_signatures": self.thought_signatures,
            "token_usage": self.token_usage,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraceStep":
        """Deserialize from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now()

        return cls(
            stage=data["stage"],
            timestamp=timestamp,
            inputs=data.get("inputs", {}),
            outputs=data.get("outputs", {}),
            decision=data.get("decision", ""),
            details=data.get("details", {}),
            model=data.get("model"),
            prompt=data.get("prompt"),
            raw_response=data.get("raw_response"),
            thinking=data.get("thinking"),
            thought_signatures=data.get("thought_signatures"),
            token_usage=data.get("token_usage"),
        )


@dataclass
class TraceContext:
    """Context that flows through the pipeline, accumulating trace steps."""

    channel: str
    trigger_messages: list[str]
    config_snapshot: dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid4()))
    started_at: datetime = field(default_factory=datetime.now)
    steps: list[TraceStep] = field(default_factory=list)
    final_result: list[str] | None = None
    is_replay: bool = False
    original_trace_id: str | None = None

    def add_step(
        self,
        stage: str,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        decision: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Add a non-LLM step to the trace."""
        self.steps.append(
            TraceStep(
                stage=stage,
                inputs=inputs,
                outputs=outputs,
                decision=decision,
                details=details or {},
            )
        )

    def add_llm_step(
        self,
        stage: str,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        decision: str,
        model: str,
        prompt: str,
        raw_response: str,
        thinking: str | None = None,
        thought_signatures: list[str] | None = None,
        token_usage: dict[str, int] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Add an LLM step to the trace."""
        self.steps.append(
            TraceStep(
                stage=stage,
                inputs=inputs,
                outputs=outputs,
                decision=decision,
                details=details or {},
                model=model,
                prompt=prompt,
                raw_response=raw_response,
                thinking=thinking,
                thought_signatures=thought_signatures,
                token_usage=token_usage,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "started_at": self.started_at.isoformat(),
            "channel": self.channel,
            "trigger_messages": self.trigger_messages,
            "config_snapshot": self.config_snapshot,
            "steps": [step.to_dict() for step in self.steps],
            "final_result": self.final_result,
            "is_replay": self.is_replay,
            "original_trace_id": self.original_trace_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraceContext":
        """Deserialize from dictionary."""
        started_at = data.get("started_at")
        if isinstance(started_at, str):
            started_at = datetime.fromisoformat(started_at)
        elif started_at is None:
            started_at = datetime.now()

        ctx = cls(
            id=data["id"],
            started_at=started_at,
            channel=data["channel"],
            trigger_messages=data.get("trigger_messages", []),
            config_snapshot=data.get("config_snapshot", {}),
            final_result=data.get("final_result"),
            is_replay=data.get("is_replay", False),
            original_trace_id=data.get("original_trace_id"),
        )
        ctx.steps = [TraceStep.from_dict(s) for s in data.get("steps", [])]
        return ctx
