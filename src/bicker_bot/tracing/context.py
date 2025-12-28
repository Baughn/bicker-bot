"""Trace context and step data models."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


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
