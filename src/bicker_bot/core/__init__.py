"""Core bot logic."""

from .context import ContextBuilder, ContextResult
from .engagement import EngagementChecker, EngagementResult
from .gate import GateFactors, GateResult, ResponseGate
from .responder import ResponseGenerator, ResponseResult
from .router import ChannelBuffer, MessageRouter, TimestampedMessage

__all__ = [
    "ChannelBuffer",
    "ContextBuilder",
    "ContextResult",
    "EngagementChecker",
    "EngagementResult",
    "GateFactors",
    "GateResult",
    "MessageRouter",
    "ResponseGate",
    "ResponseGenerator",
    "ResponseResult",
    "TimestampedMessage",
]
