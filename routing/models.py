"""Local model identities and immutable routing decisions."""

from dataclasses import dataclass
from enum import Enum


SEARCH_MODEL = 'ollama/qwen2.5:3b'
FAST_SYNTHESIS_MODEL = 'ollama/qwen3:1.7b'
QUALITY_SYNTHESIS_MODEL = SEARCH_MODEL
ROUTER_POLICY_VERSION = 'v1'


class ModelRoute(str, Enum):
    FAST = 'fast'
    QUALITY = 'quality'


class RoutingMode(str, Enum):
    AUTO = 'auto'
    FAST = 'fast'
    QUALITY = 'quality'


@dataclass(frozen=True)
class RoutingDecision:
    requested_mode: RoutingMode
    selected_route: ModelRoute
    search_model: str
    synthesis_model: str
    score: int
    reasons: tuple[str, ...]
    policy_version: str = ROUTER_POLICY_VERSION
