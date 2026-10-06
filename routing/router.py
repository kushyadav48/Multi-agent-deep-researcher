"""Small, transparent v1 complexity policy using only Python's standard library."""

import re

from routing.models import (
    FAST_SYNTHESIS_MODEL, QUALITY_SYNTHESIS_MODEL, SEARCH_MODEL,
    ModelRoute, RoutingDecision, RoutingMode,
)


QUALITY_THRESHOLD = 3
MODERATE_QUERY_WORDS = 30
LONG_QUERY_WORDS = 80

# Each category contributes at most once. Match actions/phrases rather than
# isolated topics so definitions such as 'What is architecture?' stay simple.
_RULES = (
    (2, 'Comparison requested (+2)',
     r'\b(?:compare|comparison|versus|vs\.?)\b'),
    (2, 'Trade-offs or pros/cons requested (+2)',
     r'\btrade[\s-]?offs?\b|\bpros\s+(?:and|&)\s+cons\b|'
     r'\badvantages\s+(?:and|&)\s+disadvantages\b'),
    (3, 'Evaluation, critique, or risk analysis requested (+3)',
     r'\b(?:evaluate|evaluation|critique|assess|assessment)\b|'
     r'\b(?:analy[sz]e\s+(?:\w+\s+){0,3}(?:risks?|evidence)|risk\s+analysis|'
     r'contradictions?|evidence\s+quality)\b'),
    (3, 'Recommendation or architecture design requested (+3)',
     r'\b(?:recommend|recommendation)\b|\barchitecture\s+design\b|'
     r'\bdesign\s+(?:[\w-]+\s+){0,4}(?:architecture|system)\b'),
    (1, 'Multiple sources or documents to synthesize (+1)',
     r'\b(?:documents|sources)\b|\bmulti[\s-](?:source|document)\b'),
)
_ACTIONS = re.compile(
    r'\b(?:compare|explain|summarize|analy[sz]e|evaluate|critique|assess|'
    r'recommend|design|identify)\b'
)


def route_query(query: str, mode: RoutingMode | str = RoutingMode.AUTO) -> RoutingDecision:
    """Select once before execution; no model calls, embeddings, or retry judge."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError('Research query must be non-empty text')
    requested = RoutingMode(mode.strip().casefold()) if isinstance(mode, str) else RoutingMode(mode)
    text = query.casefold()
    words = len(query.split())
    score = 0
    reasons = []
    if words >= MODERATE_QUERY_WORDS:
        score += 1
        reasons.append('At least 30 words (+1)')
    if words >= LONG_QUERY_WORDS:
        score += 1
        reasons.append('At least 80 words (+1 additional)')
    for weight, reason, pattern in _RULES:
        if re.search(pattern, text):
            score += weight
            reasons.append(reason)
    if len(_ACTIONS.findall(text)) >= 2 and re.search(r'\band\b|[,;\n]', text):
        score += 1
        reasons.append('Multiple requested actions (+1)')
    if requested is RoutingMode.AUTO:
        selected = ModelRoute.QUALITY if score >= QUALITY_THRESHOLD else ModelRoute.FAST
        reasons.append(f'Auto: score {score} {">=" if selected is ModelRoute.QUALITY else "<"} '
                       f'{QUALITY_THRESHOLD}; selected {selected.value}')
    else:
        selected = ModelRoute(requested.value)
        reasons.append(f'Manual {requested.value} override')
    synthesis_model = FAST_SYNTHESIS_MODEL if selected is ModelRoute.FAST else QUALITY_SYNTHESIS_MODEL
    return RoutingDecision(requested, selected, SEARCH_MODEL, synthesis_model, score, tuple(reasons))
