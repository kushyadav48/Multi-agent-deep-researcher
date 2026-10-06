"""Deterministic local synthesis-model selection; no inference dependencies."""

from routing.models import ModelRoute, RoutingDecision, RoutingMode
from routing.router import route_query

__all__ = ['ModelRoute', 'RoutingDecision', 'RoutingMode', 'route_query']
