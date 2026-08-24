"""Immutable messages passed from computation to persistence workers."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


def freeze_value(value: Any) -> Any:
    """Create an immutable snapshot suitable for asynchronous persistence."""
    if isinstance(value, np.ndarray):
        snapshot = value.copy()
        snapshot.flags.writeable = False
        return snapshot
    if isinstance(value, dict):
        return MappingProxyType({key: freeze_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(freeze_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(freeze_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(freeze_value(item) for item in value)
    return value


@dataclass(frozen=True)
class OutputPayload:
    step_name: str
    key: str
    value: Any
    rendering_values: Mapping[str, Any]

    def get(self, key, default=None):
        if key == self.key:
            return self.value
        return self.rendering_values.get(key, default)

    def has(self, key):
        if key == self.key:
            return True
        return key in self.rendering_values


@dataclass(frozen=True)
class CachePayload:
    step_name: str
    fingerprint: str
    values: Mapping[str, Any]
