from __future__ import annotations

from typing import Any

import pytest


class RecordingOutputManager:
    """Small output-manager double used by DAG tests."""

    def __init__(self) -> None:
        self.saved: list[tuple[str, str]] = []
        self.cached: list[tuple[str, str]] = []

    def save_async(self, step_name, key, ctx) -> None:
        self.saved.append((step_name, key))

    def cache_async(self, ctx, fingerprint, step_name) -> None:
        self.cached.append((step_name, fingerprint))


class FakeContext:
    """Implements the narrow context protocol consumed by BaseStep/DAGEngine."""

    def __init__(self, initial: dict[str, Any] | None = None, config=None) -> None:
        self.values = dict(initial or {})
        self.dopplerview_config = dict(config or {})
        self.metadata = {"step_hashes": {}}
        self.output_manager = RecordingOutputManager()

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value) -> None:
        self.values[key] = value

    def has(self, key) -> bool:
        return key in self.values

    def require(self, key):
        if key not in self.values:
            raise RuntimeError(f"Missing required context key: '{key}'")
        return self.values[key]


@pytest.fixture
def fake_context_factory():
    return FakeContext
