from __future__ import annotations

from typing import Any
import hashlib

import pytest

from dopplerview.input_output.output_manager import OutputManager

import queue


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
        self.metadata = {"step_hashes": {}, "artifact_fingerprints": {}}
        self.output_manager = RecordingOutputManager()

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value) -> None:
        self.values[key] = value
        self.metadata["artifact_fingerprints"].pop(key, None)

    def get_artifact_fingerprint(self, key):
        return self.metadata["artifact_fingerprints"].get(key)

    def set_artifact_fingerprints(self, step_name, keys, step_fingerprint):
        for key in keys:
            payload = f"{step_name}:{key}:{step_fingerprint}"
            self.metadata["artifact_fingerprints"][key] = hashlib.sha256(
                payload.encode()
            ).hexdigest()

    def has(self, key) -> bool:
        return key in self.values

    def require(self, key):
        if key not in self.values:
            raise RuntimeError(f"Missing required context key: '{key}'")
        return self.values[key]


@pytest.fixture
def fake_context_factory():
    return FakeContext


@pytest.fixture
def bare_output_manager_factory():
    def factory():
        manager = OutputManager.__new__(OutputManager)
        manager.running = False
        manager.output_queue = queue.Queue()
        manager.cache_queue = queue.Queue()
        manager.output_worker = None
        manager.cache_worker = None
        manager.output_dir = None
        manager.output_config = {}
        manager.renderers = {}
        manager.final_h5_path = None
        manager.temporary_h5_path = None
        manager.h5_path = None
        manager._worker_errors = []
        return manager

    return factory
