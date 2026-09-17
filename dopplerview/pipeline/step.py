from typing import List
import hashlib
import json
import numpy as np
from pathlib import Path
from abc import ABC

import logging

from dopplerview._version import __version__


class BaseStep(ABC):
    """
    Base class for pipeline steps.

    Each step must define:
        - name
        - requires (list of data keys)
        - produces (list of data keys)
    """

    name: str = None
    requires: set[str] = []
    produces: set[str] = []
    model_tasks: set[str] = set()
    fingerprint_schema_version = 2

    logger = logging.getLogger(__name__)

    def run(self, ctx):
        raise NotImplementedError
    
    def config_fingerprint(self, ctx):
        """
        Compute deterministic fingerprint of the step configuration.
        """
        relevant_config = self._relevant_config(ctx)
        serialized = json.dumps(relevant_config, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()
    
    def fingerprint(self, ctx):
        """Compute the identity of the scientific computation."""

        payload = {
            "schema": self.fingerprint_schema_version,
            "step": self.name,
            "config": self._relevant_config(ctx),
            "inputs": self._input_signature(ctx),
            "models": self._model_signature(ctx),
            "version": __version__
        }

        serialized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def _relevant_config(self, ctx):
        """
        Override in steps if needed. Execution-only settings never participate
        in scientific cache identity.
        """
        config = dict(ctx.dopplerview_config or {})
        config.pop("Execution", None)
        config.pop("NumberOfWorkers", None)  # legacy execution setting
        return config

    def _input_signature(self, ctx):
        sig = {}
        for key in sorted(self.requires):
            get_artifact_fingerprint = getattr(ctx, "get_artifact_fingerprint", None)
            artifact_fingerprint = (
                get_artifact_fingerprint(key)
                if get_artifact_fingerprint is not None
                else None
            )
            sig[key] = artifact_fingerprint or self._hash_value(ctx.get(key))
        return sig

    def _model_signature(self, ctx):
        get_current = getattr(ctx, "get_current_model_name_for_task", None)
        get_identity = getattr(ctx, "get_model_identity", None)
        if get_current is None or get_identity is None:
            return {}

        tasks = {self.name, *self.model_tasks}
        identities = {}
        for task in sorted(tasks):
            try:
                model_name = get_current(task)
                identities[task] = get_identity(model_name)
            except (KeyError, ValueError):
                # Most non-model steps do not have a registry task.
                continue
        return identities

    def _hash_value(self, val):
        if isinstance(val, np.ndarray):
            digest = hashlib.sha256()
            digest.update(str(val.dtype).encode())
            digest.update(json.dumps(val.shape).encode())
            digest.update(np.ascontiguousarray(val).tobytes())
            return digest.hexdigest()
        if isinstance(val, Path):
            return self._file_identity(val)
        if isinstance(val, str):
            path = Path(val)
            try:
                if path.is_file():
                    return self._file_identity(path)
            except OSError:
                pass
        serialized = json.dumps(val, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    @staticmethod
    def _file_identity(path):
        """Cheap source identity suitable for multi-gigabyte HDF5 inputs."""
        path = Path(path).resolve()
        stat = path.stat()
        payload = {
            "path": str(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        serialized = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()
    
    def export(self, ctx, debug_mode=False, fingerprint=None):
        """
        Export available step outputs using the output manager.

        This is intentionally called for both freshly executed and cache-hit
        steps. Canonical H5 values and optional diagnostic figures must not
        depend on whether the computation itself was skipped.
        """
        for key in self.produces:
            if ctx.has(key):
                ctx.output_manager.save_async(self.name, key, ctx)

        if debug_mode:
            ctx.output_manager.cache_async(
                ctx,
                fingerprint or self.fingerprint(ctx),
                self.name,
            )

class NestedStep(BaseStep):
    substeps: List[BaseStep] = []

    def __init__(self):
        self.produces, self.requires = self._resolve_produces_and_requires()

    def run(self, ctx):
        for step in self.substeps:
            step.run(ctx)
    
    def _relevant_config(self, ctx):
        """By default, combine relevant config from all substeps.
        """
        d = {}
        for step in self.substeps:
            d.update(step._relevant_config(ctx))
        return d
    
    def _resolve_produces_and_requires(self):
        """Combine produces and requires from all substeps."""
        produces = set()
        requires = set()
        for step in reversed(self.substeps):
            requires.difference_update(step.produces)  # If a substep produces something, it's not required from outside
            produces.update(step.produces)
            requires.update(step.requires)

        return produces, requires
