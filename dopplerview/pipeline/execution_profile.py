"""Execution profiles used to obtain comparable performance baselines."""

from __future__ import annotations

from enum import Enum
import os


class ExecutionProfile(str, Enum):
    DEFAULT = "default"
    SEQUENTIAL_REFERENCE = "sequential_reference"

    @classmethod
    def resolve(cls, value=None) -> "ExecutionProfile":
        if isinstance(value, cls):
            return value
        if value is None:
            value = os.getenv("DOPPLERVIEW_EXECUTION_PROFILE", cls.DEFAULT.value)
        normalized = str(value).strip().lower().replace("-", "_")
        aliases = {
            "sequential": cls.SEQUENTIAL_REFERENCE,
            "reference": cls.SEQUENTIAL_REFERENCE,
            "sequential_reference": cls.SEQUENTIAL_REFERENCE,
            "default": cls.DEFAULT,
        }
        try:
            return aliases[normalized]
        except KeyError as error:
            choices = ", ".join(profile.value for profile in cls)
            raise ValueError(
                f"Unknown execution profile '{value}'. Expected one of: {choices}."
            ) from error

    @property
    def dag_max_workers(self):
        """Compatibility hint; ExecutionPolicy owns the actual machine bound."""
        return 1 if self is self.SEQUENTIAL_REFERENCE else 2

    def operation_workers(self, configured_workers):
        if self is self.SEQUENTIAL_REFERENCE:
            return 1
        return configured_workers
