from typing import List
import hashlib
import json
import numpy as np

class BaseStep:
    """
    Base class for pipeline steps.

    Each step must define:
        - name
        - requires (list of data keys)
        - produces (list of data keys)
    """

    name: str = None
    requires: List[str] = []
    produces: List[str] = []

    def run(self, ctx):
        raise NotImplementedError
    
    def fingerprint(self, ctx):
        """
        Compute deterministic fingerprint of this step.
        """

        payload = {
            "config": self._relevant_config(ctx),
            "inputs": self._input_signature(ctx)
        }

        serialized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def _relevant_config(self, ctx):
        """
        Override in steps if needed.
        By default use entire config.
        """
        return ctx.config

    def _input_signature(self, ctx):
        sig = {}
        for key in self.requires:
            val = ctx.get(key)
            sig[key] = self._hash_value(val)
        return sig

    def _hash_value(self, val):
        if isinstance(val, np.ndarray):
            return hashlib.sha256(val.tobytes()).hexdigest()
        return str(val)
    
class NestedStep(BaseStep):
    substeps: List[BaseStep] = []

    def run(self, ctx):
        for step in self.substeps:
            step.run(ctx)