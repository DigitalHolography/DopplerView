from typing import List
import hashlib
import json
import numpy as np
from abc import ABC

import logging
import threading

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

    logger = logging.getLogger(__name__)

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
        return ctx.dopplerview_config

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
    
    def export(self, ctx, debug_mode=False):
        """
        Export step outputs using the output manager.
        """
        # def _save(step_name, ctx):
        #     for key in self.produces:
        #         if ctx.has(key):
        #             ctx.output_manager.save(step_name, key, ctx)
        #     return 1
        # thread = threading.Thread(
        #     target=_save,
        #     args=(self.name, ctx),
        #     daemon=True)
        # thread.start()

        for key in self.produces:
            if ctx.has(key):
                ctx.output_manager.save(self.name, key, ctx)

        if debug_mode:
            ctx.output_manager.save_cache(ctx)

        # if debug_mode:
        #     thread_debug = threading.Thread(
        #         target=ctx.output_manager.save_cache,
        #         args=(ctx,),
        #         daemon=True)
        #     thread_debug.start()
    
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