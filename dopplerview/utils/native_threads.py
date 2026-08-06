"""Central limits for native thread pools used below Python code."""

from __future__ import annotations

import logging
import os
import sys


logger = logging.getLogger(__name__)
_threadpool_controller = None


def configure_native_threads(limit: int) -> None:
    global _threadpool_controller
    limit = max(1, int(limit))

    # These are also useful for libraries loaded after policy configuration.
    os.environ["OMP_NUM_THREADS"] = str(limit)
    os.environ["MKL_NUM_THREADS"] = str(limit)
    os.environ["OPENBLAS_NUM_THREADS"] = str(limit)

    try:
        from threadpoolctl import threadpool_limits
        _threadpool_controller = threadpool_limits(limits=limit)
    except ImportError:
        logger.debug("threadpoolctl unavailable; BLAS limits use environment only")

    try:
        import cv2
        cv2.setNumThreads(limit)
    except (ImportError, AttributeError):
        pass

    # Do not import PyTorch merely to configure it. Apply limits if a model has
    # already loaded it; wrappers also receive the policy at construction.
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            torch.set_num_threads(limit)
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # Inter-op threads can only be configured before parallel work starts.
            logger.debug("PyTorch inter-op thread limit was already initialized")
