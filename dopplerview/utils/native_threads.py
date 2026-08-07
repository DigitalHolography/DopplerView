"""Central limits for native thread pools used below Python code."""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional


logger = logging.getLogger(__name__)
_threadpool_controller = None
_original_environment = {
    name: os.environ.get(name)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
}
_original_torch_threads = None


def configure_native_threads(limit: Optional[int]) -> None:
    global _threadpool_controller, _original_torch_threads

    if _threadpool_controller is not None:
        _threadpool_controller.restore_original_limits()
        _threadpool_controller = None

    if limit is None:
        for name, original_value in _original_environment.items():
            if original_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = original_value

        try:
            import cv2
            cv2.setNumThreads(-1)
        except (ImportError, AttributeError):
            pass

        torch = sys.modules.get("torch")
        if torch is not None and _original_torch_threads is not None:
            torch.set_num_threads(_original_torch_threads)
        return

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
            if _original_torch_threads is None:
                _original_torch_threads = torch.get_num_threads()
            torch.set_num_threads(limit)
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # Inter-op threads can only be configured before parallel work starts.
            logger.debug("PyTorch inter-op thread limit was already initialized")
