import joblib
import numpy as np

import logging
from dopplerview.utils.runtime_metrics import (
    available_cpu_count,
    emit_metric,
    process_snapshot,
)
logger = logging.getLogger(__name__)

def _process_chunk(chunk, func):
    return np.stack([func(item) for item in chunk], axis=0)

def compute_n_jobs(n_jobs):
    """Compute the number of parallel jobs to run based on the input parameter.
    Args:        n_jobs: The number of parallel jobs to run. If -1, use all available cores. If -2, use all but one core. If decimal, use that fraction of the available cores.
    Returns:        The computed number of parallel jobs to run.
    """
    if n_jobs < 0:
        return max(1, joblib.cpu_count() + n_jobs + 1) # e.g. if n_jobs=-1, this will return joblib.cpu_count() (all cores); if n_jobs=-2, this will return joblib.cpu_count() - 1 (all but one core)
    elif n_jobs < 1:
        return max(1, int(joblib.cpu_count() * n_jobs)) # e.g. if n_jobs=0.5, this will return half of the available cores
    else:
        return n_jobs

def run_in_parallel(func, iterable, n_jobs=-1, chunking=True, task_name=None):
    """
    Run a function in parallel over an iterable using joblib.
    
    Args:
        func: The function to run in parallel. It should take a single argument.
        iterable: An iterable of inputs to the function.
        n_jobs: The number of parallel jobs to run. Defaults to -1 (use all available cores).
        chunking: Whether to split the iterable into chunks for more efficient parallelization. The number of chunks is determined by `n_jobs`. Defaults to True.
    Returns:
        The concatenated results of the function applied to each item in the iterable.
    """
    requested_n_jobs = n_jobs
    n_jobs = compute_n_jobs(n_jobs)
    if chunking:
        n_jobs = min(n_jobs, len(iterable))

    snapshot = process_snapshot()
    emit_metric(
        "parallel_operation",
        task=task_name or getattr(func, "__name__", "anonymous"),
        requested_workers=requested_n_jobs,
        effective_workers=n_jobs,
        available_cpus=available_cpu_count(),
        items=len(iterable),
        chunking=chunking,
        process_threads=snapshot["process_threads"],
    )

    if task_name is not None:
        logger.info(f"    - Running {task_name} in parallel with {n_jobs} jobs and chunking={chunking}")

    if chunking:
        indices = np.array_split(np.arange(len(iterable)), n_jobs)
        chunks = [[iterable[i] for i in idx] for idx in indices]

        # Use threading backend to avoid spawning a new .exe when using multiprocessing in a PyInstaller executable
        try:
            results = joblib.Parallel(n_jobs=n_jobs, backend="threading")(
                joblib.delayed(_process_chunk)(chunk, func)
                for chunk in chunks
            )
            return np.concatenate(results, axis=0)
        except KeyboardInterrupt:
            logger.info("Interrupted cleanly")

    try:
        results = joblib.Parallel(n_jobs=n_jobs, backend="threading")(
                joblib.delayed(func)(item) for item in iterable
            )
        return np.stack(results, axis=0)
    except KeyboardInterrupt:
        logger.info("Interrupted cleanly")
