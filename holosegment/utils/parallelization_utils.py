import joblib
import numpy as np


def _process_chunk(chunk, func):
    return np.stack([func(item) for item in chunk], axis=0)


def run_in_parallel(func, iterable, n_jobs=-1, chunking=True):
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
    if n_jobs == -1:
        n_jobs = joblib.cpu_count()

    if chunking:
        n_jobs = min(n_jobs, len(iterable))

        indices = np.array_split(np.arange(len(iterable)), n_jobs)
        chunks = [[iterable[i] for i in idx] for idx in indices]

        results = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(_process_chunk)(chunk, func)
            for chunk in chunks
        )

        return np.concatenate(results, axis=0)

    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(func)(item) for item in iterable
    )

    return np.stack(results, axis=0)