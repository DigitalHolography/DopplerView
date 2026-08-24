from dopplerview.utils.runtime_metrics import available_cpu_count
def compute_n_jobs(n_jobs, cpu_count=None):
    """Compute the number of parallel jobs to run based on the input parameter.
    Args:        n_jobs: The number of parallel jobs to run. If -1, use all available cores. If -2, use all but one core. If decimal, use that fraction of the available cores.
    Returns:        The computed number of parallel jobs to run.
    """
    cpu_count = int(cpu_count or available_cpu_count())
    if n_jobs < 0:
        return max(1, cpu_count + int(n_jobs) + 1) # e.g. if n_jobs=-1, this will return all cores
    elif n_jobs < 1:
        return max(1, int(cpu_count * n_jobs))
    else:
        return max(1, int(n_jobs))
