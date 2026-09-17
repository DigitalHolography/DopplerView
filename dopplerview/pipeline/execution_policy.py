"""Central machine-aware execution policy for a pipeline worker process."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Optional

from dopplerview.pipeline.execution_profile import ExecutionProfile
from dopplerview.utils.parallelization_utils import compute_n_jobs
from dopplerview.utils.runtime_metrics import available_cpu_count


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExecutionPolicy:
    profile: ExecutionProfile
    available_cpus: int
    cpu_workers: int
    dag_concurrency: int
    native_threads_per_task: Optional[int]

    @classmethod
    def from_config(cls, config=None, profile=None) -> "ExecutionPolicy":
        config = config or {}
        profile = ExecutionProfile.resolve(profile)
        execution = config.get("Execution", {})
        cpus = available_cpu_count()

        configured_workers = execution.get(
            "NumberOfWorkers",
            config.get("NumberOfWorkers", 0.5),  # legacy compatibility
        )
        configured_workers = profile.operation_workers(configured_workers)
        workers = compute_n_jobs(configured_workers, cpu_count=cpus)

        # The default allows the independent branches present in the current
        # DAG to overlap without scaling step-level concurrency with every CPU.
        configured_dag_concurrency = execution.get(
            "DagConcurrency",
            "auto",
        )
        if configured_dag_concurrency in (None, "auto", "automatic"):
            dag_concurrency = min(2, cpus)
        else:
            dag_concurrency = min(
                cpus,
                compute_n_jobs(configured_dag_concurrency, cpu_count=cpus),
            )
        if profile is ExecutionProfile.SEQUENTIAL_REFERENCE:
            dag_concurrency = 1

        # Native runtimes are automatic by default. The Override suffix is
        # intentional: early configurations persisted a value of 1 in
        # users' copied config files, which severely throttled deep models CPU inference.
        configured_native_threads = execution.get(
            "NativeThreadsPerTaskOverride",
            "auto",
        )
        if configured_native_threads in (None, "auto", "automatic", 0, -1):
            native_threads = None
        else:
            native_threads = max(1, int(configured_native_threads))
        if profile is ExecutionProfile.SEQUENTIAL_REFERENCE:
            native_threads = 1

        return cls(
            profile=profile,
            available_cpus=cpus,
            cpu_workers=workers,
            dag_concurrency=dag_concurrency,
            native_threads_per_task=native_threads,
        )

    def describe(self) -> str:
        return (
            f"profile={self.profile.value}, available CPUs={self.available_cpus}, "
            f"shared workers={self.cpu_workers}, DAG concurrency={self.dag_concurrency}, "
            "native threads/task="
            f"{self.native_threads_per_task or 'automatic'}"
        )
