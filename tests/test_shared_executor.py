from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import time

import numpy as np
import pytest

from dopplerview.pipeline.execution_policy import ExecutionPolicy
from dopplerview.pipeline.execution_profile import ExecutionProfile
from dopplerview.utils.parallelization_utils import compute_n_jobs
from dopplerview.utils.shared_executor import SharedExecutor
from dopplerview.utils.cancellation import CancellationToken, OperationCancelled


def test_worker_resolution_supports_fixed_negative_and_fractional_values():
    assert compute_n_jobs(-1, cpu_count=8) == 8
    assert compute_n_jobs(-2, cpu_count=8) == 7
    assert compute_n_jobs(0.5, cpu_count=8) == 4
    assert compute_n_jobs(3, cpu_count=8) == 3


def test_execution_policy_reads_execution_section(monkeypatch):
    monkeypatch.setattr(
        "dopplerview.pipeline.execution_policy.available_cpu_count",
        lambda: 8,
    )

    policy = ExecutionPolicy.from_config(
        {
            "Execution": {
                "NumberOfWorkers": 0.5,
                "DagConcurrency": 2,
                "NativeThreadsPerTaskOverride": 1,
            }
        },
        profile="default",
    )

    assert policy.cpu_workers == 4
    assert policy.dag_concurrency == 2
    assert policy.native_threads_per_task == 1


def test_legacy_worker_setting_remains_supported(monkeypatch):
    monkeypatch.setattr(
        "dopplerview.pipeline.execution_policy.available_cpu_count",
        lambda: 8,
    )

    policy = ExecutionPolicy.from_config(
        {"NumberOfWorkers": 0.25},
        profile="default",
    )

    assert policy.cpu_workers == 2


def test_default_policy_leaves_native_runtime_parallelism_automatic(monkeypatch):
    monkeypatch.setattr(
        "dopplerview.pipeline.execution_policy.available_cpu_count",
        lambda: 8,
    )

    policy = ExecutionPolicy.from_config({}, profile="default")

    assert policy.native_threads_per_task is None
    assert "native threads/task=automatic" in policy.describe()


def test_legacy_phase_four_native_limit_does_not_throttle_default_profile(
    monkeypatch,
):
    monkeypatch.setattr(
        "dopplerview.pipeline.execution_policy.available_cpu_count",
        lambda: 8,
    )

    policy = ExecutionPolicy.from_config(
        {"Execution": {"NativeThreadsPerTask": 1}},
        profile="default",
    )

    assert policy.native_threads_per_task is None


def test_sequential_reference_overrides_all_concurrency(monkeypatch):
    monkeypatch.setattr(
        "dopplerview.pipeline.execution_policy.available_cpu_count",
        lambda: 8,
    )

    policy = ExecutionPolicy.from_config(
        {
            "Execution": {
                "NumberOfWorkers": -1,
                "DagConcurrency": 4,
                "NativeThreadsPerTaskOverride": 4,
            }
        },
        profile=ExecutionProfile.SEQUENTIAL_REFERENCE,
    )

    assert policy.cpu_workers == 1
    assert policy.dag_concurrency == 1
    assert policy.native_threads_per_task == 1


def test_shared_executor_preserves_input_order_and_matches_sequential_result():
    inputs = np.arange(20)
    expected = np.stack([value * value for value in inputs])
    executor = SharedExecutor(max_workers=4, available_cpus=8)
    try:
        actual = executor.map(lambda value: value * value, inputs, n_jobs=-1)
    finally:
        executor.shutdown()

    np.testing.assert_array_equal(actual, expected)


def test_per_operation_worker_limit_is_respected():
    active = 0
    maximum_active = 0
    lock = threading.Lock()

    def measured(value):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return value

    executor = SharedExecutor(max_workers=4, available_cpus=8)
    try:
        executor.map(measured, range(8), n_jobs=2)
    finally:
        executor.shutdown()

    assert maximum_active <= 2


def test_simultaneous_operations_share_one_global_capacity():
    active = 0
    maximum_active = 0
    lock = threading.Lock()

    def measured(value):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.015)
        with lock:
            active -= 1
        return value

    shared = SharedExecutor(max_workers=3, available_cpus=8)
    callers = ThreadPoolExecutor(max_workers=2)
    try:
        futures = [
            callers.submit(shared.map, measured, range(9), n_jobs=-1),
            callers.submit(shared.map, measured, range(9), n_jobs=-1),
        ]
        for future in futures:
            future.result()
    finally:
        callers.shutdown()
        shared.shutdown()

    assert maximum_active <= 3


def test_executor_rejects_work_after_shutdown():
    executor = SharedExecutor(max_workers=2, available_cpus=4)
    executor.shutdown()

    try:
        executor.map(lambda value: value, [1])
    except RuntimeError as error:
        assert "closed" in str(error)
    else:
        raise AssertionError("Closed executor accepted work")


def test_empty_workload_returns_without_submitting_tasks():
    executor = SharedExecutor(max_workers=2, available_cpus=4)
    try:
        result = executor.map(lambda value: value, [])
    finally:
        executor.shutdown()

    assert result.shape == (0,)


def test_worker_exception_is_propagated_to_caller():
    def fail(value):
        if value == 2:
            raise ValueError("bad item")
        return value

    executor = SharedExecutor(max_workers=2, available_cpus=4)
    try:
        with pytest.raises(ValueError, match="bad item"):
            executor.map(fail, [1, 2, 3])
    finally:
        executor.shutdown()


def test_executor_is_reusable_across_operations():
    executor = SharedExecutor(max_workers=2, available_cpus=4)
    try:
        first = executor.map(lambda value: value + 1, [1, 2])
        second = executor.map(lambda value: value * 2, [3, 4])
    finally:
        executor.shutdown()

    np.testing.assert_array_equal(first, [2, 3])
    np.testing.assert_array_equal(second, [6, 8])


def test_cancelled_executor_rejects_new_operation():
    cancellation = CancellationToken()
    executor = SharedExecutor(
        max_workers=2,
        available_cpus=4,
        cancellation=cancellation,
    )
    cancellation.cancel()
    try:
        with pytest.raises(OperationCancelled):
            executor.map(lambda value: value, [1, 2])
    finally:
        executor.shutdown()
