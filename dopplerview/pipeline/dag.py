# dopplerview/core/dag.py

from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, Iterable, List, Optional, Set
import threading
import time
import logging

from dopplerview.pipeline.step import BaseStep

logger = logging.getLogger(__name__)


class DAGEngine:
    """
    Directed Acyclic Graph execution engine.

    - Resolves dependencies automatically
    - Executes independent steps in parallel (wave-based scheduling)
    - Executes only required steps, with cache validation
    """

    def __init__(
        self,
        steps: Iterable[BaseStep],
        debug_mode: bool = False,
        max_workers: Optional[int] = None,
    ):
        self.steps: Dict[str, BaseStep] = {s.name: s for s in steps}
        self.debug_mode = debug_mode
        self.max_workers = max_workers  # None → ThreadPoolExecutor picks a default

        self._validate_unique_names()
        self.graph: Dict[str, Set[str]] = self._build_dependency_graph()
        self.reverse_graph: Dict[str, Set[str]] = self._build_reverse_graph()
        self.execution_order: List[str] = self._topological_sort()

        # Mutable state reset between run() calls
        self._invalidated: Set[str] = set()
        self._steps_to_run: Optional[List[str]] = None
        self._lock = threading.Lock()  # Guards shared mutable state during parallel runs

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _validate_unique_names(self) -> None:
        if not self.steps:
            raise ValueError("No steps registered in DAG.")
        if len(set(self.steps)) != len(self.steps):
            raise ValueError("Duplicate step names detected.")

    def _build_dependency_graph(self) -> Dict[str, Set[str]]:
        """
        Build a producer → consumer graph based on produced/required keys.
        graph[A] = {B, C}  means A must finish before B and C can start.
        """
        key_producers: Dict[str, str] = {}
        graph: Dict[str, Set[str]] = defaultdict(set)

        for step in self.steps.values():
            for key in step.produces:
                if key in key_producers:
                    raise ValueError(f"Multiple steps produce the same key: '{key}'")
                key_producers[key] = step.name

        for step in self.steps.values():
            for required_key in step.requires:
                producer = key_producers.get(required_key)
                if producer:
                    graph[producer].add(step.name)

        # Ensure every step appears in the graph (even if it has no edges)
        for name in self.steps:
            graph.setdefault(name, set())

        return dict(graph)

    def _build_reverse_graph(self) -> Dict[str, Set[str]]:
        """consumer → set of producers (used for ancestor traversal)."""
        reverse: Dict[str, Set[str]] = defaultdict(set)
        for producer, consumers in self.graph.items():
            for consumer in consumers:
                reverse[consumer].add(producer)
        return dict(reverse)

    # ------------------------------------------------------------------
    # Topological sort (Kahn's algorithm)
    # ------------------------------------------------------------------

    def _topological_sort(self) -> List[str]:
        in_degree = {name: 0 for name in self.steps}
        for consumers in self.graph.values():
            for node in consumers:
                in_degree[node] += 1

        queue = deque(name for name, deg in in_degree.items() if deg == 0)
        order: List[str] = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in self.graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self.steps):
            raise RuntimeError("Cycle detected in pipeline DAG.")

        return order

    # ------------------------------------------------------------------
    # Wave decomposition (parallelism)
    # ------------------------------------------------------------------

    def build_execution_waves(self, steps_to_run: List[str]) -> List[List[str]]:
        """
        Partition *steps_to_run* into sequential waves.
        All steps within a wave are independent and can run in parallel.

        A step enters wave N when all its required producers (that are
        included in steps_to_run) have been assigned to waves 0..N-1.
        """
        run_set = set(steps_to_run)
        wave_index: Dict[str, int] = {}

        for name in steps_to_run:  # already in topological order
            producers_in_run = self.reverse_graph.get(name, set()) & run_set
            if producers_in_run:
                wave_index[name] = max(wave_index[p] for p in producers_in_run) + 1
            else:
                wave_index[name] = 0

        max_wave = max(wave_index.values(), default=0)
        waves: List[List[str]] = [[] for _ in range(max_wave + 1)]
        for name, idx in wave_index.items():
            waves[idx].append(name)

        return waves

    # ------------------------------------------------------------------
    # Cache / invalidation helpers
    # ------------------------------------------------------------------

    def _collect_downstream(self, step_name: str) -> Set[str]:
        visited: Set[str] = set()

        def dfs(node: str) -> None:
            for child in self.graph.get(node, set()):
                if child not in visited:
                    visited.add(child)
                    dfs(child)

        dfs(step_name)
        return visited

    def _should_run(self, step: BaseStep, ctx) -> bool:
        """
        Determine whether a step needs to execute.

        Runs when:
        1. Explicitly invalidated (upstream changed).
        2. Any output key is absent from the cache.
        3. Fingerprint changed (skipped in debug_mode when outputs present).
        """
        with self._lock:
            already_invalidated = step.name in self._invalidated

        if already_invalidated:
            return True

        missing_output = not all(ctx.has(k) for k in step.produces)
        if missing_output:
            if self.debug_mode:
                for k in step.produces:
                    if not ctx.has(k):
                        logger.info(
                            f"    - Missing output '{k}' for step '{step.name}'."
                            " Marking for execution."
                        )
            self._mark_invalidated(step.name)
            return True

        if not self.debug_mode:
            new_hash = step.fingerprint(ctx)
            old_hash = ctx.metadata["step_hashes"].get(step.name)
            if old_hash != new_hash:
                self._mark_invalidated(step.name)
                return True

        return False

    def _mark_invalidated(self, step_name: str) -> None:
        """Thread-safe invalidation of a step and all its downstream dependents."""
        with self._lock:
            self._invalidated.add(step_name)
            self._invalidated.update(self._collect_downstream(step_name))

    # ------------------------------------------------------------------
    # Target resolution
    # ------------------------------------------------------------------

    def set_targets(self, targets: Optional[List[str]]) -> None:
        """
        Restrict execution to *targets* and their transitive dependencies.
        Pass None to run the full pipeline.
        """
        with self._lock:
            self._invalidated.clear()

        if targets is None:
            self._steps_to_run = list(self.execution_order)
        else:
            if self.debug_mode:
                with self._lock:
                    self._invalidated.update(targets)
            self._steps_to_run = self._resolve_required_steps(targets)

        # Always re-run the final target even when cached
        if self._steps_to_run:
            self._mark_invalidated(self._steps_to_run[-1])

    def _resolve_required_steps(self, targets: List[str]) -> List[str]:
        """Return the minimal topologically-ordered subgraph needed for *targets*."""
        required: Set[str] = set()

        def collect(name: str) -> None:
            if name in required:
                return
            required.add(name)
            step = self.steps.get(name)
            if step is None:
                raise ValueError(f"Unknown step: '{name}'")
            for key in step.requires:
                producer = self._find_producer(key)
                if producer:
                    collect(producer)

        for t in targets:
            if t not in self.steps:
                raise ValueError(f"Unknown step: '{t}'")
            collect(t)

        return [s for s in self.execution_order if s in required]

    def _find_producer(self, key: str) -> Optional[str]:
        for step in self.steps.values():
            if key in step.produces:
                return step.name
        return None

    # ------------------------------------------------------------------
    # Step execution
    # ------------------------------------------------------------------

    def _run_step(
        self,
        ctx,
        step: BaseStep,
        callback: Optional[Callable] = None,
    ) -> None:
        """Execute a single step, update hashes, and export outputs."""
        start = time.time()

        if callback:
            callback("step_running", step.name)
            
        try :
            step.run(ctx)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error occured during {step.name} : {e}")
            
        elapsed = time.time() - start
        logger.info(f"[DAG] Finished '{step.name}' in {elapsed:.2f}s")

        if callback:
            callback("step_done", step.name, elapsed)

        step.export(ctx)
        ctx.metadata["step_hashes"][step.name] = step.fingerprint(ctx)

    # ------------------------------------------------------------------
    # Public run interface
    # ------------------------------------------------------------------

    def run(
        self,
        ctx,
        targets: Optional[List[str]] = None,
        callback: Optional[Callable] = None,
    ) -> None:
        """
        Execute the DAG, running independent steps in parallel.

        Parameters
        ----------
        ctx:
            Pipeline context (cache, metadata, …).
        targets:
            Optional list of step names to compute. Transitive dependencies
            are included automatically. Pass None to run the full pipeline.
        callback:
            Optional callable invoked with progress events:
            ``("step_start", name, index, total)``
            ``("step_running", name)``
            ``("step_done", name, elapsed)``
            ``("step_skipped", name)``
            ``("finished")``
        """
        if self._steps_to_run is None:
            self.set_targets(targets)

        steps_to_run = self._steps_to_run
        waves = self.build_execution_waves(steps_to_run)

        logger.info(
            f"[DAG] Execution plan: {len(waves)} wave(s), "
            f"{len(steps_to_run)} step(s) — {[list(w) for w in waves]}"
        )

        completed = 0
        total = len(steps_to_run)

        for wave in waves:
            if len(wave) == 1:
                # Single step: run directly, no thread overhead
                self._execute_step_in_run(
                    ctx, wave[0], completed, total, callback
                )
                completed += 1
            else:
                # Multiple independent steps: run in parallel
                futures = {}
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    for step_name in wave:
                        future = executor.submit(
                            self._execute_step_in_run,
                            ctx, step_name, completed, total, callback,
                        )
                        futures[future] = step_name

                    for future in as_completed(futures):
                        step_name = futures[future]
                        exc = future.exception()
                        if exc is not None:
                            logger.error(
                                f"[DAG] Step '{step_name}' raised an exception: {exc}"
                            )
                            raise exc
                        completed += 1

        with self._lock:
            self._invalidated.clear()
        self._steps_to_run = None

        if callback:
            callback("finished")

    def _execute_step_in_run(
        self,
        ctx,
        step_name: str,
        index: int,
        total: int,
        callback: Optional[Callable],
    ) -> None:
        """Decide whether to run or skip a step, then act accordingly."""
        step = self.steps[step_name]

        if callback:
            callback("step_start", step_name, index, total)

        with self._lock:
            is_invalidated = step_name in self._invalidated

        if is_invalidated:
            logger.info(f"[DAG] Running (invalidated): '{step_name}'")
            self._run_step(ctx, step, callback=callback)
            self._mark_invalidated(step_name)  # propagate to downstream
            return

        if not self._should_run(step, ctx):
            logger.info(f"[DAG] Skipping (valid cache): '{step_name}'")
            if callback:
                callback("step_skipped", step_name)
            step.export(ctx)
            return

        logger.info(f"[DAG] Running step: '{step_name}'")
        self._run_step(ctx, step, callback=callback)