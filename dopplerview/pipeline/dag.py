# dopplerview/core/dag.py

from collections import defaultdict, deque
from typing import Dict, List, Iterable
import time


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


class DAGEngine:
    """
    Directed Acyclic Graph execution engine.

    - Resolves dependencies automatically
    - Executes only required steps
    """

    def __init__(self, steps: Iterable[BaseStep], debug_mode=False):
        self.steps: Dict[str, BaseStep] = {s.name: s for s in steps}

        self._validate_unique_names() # Ensure all steps have unique names
        self.graph = self._build_dependency_graph() # Directed acyclic graph of step dependencies
        self.execution_order = self._topological_sort() # Cached topological order of steps for execution

        self.invalidated = set() # Steps marked for execution due to changes or missing outputs
        self.steps_to_run = None # Cache resolved execution order for given targets

        self.debug_mode = debug_mode

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _validate_unique_names(self):
        if len(self.steps) == 0:
            raise ValueError("No steps registered in DAG.")

        if len(set(self.steps.keys())) != len(self.steps):
            raise ValueError("Duplicate step names detected.")

    def _build_dependency_graph(self):
        """
        Build step-to-step dependency graph based on produced keys.
        """
        key_producers = {}
        graph = defaultdict(set)

        # Map which step produces which key
        for step in self.steps.values():
            for key in step.produces:
                if key in key_producers:
                    raise ValueError(
                        f"Multiple steps produce the same key: '{key}'"
                    )
                key_producers[key] = step.name

        # Build dependency edges
        for step in self.steps.values():
            for required_key in step.requires:
                if required_key not in key_producers:
                    continue  # Assume provided externally
                producer = key_producers[required_key]
                graph[producer].add(step.name)

        return graph

    # ------------------------------------------------------------------
    # Topological sort
    # ------------------------------------------------------------------

    def _topological_sort(self):
        """
        Kahn's algorithm.
        """
        in_degree = {name: 0 for name in self.steps}
        for deps in self.graph.values():
            for node in deps:
                in_degree[node] += 1

        queue = deque([n for n, deg in in_degree.items() if deg == 0])
        order = []

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
    # Execution
    # ------------------------------------------------------------------

    def _should_run(self, step, ctx):
        """Determine if a step needs to be executed.
        A step should run if:
        1. It is explicitly invalidated (e.g. due to upstream changes).
        2. Any of its outputs are missing from the cache.
        3. The fingerprint hash of the step's relevant config or inputs has changed since last execution.

        If in debug mode, steps will be skipped if outputs are present, regardless of hash changes, allowing to bypass expensive computations while iterating on the pipeline.
        """
        should_run = step.name in self.invalidated

        if not should_run:
            # If outputs missing -> must run
            if not all(ctx.has(k) for k in step.produces):
                if self.debug_mode:
                    for k in step.produces:
                        if not ctx.has(k):
                            print(f"    - Missing output '{k}' for step '{step.name}'. Marking for execution.")
                should_run = True

            new_hash = step.fingerprint(ctx)
            old_hash = ctx.metadata["step_hashes"].get(step.name)

            # If hashes differ, must run
            if old_hash != new_hash and not self.debug_mode:
                should_run = True

        # If should run, invalidate downstream to ensure following steps also re-run
        if should_run:
            self.invalidated.add(step.name)
            self.invalidated.update(self._collect_downstream(step.name))

        return should_run
    
    def set_targets(self, targets: List[str]):
        """
        Set specific targets for execution, invalidating necessary steps.
        """
        self.invalidated.clear()
        if targets is None:
            self.steps_to_run = self.execution_order
        else:
            if self.debug_mode:
                self.invalidated.update(targets)
            self.steps_to_run = self._resolve_required_steps(targets)
        
        # Last target is always invalidated to ensure it runs, even if cached
        if self.steps_to_run is not None and len(self.steps_to_run) > 0:
            self.invalidated.add(self.steps_to_run[-1])

    def run_step(self, ctx, step: BaseStep):
        start_time = time.time()
        step.run(ctx)
        elapsed = time.time() - start_time
        print(f"[DAG] Finished {step.name} in {elapsed:.2f}s")
        step.export(ctx)
        ctx.metadata["step_hashes"][step.name] = step.fingerprint(ctx)

    def run(self, ctx, targets: List[str] = None):
        """
        Execute the DAG.

        If targets is None, run entire pipeline

        If targets provided, run only required subset
        """
        if self.steps_to_run is None:
            self.set_targets(targets)

        print(f"[DAG] Execution order: {self.steps_to_run}")

        for step_name in self.steps_to_run:
            step = self.steps[step_name]

            if step_name in self.invalidated:
                print(f"[DAG] Running (invalidated): {step.name}")
                self.run_step(ctx, step)

                # Invalidate downstream
                downstream = self._collect_downstream(step_name)
                self.invalidated.update(downstream)
                continue

            if not self._should_run(step, ctx):
                print(f"[DAG] Skipping (valid cache): {step.name}")
                step.export(ctx)
                continue

            print(f"[DAG] Running step: {step.name}")
            self.run_step(ctx, step)

        self.invalidated.clear()
        self.steps_to_run = None
            

    # ------------------------------------------------------------------
    # Partial execution support
    # ------------------------------------------------------------------

    def _resolve_required_steps(self, targets: List[str]) -> List[str]:
        """
        Determine minimal subgraph needed to compute targets.
        """

        required = set()

        def collect(step_name):
            if step_name in required:
                return
            required.add(step_name)

            step = self.steps[step_name]
            for key in step.requires:
                producer = self._find_producer(key)
                if producer:
                    collect(producer)

        for t in targets:
            if t not in self.steps:
                raise ValueError(f"Unknown step: {t}")
            collect(t)

        # Preserve topological order
        return [s for s in self.execution_order if s in required]

    def _find_producer(self, key):
        for step in self.steps.values():
            if key in step.produces:
                return step.name
        return None
    
    def _collect_downstream(self, step_name):
        visited = set()

        def dfs(node):
            for child in self.graph[node]:
                if child not in visited:
                    visited.add(child)
                    dfs(child)

        dfs(step_name)
        return visited