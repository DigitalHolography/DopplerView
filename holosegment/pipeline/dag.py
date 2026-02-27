# holosegment/core/dag.py

from collections import defaultdict, deque
from typing import Dict, List, Iterable


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

    def __init__(self, steps: Iterable[BaseStep]):
        self.steps: Dict[str, BaseStep] = {s.name: s for s in steps}

        self._validate_unique_names()
        self.graph = self._build_dependency_graph()
        self.execution_order = self._topological_sort()

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

    def run(self, ctx, targets: List[str] = None):
        """
        Execute the DAG.

        If targets is None:
            → run entire pipeline

        If targets provided:
            → run only required subset
        """

        if targets is None:
            steps_to_run = self.execution_order
        else:
            steps_to_run = self._resolve_required_steps(targets)

        for step_name in steps_to_run:
            step = self.steps[step_name]

            # Validate runtime requirements
            for key in step.requires:
                if not ctx.has(key):
                    raise RuntimeError(
                        f"BaseStep '{step.name}' requires '{key}' "
                        f"but it is missing from context."
                    )

            print(f"[DAG] Running step: {step.name}")
            step.run(ctx)

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