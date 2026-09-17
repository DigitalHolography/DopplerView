"""Static pipeline structure, independent from any execution runtime."""

from __future__ import annotations

from collections import defaultdict, deque
from types import MappingProxyType
from typing import Dict, Iterable, List, Mapping, Set, Tuple

from dopplerview.pipeline.step import BaseStep
class PipelineDefinition:
    """Validated, deterministic description of the DopplerView DAG.

    A definition contains step metadata and graph relationships only. It does
    not allocate executors, load models or configuration, or own input/output
    state, so it is safe for the GUI and other discovery-only consumers.
    """

    def __init__(self, steps: Iterable[BaseStep]):
        registered_steps = tuple(steps)
        self._validate_unique_names(registered_steps)
        self._steps: Tuple[BaseStep, ...] = registered_steps
        self._steps_by_name: Mapping[str, BaseStep] = MappingProxyType(
            {step.name: step for step in registered_steps}
        )
        self._registration_index = {
            step.name: index for index, step in enumerate(registered_steps)
        }
        self._graph = self._build_dependency_graph()
        self._reverse_graph = self._build_reverse_graph()
        self._execution_order = tuple(self._topological_sort())

    @classmethod
    def default(cls) -> "PipelineDefinition":
        from dopplerview.pipeline.steps.arterial_waveform_analysis import ArterialWaveformAnalysisStep
        from dopplerview.pipeline.steps.av_segmentation import ChoroidalAVSegmentationStep, RetinalAVSegmentationStep
        from dopplerview.pipeline.steps.eye_laterality_classification import EyeLateralityClassificationStep
        from dopplerview.pipeline.steps.optic_disc import OpticDiscSegmentationStep
        from dopplerview.pipeline.steps.preprocess import PreprocessStep
        from dopplerview.pipeline.steps.pulse_analysis import PulseAnalysisStep
        from dopplerview.pipeline.steps.read_moments import ReadMomentsStep
        from dopplerview.pipeline.steps.vessel_segmentation import ChoroidalVesselSegmentationStep, RetinalVesselSegmentationStep
        from dopplerview.pipeline.steps.vessel_velocity_estimator import VesselVelocityEstimatorStep

        return cls(
            [
                ReadMomentsStep(),
                PreprocessStep(),
                EyeLateralityClassificationStep(),
                OpticDiscSegmentationStep(),
                RetinalVesselSegmentationStep(),
                ChoroidalVesselSegmentationStep(),
                PulseAnalysisStep(),
                RetinalAVSegmentationStep(),
                ChoroidalAVSegmentationStep(),
                VesselVelocityEstimatorStep(),
                ArterialWaveformAnalysisStep(),
            ]
        )

    @property
    def steps(self) -> Tuple[BaseStep, ...]:
        return self._steps

    @property
    def steps_by_name(self) -> Mapping[str, BaseStep]:
        return self._steps_by_name

    @property
    def graph(self) -> Dict[str, Set[str]]:
        return {name: set(consumers) for name, consumers in self._graph.items()}

    @property
    def reverse_graph(self) -> Dict[str, Set[str]]:
        return {name: set(producers) for name, producers in self._reverse_graph.items()}

    @property
    def execution_order(self) -> List[str]:
        return list(self._execution_order)

    @staticmethod
    def _validate_unique_names(steps: Tuple[BaseStep, ...]) -> None:
        if not steps:
            raise ValueError("No steps registered in pipeline definition.")
        seen = set()
        duplicates = []
        for step in steps:
            if step.name in seen and step.name not in duplicates:
                duplicates.append(step.name)
            seen.add(step.name)
        if duplicates:
            raise ValueError("Duplicate step names detected: " + ", ".join(duplicates))

    def _ordered(self, names: Iterable[str]) -> List[str]:
        return sorted(names, key=self._registration_index.__getitem__)

    def _build_dependency_graph(self) -> Dict[str, Set[str]]:
        producers: Dict[str, str] = {}
        graph: Dict[str, Set[str]] = defaultdict(set)
        for step in self._steps:
            for key in step.produces:
                if key in producers:
                    raise ValueError(f"Multiple steps produce the same key: '{key}'")
                producers[key] = step.name
        for step in self._steps:
            for key in step.requires:
                producer = producers.get(key)
                if producer:
                    graph[producer].add(step.name)
        for step in self._steps:
            graph.setdefault(step.name, set())
        return dict(graph)

    def _build_reverse_graph(self) -> Dict[str, Set[str]]:
        reverse: Dict[str, Set[str]] = defaultdict(set)
        for producer, consumers in self._graph.items():
            for consumer in consumers:
                reverse[consumer].add(producer)
        return dict(reverse)

    def _topological_sort(self) -> List[str]:
        in_degree = {name: 0 for name in self._steps_by_name}
        for consumers in self._graph.values():
            for consumer in consumers:
                in_degree[consumer] += 1
        ready = deque(self._ordered(name for name, degree in in_degree.items() if degree == 0))
        order = []
        while ready:
            name = ready.popleft()
            order.append(name)
            newly_ready = []
            for consumer in self._ordered(self._graph[name]):
                in_degree[consumer] -= 1
                if in_degree[consumer] == 0:
                    newly_ready.append(consumer)
            ready.extend(newly_ready)
            ready = deque(self._ordered(ready))
        if len(order) != len(self._steps):
            raise RuntimeError("Cycle detected in pipeline DAG.")
        return order

    def resolve_execution_graph(self, targets=None) -> List[str]:
        if targets == []:
            return []
        if targets is None:
            return self.execution_order
        unknown = [target for target in targets if target not in self._steps_by_name]
        if unknown:
            raise ValueError("Unknown target step(s): " + ", ".join(unknown))
        required = set()

        def visit(name: str) -> None:
            if name in required:
                return
            required.add(name)
            for producer in self._reverse_graph.get(name, set()):
                visit(producer)

        for target in targets:
            visit(target)
        return [name for name in self._execution_order if name in required]

    def get_downstream_steps(self, step_name: str) -> Set[str]:
        if step_name not in self._steps_by_name:
            raise ValueError(f"Unknown step: {step_name}")
        downstream = set()
        stack = list(self._graph.get(step_name, set()))
        while stack:
            name = stack.pop()
            if name not in downstream:
                downstream.add(name)
                stack.extend(self._graph.get(name, set()))
        return downstream
