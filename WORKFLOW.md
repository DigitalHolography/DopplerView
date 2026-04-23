# WORKFLOW.md

## DopplerView Workflow

This document describes the internal architecture and execution model of the DopplerView segmentation pipeline.

It focuses on system design, execution guarantees, and architectural principles.

For instructions on extending the system, see [`CONTRIBUTING.md`](CONTRIBUTING.md).

---
## Key Features

### Deterministic Execution

Each pipeline step computes a unique fingerprint based on:

* Relevant configuration
* Input data

If configuration or inputs change, only the necessary steps are recomputed.

---

### Modular Step System

Each step:

* Declares required inputs
* Declares produced outputs
* Runs only when dependencies are satisfied

The DAG engine guarantees:

* No cyclic dependencies
* No duplicate output producers
* Stable execution order

---

### Model Registry

Models are defined declaratively in a YAML registry and:

* Downloaded automatically from HuggingFace
* Version-controlled via repository revision
* Loaded lazily
* Switchable per task at runtime

Supported formats:

* PyTorch (`.pt`)
* ONNX (`.onnx`)

---

## 1. High-Level Architecture

DopplerView is built around a modular, deterministic pipeline designed for:

* Reproducibility
* Partial recomputation
* Clear dependency management
* Model version traceability
* Separation of concerns

Execution flow:

```
CLI / GUI
    ↓
Pipeline
    ↓
DAGEngine
    ↓
Steps
    ↓
Context (shared runtime state)
    ↓
OutputManager
```

The system ensures that computation is:

* Explicitly declared
* Dependency-driven
* Deterministic
* Cache-aware

---

## 2. Pipeline as a Directed Acyclic Graph (DAG)

The segmentation workflow is implemented as a **Directed Acyclic Graph (DAG)**.

Each node represents a **Step**.

Edges represent **data dependencies** between steps.

A step depends on another step if it requires a key produced by that step.

### 2.1 Step Declaration

Each step declares:

* `name` — unique identifier
* `requires` — list of input keys
* `produces` — list of output keys
* `run(ctx)` — execution logic

Example structure:

```python
class ExampleStep(BaseStep):
    name = "example"
    requires = ["input_key"]
    produces = ["output_key"]

    def run(self, ctx):
        ...
```

---

### 2.2 Dependency Resolution

During initialization:

1. All steps are registered.
2. The engine maps:

   * Which step produces which key.
3. A dependency graph is constructed automatically.
4. A topological sort determines execution order.

Constraints:

* No duplicate step names.
* No duplicate produced keys.
* No dependency cycles.

If a cycle is detected, execution fails immediately.

---

## 3. Execution Engine

The `DAGEngine` is responsible for:

* Dependency resolution
* Topological sorting
* Selective execution
* Cache validation
* Downstream invalidation

---

### 3.1 Full Execution

If no targets are specified:

* All steps are executed in topological order.

---

### 3.2 Partial Execution

The pipeline supports partial execution.

When specific targets are provided:

```python
pipeline.run(targets=["av_segmentation"])
```

The engine:

1. Resolves the minimal required subgraph.
2. Executes only necessary upstream steps.
3. Preserves global topological order.

This enables:

* Fast experimentation
* Targeted recomputation
* Step-level debugging

---

## 4. Context: Shared Runtime State

The `Context` object is the shared runtime container.

It stores:

* Configuration
* Model manager
* Model instances (lazy-loaded)
* Input folder
* Output manager
* Runtime cache (intermediate artifacts)
* Step fingerprints

All inter-step communication occurs exclusively via the context.

No global state is used.

---

### 4.1 Runtime Cache

Intermediate results are stored in:

```
ctx.cache
```

Keys represent semantic artifacts such as:

* Preprocessed images
* Segmentation masks
* Optic disc localization
* Pulse features

This design ensures:

* Explicit data flow
* Clear provenance
* Easy debugging

---

## 5. Deterministic Fingerprinting

Each step has a deterministic fingerprint.

The fingerprint depends on:

* Relevant configuration
* Input data signatures

By default:

* The entire configuration is hashed.
* All required inputs are hashed.
* NumPy arrays are hashed via raw bytes.

Fingerprint formula (conceptually):

```
fingerprint = hash(
    relevant_config +
    hashed_inputs
)
```

---

### 5.1 Cache Validation Logic

Before running a step:

1. If outputs are missing → run.
2. If fingerprint differs from previous run → run.
3. Otherwise → skip (cached result valid).

When a step is re-executed:

* All downstream dependent steps are invalidated automatically.

This guarantees:

* Correct recomputation
* Minimal redundant work
* Deterministic behavior

---

## 6. Model Registry and Model Lifecycle

Models are registered declaratively in a YAML configuration file.

Each model defines:

* Task association
* HuggingFace repository
* Filename
* Revision
* Format (`pt` or `onnx`)
* Input normalization
* Output activation
* Input channels

---

### 6.1 ModelManager Responsibilities

The `ModelManager`:

* Resolves model specifications
* Downloads weights via `hf_hub_download`
* Caches models locally
* Manages task → model mapping
* Supports runtime model switching

Models are:

* Downloaded lazily
* Loaded on first access
* Cached in memory per session

---

### 6.2 Task-Based Model Selection

Each task is associated with a default model.

Steps do not hardcode model names.

Instead, they request:

```python
ctx.get_current_model_for_task(task_name)
```

This abstraction enables:

* Model experimentation
* Version switching
* Clean separation between logic and weights

---

## 7. Output Management

Output generation is managed through `OutputManager`.

The context initializes an output directory per run.

The output system:

* Is optional (debug mode)
* Avoids mixing computation and persistence logic
* Keeps reproducibility intact

---

## 8. Architectural Guarantees

DopplerView guarantees:

#### Deterministic Execution

Given identical:

* Input data
* Configuration
* Model versions

The output is deterministic.

---

#### Explicit Dependencies

All data dependencies are declared.

No hidden coupling exists between steps.

---

#### Automatic Invalidation

Configuration changes trigger only necessary recomputation.

---

#### No Global State

All runtime information flows through `Context`.

---

#### Model Version Traceability

Model revision and source repository are explicitly defined in the registry.

---

## 9. Design Principles

The system is built around:

* Declarative computation
* Explicit data flow
* Minimal side effects
* Modular extensibility
* Reproducible research workflows

The DAG abstraction ensures the pipeline remains:

* Scalable
* Maintainable
* Safe to extend

---

## 10. Summary

The DopplerView workflow is:

* A deterministic DAG-based pipeline
* With fingerprint-based selective execution
* Backed by a declarative model registry
* Using a shared runtime context
* Designed for reproducibility and extensibility

For instructions on extending the pipeline or adding models, refer to [`CONTRIBUTING.md`](CONTRIBUTING.md).
