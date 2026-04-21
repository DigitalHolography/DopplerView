# CONTRIBUTING.md

# Contributing to HoloSegment

Thank you for contributing to **HoloSegment**.

This document explains how to:

* Add a new model to the registry
* Add a new pipeline step
* Modify the pipeline DAG
* Understand execution & caching behavior
* Respect reproducibility guarantees

This is a developer-focused document.

---

# 1. Architecture Overview

HoloSegment is built around three core components:

1. **Model Registry + Model Manager**
2. **Pipeline (DAG-based execution engine)**
3. **Context (runtime data container)**

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
Context (shared state)
```

---

# 2. The Execution Model

## 2.1 Context

`Context` is the central execution container.

It stores:

* Runtime artifacts (`ctx.cache`)
* Configuration (`ctx.eyeflow_config`)
* ModelManager
* Loaded model instances (lazy)
* Step fingerprints
* Output manager

### Access patterns

```python
ctx.get("key")
ctx.set("key", value)
ctx.require("key")  # raises if missing
ctx.has("key")
```

Steps MUST communicate **only via context keys**.

No global state.
No hidden side effects.

---

## 2.2 DAG-Based Pipeline

The pipeline is executed by `DAGEngine`.

Each step declares:

```python
name: str
requires: List[str]
produces: List[str]
```

The DAG is built automatically:

* A step depends on another step if it requires a key that the other produces.
* Cycles are forbidden.
* Multiple producers of the same key are forbidden.

Execution uses:

* Topological sorting
* Deterministic step order
* Automatic dependency resolution

---

## 2.3 Caching & Fingerprinting

Each step implements:

```python
fingerprint(ctx)
```

Default behavior hashes:

* Relevant config
* All required inputs

If:

* Outputs already exist
* AND fingerprint unchanged

→ Step is skipped.

If fingerprint changes:
→ Downstream steps are invalidated.

This guarantees:

* Deterministic recomputation
* Partial execution
* Reproducibility

---

# 3. Adding a New Pipeline Step

## 3.1 Create the Step Class

Steps must inherit from:

```python
from holosegment.pipeline.step import BaseStep
```

### Minimal Example

```python
class MyNewStep(BaseStep):

    name = "my_new_step"

    requires = ["retinal_vessel_mask"]
    produces = ["refined_mask"]

    def run(self, ctx):
        vessel_mask = ctx.require("retinal_vessel_mask")

        refined = do_something(vessel_mask)

        ctx.set("refined_mask", refined)
```

---

## 3.2 Rules for Steps

### 1. Unique name

Every step must have a unique `name`.

If duplicated → DAG construction fails.

---

### 2. No side effects

Steps must:

* Only read from `ctx`
* Only write declared `produces`
* Not modify unrelated keys

---

### 3. Always declare correct dependencies

If your step reads:

```python
ctx.get("optic_disc_mask")
```

Then:

```python
requires = ["optic_disc_mask"]
```

If you forget this:

* DAG will not enforce ordering
* Fingerprinting becomes invalid
* Caching breaks

---

### 4. Produce declared outputs only

If your step sets:

```python
ctx.set("segmentation", result)
```

Then:

```python
produces = ["segmentation"]
```

---

## 3.3 Optional: Custom Fingerprinting

By default, fingerprint hashes:

* Entire config
* All required inputs

If your step only depends on part of config:

```python
def _relevant_config(self, ctx):
    return {
        "threshold": ctx.eyeflow_config["threshold"]
    }
```

This prevents unnecessary invalidation.

---

## 3.4 Registering the Step in the Pipeline

After creating your step:

Open `pipeline.py`:

```python
self.steps = {
    PreprocessStep(),
    ...
}
```

Add:

```python
MyNewStep(),
```

That’s it.

DAGEngine will:

* Automatically compute dependencies
* Automatically insert it in correct order

---

## 3.5 Nested Steps

If your logic contains multiple atomic operations, use:

```python
class MyCompositeStep(NestedStep):
    substeps = [
        StepA(),
        StepB(),
    ]
```

Nested steps:

* Execute sequentially
* Combine fingerprints of substeps

Use them to group logically related operations.

---

# 4. Model Registry

Models are defined in a YAML file loaded by:

```python
ModelRegistryConfig
```

Each model entry must define:

```yaml
iternet5_vesselness:
  task: vessel_segmentation
  hf_repo: DigitalHolography/iternet5_vesselness
  filename: iternet5_vesselness
  format: onnx
  input_norm: minmax
  output_activation: sigmoid
  revision: main
  input_channels: ["M0_ff_image"]
```

---

# 5. Adding a New Model

## 5.1 Step 1 — Upload Model

Upload model weights to:

* A HuggingFace repository

The system downloads models via:

```python
hf_hub_download(...)
```

---

## 5.2 Step 2 — Add YAML Entry

In the registry YAML file, add:

```yaml
my_new_model:
  task: vessel_segmentation
  hf_repo: my-org/my-repo
  filename: model.onnx
  format: onnx
  input_norm: minmax
  output_activation: sigmoid
  revision: main
  input_channels: ["M0_ff_image"]
```

If your model needs new format, input normalization method, input channels or output activation, you must implement it.

---

## 5.3 Step 3 — That’s It

No code modification required.

The registry automatically:

* Registers model
* Assigns it to the task
* Makes it selectable via ModelManager

---

# 6. Model Selection & Task Binding

Each task has a default model:

```python
self.model_tasks = {
    task: models[0]
}
```

To change model during runtime:

```python
ctx.change_model_for_task("vessel_segmentation", "my_new_model")
```

To get current model inside a step:

```python
model = ctx.get_current_model_for_task("vessel_segmentation")
```

Models are:

* Downloaded on demand
* Loaded lazily
* Cached in memory

---

# 7. Model Formats

Supported formats:

* `"pt"` → TorchModelWrapper
* `"onnx"` → ONNXModelWrapper

If you add a new format:

Modify:

```python
ModelManager.build_model_wrapper()
```

---

# 8. Partial Pipeline Execution

You can run only part of the pipeline:

```python
pipeline.run(targets=["av_segmentation"])
```

DAGEngine will:

* Resolve required upstream steps
* Execute minimal subgraph

---

# 9. Output Management

`Context.create_output_folder()` initializes:

```python
OutputManager
```

If `debug=True`:

* Intermediate outputs may be saved.

Steps may optionally use:

```python
ctx.output_manager
```

Keep I/O isolated from core logic when possible.

---

# 10. Design Principles

When contributing, respect:

### Determinism

Same inputs + same config → same outputs.

### No Hidden State

Everything flows through `Context`.

### Declarative Dependencies

All step dependencies must be declared.

### Reproducibility

Fingerprinting must remain stable.

---

# 11. Common Mistakes

❌ Forgetting to declare a required key
❌ Producing the same key in two steps
❌ Modifying context without declaring `produces`
❌ Using external mutable global state
❌ Using random seeds without fixing them

---

# 12. Recommended Development Workflow

1. Create new step
2. Add it to pipeline
3. Run full pipeline once
4. Modify config
5. Ensure correct invalidation behavior
6. Test partial execution
7. Validate output determinism

---

# 13. Testing Checklist

When adding a step or model:

* [ ] Full pipeline runs
* [ ] Partial execution works
* [ ] Fingerprinting invalidates correctly
* [ ] No duplicate produced keys
* [ ] No dependency cycles
* [ ] Model loads correctly
* [ ] Output folder structure preserved

---

# 14. Architectural Guarantees

HoloSegment guarantees:

* Deterministic execution order
* Automatic dependency resolution
* Cache-aware recomputation
* Model version traceability
* Modular extension