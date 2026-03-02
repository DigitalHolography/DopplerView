# HoloSegment

Modular artery/vein segmentation pipeline for Doppler holography retinal imaging.

HoloSegment processes Doppler hologram data acquired with **Holodoppler systems** and performs deterministic, reproducible artery/vein segmentation using a DAG-based pipeline and a configurable model registry.

The project provides:

* A deterministic DAG-based processing pipeline
* Automatic dependency resolution and selective recomputation
* A model registry with HuggingFace integration
* CLI execution
* A Streamlit-based GUI

---

# Overview

HoloSegment operates on **Holodoppler acquisition folders**, not single `.holo` files.

The pipeline processes spectral moments derived from Doppler holograms and performs:

1. **Folder loading**

   * Reads Holodoppler folder structure
   * Loads configuration files
   * Initializes runtime context

2. **Moment extraction**

   * Loads spectral moments (e.g., M0 full-field image)

3. **Preprocessing**

   * Image normalization
   * Optional registration
   * Preparation for model inference

4. **Optic disc detection**

5. **Binary vessel segmentation**

   * Deep learning–based vessel mask extraction

6. **Pulse analysis**

   * Temporal signal analysis using vessel masks

7. **Artery/vein semantic segmentation**

   * Classification of vessels into arteries and veins

The entire workflow is implemented as a **Directed Acyclic Graph (DAG)** with automatic dependency resolution and fingerprint-based cache validation.

---

# Key Features

## Deterministic Execution

Each pipeline step computes a unique fingerprint based on:

* Relevant configuration
* Input data

If configuration or inputs change, only the necessary steps are recomputed.

---

## Modular Step System

Each step:

* Declares required inputs
* Declares produced outputs
* Runs only when dependencies are satisfied

The DAG engine guarantees:

* No cyclic dependencies
* No duplicate output producers
* Stable execution order

---

## Model Registry

Models are defined declaratively in a YAML registry and:

* Downloaded automatically from HuggingFace
* Version-controlled via repository revision
* Loaded lazily
* Switchable per task at runtime

Supported formats:

* PyTorch (`.pt`)
* ONNX (`.onnx`)

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your-org/holosegment.git
cd holosegment
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install in editable mode:

```bash
pip install -e .
```

Optional (development tools):

```bash
pip install -e .[dev]
```

---

# Usage

## CLI

The CLI runs the full pipeline on a Holodoppler acquisition folder.

```bash
holosegment run --input /path/to/holodoppler_folder --config config.json
```

### Arguments

* `holodoppler_folder` : Path to Holodoppler folder
* `--config` : Eyeflow configuration JSON file (optional)
* `--debug` : Enable debug mode (save intermediate outputs)

### Example

```bash
holosegment ./data/patient_01 \
    --config ./configs/default.json \
    --debug
```

## GUI (Streamlit)

Launch the graphical interface:

```bash
streamlit run app.py
```

The GUI allows you to:

* Load a Holodoppler folder
* Automatically run initial preprocessing
* Visualize M0 full-field image
* Overlay vessel and AV segmentation masks
* Run the full pipeline interactively

---

# Project Structure

```
holosegment/
│
├── holosegment/
│   ├── pipeline/          # DAG engine, steps, context
│   ├── models/            # Registry, manager, wrappers
│   ├── input_output/      # Folder reading & output handling
│   ├── utils/
│   └── ...
│
├── app.py                 # Streamlit GUI
├── cli.py                 # Command-line script
├── README.md
├── WORKFLOW.md            # Architecture documentation
├── CONTRIBUTING.md        # Developer guide
├── pyproject.toml
└── requirements.txt
```

---

# Configuration

The pipeline configuration is provided via a JSON file.

It controls:

* Preprocessing parameters
* Model-related parameters
* Task-specific thresholds
* Runtime options

Fingerprinting ensures that changing configuration only recomputes affected steps.

See `WORKFLOW.md` for details on how configuration impacts execution.

---

# Output

Each pipeline run creates a dedicated output folder.

Depending on debug mode, it may include:

* Vessel masks
* Artery/vein masks
* Pulse analysis results
* Intermediate artifacts
* Metadata and step fingerprints

Outputs are isolated per run to ensure reproducibility.

---

# Documentation

* Architecture and execution model → `WORKFLOW.md`
* How to add steps or models → `CONTRIBUTING.md`

---

# License

MIT
