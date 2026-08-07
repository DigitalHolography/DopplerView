# DopplerView

DopplerView is a deep-learning tool for image enhancement, vascular segmentation, and topology inference that processes HoloDoppler results into vascular maps and analysis-ready inputs for EyeFlow, using a DAG-based pipeline and a configurable model registry.

The project provides:

* A deterministic DAG-based processing pipeline
* Automatic dependency resolution and selective recomputation
* A model registry with HuggingFace integration
* CLI execution
* A tkinter-based App used for deployement, with minimal and advanced UI

---

# Overview

DopplerView operates on **Holodoppler acquisition folders**.

The pipeline processes spectral moments derived from Doppler holograms and performs:

1. **Preprocessing**

   * Image normalization, flat-field correction and optional registration.

2. **Optic disc detection**

   * Using model publicly available on [huggingface](https://huggingface.co/DigitalHolography/EyeFlow_OpticDiscDetector).

3. **Binary vessel segmentation**

   * Deep learning–based vessel mask extraction, of retinal and choroidal vessels.
   * Using model publicly available on [huggingface](https://huggingface.co/collections/DigitalHolography/doppler-retinal-vessel-segmentation), trained with M0 images available on [huggingface](https://huggingface.co/datasets/DigitalHolography/HoloDopplerSegISBI).

4. **Pulse analysis**

   * Computation of diastolic/systolic frames and temporal correlation map using the arterial signal obtained with the pre-classified arteries, following the strategy described in [Dubosc, Marius, et al. "Improving segmentation of retinal arteries and veins using cardiac signal in doppler holograms." arXiv preprint arXiv:2511.14654 (2025).](https://arxiv.org/abs/2511.14654) except for the pre-classification, now done by clustering the fourier harmonic features of the vessel branches.

5. **Artery/vein semantic segmentation**

   * Following the strategy described in [the same paper](https://arxiv.org/abs/2511.14654).
   * Using models publicly available on [huggingface](https://huggingface.co/collections/DigitalHolography/doppler-retinal-vessel-segmentation). The different models used in the pipelines are indicated in [models.yaml](config/models.yaml).
   * The dataset used for training is publicly available on [huggingface](https://huggingface.co/datasets/DigitalHolography/HoloDopplerSegISBI), with the M0 images and temporal cues already computed.

6. **Estimation of the velicity in the retinal vessels**

   * Using the forward scattering model described in [Fischer, Yann, et al. "Retinal arterial blood flow measured by real-time Doppler holography at 33,000 frames per second." 2024 16th Biomedical Engineering International Conference (BMEiCON). IEEE, 2024.](https://ieeexplore.ieee.org/abstract/document/10896274)

7. **ArterialWaveformAnalysisStep**

   * Per-beat signal analysis

The entire workflow is implemented as a **Directed Acyclic Graph (DAG)** with automatic dependency resolution and fingerprint-based cache validation.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your-org/DopplerView.git
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

Install in editable mode:

```bash
pip install -e .
pip install -r requirements.txt
```

---

# Usage

DopplerView runs using a .holo folder with a corresponding [HoloDoppler](https://github.com/DigitalHolography/HoloDopplerPython/tree/main) folder, with the following structure :

```
measure_id.holo
measure_id/
└── measure_id_HD/
    ├── json/
    │   └── parameters_holodoppler.json      # The rendering parameters
    ├── h5/
    │   └── measure_id_HD_output.h5          # The .h5 file used as input
    ├── mp4/                                 # Video of the rendered moments
    └── png/                                 # Visualization of the rendering
```

## Executable (InnoSetup + TKinter)

* Download the .exe of the latest release, and let the installer do its things. Configurations files will be automatically loaded in C:\\Users\\*user_name*\\AppData\\Roaming\\DopplerView\\*release_version*
* Run DopplerView.exe
   * Drag and drop one or several .holo file(s), a folder containing .holo file(s) or a .txt file with the list of inputs, and click on *Run full pipeline*
   * To select the steps and the config used in the pipeline, activate the *Advanced view* (*View* > *Advanced View*)
   * To modify the models used in the pipline, the configuration or the .h5 output format : *Config* > *Open Configuration*


## CLI

The CLI runs the full pipeline on a Holodoppler acquisition folder.

```bash
dopplerview /path/to/measure.holo --config config.json
```

### Arguments

* `-h, --help`            show this help message and exit
*  `-c CONFIG, --config CONFIG`
                        Path to JSON configuration file  
*  `-t TARGETS [TARGETS ...], --targets TARGETS [TARGETS ...]`
                        List of target steps to run
*  `-d, --debug`           Enable debug mode. In this mode, steps outputs are read from the cache.h5 (C:\\Users\\*user_name*\\.cache\\dopplerview\\cache\\*measure_name*\\cache.h5), and   
                        only targeted steps are re-run. This is useful for debugging specific       
                        steps without having to re-run the entire pipeline.
*  `--execution-profile {default,sequential_reference}`
                        Execution policy. The sequential reference profile forces DAG and
                        internal operation worker counts to one for performance baselines.

### Example

```bash
dopplerview ./data/patient_01 \
    --config ./configs/default.json \
```

For a sequential performance reference:

```bash
dopplerview ./data/patient_01 --execution-profile sequential_reference
```

The same profile can be applied to either the CLI or GUI process with the
`DOPPLERVIEW_EXECUTION_PROFILE=sequential_reference` environment variable.
---

# Project Structure

```
DopplerView/
│
├── dopplerview/
│   ├── input_output/      # Folder reading & output handling
│   ├── models/            # Registry, manager, wrappers
│   ├── pipeline/          # DAG engine, steps, context
│   ├── utils/
│   │   └── ...
│   ├── ui                 # Tkinter GUI
│   │   └── ...
│   ├── cli.py             # Command-line script
│   └── ...
│
├── README.md
├── WORKFLOW.md            # Architecture documentation
├── CONTRIBUTING.md        # Developer guide
├── CHANGELOG.md           # Releases description
└── LICENSE
```

---

# Configuration

The pipeline configuration is provided via a JSON file. It can either be the User configuration in C:\\Users\\*user_name*\\AppData\\Roaming\\DopplerView\\*release_version*\\default_DV_params.json

It controls:

* Preprocessing parameters
* Model-related parameters
* Task-specific thresholds
* Runtime options

Fingerprinting ensures that changing configuration only recomputes affected steps.

Runtime parallelism is configured separately from scientific parameters:

```json
"Execution": {
  "NumberOfWorkers": 0.5,
  "DagConcurrency": 1
}
```

`NumberOfWorkers` accepts a fixed count, `-1` for all available CPUs, `-2` for
all but one, or a fraction such as `0.5`. All internally parallel steps share
one bounded executor, so their combined Python worker count cannot exceed this
resolved capacity. Execution settings do not invalidate scientific caches.
Native libraries and inference runtimes select a machine-appropriate thread
count automatically. Advanced diagnostics can force a fixed value with
`NativeThreadsPerTaskOverride`; the sequential reference profile always uses
one native thread.

See `WORKFLOW.md` for details on how configuration impacts execution.

---

# Output

The first execution of DopplerView creates a folder named `measure_id_DV` in the parent directory of the input `measure_id_HD`folder, with following structure :
```
measure.holo
measure/
├── measure_HD/
└── measure_DV/
   ├── h5/
   │   └── measure_id_DV.h5      # The .h5 output
   ├── output/                   # Output folders used for debuging
   │   ├── output_0
   │   └── ...
   └── json/
       └── DV_params.json        # The pipeline configuration
```

Each pipeline run overwites the results in the .h5 file. The content of the .h5 file is decided by the [h5_schema.json](config/h5_schema.json).
It also creates an `output` folder, with the content produced by each step, depending on the [output_config.json](config/output_config.json).

---

# Documentation

* Architecture and execution model → `WORKFLOW.md`
* How to add steps or models → `CONTRIBUTING.md`

---

# License

GPL-3
