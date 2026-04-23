# DopplerView

DopplerView is a deep-learning tool for image enhancement, vascular segmentation, and topology inference that processes HoloDoppler results into vascular maps and analysis-ready inputs for EyeFlow, using a DAG-based pipeline and a configurable model registry.

The project provides:

* A deterministic DAG-based processing pipeline
* Automatic dependency resolution and selective recomputation
* A model registry with HuggingFace integration
* CLI execution
* A Streamlit-based GUI, for dynamic debugging
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

   * Computation of diastolic/systolic frames and temporal correlation map using the arterial signal obtained with the pre-classified arteries, following the strategy described in [Dubosc, Marius, et al. "Improving segmentation of retinal arteries and veins using cardiac signal in doppler holograms." arXiv preprint arXiv:2511.14654 (2025).](https://arxiv.org/abs/2511.14654)

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
source .venv/Script/activate
```

Install in editable mode:

```bash
pip install -e .
pip install -r requirements.txt
```

---

# Usage

DopplerView runs using a [HoloDoppler](https://github.com/DigitalHolography/HoloDopplerPython/tree/main) acquisition folder, with the following structure :

```
measure_id.holo
measure_id/
└── measure_id_HD/
    ├── raw/
    │   └── measure_id_HD_output.h5    # The .h5 file used as input
    ├── json/
    │   └── parameters.json            # The rendering parameters
    ├── mp4/
    └── ...
```

## CLI

The CLI runs the full pipeline on a Holodoppler acquisition folder.

```bash
dopplerview /path/to/holodoppler_folder --config config.json -v
```

### Arguments

* `-h, --help`            show this help message and exit
* `-v, --verbose`         Enable verbose output
*  `-c CONFIG, --config CONFIG`
                        Path to JSON configuration file
*  `-b, --batch`           Process multiple folders. Folders are either listed in a text file (one     
                        folder path per line) or provided as subfolders of the specified path.      
*  `-t TARGETS [TARGETS ...], --targets TARGETS [TARGETS ...]`
                        List of target steps to run
*  `-d, --debug`           Enable debug mode. In this mode, steps outputs are read from the .h5, and   
                        only targeted steps are re-run. This is useful for debugging specific       
                        steps without having to re-run the entire pipeline.

### Example

```bash
dopplerview ./data/patient_01 \
    --config ./configs/default.json \
    -v
```

## GUI (Streamlit)

Launch the graphical interface:

```bash
streamlit run dopplerview/app.py
```

The GUI allows you to:

* Load a Holodoppler folder
* Automatically run initial preprocessing
* Visualize M0 full-field image
* Overlay vessel and AV segmentation masks
* Run the full pipeline interactively

# Project Structure

```
DopplerView/
│
├── dopplerview/
│   ├── pipeline/          # DAG engine, steps, context
│   ├── models/            # Registry, manager, wrappers
│   ├── input_output/      # Folder reading & output handling
│   ├── utils/
│   │   └── ...
│   ├── app.py             # Streamlit GUI
│   ├── cli.py/            # Command-line script
│   └── ...
│
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

The first execution of DopplerView creates a folder named `measure_id_DV` in the parent directory of the input `measure_id_HD`folder, with following structure :
```
measure.holo
measure/
├── measure_HD/
└── measure_DV/
   ├── output/                            # Output folders used for debuging
    │   ├── output_0
    │   └── ...
    ├── json/
    │   └── doppler_view_params.json      # The pipeline configuration
    ├── h5/
    │   └── measure_id_DV.h5              # The .h5 output
    └── cache
        └── cache.h5                      # The cache used for debugging
```

Each pipeline run overwites the results in the .h5 file. The content of the .h5 file is decided by the [h5_schema.json](config/h5_schema.json).
It also creates an `output` folder, with the content produced by each step, depending on the [output_config.json](config/output_config.json).

---

# Documentation

* Architecture and execution model → `WORKFLOW.md`
* How to add steps or models → `CONTRIBUTING.md`

---

# License

MIT
