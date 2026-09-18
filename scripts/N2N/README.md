# N2N research scripts

For dataset folders containing M0_ff HDF5 files and manual/pseudo masks, start with
[Dataset workflow](DATASET_WORKFLOW.md): raw preparation by default, arterial peak
detection, optional AVI compression, training and regional evaluation.

To collect selected recordings from mapped drives, see
[Collect LDH videos](COLLECTING.md) and `collect_videos.py` (AVI copy or lossless
`moment0ff` extraction from H5).

For a configurable implementation of the article with lossless outputs and tests,
see [Reproducing Noise2Time](REPRODUCING.md) and `noise2time.py`. The historical
experiments below remain available for provenance and comparison.

This directory contains experimental preprocessing, self-supervised video-denoising, and comparison scripts for laser Doppler holography (LDH) videos. They are standalone research programs, not commands exposed by the `dopplerview` package.

Most scripts have no command-line interface. Configure the absolute paths and experiment constants near the top (and, for some trainers, model paths near the bottom), then run the file from the repository root.

## Setup

Create or activate the project environment and install the project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

The scripts use NumPy, OpenCV, PyTorch, SciPy, Matplotlib, and tqdm. The ConvLSTM experiment scripts also import `winsound`, so they are Windows-specific unless that notification code is removed. CUDA is selected automatically when available; otherwise the trainers use the CPU and will be much slower.

Before running a script, replace the checked-in `C:\Users\Novovorontsovka\...` paths with paths on your machine. These programs execute their workload immediately and can train for many hours. Run them as:

```powershell
python scripts\N2N\<script-name>.py
```

## Recommended workflow

1. Optionally use `renommer.py` to normalize source video names. Make a backup first.
2. Run `pretraitement_complet.py` to build masked and enhanced videos, vessel masks, brightness tables, and phase-aligned trimmed videos.
3. Choose one of the training scripts below and configure its input/output paths and hyperparameters.
4. Use `selecte_block.py` after producing denoised videos from several methods to generate spatial-profile comparison figures.

The baseline scripts (`udvd.py`, `blind2Unblind.py`, and `neighbor2neighbor.py`) can instead consume a folder of videos directly and do not depend on the preprocessing pipeline's brightness tables.

## Script reference

### `pretraitement_complet.py`

Runs the complete LDH preprocessing pipeline over every AVI in `INPUT_DIR`:

- applies a circular region-of-interest mask;
- creates a vessel-enhanced video;
- derives a `.npy` vessel mask;
- measures and smooths frame brightness, detects peaks, and builds phase information;
- trims each video from its first detected peak.

Set `INPUT_DIR` and `OUTPUT_DIR`. The output tree contains `01_masked_videos`, `02_enhanced_videos`, `03_vessel_masks`, `04_brightness_tables`, and `05_trimmed_videos`. The geometry and detection thresholds (`CX`, `CY`, `RADIUS`, percentile thresholds, peak distance, and related values) assume the current LDH acquisition format and may need adjustment for other data.

```powershell
python scripts\N2N\pretraitement_complet.py
```

### `n2n.py`

The main multi-video ConvLSTM experiment with vessel-aware loss. It trains a U-Net/ConvLSTM from 10-frame sequences: nine history frames establish temporal state and the final frame is block-masked and reconstructed. The loss is evaluated at masked locations and uses vessel masks to emphasize vessel pixels, vessel gradients, and fair background reconstruction.

Configure `VIDEO_DIR`, `BRIGHTNESS_DIR`, `VESSEL_MASK_DIR`, `MODEL_SAVE_PATH`, and `EPOCH_MODEL_DIR`. Inputs must currently be 512×512 AVI files with matching preprocessing artifacts. Important defaults include 32-pixel blocks, batch size 2, 8,000 training samples per epoch, and 250 epochs.

### `n2N_l1.py`

A multi-video ablation of the ConvLSTM approach that does not use vessel masks. Despite stale comments and constant names inherited from the fuller experiment, its configured experiment is the L1-only/mask-free variant. It reserves the IDs in `TEST_VIDEO_NUMBERS` from training and writes a denoised version of those videos after each epoch.

Configure `VIDEO_DIR`, `BRIGHTNESS_DIR`, `TEST_VIDEO_NUMBERS`, `TEST_DENOISED_OUTPUT_DIR`, `MODEL_SAVE_PATH`, and `EPOCH_MODEL_DIR`. It accepts 512×512 inputs and currently limits the dataset to 20 valid videos.

### `n2n_l2.py`

A single-video ConvLSTM ablation using masked L2 reconstruction. The same video is used for training and for a full-video denoising pass after every epoch. Nine history frames precede each masked target frame.

Configure `INPUT_VIDEO_PATH`, `BRIGHTNESS_DIR`, `DENOISED_OUTPUT_DIR`, `OUTPUT_VIDEO_NUMBER`, `MODEL_SAVE_PATH`, and `EPOCH_MODEL_DIR`. The input must be 512×512 and have a matching smooth-brightness table. Defaults are 3,500 samples per epoch and 250 epochs.

### `n2n_continuity_gradient_l1.py`

A multi-video, mask-free ConvLSTM experiment whose loss combines masked L1 reconstruction with first-order gradient, Hessian/line-response, and continuity constraints. Videos listed in `TEST_VIDEO_NUMBERS` are excluded from training/validation and denoised after each epoch.

Configure `VIDEO_DIR`, `BRIGHTNESS_DIR`, test/output settings, model paths, and the `LAMBDA_VFC*` weights. The default experiment uses up to 200 valid 512×512 videos, 8,000 samples per epoch, and 250 epochs.

### `n2n_continuity_gradiant_hessain.py`

An apparent duplicate/earlier spelling variant of `n2n_continuity_gradient_l1.py` (the filename misspells “gradient” and “Hessian”). Its architecture, loss weights, default paths, and experiment settings are effectively the same. Keep it only when reproducing results tied to that exact file; for new runs, prefer the correctly named script and verify the two files before assuming they remain equivalent.

### `udvd.py`

Implements the UDVD blind-spot baseline. It samples consecutive five-frame sequences, crops 96×96 patches, and predicts the center frame with a blind-spot U-Net. Each epoch saves a checkpoint; starting at epoch 5, every fifth epoch denoises all source videos.

Edit the `Config` dataclass (`CFG`), especially `video_folder`, `output_folder`, epoch/sample counts, batch size, and optional resume settings. Its output contains `epoch_models/` plus per-epoch denoised-video directories.

### `blind2Unblind.py`

Implements the Blind2Unblind CVPR 2022 baseline for individual noisy frames. It creates global-aware masks over 4×4 cells, replaces blind pixels by neighbor interpolation, trains a U-Net with the re-visible loss, and combines masked and fully visible predictions for inference. It does not use temporal sequences, phase tables, or vessel masks.

Edit the `Config` dataclass. Defaults are 3,000 random 96×96 patches per epoch, batch size 8, and 100 epochs. It saves each epoch model and denoises all videos every five epochs beginning at epoch 5.

### `neighbor2neighbor.py`

Implements the Neighbor2Neighbor single-frame baseline. Each noisy 96×96 patch is split into paired 48×48 neighbor sub-images sampled from 2×2 neighborhoods. Training combines reconstruction and Neighbor2Neighbor regularization; inference processes full frames without changing their size. It does not use temporal or phase information.

Edit the `Config` dataclass, particularly the video/output folders and optional checkpoint resume fields. Its training and denoising schedule matches the other baseline scripts by default.

### `selecte_block.py`

Creates publication-style spatial brightness-profile comparisons for selected frames from Original, Proposed, UDVD, Sliding Average, Blind2Unblind, and Neighbor2Neighbor outputs. Each figure shows a frame with a vertical sampling line and the raw intensity profile along that line. It intentionally performs no smoothing, normalization, interpolation, enhancement, or automatic resizing.

Configure `DOWNLOADS` and the method folders, then update `VIDEO_SETTINGS`, `OUTPUT_NAMES`, and plotting constants as needed. All compared frames for one video must have identical dimensions. Frame numbers are one-based. Results are written to `OUTPUT_FOLDER`.

### `renommer.py`

Renames every supported video in `folder` to sequential names (`1.avi`, `2.mp4`, and so on), ordered by the original filename. It first uses temporary names to avoid collisions.

**Warning:** this modifies source filenames in place and provides no undo operation. Back up the directory, set `folder` carefully, and ensure no leftover `__temp_video_*` files are present before running it.

### `difference_video.py`

Currently an empty placeholder. Running it does nothing.

## Data and naming assumptions

- The ConvLSTM scripts expect 512×512 grayscale-compatible AVI videos and skip incompatible files.
- Brightness tables and vessel masks are matched to videos by filename conventions implemented in each script. Preserve the preprocessing output names unless you also update the matching functions.
- Several scripts infer a numeric video ID from either the first or last underscore-separated numeric filename component. Check this before changing names.
- Outputs are generally MJPG AVI files and PyTorch `state_dict` checkpoints (`.pth`).
- The scripts hold substantial video/model data in memory. Start with fewer videos, samples, and epochs when validating a new setup.

## Operational cautions

- There is no shared configuration file or argument parser; edits to one experiment do not configure the others.
- Output folders and checkpoint names are not consistently aligned with the active experiment (some are historical names). Review every path before starting a run to avoid mixing results.
- The source comments contain some mojibake from incorrectly decoded Chinese text. This does not normally affect execution, but it can make inherited descriptions misleading; the behavior summarized here is based on the code paths and constants.
- Use a copied dataset for experiments that rename files or write alongside source data.
