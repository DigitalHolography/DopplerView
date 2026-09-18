# Reproducing Noise2Time

**Current dataset workflow:** see [DATASET_WORKFLOW.md](DATASET_WORKFLOW.md).
It reads raw `doppler_signal/M0_ff` directly, uses manual arterial peak detection
by default, supports an explicit `--avi` compression experiment, and organizes
outputs as `prepared/`, `runs/`, and `evaluation/`. The file-based commands below
remain available for earlier experiments.

`noise2time.py` implements the **proposed method** in `backup_260708.pdf`, with
separate preprocessing, training, inference, and evaluation commands. It does
not launch experiments when imported, and does not modify the historical scripts.

The paper and historical scripts disagree about the loss, training cohort,
brightness definition, and validation split. This implementation makes those
choices explicit. It reproduces a specified method; reproducing the published
numerical results still requires the original recordings, masks, experiment
configuration, and model provenance. Baseline training and publication figures
are not automatically reproduced by this script.

## Implemented methods

| Setting | Article preset | L2 preset |
|---|---|---|
| Configuration | `noise2time_article.json` | `noise2time_l2.json` |
| Reconstruction | Masked L1 | Masked squared error |
| Spatial constraints | Eq. 19: 0.10 gradient + 0.05 direct Hessian | None |
| Samples per epoch | 8,000 | 3,500, matching the supplied single-video L2 script |
| Intended input | Multiple recordings (paper says more than ten) | One recording |
| History | Nine frames | Nine frames |
| Patches | 32 non-overlapping patches, 32 x 32 | Same |
| Optimizer | AdamW, learning rate 5e-5, batch size 2 | Same |
| Stopping | 250 epochs maximum; patience 10 | Same |

The spatial loss uses forward first differences, central Dxx/Dyy differences,
and the forward four-pixel mixed difference from the article. The mixed Hessian
component receives weight two in both numerator and denominator. Every required
pixel must lie within the replacement mask. Each loss is normalized by its own
valid support; sample losses are summed for backpropagation. Logged losses are
means per sample, so training and validation logs have the same scale.

The architecture follows the supplied U-Net: six spatial levels, channel widths
32/64/128/256/512/512, GroupNorm, SiLU, a 512-channel ConvLSTM at the deepest level,
skip connections, and a residual output. For 512 x 512 images the deepest level
is 16 x 16. The historical decoder passes are omitted because their outputs are
unused; gradients still propagate through the historical encoder and ConvLSTM.
The state resets for every window. New checkpoint parameter names differ from
the old scripts: legacy `.pth` files cannot be loaded directly.

## Installation

Use a Python environment containing NumPy, SciPy, OpenCV, and PyTorch. These are
already project dependencies. CUDA training needs a CUDA-enabled PyTorch build.
Full-size training is expensive; the tests use a smaller CPU model solely for
verification. A CPU smoke test does not establish full-size GPU memory usage.

Run the commands below from the repository root. Replace example paths and FPS
with your actual data. No patient dataset is bundled.

## 1. Prepare each original recording

### Prepare an entire collected folder

Pass a folder to the same `prepare` command:

```powershell
python scripts/N2N/noise2time.py prepare --input "D:/N2T/videos" --output "D:/N2T/prepared" --fps 30
```

This prepares every AVI and NPY file directly inside the input folder, one at a
time to limit memory use. Collection JSON reports are ignored. Each input gets
its own recording folder: `measure_HD_M0.avi` becomes
`D:/N2T/prepared/measure_HD_M0/`, containing the arrays and metadata used by training.
Subdirectories are not searched.

All preprocessing options below apply to every selected recording. `--fps 30`
is only an example; supply the actual acquisition rate. If all files are AVI,
omit `--fps` to use each video's container rate. NPY files require explicit FPS.
An optional `--brightness-mask` is shared across the batch, so only use it if the
same mask is appropriate for every recording.

- `--pattern "*.avi"` or `--pattern "*.npy"` selects one format. This is required
  if AVI and NPY files have identical stems; the script never silently picks one.
- `--skip-existing` leaves existing recording folders untouched, without
  verifying their content or settings. Without it, existing folders are reported
  as errors; they are never overwritten.
- A bad recording is reported and the remaining recordings continue.
- Each recording is published only after all its preparation files are written.
- A unique `preparation_*.json` summarizes successes, skips, and errors. The exit
  code is 1 if any recording failed, otherwise 0.

H5-extracted moments still need an explicit calibrated scaling into [0,1] if
their raw floating-point values are outside this range. Batch preparation does
not silently normalize those recordings.

### Prepare a single recording

```powershell
python scripts/N2N/noise2time.py prepare --input "D:/LDH/video01.avi" --output "D:/N2T/prepared/video01" --fps 30
python scripts/N2N/noise2time.py prepare --input "D:/LDH/video02.avi" --output "D:/N2T/prepared/video02" --fps 30
```

**30 is an example, not an assumed acquisition rate.** Use the physical frame
rate of the reconstructed LDH sequence; video playback FPS may differ. If omitted
for a video, its container FPS is used. NPY inputs require `--fps`.

Inputs can be grayscale videos or arrays with shape `(T,H,W)`. Arrays must be
uint8 (divided by 255) or finite floating-point values in [0,1]. Higher-bit-depth
data must be converted using an explicit calibrated scale before this step.
There is no independent per-video contrast normalization.

Preparation:

1. Masks the circular field of view (defaults cx=255, cy=255, radius=260).
2. Calculates whole-image mean brightness after masking, following Eq. 2.
3. Smooths brightness with a five-frame moving mean and edge padding.
4. Detects peaks (minimum distance 15 frames; prominence 0.10 of the 95th-5th
   percentile range). These settings come from the historical preprocessing.
5. Trims to the first peak and labels each frame by its offset since its peak.
6. Writes `frames.npy`, `roi.npy`, `brightness.npy`, `brightness_raw.npy`,
   `phase.npy`, and metadata.

Both single-file and folder preparation also save these visualization files
automatically in each recording folder:

- `prepared.avi`: the circularly masked and trimmed video, at the acquisition
  FPS. MJPG preview, fixed [0,1] to uint8 conversion; no contrast normalization.
- `brightness.png`: raw and smoothed brightness curves with detected peaks,
  plus a phase plot. Time is measured from the prepared video's start.
- `brightness.csv`: zero-based prepared/original frame indices, times in seconds
  for both timelines, raw/smoothed brightness, phase, and a detected-peak flag.

All rows, curves, and preview frames refer to the same trimmed recording. NPY
files remain the scientific inputs; AVI compression is only for visualization.
Preparation requires even image dimensions to avoid silent AVI codec cropping.
Plots are generated headlessly. Existing folders skipped with `--skip-existing`
are not backfilled; prepare into a new output folder to generate these artifacts.

At least two detected peaks are required. Peak detection is not ECG-validated;
inspect the saved peak indices before a long run. The final incomplete cycle is
retained, as in the existing preprocessing. Phase offsets are **not normalized
by cycle duration**; this intentionally retains the historical pairing rule.

Use `--brightness-mask vessel.npy` to use a supplied vessel mask for mean
brightness, matching the historical scripts instead of Eq. 2. It must match the
input geometry. This script does not generate vessel masks or perform CLAHE.
Use original inputs: passing an already trimmed video will trim it again.

For other image sizes, supply `--cx`, `--cy`, and `--radius`. Model inputs must
have height and width divisible by 32; the paper uses 512 x 512.

Prepared arrays retain scientific intensities without introducing another lossy
video encoding. Metadata retain the original frame offset, physical FPS, source
hash, detected peaks, geometry, and brightness definition. Every output directory
must be new to prevent accidental replacement of a previous experiment.

## 2. Train

```powershell
python scripts/N2N/noise2time.py train --records "D:/N2T/prepared/video01" "D:/N2T/prepared/video02" --config scripts/N2N/noise2time_article.json --output "D:/N2T/runs/article01" --device cuda
```

List all intended training recordings; the two above only illustrate the syntax.
The paper describes more than ten. Use `--device auto` (the default) to select
CUDA when available, otherwise CPU. All recordings in a run must have the same
spatial dimensions and unique folder names.

For the single-video L2 experiment:

```powershell
python scripts/N2N/noise2time.py train --records "D:/N2T/prepared/video01" --config scripts/N2N/noise2time_l2.json --output "D:/N2T/runs/l2_video01" --device cuda
```

The target recording may participate in training for the article's intended
target-adapted protocol. It must not subsequently be described as an unseen test
recording.

### Validation choices

The default fixes a random validation subset before training, as in the source
scripts. The article says the split changes each epoch while replacement choices
stay fixed; those statements do not specify a consistent protocol. We use fixed
targets and fixed replacement choices for comparable epoch losses.

**Default validation is not independent:** sequences overlap and donors may
come from any eligible frame of the same recording. This limitation is recorded
in `split.json` and printed at startup. Thirty-five validation samples and AdamW
weight decay 0.01 are explicit choices from the supplied L2 implementation, not
fully specified article parameters.

For recording-disjoint validation, copy the configuration and set:

```json
{
  "validation_records": ["video02"],
  "validation_samples": 35
}
```

A configuration may contain only overridden fields. Include both recording
folders in `--records`; `video02` then supplies no training targets or donors.
Held-out recording validation is an additional protocol, not the original
target-adapted experiment. Patient-level separation must be enforced by your
choice of folders; the script does not infer patient identity.

### Saved results

The console reports the selected device, loading and hashing of recordings, and
model initialization. Each epoch shows separate training and validation progress
bars with completed/total batches, percentage, elapsed time, estimated time left
for that stage, running mean loss, and processed sample count. Timing estimates
appear after batches complete and may fluctuate during startup. Checkpoint saves,
epoch duration, and early-stopping patience are also reported. Progress refreshes
are limited to about once per second; the first batch may take longer.

An already running process does not pick up this change. It continues to print
the old epoch-level summary. These indicators appear on the next launch; there
is no need to interrupt an existing training run just to install the update.

- `best.pt`: best validation checkpoint, including configuration and optimizer.
- `last.pt`: latest checkpoint.
- `config.json`: all effective hyperparameters.
- `split.json`: exact target indices and recording paths.
- `provenance.json`: input hashes, source-script hash, runtime versions.
- `metrics.jsonl`: per-epoch reconstruction, gradient, Hessian, and total losses.
- `previews/epoch_001/<record>.avi` (and subsequent epoch folders): denoised AVI
  for every recording supplied to training, including validation recordings.
  Each preview has a JSON sidecar identifying the epoch and source recording.

AVI previews are produced after checkpoint saving at **every epoch**, including
the final epoch that triggers early stopping. They use that epoch's weights,
not necessarily the best checkpoint. This adds a full inference pass per
recording per epoch and uses additional disk space. Pass `--no-epoch-previews`
to disable previews. Preview export has its own frame progress bar and does not
update model weights. Only AVIs and sidecars are retained for epoch previews;
full float arrays are saved by the standalone `denoise` command.

Previews preserve frame count and acquisition FPS. The first nine frames are
copied, matching inference. They use MJPG, clipping to [0,1] and rounding to
8-bit grayscale (stored as BGR for codec compatibility). There is no per-frame
contrast normalization, so the display scale is consistent across epochs.
MJPG previews are for viewing; compute quantitative metrics on the NPY output.
An already running training process will not acquire these changes automatically.

To limit disk use, only best and latest weights are retained. There is currently
no resume command; use a new output directory for a fresh run. Seeds and
deterministic operations are configured, but identical results across different
PyTorch/CUDA versions and hardware are not promised. Unsupported deterministic
operations raise an error instead of silently switching behavior.

Empty supervision, unavailable donors, nonfinite losses, and unsuccessful patch
placement raise errors. Samples lacking another same-phase donor are excluded
before splitting. This avoids treating an empty validation pass as zero error.

## 3. Denoise from a saved model

Pass a parent output folder to create a complete measurement bundle:

```powershell
python scripts/N2N/noise2time.py denoise --checkpoint "D:/N2T/runs/article01/best.pt" --record "D:/N2T/prepared/measure_HD_M0" --output "D:/N2T/results" --device cuda
```

Output:

```text
D:/N2T/results/measure/
    original.avi
    denoised.avi
    denoised.npy
    denoised.json
```

The subfolder uses the prepared recording folder's name, removing a trailing
`_HD_M0` when present. An existing measurement subfolder is not overwritten.
All four files are written before the subfolder is published.

`original.avi` shows the **prepared input before denoising**, including the
circular mask and trimming performed during preparation. It is not a byte copy
of the original acquisition file: this makes its frame count, timing, geometry,
and fixed display scale match `denoised.avi` for direct comparison. Both videos
use the stored acquisition FPS. The JSON records the original acquisition frame
offset and the meaning of the original preview. The NPY remains unclipped float32.

The previous explicit `.npy` and `.avi` filename output modes also remain available:

```powershell
python scripts/N2N/noise2time.py denoise --checkpoint "D:/N2T/runs/article01/best.pt" --record "D:/N2T/prepared/video01" --output "D:/N2T/video01_denoised.npy" --device cuda
```

To save both the float array and its AVI conversion in one inference pass:

```powershell
python scripts/N2N/noise2time.py denoise --checkpoint "D:/N2T/runs/article01/best.pt" --record "D:/N2T/prepared/video01" --output "D:/N2T/measure.avi" --device cuda
```

This writes `measure.npy`, `measure.avi`, and `measure.json`. An `.npy` output
continues to produce only the array and JSON. Existing output files are never
overwritten. The AVI uses the same fixed-scale conversion as the epoch previews;
the NPY retains unclipped float32 values. Temporary inference files are cleaned
up on failure, and output publication happens after the inference pass completes.

Inference follows the legacy code: nine preceding frames plus the fully visible
current frame. No donor replacement is used during inference. The first nine
frames are copied, with this fact recorded in the output JSON sidecar. This
train/inference input difference remains an experimental assumption.

The output is float32, without clipping, quantization, or compression. Values
outside [0,1] are retained so saturation errors can be measured. Use a separately
clipped copy for display. The JSON sidecar records alignment, checkpoint hash,
source-array hash, FPS, and the copied prefix.

## 4. Recreate the quantitative metrics

Provide independently selected binary vessel/background NPY or PNG masks, each `(H,W)`.
They must be nonempty, disjoint, and inside the circular field of view.

Handmade PNG masks are supported directly by `--vessel-mask` and
`--background-mask`. Black pixels are excluded; any nonzero grayscale value or
color channel selects the pixel. Fully transparent pixels are ignored, including
hidden color values. Thus use white (or colored) vessels on black, not black
vessels on white, and supply a mask rather than annotations over the original
image. Nonzero antialiased edge pixels are included. Image dimensions and pixel
alignment must match the prepared video. Different dimensions trigger nearest-neighbor
resizing with a warning; this does not correct translation, cropping, rotation or
registration errors. Masks are never inverted. NPY and PNG can be mixed.

```powershell
python scripts/N2N/noise2time.py evaluate --record "D:/N2T/prepared/measure_HD_M0" --denoised "D:/N2T/results/measure/denoised.npy" --vessel-mask "D:/N2T/masks/vessels.png" --background-mask "D:/N2T/masks/background.npy" --output "D:/N2T/measure_metrics.json"
```

```powershell
python scripts/N2N/noise2time.py evaluate --record "D:/N2T/prepared/video01" --denoised "D:/N2T/video01_denoised.npy" --vessel-mask "D:/N2T/masks/video01_vessel.npy" --background-mask "D:/N2T/masks/video01_background.npy" --output "D:/N2T/video01_metrics.json"
```

This calculates the paper's average pixel-wise background temporal standard
deviation, NRR, vessel-mean temporal correlation, and cardiac amplitude ratio.
It checks source correspondence and excludes the copied prefix for both inputs.

The paper does not fully specify cardiac amplitude estimation. Here the largest
original vessel-curve Fourier component between `--min-hz` (0.5 default) and
`--max-hz` (3.0 default) selects a frequency. A sinusoid plus constant is fitted by
least squares to **both curves at that same frequency**. This convention is
explicit, but may differ from the one used for Table I. Inspect the chosen
frequency; a harmonic can dominate the fundamental. Undefined metrics are saved
as null, never as artificial perfect scores. Pixel temporal SD uses ddof=0.

Aggregate separately evaluated recordings into Table I-style means and sample SD:

```powershell
python scripts/N2N/noise2time.py summarize --metrics "D:/N2T/video01_metrics.json" "D:/N2T/video02_metrics.json" --output "D:/N2T/cohort_metrics.json"
```

Each recording has equal weight. Summary SD uses ddof=1, matching the article's
printed table. Missing-value counts are reported separately for each metric.
Include one recording per patient if claiming patient-wise statistics; duplicate
patient identity across differently named records is not detected automatically.

These metrics assess fluctuations and broad waveform preservation. They do not
establish recovery of the unknown clean signal. For scientific validation add
known-signal simulations, phase-resolved residuals, and small-vessel measurements.

## Arterial peak detection

Preparation now accepts `--artery-mask path/to/manual_artery.png`. This selects
the robust arterial detector for peak timing while preserving the existing
brightness-normalization choice. The detector, experiments on all 22 recordings
in `D:/dataset_choroid2`, comparisons with DopplerView, and limitations are
documented in [ARTERIAL_PEAKS.md](ARTERIAL_PEAKS.md).

## 5. Regional evaluation report

Provide separate retinal artery, retinal vein and choroidal masks. Background is derived automatically:

```powershell
python scripts/N2N/noise2time.py evaluate `
  --record "D:/N2T/prepared/measure_HD_M0" `
  --denoised "D:/N2T/results/measure/denoised.npy" `
  --retinal-artery-mask "D:/N2T/masks/arteries.png" `
  --retinal-vein-mask "D:/N2T/masks/veins.png" `
  --choroidal-masks "D:/N2T/masks/choroidal.png" `
  --background-dilation-radius 2 `
  --output "D:/N2T/results/measure/evaluation"
```

All three vessel mask options accept multiple paths, combined by union within each
group. Do not provide a background mask in regional mode. The old
`--vessel-mask` evaluation remains available and produces a single JSON file.
Regional mode requires a new output directory and publishes it only on success.

### Mask exclusions

Let A, V and C be the input artery, vein and choroidal unions after clipping to
the prepared ROI. Evaluation uses `A & ~(V | C)`, `V & ~(A | C)`, and
`C & ~(A | V)`. These exclusions are simultaneous: an overlapping pixel belongs
to **none** of the vessel groups. Background is computed as
`ROI & ~(dilate(original_A | original_V) | original_C)` using the original masks
before overlap removal. Only retinal masks are dilated; choroidal pixels are
excluded without dilation. The dilation uses a Euclidean disk with radius
`--background-dilation-radius` in prepared-image pixels (default 2, 0 disables
dilation). Dilation occurs before ROI clipping, so nearby retinal pixels outside
the ROI still exclude their surrounding pixels inside it. The radius and method
are saved in the report. Empty resulting masks stop evaluation.

Removing overlapping mask pixels cannot unmix depth contributions already present
in a pixel's measured signal. Inspect alignment and the excluded-pixel counts.
The input masks are preserved; cleaned masks are saved separately as PNG and NPY.
Source paths, hashes and resizing warnings are recorded in `metrics.json`.

### Outputs

Open `report.html` in the output folder. Keep the folder together when sharing it.
It includes:

- Per-group temporal correlation, cardiac amplitude ratio, mean intensity change,
  estimated delay, harmonics and vessel-background contrast (full values in JSON).
- Shared background temporal SD and NRR. NRR is not an independent score for each
  vascular group; it measures the same selected background in all three rows.
- Mask overlays, exclusions and up to three local patches per group.
- Original/denoised/residual waveforms, a three-cycle zoom and frequency spectra.
- Mean images, signed mean residuals, temporal SD and SD reduction maps.
- Four phase-resolved residual maps using fractional preprocessing peak-to-peak
  intervals. Frames outside complete peak pairs are excluded from these maps.
- Vessel profiles and time-distance images at identical coordinates.
- `comparison.avi`: synchronized original, denoised and signed residual panels,
  using acquisition FPS and excluding the copied prefix.
- `metrics.json`, `metrics.csv`, waveform CSVs, profile NPZs and `spatial_maps.npz`.

All metrics use unquantized NPY values. Original/denoised display scales are fixed
to [0,1]; residual scales are symmetric and fixed per plot/video. Display clipping
does not affect metrics. Undefined ratios/correlations are JSON null.

By default, one common frequency is selected from the original pooled vessel
waveform in `--min-hz` / `--max-hz`. A dominant harmonic can be mistaken for the
fundamental: inspect the spectra or specify a verified `--cardiac-hz 0.82`.
Amplitude is fitted at the same frequency before and after denoising; the reported
frequency resolution is FPS divided by scored frame count.

`--local-size 32 --local-count 3` controls automatic local patches. They favor
high mask occupancy, so they do not establish performance on faint small vessels.
`--max-lag-seconds 0.25` bounds the integer-frame delay search. Positive delay means
denoised output is late. Lag-corrected correlation is supplemental; inspect the
zero-lag correlation too.

Automatic profiles are approximately perpendicular to the largest local mask
patch. For meaningful individual-vessel sections, supply `--profiles profiles.json`:

```json
{
  "retinal_artery": {"start": [100, 120], "end": [130, 120]},
  "retinal_vein": {"start": [200, 220], "end": [230, 220]},
  "choroidal": {"start": [300, 320], "end": [330, 320]}
}
```

Coordinates are `[x,y]` in the prepared image; omitted groups use automatic lines.
Profiles include neighboring pixels along the line, unlike mask-restricted metrics.
The matched snapshot is selected from the original region waveform's maximum.
Temporal means are structural references, not clean ground truth. This report
does not supply uncertainty estimates or demonstrate held-out generalization.
The existing `summarize` command currently accepts legacy single-region JSON only.

## Ablations

Copy a configuration and change one factor at a time:

- `"objective": "l1"`: reconstruction only.
- `"objective": "l2"`: squared reconstruction only.
- `"convlstm": false`: removes temporal memory; only the current frame is used.
- `"blocks": 1`: one replacement patch.
- `"block_size": 1, "objective": "l2"`: one-pixel replacements as in the article's
  L2 ablation; mask size is derived directly from replacement support.

Changing patch number/size changes the amount of supervision. Do not interpret
this comparison as isolating spatial context unless that factor is controlled.

## Verification

```powershell
python -m pytest tests/test_noise2time.py tests/test_noise2time_report.py -q
```

Tests check analytic derivative values and normalization, no loss dependence on
unmasked pixels, exact patch support including 1 x 1, donor exclusion, brightness
scaling/clipping, disjoint recording splits, invalid supervision, and an actual
small CPU prepare/train/checkpoint/inference/metric-aggregation run. The identity
metric control requires NRR=0, correlation=1, and amplitude ratio=1.
