# Dataset workflow: raw M0_ff by default

The dataset is organized as:

```text
dataset/
  measurement/
    measurement_DV.h5
    manual/
      retina_artery_mask.png
      retina_vein_mask.png
    pseudo/
      choroidal_vessel_segmentation_choroidal_vessel_mask.png
```

`retinal_artery_mask.png` / `retinal_vein_mask.png` are accepted spellings too.
Each measurement must have exactly one `.h5`, with `doppler_signal/M0_ff` in
`(time,height,width)` order. Masks must match the image geometry. No resizing,
transposition guessing, enhancement or registration is applied.

**Only the pseudo choroidal vessel mask is used for evaluation**, even if manual
choroidal annotations exist. The canonical unsuffixed `*choroidal_vessel_mask.png`
is required; versioned `_2` copies and `*_raw` masks are not substituted. A missing
or ambiguous pseudo mask is reported as an error, never replaced with a manual one.

## Commands

Run from the repository root. Use the same output root for all stages:

```powershell
python scripts/N2N/noise2time.py prepare --input "D:/dataset_choroid2" --output "D:/N2T/raw"

python scripts/N2N/noise2time.py train --input "D:/dataset_choroid2" --output "D:/N2T/raw" --config scripts/N2N/noise2time_l2.json --device cuda

python scripts/N2N/noise2time.py evaluate --input "D:/dataset_choroid2" --output "D:/N2T/raw" --device cuda
```

Evaluation uses `runs/best.pt`, runs inference for missing denoised measurements,
then creates the regional reports. An existing denoised result must match the
selected checkpoint and prepared frames. Use `--checkpoint` to select another
checkpoint in a new output workflow. Existing reports are not overwritten.

Inference can also be run separately:

```powershell
python scripts/N2N/noise2time.py denoise --input "D:/dataset_choroid2" --output "D:/N2T/raw" --device cuda
```

All stages accept `--measures 251031_ALA_L_1 260626_COY_choroid_6` to select a
subset. Preparation can resume with `--skip-existing`; source hashes, detector,
FPS, input mode, peak-frequency bounds and ROI settings must match. Preparation
and evaluation record per-measure failures in root-level summary JSON files and
return a nonzero exit status if any measurement fails.

The default effective frame rate is **37000 / 256 = 144.53125 Hz**, as confirmed
for this dataset. Supply `--fps` during preparation for another acquisition rate.
The code does not infer acquisition timing from AVI playback metadata.

## What prepare does

1. Read raw M0_ff in bounded blocks and find its maximum inside the circular diaphragm.
2. Cache float32 frames using **one fixed division by that maximum**. This is
   numerical scaling for the existing neural network, not contrast enhancement.
   Pixels outside the diaphragm are set to zero. No percentile clipping,
   per-frame normalization, temporal filtering of the video, or first-peak trimming is performed.
3. Save `prepared.avi` with a fixed display mapping. In the default raw mode this
   file is visualization only; training and analysis use the uncompressed cache.
4. Extract the mean signal inside the handmade retinal artery mask intersected with the diaphragm. Detect
   peaks with the robust method from [ARTERIAL_PEAKS.md](ARTERIAL_PEAKS.md).
5. Compute brightness tables and save the arterial signal/peak/cycle plot.

The frame cache is a float32 numerical representation of the raw values, subject
to normal floating-point rounding, with no 8-bit quantization in raw mode.
`intensity_scale` records the multiplier that converts cache/denoised values back
to original M0 units inside the diaphragm. Exterior values are intentionally discarded.
The source HDF5 is read-only and remains authoritative.
CSV signal intensities are already expressed in original M0 units. Stored
brightness NPY values use the same scale as the training frames.

Brightness normalization uses a 20 ms-sigma Gaussian smoothing of the **measured
arterial mean**, without interpolating artifacts. Peak detection separately uses
its robust, repaired working trace. Both curves are shown in the plot.

All original frames remain in the cache and preview. Phase is elapsed frames
since a peak, with `-1` outside complete peak-to-peak intervals. Targets/donors
outside complete cycles or flagged as artifacts are excluded from sampling.
History windows may still contain those frames; this is not video repair.
At least three detected peaks are needed to form two complete pairing cycles.
Quality warnings are saved in metadata and require scientific inspection.

The circular ROI masks prepared videos, brightness extraction, and denoised
outputs, and restricts training patch placement and evaluation.
Defaults remain center (255,255), radius 260; smaller images
need appropriate `--cx`, `--cy`, and `--radius`. Spatial dimensions must be
divisible by 32 for the existing network.

Preparations created before diaphragm masking was applied must be regenerated
in a new output root, followed by training and evaluation. They are rejected by
`--skip-existing` and the dataset training/denoising/evaluation commands.

## Resume training and track progress

Use the same dataset, measurement selection and output folder as the interrupted run:

```powershell
python scripts/N2N/noise2time.py train --input "D:/dataset_choroid2" --output "D:/N2T/raw" --resume --device cuda
```

The command loads `runs/last.pt`, restores the model and AdamW optimizer, and starts
at the next epoch. An interrupted partial epoch is repeated. `--epochs 100` means
100 epochs in total, including completed epochs; omit it to retain the saved target.
The configuration is loaded from the checkpoint when `--config` is omitted. Only
the epoch target may change on resume. Prepared-data hashes and the split must match.
Legacy `--records` mode also supports resume, with `last.pt` directly in `--output`.

New checkpoints also save the sampling/Torch/Python random-generator states and
early-stopping counter. Older checkpoints can restore weights and optimizer but
cannot reproduce the exact random sampling continuation; a warning explains this.
Checkpoints are replaced atomically. If no completed epoch was saved, there is
no checkpoint to resume. A run that already reached early stopping stays stopped.

Each epoch updates `runs/metrics.jsonl`, `runs/metrics.csv`, and `runs/metrics.png`:

- Training and validation losses, including reconstruction/gradient/Hessian components in the tables.
- Mean per-pixel **temporal** background standard deviation, for original and denoised data, and NRR in the tables.
- Artery, vein and choroid waveform correlation, temporal standard-deviation ratio,
  and original/denoised means in the tables.

Diagnostics use full, unclipped inference sequences, excluding the copied history
prefix. Background is derived exactly as in evaluation, inside the diaphragm;
the black exterior is never counted as background. The manual retinal and pseudo
choroidal masks are discovered automatically and their hashes saved in
`monitor_masks.json`. `--background-dilation-radius` defaults to 2 pixels.
Masks and monitoring settings must remain unchanged on resume. These diagnostics
describe supplied recordings, including training recordings; they are not independent
test scores and do not select `best.pt`, which still uses validation loss.

Lower background temporal variability is useful only together with preserved vessel
waveforms. Zero variability also occurs for a constant output. Look for stable vessel
means, correlation near 1 and waveform standard-deviation ratios near 1, recognizing
that removal of vessel noise can itself lower that ratio. No clean reference is assumed.
All intensities use the prepared fixed scale, so compare records relative to their own
original baseline. NRR and correlation are null when their denominator/variance is zero.

Metrics share the inference pass with epoch previews. `--no-epoch-previews` skips
AVIs but still computes metrics; `--no-epoch-metrics` skips regional diagnostics.
Loss logging always remains active. Full-video diagnostics require all three vessel
masks; legacy records without dataset metadata only receive loss plots.
If interrupted during diagnostics, the completed epoch checkpoint is retained, but
that epoch may lack regional metrics or have incomplete previews; resume continues
with the next epoch. Earlier epochs cannot be retrospectively scored without their weights.

## AVI compression experiment

Use a separate output root and add `--avi` **only to prepare**:

```powershell
python scripts/N2N/noise2time.py prepare --input "D:/dataset_choroid2" --output "D:/N2T/avi" --avi
python scripts/N2N/noise2time.py train --input "D:/dataset_choroid2" --output "D:/N2T/avi" --config scripts/N2N/noise2time_l2.json --device cuda
python scripts/N2N/noise2time.py evaluate --input "D:/dataset_choroid2" --output "D:/N2T/avi" --device cuda
```

The sequence is mapped to uint8 by `round(255 * raw / intensity_scale)`, encoded
as MJPG AVI, decoded, and cached as float32/255. Exterior pixels are zeroed again
after decoding to remove codec ringing outside the aperture. Exported MJPG previews
may show slight boundary ringing; NPY exterior pixels are exactly zero.
**Peaks, brightness, training and
evaluation all use that decoded sequence.** This tests 8-bit quantization plus
MJPG compression together, not codec effects alone. The raw and AVI paths use
the same scaling convention and preserve frame count/alignment.

Compare separate experiments with matching model configuration and seeds.
Compression may change peak locations and therefore pairing/sampling, so a
raw-versus-AVI retraining comparison measures the combined pipeline effect.
To isolate inference input sensitivity, use the same checkpoint for both paths;
pass its path explicitly. Evaluation compares each denoised output against its
own selected input, so the regional report is not itself a raw-versus-AVI error
measurement. Do not infer better physical accuracy merely from a higher NRR.

## Evaluation masks

Masks are taken from each dataset measurement:

- Retinal artery and vein: handmade `manual/` masks.
- Choroidal vessels: the canonical `pseudo/` mask, always.
- Background: `ROI & ~(dilate(original_artery | original_vein) | original_choroidal)`.

`--background-dilation-radius 2` controls the retinal disk dilation (pixels);
choroid is not dilated. Vessel-group overlaps are simultaneously removed from
both groups. Background construction uses the original masks, including ambiguous
vessel pixels, so these cannot become background accidentally. Empty resulting
groups are errors. The report explicitly identifies its pseudo choroidal source.
This excludes overlapping pixels; it cannot unmix signals within a pixel.

## Output tree

```text
output_folder/
  prepared/
    measurement/
      prepared.avi
      brightness.png              # raw signal, both smoothings, peaks, cycles
      brightness.csv
      frames.npy                  # raw-scaled OR AVI-decoded input cache
      brightness.npy
      brightness_raw.npy
      phase.npy
      valid_frames.npy
      roi.npy
      artery_mask.npy
      arterial_peaks.json
      arterial_peak_diagnostics.npz
      metadata.json
  runs/
    best.pt
    last.pt
    config.json
    split.json
    provenance.json
    metrics.jsonl
    previews/epoch_001/measurement.avi
    denoised/measurement/
      original.avi
      denoised.avi
      denoised.npy
      denoised.json
  evaluation/
    measurement/
      masks/
      plots/
      report.html
      metrics.json
      metrics.csv
      comparison.avi
      spatial_maps.npz
      dataset_sources.json
  preparation_summary.json
  evaluation_summary.json
```

Denoising still copies the initial history-length prefix; evaluation excludes
those copied frames. No training or evaluation starts during preparation.
The older explicit file/record flags remain available for previous experiments.

## Verification

```powershell
python -m pytest tests/test_noise2time_dataset.py tests/test_noise2time.py tests/test_arterial_peaks.py tests/test_noise2time_report.py
```

Tests verify raw scaling inside the diaphragm, unchanged source data, no trimming,
zero exterior pixels, decoded AVI pixels inside the aperture, phase/donor exclusions, reuse safeguards, strict pseudo-only
choroidal selection, and a small CPU prepare/train/infer/evaluate workflow.
