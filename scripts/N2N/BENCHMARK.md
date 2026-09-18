# One-factor-at-a-time Noise2Time benchmark

The runner executes **six experiments serially** on one device. It shares existing
prepared videos across runs, without copying the large frame caches. Every run has
its own checkpoints, optimizer, log, epoch metrics, previews and regional reports.

| Strategy | Change from baseline |
|---|---|
| `baseline` | Patched current frame + previous frames, mixed split, ConvLSTM, brightness correction |
| `history_only` | Predict frame 10 from frames 1–9 only |
| `temporal_split` | Separate temporal blocks within every development video |
| `video_validation` | Reserve one development video for validation/model selection |
| `no_convlstm` | Existing plain U-Net without recurrent memory |
| `no_brightness` | Do not rescale replacement patches using brightness ratios |

All other settings come from the same baseline config: seed, optimizer, loss,
training sample budget, epoch limit, early stopping, patch size/count, etc.
The default history is nine frames; a custom config can change it for all runs.
This is a single-seed screening experiment, not an estimate of uncertainty across seeds.

## Run overnight

Prepare the development dataset first, using the normal dataset workflow. Then:

```powershell
python scripts/N2N/benchmark_noise2time.py --prepared "D:/N2T/raw/prepared" --output "D:/N2T/benchmark" --config scripts/N2N/noise2time_l2.json --device cuda
```

Use your actual prepared folder. The baseline config must enable patched input,
ConvLSTM and brightness correction, with mixed validation. The current example
config supplies ten epochs and 3,500 training samples per epoch; six runs may take
longer than one night depending on your GPU and number of videos. Set the common
budget in the config before starting. Early stopping can make run lengths differ.

At least **two development videos** are required. By default the alphabetically
last one becomes validation-only in `video_validation`. Override it with
`--validation-video MEASURE_NAME`; it must differ from the first preview video.
All development videos participate in training in the other five experiments.
Use `--measures NAME_1 NAME_2 ...` to select a subset of the prepared folder.

Add `--dry-run` to validate all splits and write `plan.json` without training.
To start from that plan, or resume an interrupted benchmark:

```powershell
python scripts/N2N/benchmark_noise2time.py --output "D:/N2T/benchmark" --resume --device cuda
```

Completed strategies are retained. Incomplete training resumes from `last.pt`;
an interrupted partial epoch is repeated. A run interrupted before its first
checkpoint is preserved in an `unfinished_runs_*` folder and started again.
An individual strategy failure is logged and the runner proceeds to the others.
It returns a nonzero exit code if any strategy fails. Check `index.html` for status.

## Separate evaluation dataset

Your external dataset is never used for gradient updates, early stopping or
checkpoint selection. Supply it with `--evaluation-input` on the initial command,
or score it tomorrow without retraining:

```powershell
python scripts/N2N/benchmark_noise2time.py --output "D:/N2T/benchmark" --evaluate-only --evaluation-input "D:/separate_evaluation_dataset" --device cuda
```

The dataset must have the same HDF5/manual-retinal/pseudo-choroidal layout as the
development dataset. Preparation is cached under `evaluation_data/`. It uses the
development preparation's frame rate, diaphragm geometry, peak bounds and raw/AVI
mode, so supply a compatible acquisition dataset. Exact HDF5 duplicates of
development data are rejected; you must also ensure the recordings/subjects are
independent if you want subject-level generalization claims.

Each strategy's `best.pt` is evaluated on every external recording. Development
reports cover only the first development video and are explicitly labeled
`development`; external reports are labeled `unseen`. A video used for validation
is not an unseen test video because validation determines checkpoint selection.

## Exactly what changes

### History-only prediction

Training targets and sampled loss patches remain the same as baseline, but the
entire tenth input frame is removed. It cannot leak through encoder skips or the
residual connection: that connection uses the ninth frame. Inference also uses
only the preceding nine frames. Brightness correction applies to donor patches,
so it has no effect on this variant's input after the current frame is removed.
This experiment can expose temporal smoothing or prediction lag, not just noise
removal; inspect the report's waveform lag and amplitude metrics.

### Temporal separation

The final complete cardiac cycle of each video is held out for validation. A gap
equal to the brightness filter's support separates it from training; histories and
validation targets remain in the held-out cycle. Validation donors come from the
preceding training cycles, since one validation cycle still needs same-phase donors.
This requires at least three complete cycles (four peaks): two training cycles plus
one validation cycle. It never silently falls back to the mixed split.

The prepared `brightness.png` shows the whole video. Five peaks define four cycles:
the first three supply training donors and the last one is validation. Four peaks
define three complete cycles and are also enough. Changing frequency bounds to
manufacture more peaks is not a solution; select out recordings with fewer cycles.

Spatial masks, prepared brightness and fixed intensity scale remain common
calibration. Peak timing and phase labels come from the prepared full-video trace;
target and history frame support remains temporally disjoint. Validation samples
within the held-out cycle may overlap each other; auxiliary donors are explicitly
training frames and are recorded in `split.json`. Compare external reports rather
than validation loss alone across split variants.

### Plain U-Net and brightness correction

The existing plain U-Net processes only the last supplied frame, so removing
ConvLSTM also removes temporal memory. It is not a channel-stacked nine-frame U-Net.
The brightness ablation changes only donor-to-target brightness rescaling;
phase matching, intensity normalization and the loss locations are unchanged.

## What to open tomorrow

Open `D:/N2T/benchmark/index.html` in a browser. It contains run statuses, links to
all reports, and one comparison curve for every logged numerical metric.

```text
benchmark/
  index.html                  # Comparison dashboard
  plan.json                   # Exact configurations and record selection
  comparison.csv              # All epoch metrics for all strategies
  final_comparison.csv        # Regional best-checkpoint report metrics
  comparison/*.png            # Overlay plots, one per metric
  baseline/                   # Also five alternative strategy folders
    training.log
    status.json
    config.json
    runs/
      best.pt, last.pt
      split.json, provenance.json, monitor_masks.json
      metrics.jsonl, metrics.csv, metrics.png
      previews/epoch_001/FIRST_MEASURE.avi
    reports/development/FIRST_MEASURE/report.html
    reports/unseen/EXTERNAL_MEASURE/report.html
    denoised/...
  evaluation_data/prepared/...
```

Epoch inference, diagnostics and AVI previews cover **only the first development
measurement**, for every strategy. This reduces benchmark time; all selected
development measurements still contribute to their configured train/validation
split. The dashboard refreshes after each strategy; individual metric files update
each epoch. To refresh comparisons during training, run:

```powershell
python scripts/N2N/benchmark_noise2time.py --output "D:/N2T/benchmark" --report-only
```

Compare background temporal standard deviation together with vessel waveform
correlation, variability ratio and mean intensity. Final regional reports add
cardiac amplitude ratios, temporal lag, spatial diagnostics and videos. A constant
output has zero background standard deviation but is not a successful denoiser.
The development curves are diagnostics, not unbiased accuracy scores.
