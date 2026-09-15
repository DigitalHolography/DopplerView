# Choroid clustering benchmark: metrics and evaluation protocol

## How to read a metric name

The CSV uses compositional names. For example,
`heldout_partial_assigned_weighted_ARI` means:

- `heldout_`: only reliable partial labels not used for constraints or label mapping;
- `partial_`: branch-level comparison against partial annotations;
- `assigned_`: noise/rejected branches are removed before scoring;
- `weighted_`: branch-size weights multiplied by annotation confidence are used;
- `ARI`: adjusted Rand index.

`all_` uses every reliably annotated branch. It is descriptive, but it is
optimistic for constrained methods and every column ending in
`_resubstitution`, because the same labels contributed to fitting or naming.
Use `heldout_` columns for method comparison.

## Annotation and support columns

The partial masks are converted to one target per branch. Pixels belonging to
more than one class are ambiguous and do not provide class evidence. A branch
is labeled only when it passes the configured minimum pixel and dominance
rules. Its confidence is:

`dominant fraction × min(1, dominant annotated pixels / saturation pixels)`.

The following suffixes describe the evaluated data, not performance:

| Suffix | Meaning |
|---|---|
| `branch_count` | Number of branches in the complete branch map. |
| `labeled_branch_count` | Branches with sufficiently reliable partial labels. |
| `ambiguous_branch_count` | Branches with conflicting class evidence. |
| `unlabeled_branch_count` | Branches without enough evidence. |
| `cluster_count` | Non-noise clusters returned by the method. |
| `noise_branch_count` | Branches labeled `-1` (rejected/noise). |
| `labeled_coverage` | Fraction of labeled branches that the method assigns. Higher is better. |
| `overall_coverage` | Fraction of all branches that the method assigns. Higher is not automatically better: assigning noise can reduce precision. |
| `effective_label_weight_sum` | Total confidence-adjusted evaluation weight. Use it to detect samples with very little evidence. |

Always interpret an `assigned_` score beside `labeled_coverage`. A method can
obtain an excellent assigned-only score by rejecting difficult branches.

## Anonymous-partition metrics

These metrics are invariant to numeric cluster IDs and therefore measure the
clustering before artery/vein names are attached.

| Metric | Range/direction | Interpretation and limitation |
|---|---|---|
| `ARI` | Usually -1 to 1; higher | Pair agreement corrected for chance. It penalizes both class splitting and class merging. This is the preferred single partition score. Small labeled sets can make it variable. |
| `NMI` | 0 to 1; higher | Shared information between clusters and classes. It is permutation-invariant but not chance-adjusted and can favor solutions with more clusters. |
| `homogeneity` | 0 to 1; higher | Each cluster contains only one annotated class. Over-clustering can make it artificially high. |
| `completeness` | 0 to 1; higher | All branches of a class fall in one cluster. Under-clustering can make it high. |
| `v_measure` | 0 to 1; higher | Harmonic mean of homogeneity and completeness. Use it to diagnose whether ARI loss comes from splitting or merging. |

`weighted_*` versions use the configured branch weight (currently clipped
square-root area) multiplied by annotation confidence. They reduce the impact
of tiny uncertain fragments without allowing trunks to dominate as strongly as
raw pixel area. Weighted ARI is a continuous pair-mass analogue, not
scikit-learn's standard ARI; report standard and weighted results together.

`assigned_*` repeats these calculations after excluding noise. The unqualified
version includes noise as a common predicted label and is the conservative
score.

## Class-name mapping metrics

Each anonymous cluster is mapped independently to the semantic class with the
largest weighted partial-label evidence on the constraint subset. Multiple
clusters may therefore map to the same class. A cluster with no constraint
evidence remains unmapped. This is the original partial-ground-truth mapping
and avoids forcing a globally one-to-one correspondence when a physiological
class is represented by several clusters.

| Metric | Range/direction | Interpretation |
|---|---|---|
| `heldout_mapped_accuracy` | 0 to 1; higher | Fraction of held-out labeled branches assigned the correct semantic class. Dominant classes can control it. |
| `heldout_mapped_balanced_accuracy` | 0 to 1; higher | Mean recall over artery, vein, and aliased artery. Each class counts equally. |
| `heldout_mapped_macro_f1` | 0 to 1; higher | Mean class F1; penalizes both missed classes and false assignments. Preferred offline semantic score. |
| `heldout_weighted_mapped_accuracy` | 0 to 1; higher | Accuracy weighted by branch size and annotation confidence. |
| `heldout_weighted_mapped_macro_f1` | 0 to 1; higher | Confidence/size-weighted macro F1. Report beside its unweighted counterpart. |

Columns ending in `_resubstitution` map and score on the same branches. They are
upper-bound diagnostics only and must not be used to select a method.

## Deployment/physiology mapping metrics

`semantic_mapping` identifies which names are used for saved masks,
visualizations, and signal extraction. It is `partial_ground_truth_majority`
for anonymous one-step clusters and `method_assignment` for pipelines that
produce semantic masks directly.

`deployment_mapping` identifies the separate label-free diagnostic:

- `correlation_prototypes`: a label-free one-to-one match between cluster
  HF/M0/LF correlation profiles and provisional artery, vein, and aliased-
  artery prototypes;
- `method_assignment`: the two-step or threshold method produced semantic
  labels directly.

`heldout_physiology_accuracy`, `balanced_accuracy`, `macro_f1`, and weighted
accuracy compare that deployable mapping with held-out labels. These are more
representative of real use than partial-ground-truth mapping scores, but they also test the
physiological mapping hypothesis, not only clustering quality.

`heldout_physiology_mapped_coverage` is the confidence/size-weighted fraction of
held-out branches receiving a semantic name. Read it beside physiology macro
F1; a mapper can improve conditional correctness by leaving clusters unmapped.

## Signal-similarity metrics

For each band (`HF`, `M0`, `LF`) and class (`artery`, `vein`,
`aliased_artery`), the benchmark compares the signal from predicted pixels with
the manually labeled reference signal. Manual-reference pixels are removed from
the predicted mask, preventing direct pixel leakage. With the default
`clean-all-valid` protocol, these are descriptive in-sample physiology metrics:
clustering and signal evaluation both use all quality-controlled cycles. They
must not be described as held-out validation. The independent partial-branch
label split used by `heldout_*` clustering metrics is unchanged.

Before feature construction, short reversible impulses are detected on the
automatic choroid-candidate signal and interpolated in extracted branch/mask
signals. The raw videos are never overwritten. A nearby autocorrelation peak
can correct a biased spectral period estimate, complete cycles are phase
aligned, and cycles with anomalous shape, amplitude, or artifact burden are
rejected. Correlations computed directly from video omit individual artifact
frames. Each global-benchmark measure saves `temporal_cleaning/summary.json`,
`cycles.csv`, and `masks.npz` so these decisions can be audited.
The same configuration is copied into every metrics row as
`temporal_protocol` and `signal_cleaning_*` columns, preventing results from
different temporal protocols from being pooled accidentally.

Use `--temporal-protocol alternating` only to reproduce the former 50/50 cycle
split. It gives temporally held-out signal metrics, but can discard too much of
a short acquisition and is no longer the recommended clustering protocol.
In the notebook, set `BENCHMARK_TEMPORAL_PROTOCOL` to either
`"clean-all-valid"` or `"alternating"` before running the benchmark preparation
cell. The earlier method examples have the equivalent
`CHOROID_TEMPORAL_PROTOCOL` control.

| Suffix | Range/direction | Meaning |
|---|---|---|
| `predicted_pixel_count` / `reference_pixel_count` | count | Support diagnostics. Empty or tiny masks make signal scores undefined or unstable. |
| `pearson` | -1 to 1; higher | Zero-lag linear similarity of standardized median cycle shapes. |
| `spearman` | -1 to 1; higher | Rank/monotonic similarity; less sensitive to waveform amplitude outliers. |
| `max_correlation` | -1 to 1; higher | Best cross-correlation within ±25% of a cardiac cycle. It can hide a physiologically wrong phase shift. |
| `max_correlation_lag_samples` | signed samples; closer to 0 usually better | Shift producing maximum correlation. Convert to radians with `2π × lag / beat_period` if needed. |
| `phase_error_radians` | 0 to π; lower | Absolute phase difference of the fundamental cardiac harmonic. |
| `cardiac_coherence` | 0 to 1; higher | Spectral coherence at the estimated cardiac frequency. It is noisy when few usable cycles are available. |
| `normalized_rmse` | 0 upward; lower | RMSE between standardized cycle templates. It is largely redundant with zero-lag Pearson correlation. |
| `soft_dtw_divergence` | 0 upward; lower | Shape discrepancy allowing limited nonlinear temporal alignment. A low value can tolerate timing changes that are physiologically important. |

`signal_<band>_<metric>_macro` is the unweighted mean across classes with a
finite value. Inspect all class-level columns and pixel counts: silently missing
classes are excluded from the macro mean.

Signal scores are supporting physiological evidence, not ground-truth
segmentation accuracy. A large anatomically wrong region can still have a
similar average waveform.

## Stability, runtime, and status metrics

Stability columns appear only when `stability_runs >= 2` and the method supports
subsampling:

- `stability_run_count`, `valid_run_pair_count`, and
  `observation_fraction_mean` describe the resampling evidence;
- `cluster_count_mean/std` and `noise_fraction_mean` show structural variation;
- `stability_ARI_mean/std/min` compare repeated partitions on branches observed
  in both runs;
- `stability_weighted_ARI_*` uses branch weights;
- `stability_assigned_ARI_*` excludes noise and must be paired with noise and
  observation fractions.

`clustering_seconds`, `evaluation_seconds`, and `runtime_seconds` are engineering
metrics, not scientific quality. `error` and `visualization_error` must be
reported rather than silently dropping failed methods. `temporal_leakage=True`
marks a result that must not enter the primary comparison.

## Saving and re-evaluating clusters

Set `cluster_archive_dir` independently of `visualization_dir`:

```python
result = run_single_sample_benchmark(
    ...,
    cluster_archive_dir=Path("D:/benchmarks") / sample_name / "clusters",
)
```

Each successful method is progressively and atomically written to
`<cluster_archive_dir>/<representation>_<method>/clusters.npz`. The archive
contains anonymous labels, exact branch IDs, class names, weights, method-native
deployment labels and two-step masks when present, and raw three-band
correlation features used by the label-free mapper.

The loader also accepts the existing `labels_and_masks.npz` files inside your
two visualization directories. When the adjacent benchmark CSV can be found,
method identity and `temporal_leakage` are recovered from it. These legacy files
can reproduce partition and saved-semantic-mask metrics, but they do not contain
branch weights or raw correlation features; supply weights explicitly if needed,
and use the new archives for future mapping-rule experiments.

After editing the partial masks, rebuild `PartialBranchTargets` with the same
branch map and call:

```python
updated = reevaluate_cluster_archive(
    archive_path,
    updated_partial_targets,
    labeled_vessels=choroid_labeled_vessels,
    signal_videos={"HF": HF_M0_ff, "M0": M0_ff, "LF": LF_M0_ff},
    signal_reference_masks=updated_known_positive_masks,
    sampling_frequency=sampling_freq,
    beat_period=beat_period,
    signal_frame_mask=benchmark_signal_frames,
    signal_artifact_mask=benchmark_artifact_frames,
)
updated.table
```

To update all methods for one measure and checkpoint a new CSV progressively:

```python
updated_table = choroid_benchmark.reevaluate_cluster_archives(
    benchmark_cluster_archive_dir,
    updated_partial_targets,
    csv_path=Path("D:/benchmarks") / f"{measure_name}_updated_metrics.csv",
    labeled_vessels=choroid_labeled_vessels,
    signal_videos={"HF": HF_M0_ff, "M0": M0_ff, "LF": LF_M0_ff},
    signal_reference_masks=updated_known_positive_masks,
    sampling_frequency=sampling_freq,
    beat_period=beat_period,
    signal_frame_mask=benchmark_signal_frames,
    signal_artifact_mask=benchmark_artifact_frames,
)
```

The same branch IDs are required. If vessel pre-masking or branch splitting
changes, clustering must be rerun. Runtime and resampling stability also cannot
be reconstructed from one saved partition.

## Recommended 15-measure protocol

1. Freeze the branch extraction, cardiac-cycle cleaning/selection, annotation rules, class
   order, method list, random seed, and hyperparameters before the global run.
2. Treat the two samples already used to inspect and tune methods as development
   samples. Do not describe their final scores as independent confirmation.
3. Run every retained method on every measure, with one CSV and one cluster
   archive directory per measure. Disable exhaustive figures if storage or time
   is a concern; cluster saving is independent.
4. Exclude `temporal_leakage=True` from primary analysis. Keep failures as
   failures and record the number of valid samples for every method.
5. Use one measure as one statistical unit. If several measures belong to one
   patient, aggregate or bootstrap at patient level rather than treating them as
   independent.
6. Use a prespecified hierarchy instead of optimizing hundreds of columns:
   - primary partition endpoint: `heldout_partial_weighted_ARI`, with ordinary
     `heldout_partial_ARI` as sensitivity analysis;
   - primary offline semantic endpoint: `heldout_mapped_macro_f1`;
   - deployment endpoint: `heldout_physiology_macro_f1` together with
     `heldout_physiology_mapped_coverage`;
   - supporting physiology: class-level M0/HF/LF Pearson and phase error;
   - safeguards: coverage, cluster count, noise fraction, errors, and runtime.
7. Aggregate paired per-measure scores using median and interquartile range,
   bootstrap confidence intervals over measures/patients, average rank, and win
   count. Also show each measure as a point; do not pool all branches as though
   they were independent.
8. Shortlist a small number of methods using the primary hierarchy, then inspect
   their saved overlays and per-class failures. With about 191 configurations
   and only 15 measures, broad pairwise significance testing would be dominated
   by multiple comparisons and selection bias.
9. Perform sensitivity analyses for unweighted versus weighted scores,
   conservative versus assigned-only scores, and alternative annotation
   thresholds. Re-evaluate from the archives when only ground truth changes.
10. Report the final method on untouched measures if possible. If all 15 samples
    influence method choice, describe the result as exploratory or use grouped
    nested cross-validation for any supervised tuning.
