"""Run the choroid clustering benchmark over a folder of measures.

The loader accepts both layouts used by ``choroid_segmentation.ipynb``::

    input/measure_id/manual/*.png
    input/measure_id/measure_id.h5

and::

    input/measure_id/ground_truths/*.png
    input/measure_id/measure_id_DV/h5/measure_id_DV.h5

Run ``python sandbox/run_choroid_global_benchmark.py --help`` for controls.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
import gc
import json
import logging
import os
from pathlib import Path
import sys
from time import perf_counter

os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from skimage import exposure
from skimage.filters import frangi
from skimage.measure import label
from skimage.morphology import dilation, disk, opening

from dopplerview.segmentation import clustering, process_masks, pulse_analysis as pa
from dopplerview.segmentation import signal_processing
from dopplerview.segmentation.embedding import (
    PCA_embedding,
    autocorrelation_embedding,
    complex_fourier_embedding,
    correlation_stack_per_pixel,
    harmonic_embedding,
)
from dopplerview.utils import image_utils
from sandbox import choroid_benchmark, experimental_clustering, signal_evaluation
from sandbox import signal_preprocessing
from sandbox.partial_branch_evaluation import build_partial_branch_targets


LOGGER = logging.getLogger("sandbox.choroid_global_benchmark")
CLASS_NAMES = ("artery", "vein", "aliased_artery")
MASK_CANDIDATES = {
    "retina_artery": ("retina_artery_mask.png",),
    "retina_vein": ("retina_vein_mask.png",),
    "artery": ("choroid_artery_mask.png", "choroidal_artery_mask.png"),
    "vein": ("choroid_vein_mask.png", "choroidal_vein_mask.png"),
    "aliased_artery": (
        "choroid_aliased_artery_mask.png",
        "choroidal_aliased_artery_mask.png",
    ),
}
EXPLICIT_VESSEL_MASK_NAMES = (
    "choroid_vessel_mask.png",
    "choroidal_vessel_mask.png",
    "choroid_mask.png",
)


@dataclass(frozen=True)
class MeasureFiles:
    name: str
    folder: Path
    h5_path: Path
    masks: dict[str, Path]
    candidate_vessel_mask: Path | None


@dataclass
class MeasureResources:
    files: MeasureFiles
    videos: dict[str, np.ndarray]
    reference_masks: dict[str, np.ndarray]
    retinal_artery_mask: np.ndarray
    labeled_vessels: np.ndarray
    partial_targets: object
    branch_weights: np.ndarray
    beat_period: int
    sampling_frequency: float
    fit_frames: np.ndarray
    cycle_frames: np.ndarray
    signal_frames: np.ndarray
    artifact_frames: np.ndarray
    branch_signals: np.ndarray
    cleaned_branch_signals: np.ndarray
    cycle_templates: np.ndarray
    cleaning_result: signal_preprocessing.SignalCleaningResult | None
    embeddings: dict[str, np.ndarray]
    component_names: dict[str, tuple[str, ...]]
    correlation_features: np.ndarray
    threshold_labels: np.ndarray | None
    visualization_image: np.ndarray


def _atomic_csv(table, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        table.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _find_mask(folder, names):
    search_folders = (folder / "ground_truths", folder / "manual", folder)
    for search_folder in search_folders:
        for name in names:
            path = search_folder / name
            if path.is_file():
                return path
    return None


def _contains_required_bands(path):
    try:
        with h5py.File(path, "r") as h5:
            container = h5["doppler_signal"] if "doppler_signal" in h5 else h5
            return all(name in container for name in ("M0_ff", "HF_M0_ff", "LF_M0_ff"))
    except (OSError, KeyError):
        return False


def _find_h5(folder, measure_name):
    preferred = (
        folder / f"{measure_name}.h5",
        folder / f"{measure_name}_DV" / "h5" / f"{measure_name}_DV.h5",
    )
    for path in preferred:
        if path.is_file() and _contains_required_bands(path):
            return path
    for path in sorted(folder.rglob("*.h5")):
        if _contains_required_bands(path):
            return path
    return None


def discover_measure(folder):
    """Return resources for one measure, or a skip reason for missing GT."""
    folder = Path(folder).resolve()
    masks = {
        name: _find_mask(folder, candidates)
        for name, candidates in MASK_CANDIDATES.items()
    }
    missing_choroid = [name for name in CLASS_NAMES if masks[name] is None]
    if missing_choroid:
        return None, "missing choroid masks: " + ", ".join(missing_choroid)
    missing_retina = [
        name for name in ("retina_artery", "retina_vein") if masks[name] is None
    ]
    if missing_retina:
        raise FileNotFoundError("missing retinal masks: " + ", ".join(missing_retina))
    h5_path = _find_h5(folder, folder.name)
    if h5_path is None:
        raise FileNotFoundError("no HDF5 file containing M0_ff, HF_M0_ff and LF_M0_ff")
    candidate_mask = _find_mask(folder, EXPLICIT_VESSEL_MASK_NAMES)
    return MeasureFiles(folder.name, folder, h5_path, masks, candidate_mask), None


def _load_mask(path, shape):
    target_size = (shape[1], shape[0])
    with Image.open(path) as image:
        return np.asarray(
            image.convert("L").resize(target_size, resample=Image.Resampling.NEAREST)
        ) > 0


def load_measure_arrays(files):
    """Load the three videos and all manual masks with safe HDF5 closure."""
    with h5py.File(files.h5_path, "r") as h5:
        container = h5["doppler_signal"] if "doppler_signal" in h5 else h5
        videos = {
            "M0": container["M0_ff"][()],
            "HF": container["HF_M0_ff"][()],
            "LF": container["LF_M0_ff"][()],
        }
    shapes = {name: np.asarray(video).shape for name, video in videos.items()}
    if any(len(shape) != 3 for shape in shapes.values()) or len(set(shapes.values())) != 1:
        raise ValueError(f"frequency-band videos must share one (T,H,W) shape, got {shapes}")
    spatial_shape = videos["M0"].shape[1:]
    masks = {name: _load_mask(path, spatial_shape) for name, path in files.masks.items()}
    candidate_mask = (
        None
        if files.candidate_vessel_mask is None
        else _load_mask(files.candidate_vessel_mask, spatial_shape)
    )
    return videos, masks, candidate_mask


def _notebook_candidate_mask(m0_video):
    """Reproduce the current notebook pre-mask exactly for comparability."""
    m0_image = image_utils.normalize_to_uint8(np.mean(m0_video, axis=0))
    normalized = image_utils.normalize_image(m0_image)
    contrasted = (exposure.equalize_adapthist(normalized, clip_limit=0.02) * 255).astype(
        np.uint8
    )
    vesselness = frangi(contrasted)
    # This intentionally matches the notebook's current, visually validated
    # baseline. It should be replaced only as a separate controlled experiment.
    return (~vesselness.astype(bool)) & process_masks.disk_mask(*m0_image.shape, 0.45)


def _retinal_resources(videos, masks):
    retinal_union = masks["retina_artery"] | masks["retina_vein"]
    labeled, _ = process_masks.get_labeled_vessels(retinal_union, mask_optic_disc=False)
    overlaps = process_masks.compute_branch_overlaps(
        labeled, masks["retina_artery"], masks["retina_vein"]
    )
    _, labeled = process_masks.compute_branch_label(
        overlaps, remove_overlaps=True, labeled_vessels=labeled
    )
    artery = (labeled > 0) & masks["retina_artery"]
    if not np.any(artery):
        raise ValueError("retinal artery mask contains no retained vessel branch")
    return labeled, artery


def _threshold_baseline(videos, retinal_artery_mask, retinal_masks):
    retinal_signal = signal_processing.get_pulse_from_mask(
        videos["HF"], retinal_artery_mask
    )
    corr_m0 = signal_processing.compute_correlation(
        videos["M0"], retinal_signal, normalization_interval=[-1, 1]
    )
    retinal_vessels = dilation(retinal_masks["retina_artery"] | retinal_masks["retina_vein"])
    pre_aliased = signal_processing.compute_correlation(
        videos["HF"], retinal_signal, normalization_interval=[-1, 1]
    ) < -0.47
    connected = process_masks.connect_components(pre_aliased, max_distance=5)
    large_arteries = opening(connected, disk(2))
    anti_correlated = corr_m0 < -0.3
    aliased = process_masks.keep_connected_components(anti_correlated, large_arteries)
    aliased = aliased & ~retinal_vessels
    veins = process_masks.keep_connected_components(
        corr_m0 < -0.3, aliased, negative=True
    )
    veins = (process_masks.remove_small_vessels(label(veins), min_size=10) > 0)
    veins &= ~retinal_vessels
    arteries = process_masks.remove_small_vessels(label(corr_m0 > 0.24), min_size=10) > 0
    arteries &= ~retinal_vessels
    return {"artery": arteries, "vein": veins, "aliased_artery": aliased}


def _masks_to_branch_labels(class_masks, targets, labeled_vessels, minimum=0.4):
    fractions = np.asarray(
        [
            [
                np.mean(class_masks[name][labeled_vessels == branch_id])
                for name in targets.class_names
            ]
            for branch_id in targets.branch_ids
        ]
    )
    labels = np.argmax(fractions, axis=1).astype(int)
    labels[np.max(fractions, axis=1) < minimum] = -1
    return labels


def prepare_measure(files, args):
    videos, masks, explicit_candidate_mask = load_measure_arrays(files)
    sampling_frequency = (
        args.sampling_frequency
        if args.sampling_frequency is not None
        else pa.get_effective_sampling_frequency(
            args.camera_frequency, args.temporal_window_size
        )
    )
    retinal_branches, retinal_artery = _retinal_resources(videos, masks)
    retina_signals = np.asarray(
        pa.get_filtered_branch_signals(videos["M0"], retinal_branches, sampling_frequency)
    )
    beat_period = pa.compute_period(retina_signals, sampling_frequency)
    if beat_period is None:
        raise ValueError("cardiac period could not be estimated from retinal branches")
    if videos["M0"].shape[0] // beat_period < 2:
        raise ValueError(
            f"only {videos['M0'].shape[0] // beat_period} complete cardiac cycle(s)"
        )

    if args.candidate_mask_source == "file":
        if explicit_candidate_mask is None:
            raise FileNotFoundError(
                "candidate-mask source is 'file', but no choroid vessel mask was found"
            )
        candidate_mask = explicit_candidate_mask
    else:
        candidate_mask = _notebook_candidate_mask(videos["M0"])
    labeled_vessels, _ = process_masks.get_labeled_vessels(
        candidate_mask, mask_optic_disc=False
    )
    if args.min_branch_size > 0:
        labeled_vessels = process_masks.remove_small_vessels(
            labeled_vessels.copy(), min_size=args.min_branch_size
        )
    if not np.any(labeled_vessels > 0):
        raise ValueError("candidate vessel mask produced no retained choroid branches")

    reference_masks = {name: masks[name] for name in CLASS_NAMES}
    targets = build_partial_branch_targets(
        labeled_vessels,
        reference_masks,
        min_annotated_pixels=args.min_annotated_pixels,
        min_dominance=args.min_dominance,
        min_annotated_fraction=args.min_annotated_fraction,
        evidence_saturation_pixels=args.evidence_saturation_pixels,
    )
    if not np.any(targets.labeled_mask):
        raise ValueError("no candidate branch overlaps the partial choroid labels")
    for class_index, class_name in enumerate(targets.class_names):
        if np.count_nonzero(targets.labels == class_index) < 2:
            raise ValueError(
                f"class {class_name!r} has fewer than two labeled branches; "
                "the constraint/held-out branch split is impossible"
            )

    branch_signals = np.asarray(
        pa.get_filtered_branch_signals(videos["M0"], labeled_vessels, sampling_frequency)
    )
    cleaning_result = None
    if args.temporal_protocol == "alternating":
        fit_frames, signal_frames = signal_evaluation.alternating_cycle_split(
            videos["M0"].shape[0], beat_period
        )
        cycle_frames = fit_frames.copy()
        artifact_frames = np.zeros(videos["M0"].shape[0], dtype=bool)
        cleaned_branch_signals = branch_signals.copy()
    else:
        raw_reference = signal_processing.get_pulse_from_mask(
            videos["M0"], candidate_mask
        )
        reference_period = pa.compute_period(raw_reference, sampling_frequency)
        if reference_period is None:
            reference_period = beat_period
        cleaning_result = signal_preprocessing.preprocess_cardiac_signal(
            raw_reference,
            sampling_frequency,
            reference_period,
            derivative_z_threshold=args.artifact_derivative_z,
            max_artifact_duration_seconds=args.max_artifact_duration_seconds,
            minimum_cycle_correlation=args.minimum_cycle_correlation,
            cycle_outlier_z=args.cycle_outlier_z,
            maximum_cycle_artifact_fraction=args.maximum_cycle_artifact_fraction,
            minimum_valid_cycles=args.minimum_valid_cycles,
        )
        beat_period = cleaning_result.beat_period
        fit_frames = cleaning_result.fit_frame_mask
        cycle_frames = cleaning_result.valid_cycle_frame_mask
        signal_frames = cycle_frames.copy()
        artifact_frames = cleaning_result.frame_artifact_mask
        cleaned_branch_signals = signal_preprocessing.interpolate_artifact_samples(
            branch_signals, artifact_frames, axis=1
        )
        LOGGER.info(
            "%s temporal cleaning: period %d -> %d; %d/%d cycles retained; "
            "%d artifact frames; fallback=%s",
            files.name,
            cleaning_result.initial_beat_period,
            cleaning_result.beat_period,
            cleaning_result.n_valid_cycles,
            cleaning_result.n_cycles,
            np.count_nonzero(artifact_frames),
            cleaning_result.fallback_used,
        )

    fit_videos = {name: video[fit_frames] for name, video in videos.items()}
    templates = signal_evaluation.cycle_templates_from_frame_mask(
        cleaned_branch_signals, cycle_frames, beat_period, reducer="median"
    )
    correlations = correlation_stack_per_pixel(
        retinal_artery,
        [fit_videos["HF"], fit_videos["M0"], fit_videos["LF"]],
        labeled_vessels,
        include_std=False,
        normalization_interval=None,
    )
    embeddings = {
        "correlation_HF_M0_LF": correlations,
        "complex_fourier_M0": complex_fourier_embedding(templates, n_harmonics=3),
        "harmonic_M0": harmonic_embedding(templates, n_harmonics=3),
        "PCA_M0": PCA_embedding(templates, n_components=3, gradient=False),
        "gradient_PCA_M0": PCA_embedding(templates, n_components=3, gradient=True),
        "autocorrelation_M0": autocorrelation_embedding(
            templates, n_lags=min(10, templates.shape[1] - 1)
        ),
    }
    component_names = {
        "correlation_HF_M0_LF": ("HF correlation", "M0 correlation", "LF correlation"),
        "complex_fourier_M0": tuple(
            name for harmonic in range(1, 4)
            for name in (f"Re(H{harmonic})", f"Im(H{harmonic})")
        ),
        "harmonic_M0": tuple(
            name for harmonic in range(1, 4)
            for name in (
                f"cos phase H{harmonic}",
                f"sin phase H{harmonic}",
                f"relative amplitude H{harmonic}",
            )
        ),
        "PCA_M0": ("PC1", "PC2", "PC3"),
        "gradient_PCA_M0": ("gradient PC1", "gradient PC2", "gradient PC3"),
        "autocorrelation_M0": tuple(
            f"autocorrelation lag {lag}"
            for lag in range(1, embeddings["autocorrelation_M0"].shape[1] + 1)
        ),
    }
    threshold_labels = None
    if not args.skip_threshold_baseline:
        threshold_masks = _threshold_baseline(fit_videos, retinal_artery, masks)
        threshold_labels = _masks_to_branch_labels(
            threshold_masks, targets, labeled_vessels
        )
    weights = clustering.branch_size_weights(
        labeled_vessels, mode="sqrt_area", clip_quantiles=(0.05, 0.95)
    )
    return MeasureResources(
        files=files,
        videos=videos,
        reference_masks=reference_masks,
        retinal_artery_mask=retinal_artery,
        labeled_vessels=labeled_vessels,
        partial_targets=targets,
        branch_weights=weights,
        beat_period=beat_period,
        sampling_frequency=sampling_frequency,
        fit_frames=fit_frames,
        cycle_frames=cycle_frames,
        signal_frames=signal_frames,
        artifact_frames=artifact_frames,
        branch_signals=branch_signals,
        cleaned_branch_signals=cleaned_branch_signals,
        cycle_templates=templates,
        cleaning_result=cleaning_result,
        embeddings=embeddings,
        component_names=component_names,
        correlation_features=correlations,
        threshold_labels=threshold_labels,
        visualization_image=image_utils.normalize_to_uint8(np.mean(videos["M0"], axis=0)),
    )


def build_two_step_candidates(resources):
    """Create the same 42 cached two-step strategies as the notebook."""
    stage1_embeddings = {
        "fourier3": partial(complex_fourier_embedding, n_harmonics=3),
        "gradient_pca3": partial(PCA_embedding, n_components=3, gradient=True),
    }
    binary_clusterers = {
        "gmm": partial(clustering.gmm_cluster, n_clusters=2),
        "kmeans": partial(clustering.kmeans_cluster, n_clusters=2),
        "agglomerative_ward": partial(clustering.agglomerative_cluster, n_clusters=2),
    }
    correlation_clusterers = {
        **binary_clusterers,
        "threshold": partial(clustering.correlation_clustering, thresholds=[0, -0.05]),
    }
    fit_videos = {
        name: video[resources.fit_frames] for name, video in resources.videos.items()
    }
    # Filter full signals before selecting complete accepted cycles. This avoids
    # filter-edge artifacts at joins between non-consecutive cycles.
    cycle_signals = resources.cleaned_branch_signals[:, resources.cycle_frames]
    stage1_cache = {}
    inputs_cache = {}

    def stage1(embedding_name, clusterer_name):
        key = (embedding_name, clusterer_name)
        if key not in stage1_cache:
            LOGGER.info("Computing shared stage 1: %s / %s", *key)
            stage1_cache[key] = clustering.run_clustering_pipeline(
                cycle_signals,
                resources.labeled_vessels,
                resources.sampling_frequency,
                embedding_func=stage1_embeddings[embedding_name],
                clustering_func=binary_clusterers[clusterer_name],
                video=fit_videos["M0"],
                correct_signals=False,
                beat_period=resources.beat_period,
            )
        return stage1_cache[key]

    def stage2_inputs(embedding_name, clusterer_name):
        key = (embedding_name, clusterer_name)
        if key not in inputs_cache:
            first = stage1(*key)
            branches, _ = process_masks.get_labeled_vessels(
                first.vein_mask, mask_optic_disc=(embedding_name == "fourier3")
            )
            if branches.max() == 0:
                raise ValueError("stage 1 left no branches for stage 2")
            correlations = correlation_stack_per_pixel(
                resources.retinal_artery_mask,
                [fit_videos["HF"], fit_videos["LF"]],
                branches,
                include_std=False,
                normalization_interval=[-1, 1],
            )
            full_signals = pa.get_filtered_branch_signals(
                resources.videos["LF"], branches, resources.sampling_frequency
            )
            cleaned_signals = signal_preprocessing.interpolate_artifact_samples(
                full_signals, resources.artifact_frames, axis=1
            )
            signals = cleaned_signals[:, resources.cycle_frames]
            targets = build_partial_branch_targets(branches, resources.reference_masks)
            inputs_cache[key] = branches, correlations, signals, targets
        return inputs_cache[key]

    def run_candidate(
        _unused_X,
        *,
        stage1_embedding_name,
        stage1_clusterer_name,
        stage2_representation,
        stage2_clusterer_name,
        **_kwargs,
    ):
        first = stage1(stage1_embedding_name, stage1_clusterer_name)
        branches, correlations, signals, targets = stage2_inputs(
            stage1_embedding_name, stage1_clusterer_name
        )
        if stage2_representation == "correlation_HF_LF":
            second = clustering.run_clustering_pipeline(
                correlations,
                branches,
                resources.sampling_frequency,
                embedding_func=None,
                clustering_func=correlation_clusterers[stage2_clusterer_name],
                video=fit_videos["M0"],
                correct_signals=False,
                beat_period=resources.beat_period,
                assign_to_av=False,
            )
            aliased, vein, _ = pa.assign_corr_stack_to_av(
                second.X, second.cluster_labels, branches, negative=True
            )
        else:
            second = clustering.run_clustering_pipeline(
                signals,
                branches,
                resources.sampling_frequency,
                embedding_func=partial(PCA_embedding, n_components=3, gradient=True),
                clustering_func=binary_clusterers[stage2_clusterer_name],
                video=fit_videos["M0"],
                correct_signals=False,
                beat_period=resources.beat_period,
            )
            aliased, vein = second.artery_mask, second.vein_mask
        class_masks = {"artery": first.artery_mask, "vein": vein, "aliased_artery": aliased}
        final = choroid_benchmark.class_masks_to_branch_labels(
            resources.labeled_vessels,
            class_masks,
            class_names=resources.partial_targets.class_names,
        )
        first_names = (
            tuple(
                name for harmonic in range(1, 4)
                for name in (f"Re(H{harmonic})", f"Im(H{harmonic})")
            )
            if stage1_embedding_name == "fourier3"
            else ("gradient PC1", "gradient PC2", "gradient PC3")
        )
        second_names = (
            ("HF correlation", "LF correlation")
            if stage2_representation == "correlation_HF_LF"
            else ("gradient PC1", "gradient PC2", "gradient PC3")
        )
        return choroid_benchmark.BenchmarkPrediction(
            cluster_labels=final,
            deployment_labels=final.copy(),
            embedding_views=(
                choroid_benchmark.BenchmarkEmbeddingView(
                    f"stage 1: {stage1_embedding_name} + {stage1_clusterer_name}",
                    first.X,
                    first.cluster_labels,
                    component_names=first_names,
                    partial_labels=resources.partial_targets.labels,
                    masks=(("artery", first.artery_mask), ("remaining candidates", first.vein_mask)),
                ),
                choroid_benchmark.BenchmarkEmbeddingView(
                    f"stage 2: {stage2_representation} + {stage2_clusterer_name}",
                    second.X,
                    second.cluster_labels,
                    component_names=second_names,
                    partial_labels=targets.labels,
                    masks=(("aliased artery", aliased), ("vein", vein)),
                ),
            ),
            class_masks=class_masks,
        )

    candidates = []
    for embedding_name in stage1_embeddings:
        for first_clusterer in binary_clusterers:
            for second_clusterer in correlation_clusterers:
                candidates.append(
                    choroid_benchmark.BenchmarkCandidate(
                        name=f"{first_clusterer}_then_correlation_{second_clusterer}",
                        representation=f"two_step_{embedding_name}",
                        X=None,
                        run=partial(
                            run_candidate,
                            stage1_embedding_name=embedding_name,
                            stage1_clusterer_name=first_clusterer,
                            stage2_representation="correlation_HF_LF",
                            stage2_clusterer_name=second_clusterer,
                        ),
                        subsample_safe=False,
                    )
                )
            for second_clusterer in binary_clusterers:
                candidates.append(
                    choroid_benchmark.BenchmarkCandidate(
                        name=f"{first_clusterer}_then_gradient_PCA_{second_clusterer}",
                        representation=f"two_step_{embedding_name}",
                        X=None,
                        run=partial(
                            run_candidate,
                            stage1_embedding_name=embedding_name,
                            stage1_clusterer_name=first_clusterer,
                            stage2_representation="gradient_PCA_LF",
                            stage2_clusterer_name=second_clusterer,
                        ),
                        subsample_safe=False,
                    )
                )
    return tuple(candidates)


def benchmark_measure(resources, output_folder, args):
    output_folder.mkdir(parents=True, exist_ok=True)
    if resources.cleaning_result is not None:
        cleaning_folder = output_folder / "temporal_cleaning"
        cleaning_folder.mkdir(parents=True, exist_ok=True)
        summary_path = cleaning_folder / "summary.json"
        summary_path.write_text(
            json.dumps(resources.cleaning_result.summary(), indent=2) + "\n",
            encoding="utf-8",
        )
        _atomic_csv(
            pd.DataFrame(resources.cleaning_result.cycle_rows()),
            cleaning_folder / "cycles.csv",
        )
        np.savez_compressed(
            cleaning_folder / "masks.npz",
            frame_artifact_mask=resources.artifact_frames,
            valid_cycle_frame_mask=resources.cycle_frames,
            fit_frame_mask=resources.fit_frames,
        )
    candidates = () if args.skip_two_step else build_two_step_candidates(resources)
    threshold_labelings = (
        {}
        if resources.threshold_labels is None
        else {"current_pixel_pipeline": resources.threshold_labels}
    )
    return choroid_benchmark.run_single_sample_benchmark(
        embeddings=resources.embeddings,
        partial_targets=resources.partial_targets,
        custom_candidates=candidates,
        custom_candidates_first=True,
        embedding_component_names=resources.component_names,
        csv_path=output_folder / "metrics.csv",
        cluster_archive_dir=None if args.no_save_clusters else output_folder / "clusters",
        visualization_dir=(output_folder / "visualizations") if args.save_visualizations else None,
        visualization_image=resources.visualization_image,
        temporal_signals=resources.cycle_templates,
        threshold_labelings=threshold_labelings,
        cluster_counts=tuple(args.cluster_counts),
        branch_weights=resources.branch_weights,
        deployment_correlation_features=resources.correlation_features,
        labeled_vessels=resources.labeled_vessels,
        signal_videos=resources.videos,
        signal_reference_masks=resources.reference_masks,
        sampling_frequency=resources.sampling_frequency,
        beat_period=resources.beat_period,
        signal_frame_mask=resources.signal_frames,
        signal_artifact_mask=resources.artifact_frames,
        temporal_protocol=args.temporal_protocol,
        signal_cleaning_summary=(
            None
            if resources.cleaning_result is None
            else resources.cleaning_result.summary()
        ),
        constraint_fraction=args.constraint_fraction,
        random_state=args.random_state,
        stability_runs=args.stability_runs,
        stability_sample_fraction=args.stability_sample_fraction,
        soft_dtw_max_pairwise_samples=args.soft_dtw_max_pairwise_samples,
        soft_dtw_max_template_length=args.soft_dtw_max_template_length,
        soft_dtw_window=args.soft_dtw_window,
        include_adaptive=not args.skip_adaptive,
        candidate_keys=set(args.candidate_key) if args.candidate_key else None,
    )


def _completion_path(folder):
    return folder / "complete.json"


def _write_completion(folder, resources, method_count):
    payload = {
        "measure": resources.files.name,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "h5_path": str(resources.files.h5_path),
        "beat_period": resources.beat_period,
        "temporal_protocol": (
            "clean-all-valid" if resources.cleaning_result is not None else "alternating"
        ),
        "sampling_frequency": resources.sampling_frequency,
        "branch_count": len(resources.partial_targets.branch_ids),
        "labeled_branch_count": int(np.count_nonzero(resources.partial_targets.labeled_mask)),
        "method_count": int(method_count),
    }
    path = _completion_path(folder)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_global_benchmark(args):
    input_folder = Path(args.input_folder).expanduser().resolve()
    output_folder = Path(args.output_folder).expanduser().resolve()
    if not input_folder.is_dir():
        raise NotADirectoryError(input_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    measures = sorted(path for path in input_folder.iterdir() if path.is_dir())
    if args.measure:
        selected = set(args.measure)
        measures = [path for path in measures if path.name in selected]
        missing = selected - {path.name for path in measures}
        if missing:
            raise ValueError("unknown requested measures: " + ", ".join(sorted(missing)))
    if not measures:
        raise ValueError("input folder contains no selected measure directories")

    status_path = output_folder / "global_measure_status.csv"
    global_metrics_path = output_folder / "global_metrics.csv"
    statuses = []
    metric_tables = []
    for index, folder in enumerate(measures, start=1):
        name = folder.name
        measure_output = output_folder / name
        started = perf_counter()
        LOGGER.info("[%d/%d] Inspecting %s", index, len(measures), name)
        row = {"measure": name, "status": None, "reason": None}
        try:
            files, skip_reason = discover_measure(folder)
            if files is None:
                row.update(status="skipped", reason=skip_reason)
                LOGGER.info("[%d/%d] Skipping %s: %s", index, len(measures), name, skip_reason)
                continue
            if args.dry_run:
                row.update(status="eligible", reason=None, h5_path=str(files.h5_path))
                LOGGER.info("[%d/%d] Eligible: %s", index, len(measures), name)
                continue
            metrics_path = measure_output / "metrics.csv"
            if (
                not args.overwrite
                and _completion_path(measure_output).is_file()
                and metrics_path.is_file()
            ):
                table = pd.read_csv(metrics_path)
                table.insert(0, "measure", name)
                metric_tables.append(table)
                row.update(status="resumed", reason="existing completion marker")
                LOGGER.info("[%d/%d] Reusing completed %s", index, len(measures), name)
                continue

            resources = prepare_measure(files, args)
            LOGGER.info(
                "[%d/%d] Running %s: %d branches, %d labeled, period=%d, usable cycles=%d",
                index,
                len(measures),
                name,
                len(resources.partial_targets.branch_ids),
                np.count_nonzero(resources.partial_targets.labeled_mask),
                resources.beat_period,
                int(resources.cycle_frames.sum() // resources.beat_period),
            )
            result = benchmark_measure(resources, measure_output, args)
            table = result.table.copy()
            table.insert(0, "measure", name)
            metric_tables.append(table)
            _write_completion(measure_output, resources, len(table))
            row.update(
                status="completed",
                reason=None,
                branch_count=len(resources.partial_targets.branch_ids),
                labeled_branch_count=int(np.count_nonzero(resources.partial_targets.labeled_mask)),
                beat_period=resources.beat_period,
                method_count=len(table),
            )
            LOGGER.info("[%d/%d] Completed %s (%d methods)", index, len(measures), name, len(table))
        except KeyboardInterrupt:
            row.update(status="interrupted", reason="KeyboardInterrupt")
            raise
        except Exception as error:
            row.update(status="failed", reason=f"{type(error).__name__}: {error}")
            LOGGER.exception("[%d/%d] Failed %s", index, len(measures), name)
            if args.fail_fast:
                raise
        finally:
            row["duration_seconds"] = perf_counter() - started
            statuses.append(row)
            _atomic_csv(pd.DataFrame(statuses), status_path)
            if metric_tables:
                _atomic_csv(pd.concat(metric_tables, ignore_index=True), global_metrics_path)
            gc.collect()
    return pd.DataFrame(statuses)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_folder", help="Folder containing one directory per measure")
    parser.add_argument("output_folder", help="Destination for metrics, clusters and figures")
    parser.add_argument("--measure", action="append", help="Run only this measure; repeat as needed")
    parser.add_argument("--candidate-key", action="append", help="Run only this representation/method key")
    parser.add_argument("--cluster-counts", nargs="+", type=int, default=(3, 4))
    parser.add_argument("--candidate-mask-source", choices=("notebook", "file"), default="notebook")
    parser.add_argument("--min-branch-size", type=int, default=25)
    parser.add_argument("--min-annotated-pixels", type=int, default=3)
    parser.add_argument("--min-dominance", type=float, default=0.8)
    parser.add_argument("--min-annotated-fraction", type=float, default=0.0)
    parser.add_argument("--evidence-saturation-pixels", type=int, default=10)
    parser.add_argument("--sampling-frequency", type=float)
    parser.add_argument("--camera-frequency", type=float, default=37037.0)
    parser.add_argument("--temporal-window-size", type=int, default=256)
    parser.add_argument(
        "--temporal-protocol",
        choices=("clean-all-valid", "alternating"),
        default="clean-all-valid",
        help=(
            "Use every quality-controlled cycle (default), or reproduce the old "
            "alternating-cycle fit/evaluation split"
        ),
    )
    parser.add_argument("--artifact-derivative-z", type=float, default=6.0)
    parser.add_argument("--max-artifact-duration-seconds", type=float, default=0.12)
    parser.add_argument("--minimum-cycle-correlation", type=float, default=0.5)
    parser.add_argument("--cycle-outlier-z", type=float, default=3.5)
    parser.add_argument("--maximum-cycle-artifact-fraction", type=float, default=0.15)
    parser.add_argument("--minimum-valid-cycles", type=int, default=2)
    parser.add_argument("--constraint-fraction", type=float, default=0.5)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--stability-runs", type=int, default=0)
    parser.add_argument("--stability-sample-fraction", type=float, default=0.8)
    parser.add_argument("--soft-dtw-max-pairwise-samples", type=int, default=96)
    parser.add_argument("--soft-dtw-max-template-length", type=int, default=32)
    parser.add_argument("--soft-dtw-window", type=int, default=4)
    parser.add_argument("--save-visualizations", action="store_true")
    parser.add_argument("--no-save-clusters", action="store_true")
    parser.add_argument("--skip-two-step", action="store_true")
    parser.add_argument("--skip-adaptive", action="store_true")
    parser.add_argument("--skip-threshold-baseline", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Rerun completed measures")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Only report eligible/skipped measures")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser


def configure_logging(output_folder, level):
    output_folder = Path(output_folder).expanduser().resolve()
    output_folder.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handlers = [logging.StreamHandler(), logging.FileHandler(output_folder / "global_benchmark.log")]
    for handler in handlers:
        handler.setFormatter(formatter)
    logging.basicConfig(level=getattr(logging, level), handlers=handlers, force=True)


def main(argv=None):
    args = build_parser().parse_args(argv)
    configure_logging(args.output_folder, args.log_level)
    LOGGER.info("Input: %s", Path(args.input_folder).expanduser().resolve())
    LOGGER.info("Output: %s", Path(args.output_folder).expanduser().resolve())
    statuses = run_global_benchmark(args)
    LOGGER.info("Final status counts: %s", statuses["status"].value_counts().to_dict())
    return 0 if not np.any(statuses["status"] == "failed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
