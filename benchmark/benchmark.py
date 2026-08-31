from dataclasses import dataclass
from typing import Callable
import numpy as np
import pandas as pd
import h5py
import dopplerview.segmentation.pulse_analysis as pa
import dopplerview.segmentation.signal_processing as sp
import evaluation
import clustering

@dataclass
class ExperimentConfig:
    input_name: str
    embedding_name: str
    clustering_name: str

    embedding_func: Callable
    clustering_func: Callable

    correct_signals: bool = False
    beat_period: int | None = None


@dataclass
class ExperimentResult:
    config: ExperimentConfig

    embedding_matrix: np.ndarray
    cluster_labels: np.ndarray
    mask_labels: np.ndarray

    artery_mask: np.ndarray
    vein_mask: np.ndarray

    metrics: dict

def run_clustering_pipeline(
    signals,
    labeled_vessels,
    sampling_frequency,
    embedding_func,
    clustering_func,
    correct_signals=False,
    beat_period=None,
):
    """
    Complete clustering pipeline.

    Returns
    -------
    ClusteringResult
    """

    if correct_signals:
        if beat_period is None:
            beat_period = pa.compute_period(signals, sampling_frequency)
        if beat_period is None:
            raise ValueError("Unable to estimate a cardiac period for signal correction")
        corrected = [
            pa.remove_bad_beats(
                branch,
                beat_period,
            )[0]
            for branch in signals
        ]

        cycle_templates = [
            pa.get_cycle_template(
                branch,
                sampling_frequency,
                return_period=True,
            )
            for branch in corrected
        ]

    else:

        cycle_templates = [
            pa.get_cycle_template(
                branch,
                sampling_freq=sampling_frequency,
                beat_period=beat_period,
                return_period=True,
            )
            for branch in signals
        ]

    templates, periods = zip(*cycle_templates)

    templates = np.asarray(templates)
    periods = np.asarray(periods)

    X = embedding_func(templates)

    cluster_labels = clustering_func(X)

    (
        artery_mask,
        vein_mask,
        mask_labels,
    ) = evaluation.assign_clusters_to_av(
        cluster_labels,
        signals,
        periods,
        labeled_vessels,
    )

    return clustering.ClusteringResult(
        templates=templates,
        periods=periods,
        X=X,
        cluster_labels=cluster_labels,
        mask_labels=mask_labels,
        artery_mask=artery_mask,
        vein_mask=vein_mask,
    )

def save_experiment_h5(
    h5_path,
    experiment_name,
    result,
    metadata,
):
    """
    Save one experiment.
    """

    with h5py.File(
        h5_path,
        "a",
    ) as f:

        if experiment_name in f:
            del f[experiment_name]

        g = f.create_group(
            experiment_name
        )

        g.create_dataset(
            "embedding_matrix",
            data=result.X,
            compression="gzip",
        )

        g.create_dataset(
            "cluster_labels",
            data=result.cluster_labels,
        )

        g.create_dataset(
            "mask_labels",
            data=result.mask_labels,
        )

        g.create_dataset(
            "artery_mask",
            data=result.artery_mask.astype(
                np.uint8
            ),
        )

        g.create_dataset(
            "vein_mask",
            data=result.vein_mask.astype(
                np.uint8
            ),
        )

        for k, v in metadata.items():
            g.attrs[k] = v


def run_benchmark(
    videos,
    labeled_vessels,
    gt_branch_labels,
    gt_artery_mask,
    gt_vein_mask,
    embeddings,
    clusterings,
    sampling_frequency,
    h5_path,
    beat_period=None
):
    rows = []

    experiment_id = 0

    for input_name, video in videos.items():

        branch_ids = np.unique(labeled_vessels)
        branch_ids = branch_ids[branch_ids > 0]
        signals = np.array([
            sp.get_pulse_from_mask(
                video,
                labeled_vessels == branch_id
            )
            for branch_id in branch_ids
        ])

        for embedding_name, embedding_func in embeddings.items():

            for clustering_name, clustering_func in clusterings.items():

                experiment_id += 1

                result = run_clustering_pipeline(
                    signals=signals,
                    labeled_vessels=labeled_vessels,
                    sampling_frequency=sampling_frequency,
                    embedding_func=embedding_func,
                    clustering_func=clustering_func,
                    beat_period=beat_period
                )

                metrics = evaluation.evaluate_experiment(
                    result,
                    gt_branch_labels,
                    gt_artery_mask,
                    gt_vein_mask,
                )

                metadata = {
                    "input_name": input_name,
                    "embedding_name": embedding_name,
                    "clustering_name": clustering_name,
                }

                experiment_name = (
                    f"experiment_"
                    f"{experiment_id:04d}"
                )

                save_experiment_h5(
                    h5_path,
                    experiment_name,
                    result,
                    metadata,
                )

                row = {
                    **metadata,
                    **metrics,
                }

                rows.append(row)

    return pd.DataFrame(rows)
