from dataclasses import dataclass
from typing import Callable
import numpy as np
import pandas as pd
import dopplerview.segmentation.pulse_analysis as pa
import dopplerview.segmentation.signal_processing as sp
from . import evaluation
from dopplerview.segmentation import clustering

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
    """Run the Cartesian product of inputs, embeddings, and clusterings."""
    if not videos:
        raise ValueError("videos cannot be empty")
    if not embeddings:
        raise ValueError("embeddings cannot be empty")
    if not clusterings:
        raise ValueError("clusterings cannot be empty")

    labeled_vessels = np.asarray(labeled_vessels)
    if labeled_vessels.ndim != 2:
        raise ValueError("labeled_vessels must be a 2-D label image")
    branch_ids = np.unique(labeled_vessels)
    branch_ids = branch_ids[branch_ids > 0]
    if branch_ids.size == 0:
        raise ValueError("labeled_vessels does not contain any positive branch ID")

    rows = []

    experiment_id = 0

    for input_name, video in videos.items():
        video = np.asarray(video)
        if video.ndim != 3 or video.shape[1:] != labeled_vessels.shape:
            raise ValueError(
                f"video {input_name!r} and labeled_vessels have incompatible shapes"
            )
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
                    video=video,
                    beat_period=beat_period
                )

                metrics = evaluation.evaluate_experiment(
                    result,
                    gt_branch_labels,
                    gt_artery_mask,
                    gt_vein_mask,
                )

                experiment_name = f"experiment_{experiment_id:04d}"
                metadata = {
                    "experiment_name": experiment_name,
                    "input_name": input_name,
                    "embedding_name": embedding_name,
                    "clustering_name": clustering_name,
                }

                evaluation.save_experiment_h5(
                    h5_path,
                    experiment_name,
                    result,
                    {**metadata, "branch_ids": branch_ids},
                    metrics=metrics,
                )

                row = {
                    **metadata,
                    **metrics,
                }

                rows.append(row)

    return pd.DataFrame(rows)
