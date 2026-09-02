from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from dataclasses import dataclass
from typing import Callable

import numpy as np
import dopplerview.segmentation.pulse_analysis as pa

def kmeans_cluster(X, n_clusters=2):
    return KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=20,
        random_state=0,
        algorithm="lloyd",
    ).fit_predict(X)


def agglomerative_cluster(X, n_clusters=2):
    return AgglomerativeClustering(
        n_clusters=n_clusters
    ).fit_predict(X)


def gmm_cluster(X, n_clusters=2):
    return GaussianMixture(
        n_components=n_clusters,
        random_state=0
    ).fit(X).predict(X)


@dataclass
class ClusteringResult:
    templates: np.ndarray
    periods: np.ndarray

    X: np.ndarray

    cluster_labels: np.ndarray
    mask_labels: np.ndarray

    artery_mask: np.ndarray
    vein_mask: np.ndarray

def run_clustering_pipeline(
    signals,
    labeled_vessels,
    sampling_frequency,
    embedding_func,
    clustering_func,
    video,
    correct_signals=False,
    beat_period=None,
    assign_to_av=True
):
    """
    Complete clustering pipeline.
    signals: array of shape (n_branches, n_timepoints)
    labeled_vessels: array of shape (height, width) with branch labels
    sampling_frequency: sampling frequency of the signals
    embedding_func: function to embed the signals (e.g., PCA). If the signals are already embedded, this can be None.
    clustering_func: function to perform clustering
    video: the video data
    correct_signals: whether to correct the signals
    beat_period: the period of the heartbeats
    assign_to_av: whether to assign clusters to artery/vein

    Returns
    -------
    ClusteringResult
    """
    signals = np.asarray(signals)
    labeled_vessels = np.asarray(labeled_vessels)
    video = np.asarray(video)
    if signals.ndim != 2:
        raise ValueError("signals must be a 2-D branch-by-time array")
    if labeled_vessels.ndim != 2:
        raise ValueError("labeled_vessels must be a 2-D label image")
    if video.ndim != 3 or video.shape[1:] != labeled_vessels.shape:
        raise ValueError("video and labeled_vessels must have matching spatial shapes")
    branch_ids = np.unique(labeled_vessels)
    branch_ids = branch_ids[branch_ids > 0]
    if len(signals) != len(branch_ids):
        raise ValueError("signals must contain exactly one row per labeled branch")

    if embedding_func is not None:
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

    else:
        X = signals
        templates = None
        periods = np.asarray([beat_period] * len(signals))

    cluster_labels = clustering_func(X)
    cluster_labels = np.asarray(cluster_labels)
    if cluster_labels.ndim != 1 or len(cluster_labels) != len(branch_ids):
        raise ValueError("clustering must return exactly one label per branch")
    if np.unique(cluster_labels).size == 2:
        cluster_labels = pa.canonicalize_binary_cluster_labels(cluster_labels, X)

    if assign_to_av:
        (artery_mask, vein_mask, mask_labels,) = pa.assign_clusters_to_av(
            cluster_labels,
            video,
            periods,
            labeled_vessels,
            sampling_freq=sampling_frequency
        )
    else:
        mask_labels = np.zeros_like(cluster_labels, dtype=int)
        artery_mask = np.zeros_like(labeled_vessels, dtype=bool)
        vein_mask = np.zeros_like(labeled_vessels, dtype=bool)

    return ClusteringResult(
        templates=templates,
        periods=periods,
        X=X,
        cluster_labels=cluster_labels,
        mask_labels=mask_labels,
        artery_mask=artery_mask,
        vein_mask=vein_mask,
    )