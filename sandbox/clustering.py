from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from dataclasses import dataclass
from typing import Callable

import numpy as np

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
