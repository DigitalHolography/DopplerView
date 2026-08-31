import pandas as pd
import os

def dataframe_from_dataset(dataset):
    df = pd.DataFrame(columns=["person", "nb_measure", "R", "L", "unknown"])
    for measure in os.listdir(dataset):
        if not os.path.isdir(os.path.join(dataset, measure)):
            continue
        
        parts = measure.split('_')
        if '' in parts:
            parts.remove('')
    
        if len(parts) == 5:
            date, name, eye, HD, processing_number = parts[:5]            
        elif len(parts) == 6:
            date, name, eye, measure_number, HD, processing_number = parts[:6]
    
        if not name in df["person"].values:
            df.loc[len(df)] = {"person": name, "nb_measure": 0, "R":0, "L":0, "unknown":0}
        
        df.loc[df["person"] == name, "nb_measure"] += 1
    
        if "OD" in eye.upper() or "R" in eye.upper():
            df.loc[df["person"] == name, "R"] += 1
        elif "OS" in eye.upper() or "L" in eye.upper():
            df.loc[df["person"] == name, "L"] += 1
        else:
            df.loc[df["person"] == name, "unknown"] += 1
            
    return df.sort_values(by='nb_measure', ascending=False)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

def project(X, use_pca=False, component_names=None, project_dim=None):
    """
    Project features to 2D or 3D for visualization.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Feature embedding.
    use_pca : bool
        If True and X has >2 dimensions, project using PCA.
    component_names : list
        Names for the feature components (used in axis labels).
    project_dim : int
        Number of dimensions to project to (if using PCA).
    """
    xlabel, ylabel, zlabel = None, None, None
    if use_pca:
        if project_dim is not None:
            n_components = min(project_dim, X.shape[1])
        else:
            n_components = min(3, X.shape[1])
        pca = PCA(n_components=n_components)
        X_plot = pca.fit_transform(X)
        if n_components >= 1:
            xlabel = f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)"
        if n_components >= 2:
            ylabel = f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)"
        if n_components >= 3:
            zlabel = f"PC3 ({100*pca.explained_variance_ratio_[2]:.1f}%)"
    elif X.shape[1] == 1:
        X_plot = X
        xlabel = component_names[0] if component_names is not None else "Feature 1"
    elif X.shape[1] == 2:
        X_plot = X
        xlabel = component_names[0] if component_names is not None else "Feature 1"
        ylabel = component_names[1] if component_names is not None else "Feature 2"
    elif X.shape[1] == 3:
        X_plot = X[:, :3]
        xlabel = component_names[0] if component_names is not None else "Feature 1"
        ylabel = component_names[1] if component_names is not None else "Feature 2"
        zlabel = component_names[2] if component_names is not None else "Feature 3"
    else:
        raise ValueError("Cannot visualize >3D features without PCA")
    
    return X_plot, xlabel, ylabel, zlabel

def plot(ax, X_plot, labels, xlabel=None, ylabel=None, zlabel=None, title=None, label_names=None):
    for lab in np.unique(labels):
        idx = labels == lab

        ax.scatter(
            X_plot[idx, 0],
            X_plot[idx, 1] if X_plot.shape[1] >= 2 else None,
            X_plot[idx, 2] if X_plot.shape[1] >= 3 else None,
            label=f"cluster {lab}" if label_names is None else label_names.get(lab, f"cluster {lab}")
        )

    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if zlabel:
        ax.set_zlabel(zlabel)

    if title:
        ax.set_title(title)

    ax.legend()
    ax.axis("equal")
    ax.grid(alpha=0.3)
    return ax

def plot_clustering(X, labels, title=None, use_pca=False, gt_labels=None, label_names=None, component_names=None, project_dim=None):
    """
    Visualize clustering results for arbitrary feature dimensions.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Feature embedding.
    labels : ndarray, shape (n_samples,)
        Cluster labels.
    gt_labels : ndarray, shape (n_samples,)
        Ground truth labels.
    title : str
        Plot title.
    use_pca : bool
        If True and X has >2 dimensions, project using PCA.
    label_names : dict
        Mapping from label values to names for plotting.
    component_names : list
        Names for the feature components (used in axis labels).
    """

    X = np.asarray(X)
    labels = np.asarray(labels)

    # --- Projection ---
    X_plot, xlabel, ylabel, zlabel = project(X, use_pca=use_pca, component_names=component_names, project_dim=project_dim)

    # --- Plot ---
    fig=plt.figure(figsize=(13, 6)) if gt_labels is not None else plt.figure(figsize=(6, 6))
    subplot_args = 121 if gt_labels is not None else 111
    ax = fig.add_subplot(subplot_args, projection="3d") if X_plot.shape[1] == 3 else fig.add_subplot(subplot_args)
    plot(ax, X_plot, labels, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, title=title)
    if gt_labels is not None:
        gt_labels = np.asarray(gt_labels)
        ax_gt = fig.add_subplot(122, projection="3d") if X_plot.shape[1] == 3 else fig.add_subplot(122)
        plot(ax_gt, X_plot, gt_labels, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, title="Ground Truth", label_names=label_names)

    plt.show()