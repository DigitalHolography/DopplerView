import numpy as np
import pytest

from sandbox.choroid_benchmark import (
    constraints_from_partial_targets,
    map_clusters_to_classes,
    run_single_sample_benchmark,
    stratified_partial_label_split,
)
from sandbox.partial_branch_evaluation import build_partial_branch_targets
import sandbox.choroid_benchmark as benchmark_module


def _targets():
    branch_ids = np.arange(1, 13).reshape(3, 4)
    masks = {}
    for class_label, class_name in enumerate(("artery", "vein", "aliased_artery")):
        mask = np.zeros_like(branch_ids, dtype=bool)
        ids = np.arange(1 + 4 * class_label, 5 + 4 * class_label)
        mask[np.isin(branch_ids, ids)] = True
        masks[class_name] = mask
    return branch_ids, build_partial_branch_targets(
        branch_ids, masks, min_annotated_pixels=1
    )


def test_partial_split_is_disjoint_and_stratified():
    _, targets = _targets()
    constraint, evaluation = stratified_partial_label_split(targets, random_state=2)

    assert not np.any(constraint & evaluation)
    for class_label in range(3):
        assert np.any(constraint & (targets.labels == class_label))
        assert np.any(evaluation & (targets.labels == class_label))


def test_constraints_and_mapping_use_selected_training_branches():
    _, targets = _targets()
    selected = np.zeros(12, dtype=bool)
    selected[[0, 1, 4, 5, 8, 9]] = True
    must_link, cannot_link = constraints_from_partial_targets(targets, selected)

    assert any(np.array_equal(pair, [0, 1]) for pair in must_link)
    assert any(np.array_equal(pair, [0, 4]) for pair in cannot_link)
    clusters = np.repeat([9, 3, 7], 4)
    mapped = map_clusters_to_classes(clusters, targets, selected)
    np.testing.assert_array_equal(mapped, np.repeat([0, 1, 2], 4))


def test_single_sample_benchmark_runs_required_euclidean_families(tmp_path):
    _, targets = _targets()
    rng = np.random.default_rng(1)
    X = np.repeat([[-3.0, 0.0], [0.0, 3.0], [3.0, 0.0]], 4, axis=0)
    X += rng.normal(0, 0.05, X.shape)

    csv_path = tmp_path / "nested" / "metrics.csv"
    result = run_single_sample_benchmark(
        {"correlation_3band": X},
        targets,
        temporal_signals=None,
        threshold_labelings={"manual_grid": np.repeat([0, 1, 2], 4)},
        cluster_counts=(3,),
        include_adaptive=False,
        random_state=2,
        csv_path=csv_path,
    )

    methods = set(result.table["method"])
    assert {
        "kmeans_k3",
        "gmm_k3",
        "hierarchical_ward_k3",
        "trimmed_kmeans_k3",
        "cop_kmeans_k3",
        "threshold_manual_grid",
    }.issubset(methods)
    assert result.table["error"].isna().all() if "error" in result.table else True
    kmeans = result.table[result.table["method"] == "kmeans_k3"].iloc[0]
    assert kmeans["heldout_partial_ARI"] == 1.0
    assert kmeans["heldout_mapped_macro_f1"] == 1.0
    assert "heldout_partial_mapped_macro_f1_resubstitution" not in result.table
    assert result.csv_path == csv_path.resolve()
    assert csv_path.is_file()
    saved = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    assert len(saved) == len(result.table)


def test_csv_is_checkpointed_before_a_later_method_is_interrupted(
    tmp_path, monkeypatch
):
    _, targets = _targets()
    X = np.repeat([[-3.0], [0.0], [3.0]], 4, axis=0)
    csv_path = tmp_path / "checkpoint.csv"

    class InterruptedGMM:
        def __init__(self, **_kwargs):
            pass

        def fit_predict(self, _X):
            raise KeyboardInterrupt

    monkeypatch.setattr(benchmark_module, "GaussianMixture", InterruptedGMM)
    with pytest.raises(KeyboardInterrupt):
        run_single_sample_benchmark(
            {"embedding": X},
            targets,
            cluster_counts=(3,),
            include_adaptive=False,
            candidate_keys={"embedding/kmeans_k3", "embedding/gmm_k3"},
            csv_path=csv_path,
        )

    checkpoint = np.genfromtxt(
        csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8"
    )
    assert checkpoint["method"].item() == "kmeans_k3"
