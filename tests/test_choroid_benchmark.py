import json
from pathlib import Path

import numpy as np
import pytest

from sandbox.choroid_benchmark import (
    BenchmarkCandidate,
    BenchmarkEmbeddingView,
    BenchmarkPrediction,
    class_masks_to_branch_labels,
    constraints_from_partial_targets,
    load_cluster_archive,
    map_clusters_by_correlation_physiology,
    map_clusters_to_classes,
    reevaluate_cluster_archive,
    reevaluate_cluster_archives,
    run_single_sample_benchmark,
    stratified_partial_label_split,
)
from sandbox.partial_branch_evaluation import build_partial_branch_targets
import sandbox.choroid_benchmark as benchmark_module


def test_choroid_notebook_code_cells_are_valid_python():
    notebook_path = (
        Path(__file__).resolve().parents[1]
        / "sandbox"
        / "choroid_segmentation.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        source = "\n".join(
            line
            for line in source.splitlines()
            if not line.lstrip().startswith(("%", "!"))
        )
        compile(source, f"{notebook_path.name}:{cell['id']}", "exec")


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


def test_partial_mapping_assigns_each_cluster_independently():
    _, targets = _targets()
    selected = np.ones(12, dtype=bool)
    clusters = np.array([9, 9, 8, 8, 3, 3, 3, 3, 7, 7, 7, 7])

    mapped = map_clusters_to_classes(clusters, targets, selected)

    assert np.all(mapped[clusters == 9] == 0)
    assert np.all(mapped[clusters == 8] == 0)
    assert np.all(mapped[clusters == 3] == 1)
    assert np.all(mapped[clusters == 7] == 2)


def test_correlation_physiology_mapping_names_three_signatures_and_rejects_extra():
    clusters = np.repeat([17, 4, 12, 99], 2)
    correlations = np.array(
        [
            [0.8, 0.7, 0.6],
            [0.7, 0.8, 0.7],
            [0.05, -0.6, -0.7],
            [0.10, -0.5, -0.6],
            [-0.8, -0.7, -0.6],
            [-0.7, -0.8, -0.7],
            [0.2, 0.0, -0.1],
            [0.1, 0.1, -0.2],
        ]
    )

    result = map_clusters_by_correlation_physiology(clusters, correlations)

    assert result.cluster_to_class == {
        4: "vein",
        12: "aliased_artery",
        17: "artery",
    }
    assert np.all(result.mapped_labels[clusters == 99] == -1)
    assert result.similarity_matrix.shape == (4, 3)


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
        temporal_protocol="clean-all-valid",
        signal_cleaning_summary={"valid_cycle_count": 3, "fallback_used": False},
        csv_path=csv_path,
    )

    methods = set(result.table["method"])
    assert {
        "kmeans_k3",
        "gmm_k3",
        "agglomerative_ward_k3",
        "agglomerative_average_k3",
        "agglomerative_complete_k3",
        "agglomerative_single_k3",
        "trimmed_kmeans_k3",
        "cop_kmeans_k3",
        "threshold_manual_grid",
    }.issubset(methods)
    assert result.table["error"].isna().all() if "error" in result.table else True
    kmeans = result.table[result.table["method"] == "kmeans_k3"].iloc[0]
    assert kmeans["heldout_partial_ARI"] == 1.0
    assert kmeans["heldout_mapped_macro_f1"] == 1.0
    assert "heldout_partial_mapped_macro_f1_resubstitution" not in result.table
    assert set(result.table["temporal_protocol"]) == {"clean-all-valid"}
    assert set(result.table["signal_cleaning_valid_cycle_count"]) == {3}
    assert not result.table["signal_cleaning_fallback_used"].any()
    assert result.csv_path == csv_path.resolve()
    assert csv_path.is_file()
    saved = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    assert len(saved) == len(result.table)


def test_benchmark_reports_label_free_physiology_mapping(tmp_path):
    _, targets = _targets()
    correlations = np.repeat(
        [[0.8, 0.7, 0.6], [0.0, -0.6, -0.7], [-0.8, -0.7, -0.6]],
        4,
        axis=0,
    )
    result = run_single_sample_benchmark(
        {"correlation_3band": correlations},
        targets,
        cluster_counts=(3,),
        include_adaptive=False,
        candidate_keys={"correlation_3band/kmeans_k3"},
        deployment_correlation_features=correlations,
        csv_path=tmp_path / "physiology.csv",
        random_state=2,
    )

    row = result.table.iloc[0]
    assert row["heldout_physiology_macro_f1"] == 1.0
    assert row["heldout_physiology_mapped_coverage"] == 1.0
    assert row["semantic_mapping"] == "partial_ground_truth_majority"
    assert set(result.physiology_mapped_class_labels) == {
        "correlation_3band/kmeans_k3"
    }


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


def test_cluster_archive_is_independent_of_visualization_and_can_be_reevaluated(
    tmp_path,
):
    _, targets = _targets()
    X = np.repeat([[-3.0], [0.0], [3.0]], 4, axis=0)
    weights = np.linspace(1.0, 2.0, len(X))
    archive_directory = tmp_path / "clusters"

    original = run_single_sample_benchmark(
        {"embedding": X},
        targets,
        cluster_counts=(3,),
        include_adaptive=False,
        candidate_keys={"embedding/kmeans_k3"},
        branch_weights=weights,
        cluster_archive_dir=archive_directory,
    )

    archive_path = archive_directory / "embedding_kmeans_k3" / "clusters.npz"
    assert archive_path.is_file()
    assert "artifact_directory" not in original.table
    archive = load_cluster_archive(archive_path)
    np.testing.assert_array_equal(archive["branch_ids"], targets.branch_ids)
    np.testing.assert_array_equal(
        archive["cluster_labels"], original.cluster_labels["embedding/kmeans_k3"]
    )
    np.testing.assert_array_equal(archive["branch_weights"], weights)

    reevaluated = reevaluate_cluster_archive(archive_path, targets)

    assert reevaluated.table.loc[0, "heldout_partial_weighted_ARI"] == pytest.approx(
        original.table.loc[0, "heldout_partial_weighted_ARI"]
    )
    assert "runtime_seconds" not in reevaluated.table

    updated_csv = tmp_path / "updated_metrics.csv"
    updated_table = reevaluate_cluster_archives(
        archive_directory,
        targets,
        csv_path=updated_csv,
    )
    assert len(updated_table) == 1
    assert updated_csv.is_file()
    assert updated_table.loc[0, "heldout_partial_weighted_ARI"] == pytest.approx(
        original.table.loc[0, "heldout_partial_weighted_ARI"]
    )


def test_legacy_visualization_archive_can_be_loaded_and_reevaluated(tmp_path):
    branch_map, targets = _targets()
    method_directory = tmp_path / "legacy_visualizations" / "old_method"
    method_directory.mkdir(parents=True)
    archive_path = method_directory / "labels_and_masks.npz"
    semantic = np.repeat([0, 1, 2], 4)
    np.savez_compressed(
        archive_path,
        branch_ids=targets.branch_ids,
        cluster_labels=np.repeat([8, 3, 5], 4),
        semantic_labels=semantic,
        mask_artery=np.isin(branch_map, [1, 2, 3, 4]),
        mask_vein=np.isin(branch_map, [5, 6, 7, 8]),
        mask_aliased_artery=np.isin(branch_map, [9, 10, 11, 12]),
    )

    archive = load_cluster_archive(archive_path)
    assert archive["schema_version"] == 0
    assert archive["temporal_leakage"] is True
    result = reevaluate_cluster_archive(
        archive_path,
        targets,
        labeled_vessels=branch_map,
    )

    assert result.table.loc[0, "heldout_physiology_macro_f1"] == 1.0


def test_class_masks_are_returned_to_original_branch_alignment():
    branch_map, _ = _targets()
    masks = {
        "artery": np.isin(branch_map, [1, 2, 3, 4]),
        "vein": np.isin(branch_map, [5, 6, 7, 8]),
        "aliased_artery": np.isin(branch_map, [9, 10, 11]),
    }

    labels = class_masks_to_branch_labels(branch_map, masks)

    np.testing.assert_array_equal(labels, [0] * 4 + [1] * 4 + [2] * 3 + [-1])


def test_custom_two_step_candidate_saves_visualization_and_arrays(tmp_path):
    branch_map, targets = _targets()
    X = np.repeat([[-3.0, 0.0], [0.0, 3.0], [3.0, 0.0]], 4, axis=0)
    expected = np.repeat([8, 3, 5], 4)
    # Deliberately disagree with the method-native masks: this verifies that
    # two-step artifacts preserve their exact pixel support instead of
    # reconstructing whole branches from the evaluation labels.
    semantic = np.zeros(12, dtype=int)
    method_masks = {
        "artery": np.isin(branch_map, [1, 2, 3, 4]),
        "vein": np.isin(branch_map, [5, 6, 7, 8]),
        "aliased_artery": np.isin(branch_map, [9, 10, 11, 12]),
    }
    time = np.arange(24)
    pulse = np.sin(2 * np.pi * time / 6)
    video = 10 + pulse[:, None, None] * branch_map[None, :, :]
    references = {}
    for class_label, class_name in enumerate(targets.class_names):
        references[class_name] = branch_map == (1 + 4 * class_label)

    candidate = BenchmarkCandidate(
        name="fourier_then_correlation",
        representation="two_step",
        X=None,
        run=lambda _X, **_: BenchmarkPrediction(
            cluster_labels=expected,
            deployment_labels=semantic,
            embedding_views=(
                BenchmarkEmbeddingView(
                    "stage 1",
                    X,
                    np.repeat([0, 1, 1], 4),
                    masks=(
                        ("artery", np.isin(branch_map, [1, 2, 3, 4])),
                        ("remaining candidates", np.isin(branch_map, np.arange(5, 13))),
                    ),
                ),
                BenchmarkEmbeddingView(
                    "stage 2",
                    X[4:],
                    np.repeat([0, 1], 4),
                    component_names=("HF correlation", "LF correlation"),
                    partial_labels=targets.labels[4:],
                    masks=(
                        ("vein", np.isin(branch_map, [5, 6, 7, 8])),
                        ("aliased artery", np.isin(branch_map, [9, 10, 11, 12])),
                    ),
                ),
            ),
            class_masks=method_masks,
        ),
        subsample_safe=False,
    )
    output = tmp_path / "visualizations"
    archive_output = tmp_path / "clusters"
    result = run_single_sample_benchmark(
        {},
        targets,
        custom_candidates=(candidate,),
        cluster_counts=(),
        include_adaptive=False,
        labeled_vessels=branch_map,
        signal_videos={"HF": video, "M0": video, "LF": video},
        signal_reference_masks=references,
        sampling_frequency=100,
        beat_period=6,
        visualization_image=branch_map.astype(float),
        visualization_dir=output,
        cluster_archive_dir=archive_output,
        csv_path=tmp_path / "metrics.csv",
    )

    assert result.table.loc[0, "deployment_mapping"] == "method_assignment"
    artifact_directory = output / "two_step_fourier_then_correlation"
    assert (artifact_directory / "1st_step" / "clusters.png").is_file()
    assert (artifact_directory / "1st_step" / "masks.png").is_file()
    assert (artifact_directory / "2nd_step" / "clusters.png").is_file()
    assert (artifact_directory / "2nd_step" / "masks.png").is_file()
    assert (artifact_directory / "final_overlays.png").is_file()
    assert (artifact_directory / "signals.png").is_file()
    assert not (artifact_directory / "clustering.png").exists()
    assert not (artifact_directory / "diagnostic.png").exists()
    arrays = np.load(artifact_directory / "labels_and_masks.npz")
    np.testing.assert_array_equal(arrays["cluster_labels"], expected)
    np.testing.assert_array_equal(arrays["semantic_labels"], semantic)
    assert arrays["mask_artery"].sum() == 4
    assert arrays["mask_vein"].sum() == 4
    assert arrays["mask_aliased_artery"].sum() == 4
    archive = load_cluster_archive(
        archive_output / "two_step_fourier_then_correlation" / "clusters.npz"
    )
    for name, mask in method_masks.items():
        np.testing.assert_array_equal(archive["class_masks"][name], mask)


def test_three_real_embedding_components_are_not_projected():
    X = np.arange(15, dtype=float).reshape(5, 3)

    coordinates, names, use_3d = benchmark_module._embedding_projection(
        X, ("HF correlation", "M0 correlation", "LF correlation")
    )

    np.testing.assert_array_equal(coordinates, X)
    assert names == ("HF correlation", "M0 correlation", "LF correlation")
    assert use_3d


def test_adjacent_cluster_ids_use_distinct_categorical_colors():
    assert benchmark_module.CLUSTER_COLORS[0] == "tab:red"
    assert benchmark_module.CLUSTER_COLORS[1] == "tab:blue"


def test_custom_two_step_candidates_can_run_before_builtin_methods():
    _, targets = _targets()
    X = np.repeat([[-3.0], [0.0], [3.0]], 4, axis=0)
    candidate = BenchmarkCandidate(
        name="two_step_first",
        representation="two_step",
        X=None,
        run=lambda _X, **_: np.repeat([0, 1, 2], 4),
        subsample_safe=False,
    )

    result = run_single_sample_benchmark(
        {"one_step": X},
        targets,
        custom_candidates=(candidate,),
        custom_candidates_first=True,
        cluster_counts=(3,),
        include_adaptive=False,
        candidate_keys={"two_step/two_step_first", "one_step/kmeans_k3"},
    )

    assert result.table["representation"].tolist() == ["two_step", "one_step"]
