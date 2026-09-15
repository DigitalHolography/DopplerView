from argparse import Namespace

import h5py
import numpy as np
from PIL import Image

from sandbox.run_choroid_global_benchmark import (
    discover_measure,
    load_measure_arrays,
    run_global_benchmark,
)


def _write_mask(path, shape=(8, 10)):
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[2:6, 3:8] = 255
    Image.fromarray(mask).save(path)


def _write_measure(folder, *, nested_h5=False, include_choroid=True):
    folder.mkdir()
    masks = folder / "manual"
    for name in ("retina_artery_mask.png", "retina_vein_mask.png"):
        _write_mask(masks / name)
    if include_choroid:
        for name in (
            "choroid_artery_mask.png",
            "choroid_vein_mask.png",
            "choroid_aliased_artery_mask.png",
        ):
            _write_mask(masks / name)
    h5_path = (
        folder / f"{folder.name}_DV" / "h5" / f"{folder.name}_DV.h5"
        if nested_h5
        else folder / f"{folder.name}.h5"
    )
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "w") as h5:
        signals = h5.create_group("doppler_signal")
        for name in ("M0_ff", "HF_M0_ff", "LF_M0_ff"):
            signals.create_dataset(name, data=np.zeros((20, 8, 10)))
    return h5_path


def test_discovery_supports_documented_flat_layout_and_loads_arrays(tmp_path):
    folder = tmp_path / "measure_a"
    h5_path = _write_measure(folder)

    files, reason = discover_measure(folder)
    videos, masks, candidate_mask = load_measure_arrays(files)

    assert reason is None
    assert files.h5_path == h5_path
    assert videos["M0"].shape == (20, 8, 10)
    assert set(masks) == {
        "retina_artery",
        "retina_vein",
        "artery",
        "vein",
        "aliased_artery",
    }
    assert candidate_mask is None


def test_discovery_supports_nested_layout_and_skips_missing_choroid(tmp_path):
    eligible = tmp_path / "measure_nested"
    h5_path = _write_measure(eligible, nested_h5=True)
    skipped = tmp_path / "measure_without_choroid"
    _write_measure(skipped, include_choroid=False)

    files, reason = discover_measure(eligible)
    missing_files, missing_reason = discover_measure(skipped)

    assert reason is None
    assert files.h5_path == h5_path
    assert missing_files is None
    assert "missing choroid masks" in missing_reason


def test_global_dry_run_checkpoints_eligible_and_skipped_measures(tmp_path):
    input_folder = tmp_path / "input"
    input_folder.mkdir()
    _write_measure(input_folder / "eligible")
    _write_measure(input_folder / "skipped", include_choroid=False)
    output_folder = tmp_path / "output"
    args = Namespace(
        input_folder=str(input_folder),
        output_folder=str(output_folder),
        measure=None,
        dry_run=True,
        overwrite=False,
        fail_fast=False,
    )

    statuses = run_global_benchmark(args)

    assert dict(zip(statuses["measure"], statuses["status"])) == {
        "eligible": "eligible",
        "skipped": "skipped",
    }
    assert (output_folder / "global_measure_status.csv").is_file()
