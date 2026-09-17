from __future__ import annotations

import h5py
import numpy as np
import pytest

from dopplerview.pipeline.pipeline import Context


class DopplerViewFolderDouble:
    measure_name = "measure"

    def __init__(self, h5_path):
        self.h5_path = h5_path

    def get_h5_path(self):
        return self.h5_path


def read_value(path):
    with h5py.File(path, "r") as h5:
        return h5["value"][()]


def test_setting_folder_does_not_truncate_previous_result(
    bare_output_manager_factory, tmp_path
):
    final_path = tmp_path / "measure_DV.h5"
    with h5py.File(final_path, "w") as h5:
        h5.create_dataset("value", data=1)

    manager = bare_output_manager_factory()
    manager.set_DV_folder(DopplerViewFolderDouble(final_path))

    assert read_value(final_path) == 1


def test_failed_run_discards_temporary_h5_and_preserves_previous_result(
    bare_output_manager_factory, tmp_path
):
    final_path = tmp_path / "measure_DV.h5"
    with h5py.File(final_path, "w") as h5:
        h5.create_dataset("value", data=1)
    manager = bare_output_manager_factory()
    manager.set_DV_folder(DopplerViewFolderDouble(final_path))

    manager.begin_run()
    temporary_path = manager.temporary_h5_path
    with h5py.File(manager.h5_path, "a") as h5:
        h5.create_dataset("value", data=2)
    manager.abort_run()

    assert read_value(final_path) == 1
    assert not temporary_path.exists()


def test_successful_run_atomically_replaces_previous_result(
    bare_output_manager_factory, tmp_path
):
    final_path = tmp_path / "measure_DV.h5"
    with h5py.File(final_path, "w") as h5:
        h5.create_dataset("value", data=1)
    manager = bare_output_manager_factory()
    manager.set_DV_folder(DopplerViewFolderDouble(final_path))

    manager.begin_run()
    temporary_path = manager.temporary_h5_path
    with h5py.File(manager.h5_path, "a") as h5:
        h5.create_dataset("value", data=2)
    manager.commit_run()

    assert read_value(final_path) == 2
    assert not temporary_path.exists()
    assert manager.h5_path == final_path


def test_enqueued_output_is_an_immutable_snapshot(bare_output_manager_factory):
    manager = bare_output_manager_factory()
    manager.running = True
    original = np.array([1, 2, 3])

    class Context:
        def get(self, key):
            return original

        def has(self, key):
            return True

    manager.save_async("step", "artifact", Context())
    payload = manager.output_queue.get_nowait()
    original[0] = 99

    np.testing.assert_array_equal(payload.value, [1, 2, 3])
    assert payload.value.flags.writeable is False
    with pytest.raises(ValueError):
        payload.value[0] = 10


def test_async_writer_failure_aborts_publication_and_preserves_previous_result(
    bare_output_manager_factory, tmp_path
):
    final_path = tmp_path / "measure_DV.h5"
    with h5py.File(final_path, "w") as h5:
        h5.create_dataset("value", data=1)
    manager = bare_output_manager_factory()
    manager.schema = {"artifact": "value"}
    manager.output_enabled = False
    manager.set_DV_folder(DopplerViewFolderDouble(final_path))
    manager.begin_run()
    manager.start()

    class UnsupportedValue:
        pass

    class PayloadContext:
        def get(self, key):
            return UnsupportedValue()

        def has(self, key):
            return True

    manager.save_async("step", "artifact", PayloadContext())
    ctx = Context(output_manager=manager)

    with pytest.raises(RuntimeError, match="persistence failed"):
        ctx.finish_output_manager(success=True)

    assert read_value(final_path) == 1
    assert manager.temporary_h5_path is None
