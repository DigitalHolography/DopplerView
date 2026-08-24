from __future__ import annotations

import logging

from dopplerview.input_output import log_config


def test_setup_logging_separates_readable_and_debug_logs(tmp_path, monkeypatch):
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    monkeypatch.setattr(log_config, "get_log_dir", lambda: tmp_path)

    try:
        log_config.setup_logging()
        logging.getLogger("test.instrumentation").debug("detailed measurement")
        logging.getLogger("test.application").info("readable message")

        for handler in root.handlers:
            handler.flush()

        readable = log_config.get_log_file().read_text(encoding="utf-8")
        debug = log_config.get_debug_log_file().read_text(encoding="utf-8")

        assert "readable message" in readable
        assert "detailed measurement" not in readable
        assert "readable message" in debug
        assert "detailed measurement" in debug
    finally:
        for handler in root.handlers:
            handler.close()
        root.handlers[:] = original_handlers
        root.setLevel(original_level)
