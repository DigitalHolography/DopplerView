import logging
import sys
import traceback
from pathlib import Path

from dopplerview.input_output.output_manager import OutputManager
from dopplerview.pipeline.pipeline import Pipeline

from dopplerview.ui.image_utils import build_step_preview

logger = logging.getLogger(__name__)


class _MultiprocessingQueueLogHandler(logging.Handler):
    """Forward log messages from the pipeline child process to the Tk parent process."""

    def __init__(self, queue_out):
        super().__init__()
        self.queue_out = queue_out

    def emit(self, record):
        try:
            self.queue_out.put((
                "log",
                {
                    "name": record.name,
                    "levelno": record.levelno,
                    "levelname": record.levelname,
                    "message": self.format(record),
                    "pathname": record.pathname,
                    "lineno": record.lineno,
                },
            ))
        except Exception:
            # Logging must never crash the worker.
            pass


class _QueueTextStream:
    """Redirect stdout/stderr lines from the child process to the parent logger."""

    def __init__(self, queue_out, *, levelno: int, name: str):
        self.queue_out = queue_out
        self.levelno = levelno
        self.name = name
        self._buffer = ""

    def write(self, text):
        if not text:
            return
        self._buffer += str(text)
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self.queue_out.put((
                    "log",
                    {
                        "name": self.name,
                        "levelno": self.levelno,
                        "levelname": logging.getLevelName(self.levelno),
                        "message": line,
                        "pathname": "",
                        "lineno": 0,
                    },
                ))

    def flush(self):
        if self._buffer.strip():
            self.queue_out.put((
                "log",
                {
                    "name": self.name,
                    "levelno": self.levelno,
                    "levelname": logging.getLevelName(self.levelno),
                    "message": self._buffer.strip(),
                    "pathname": "",
                    "lineno": 0,
                },
            ))
        self._buffer = ""


def _configure_child_process_logging(queue_out):
    """
    Configure logging in the spawned pipeline process.

    On Windows, multiprocessing uses spawn: the child process starts with a fresh
    interpreter and does not reliably inherit the parent logging handlers.  This
    handler forwards child logs to the parent process through the same event queue
    used for pipeline progress.
    """
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.DEBUG)

    handler = _MultiprocessingQueueLogHandler(queue_out)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(handler)

    sys.stdout = _QueueTextStream(queue_out, levelno=logging.INFO, name="pipeline.stdout")
    sys.stderr = _QueueTextStream(queue_out, levelno=logging.ERROR, name="pipeline.stderr")

def pipeline_process_worker(run_spec, queue_out):
    """
    Run the heavy DopplerView pipeline in a child process.

    The Tk process must remain UI-only.  This isolates native crashes/hangs from
    OpenCV, ONNXRuntime, PyTorch, h5py, etc. from Tkinter's event loop.
    """
    _configure_child_process_logging(queue_out)

    try:
        h5_schema_path = run_spec["h5_schema_path"]
        output_config_path = run_spec["output_config_path"]
        models_config_path = run_spec["models_config_path"]
        dopplerview_config_path = run_spec.get("dopplerview_config_path")
        input_list = [Path(p) for p in run_spec["input_list"]]
        targets = run_spec.get("steps")
        selected_models = run_spec.get("selected_models", {})
        config_mode = run_spec.get("config_mode", "default")
        output_enabled = bool(run_spec.get("output_enabled", False))

        output_manager = OutputManager(
            h5_schema_path,
            output_config_path,
            output_enabled=output_enabled,
        )
        pipeline = Pipeline(output_manager=output_manager)

        pipeline.load_model_registry(models_config_path)
        if dopplerview_config_path and Path(dopplerview_config_path).exists():
            pipeline.load_dopplerview_config(dopplerview_config_path)

        try:
            pipeline.set_config_mode(config_mode)
        except Exception:
            logger.exception("Failed to set pipeline config mode in worker")

        for task_name, model_name in selected_models.items():
            if model_name:
                try:
                    pipeline.ctx.change_model_for_task(task_name, model_name)
                except Exception:
                    logger.exception("Failed to select model %s for task %s", model_name, task_name)

        pipeline.ctx.clear_input_list()
        pipeline.load_input_list_from_list(input_list)

        def callback(event, *args):
            queue_out.put((event, args))

            if event == "step_done" and args:
                step_name = args[0]
                preview = build_step_preview(pipeline, step_name)
                if preview is not None:
                    queue_out.put(("preview_image", (preview,)))

        pipeline.run_batch(targets=targets, callback=callback)
        queue_out.put(("worker_done", None))

    except BaseException:
        queue_out.put(("error", traceback.format_exc()))
