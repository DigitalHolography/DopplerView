"""Cooperative cancellation shared by pipeline operations."""

import threading


class OperationCancelled(RuntimeError):
    pass


class CancellationToken:
    def __init__(self):
        self._event = threading.Event()

    @property
    def cancelled(self):
        return self._event.is_set()

    def cancel(self):
        self._event.set()

    def reset(self):
        self._event.clear()

    def check(self):
        if self.cancelled:
            raise OperationCancelled("Pipeline operation was cancelled.")
