from __future__ import annotations

from threading import Event, RLock


class RunControlRegistry:
    def __init__(self):
        self._events: dict[str, Event] = {}
        self._lock = RLock()

    def _event(self, run_id: str) -> Event:
        with self._lock:
            return self._events.setdefault(run_id, Event())

    def request_stop(self, run_id: str) -> None:
        self._event(run_id).set()

    def clear_stop(self, run_id: str) -> None:
        self._event(run_id).clear()

    def is_stop_requested(self, run_id: str) -> bool:
        return self._event(run_id).is_set()

    def discard(self, run_id: str) -> None:
        with self._lock:
            self._events.pop(run_id, None)
