"""Event emitter — typed event system for Glassbox lifecycle events."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable

logger = logging.getLogger("glassbox")

Listener = Callable[[Any], None]


class GlassboxEmitter:
    """Simple observer pattern. Listener errors are silently suppressed."""

    def __init__(self) -> None:
        self._listeners: dict[str, list[Listener]] = defaultdict(list)

    def on(self, event: str, listener: Listener) -> Callable[[], None]:
        """Subscribe to an event. Returns an unsubscribe function."""
        self._listeners[event].append(listener)

        def unsubscribe() -> None:
            try:
                self._listeners[event].remove(listener)
            except ValueError:
                pass

        return unsubscribe

    def off(self, event: str, listener: Listener) -> None:
        try:
            self._listeners[event].remove(listener)
        except ValueError:
            pass

    def emit(self, event: str, payload: Any) -> None:
        for listener in self._listeners.get(event, []):
            try:
                listener(payload)
            except Exception:
                # Listener errors never propagate to the caller's hot path
                logger.debug("Listener error on %s", event, exc_info=True)

    def remove_all_listeners(self) -> None:
        self._listeners.clear()


# Global singleton — shared by all wrappers
emitter = GlassboxEmitter()
