"""Async context propagation — Python's equivalent of Node.js AsyncLocalStorage."""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Callable, Optional, TypeVar

if TYPE_CHECKING:
    from .run import GlassboxRun

T = TypeVar("T")

_run_var: ContextVar[Optional["GlassboxRun"]] = ContextVar("glassbox_run", default=None)


def get_current_run() -> Optional["GlassboxRun"]:
    """Get the active GlassboxRun in the current async/thread context."""
    return _run_var.get()


def with_run(run: "GlassboxRun", fn: Callable[[], T]) -> T:
    """Execute fn with the given run as the active context."""
    token = _run_var.set(run)
    try:
        return fn()
    finally:
        _run_var.reset(token)
