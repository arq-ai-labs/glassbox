"""Tests for GlassboxEmitter — event system with error isolation."""

import pytest

from glassbox.core.emitter import GlassboxEmitter


class TestGlassboxEmitter:
    def test_on_and_emit(self):
        emitter = GlassboxEmitter()
        received = []
        emitter.on("test", lambda payload: received.append(payload))
        emitter.emit("test", "hello")
        assert received == ["hello"]

    def test_multiple_listeners(self):
        emitter = GlassboxEmitter()
        results = []
        emitter.on("test", lambda p: results.append(f"a:{p}"))
        emitter.on("test", lambda p: results.append(f"b:{p}"))
        emitter.emit("test", "data")
        assert results == ["a:data", "b:data"]

    def test_different_events(self):
        emitter = GlassboxEmitter()
        a_results = []
        b_results = []
        emitter.on("event_a", lambda p: a_results.append(p))
        emitter.on("event_b", lambda p: b_results.append(p))

        emitter.emit("event_a", "for_a")
        emitter.emit("event_b", "for_b")

        assert a_results == ["for_a"]
        assert b_results == ["for_b"]

    def test_unsubscribe_function(self):
        emitter = GlassboxEmitter()
        received = []
        unsub = emitter.on("test", lambda p: received.append(p))

        emitter.emit("test", 1)
        unsub()
        emitter.emit("test", 2)

        assert received == [1]  # 2 should not appear

    def test_off(self):
        emitter = GlassboxEmitter()
        received = []
        listener = lambda p: received.append(p)
        emitter.on("test", listener)
        emitter.off("test", listener)
        emitter.emit("test", "data")
        assert received == []

    def test_off_nonexistent_listener(self):
        """off() with unknown listener should not raise."""
        emitter = GlassboxEmitter()
        emitter.off("test", lambda p: None)  # should not raise

    def test_listener_error_suppressed(self):
        """Listener errors should not propagate to the caller."""
        emitter = GlassboxEmitter()
        results = []

        def bad_listener(p):
            raise ValueError("boom")

        def good_listener(p):
            results.append(p)

        emitter.on("test", bad_listener)
        emitter.on("test", good_listener)

        # Should not raise, and good_listener should still fire
        emitter.emit("test", "data")
        assert results == ["data"]

    def test_emit_no_listeners(self):
        """Emitting to an event with no listeners should not raise."""
        emitter = GlassboxEmitter()
        emitter.emit("nonexistent", "data")  # should not raise

    def test_remove_all_listeners(self):
        emitter = GlassboxEmitter()
        received = []
        emitter.on("a", lambda p: received.append(p))
        emitter.on("b", lambda p: received.append(p))

        emitter.remove_all_listeners()

        emitter.emit("a", "data")
        emitter.emit("b", "data")
        assert received == []

    def test_double_unsubscribe(self):
        """Calling unsubscribe twice should not raise."""
        emitter = GlassboxEmitter()
        unsub = emitter.on("test", lambda p: None)
        unsub()
        unsub()  # should not raise

    def test_complex_payload(self):
        """Payloads can be any type."""
        emitter = GlassboxEmitter()
        received = []
        emitter.on("test", lambda p: received.append(p))

        emitter.emit("test", {"key": "value", "nested": [1, 2, 3]})
        assert received[0]["key"] == "value"
        assert received[0]["nested"] == [1, 2, 3]
