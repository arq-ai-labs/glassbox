"""SDK wrappers — auto-detecting wrap() for OpenAI and Anthropic clients."""

from __future__ import annotations

from typing import Any, Optional

from .openai_wrapper import WrapOptions


def wrap(client: Any, **kwargs: Any) -> Any:
    """Wrap an LLM client to auto-capture ContextPacks.

    Detects whether the client is OpenAI or Anthropic and applies
    the appropriate wrapper.

    Args:
        client: An OpenAI or Anthropic client instance.
        **kwargs: Passed to WrapOptions (agent_name, app_name, storage_dir, etc.)

    Returns:
        The same client, monkey-patched with ContextPack capture.
    """
    options = WrapOptions(**kwargs) if kwargs else None

    # Anthropic: has client.messages.create
    if hasattr(client, "messages") and hasattr(getattr(client, "messages", None), "create"):
        from .anthropic_wrapper import wrap_anthropic
        return wrap_anthropic(client, options)

    # OpenAI: has client.chat.completions.create
    if (
        hasattr(client, "chat")
        and hasattr(getattr(client, "chat", None), "completions")
        and hasattr(getattr(getattr(client, "chat", None), "completions", None), "create")
    ):
        from .openai_wrapper import wrap_openai
        return wrap_openai(client, options)

    raise ValueError(
        f"Unrecognized client type: {type(client).__name__}. "
        "wrap() supports OpenAI and Anthropic clients. "
        "For LangGraph/LangChain, use observe() or GlassboxCallbackHandler instead."
    )


__all__ = ["wrap", "WrapOptions"]
