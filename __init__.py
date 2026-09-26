"""Hermes plugin: Jev (TypeSafe System One) behind the tool-risk gate and the done-check.

Thin adapter — the decision logic lives in ``system_one_guard.py``, which also runs
standalone as a shell hook (see README). Hooks fail open: an exception is logged
and the agent proceeds.

``pre_llm_call`` is registered for one reason: it is the only hook that carries the turn's
user message, and the risk gate needs it. It makes no Jev call. A previous version asked
Jev for a route, a lane, a complexity, and a model tier here; the measured plan text was
ignored 4 times out of 4, and a mid-conversation model swap breaks the prompt cache, so
all of that is gone.
"""
from __future__ import annotations

import logging

from . import system_one_guard

logger = logging.getLogger(__name__)

_EVENTS = ("pre_llm_call", "pre_tool_call", "pre_verify")
_TOP_LEVEL = ("tool_name", "args", "session_id", "cwd", "profile")


def _as_payload(event: str, kwargs: dict) -> dict:
    """Map plugin-hook kwargs onto the payload shape system_one_guard.handle() expects."""
    return {
        "hook_event_name": event,
        "tool_name": kwargs.get("tool_name"),
        "tool_input": kwargs.get("args") if isinstance(kwargs.get("args"), dict) else None,
        "session_id": kwargs.get("session_id") or "",
        "cwd": "",
        "profile": "",
        "extra": {k: v for k, v in kwargs.items() if k not in _TOP_LEVEL},
    }


def _make_hook(event: str):
    def hook(**kwargs):
        try:
            return system_one_guard.handle(_as_payload(event, kwargs)) or None
        except Exception:  # fail open, same policy as the standalone script
            logger.warning("jev-guard: %s failed open", event, exc_info=True)
            return None

    hook.__name__ = f"system_one_guard_{event}"
    return hook


def register(ctx):
    """Resolve settings, then register the two gate hooks and the message recorder."""
    try:  # profile-scoped flow log; older loaders without ctx.state keep the default path
        log_path = str(ctx.state.data_dir / "jev-flow.jsonl")
    except AttributeError:
        log_path = system_one_guard.LOG_PATH
    system_one_guard.configure(
        log_path=log_path,
        timeout=ctx.get_config("timeout", default=system_one_guard.TIMEOUT),
        approve_at=ctx.get_config("approve_at", default=system_one_guard.APPROVE_AT),
        block_at=ctx.get_config("block_at", default=system_one_guard.URGENT_AT),
        verify_at=ctx.get_config("verify_at", default=system_one_guard.VERIFY_AT),
        code_chars=ctx.get_config("code_chars", default=system_one_guard.CODE_CHARS),
        max_state_chars=ctx.get_config("max_state_chars", default=system_one_guard.MAX_STATE_CHARS),
    )
    for event in _EVENTS:
        ctx.register_hook(event, _make_hook(event))
