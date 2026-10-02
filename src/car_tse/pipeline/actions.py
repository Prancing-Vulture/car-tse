"""Stub vehicle actions and spoken replies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ActionResult:
    intent_id: str
    action: str
    reply_wav: Path | None = None


def intent_to_action(intent_id: str, label_map: dict[str, str] | None = None) -> ActionResult:
    label_map = label_map or {}
    action = label_map.get(intent_id, f"noop_{intent_id}")
    reply = None
    if action.startswith("reply:"):
        parts = action.split(":", 2)
        if len(parts) >= 3:
            reply = Path(parts[2])
    return ActionResult(intent_id=intent_id, action=action, reply_wav=reply)
