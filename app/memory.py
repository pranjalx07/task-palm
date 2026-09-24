import json
from typing import Any

import redis

from app.config import Settings


class ConversationMemory:
    def __init__(self, settings: Settings) -> None:
        self.client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        self.ttl_seconds = 60 * 60 * 24

    def get_messages(self, session_id: str) -> list[dict[str, str]]:
        value = self.client.get(f"chat:{session_id}")
        return json.loads(value) if value else []

    def append(self, session_id: str, role: str, content: str) -> None:
        messages = self.get_messages(session_id)
        messages.append({"role": role, "content": content})
        self.client.setex(f"chat:{session_id}", self.ttl_seconds, json.dumps(messages[-20:]))

    def get_booking(self, session_id: str) -> dict[str, str]:
        value = self.client.get(f"booking:{session_id}")
        return json.loads(value) if value else {}

    def set_booking(self, session_id: str, fields: dict[str, str]) -> None:
        self.client.setex(f"booking:{session_id}", self.ttl_seconds, json.dumps(fields))

    def clear_booking(self, session_id: str) -> None:
        self.client.delete(f"booking:{session_id}")


def build_memory(settings: Settings) -> ConversationMemory:
    return ConversationMemory(settings)