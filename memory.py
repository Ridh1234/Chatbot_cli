"""Conversation memory management with a sliding window."""
from __future__ import annotations
from collections import deque
from typing import Deque, Dict, List

class SlidingWindowMemory:
    def __init__(self, max_turns: int = 5):
        # Each turn: user + assistant messages stored as separate entries
        self.max_turns = max_turns * 2  # store both roles
        self.buffer: Deque[Dict[str, str]] = deque()

    def add(self, role: str, content: str):
        self.buffer.append({"role": role, "content": content})
        while len(self.buffer) > self.max_turns:
            self.buffer.popleft()

    def get(self) -> List[Dict[str, str]]:
        return list(self.buffer)

    def clear(self):
        self.buffer.clear()
