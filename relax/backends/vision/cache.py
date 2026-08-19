# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Memory-bounded cache primitives for frozen vision features."""

from collections import OrderedDict
from typing import Any, Hashable


class ByteBoundedLRUCache:
    """Least-recently-used cache bounded by caller-reported resident bytes."""

    def __init__(self, max_bytes: int) -> None:
        if max_bytes <= 0:
            raise ValueError(f"max_bytes must be positive, got {max_bytes}")
        self.max_bytes = max_bytes
        self._entries: OrderedDict[Hashable, tuple[Any, int]] = OrderedDict()
        self._resident_bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: Hashable) -> Any:
        """Return and refresh an entry, or ``None`` when it is absent."""
        entry = self._entries.get(key)
        if entry is None:
            self._misses += 1
            return None
        self._entries.move_to_end(key)
        self._hits += 1
        return entry[0]

    def put(self, key: Hashable, value: Any, *, size_bytes: int) -> None:
        """Insert an entry and evict least-recently-used values as necessary."""
        if size_bytes < 0:
            raise ValueError(f"size_bytes must be non-negative, got {size_bytes}")
        if size_bytes > self.max_bytes:
            return
        previous = self._entries.pop(key, None)
        if previous is not None:
            self._resident_bytes -= previous[1]

        self._entries[key] = (value, size_bytes)
        self._resident_bytes += size_bytes
        while self._resident_bytes > self.max_bytes and self._entries:
            _, (_, evicted_size) = self._entries.popitem(last=False)
            self._resident_bytes -= evicted_size
            self._evictions += 1

    @property
    def stats(self) -> dict[str, int]:
        """Return current occupancy and cumulative lookup/eviction counters."""
        return {
            "entries": len(self._entries),
            "resident_bytes": self._resident_bytes,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
        }
