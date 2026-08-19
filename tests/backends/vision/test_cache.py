# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib


def test_byte_bounded_lru_cache_evicts_least_recently_used_entry():
    cache_module = importlib.import_module("relax.backends.vision.cache")
    cache = cache_module.ByteBoundedLRUCache(max_bytes=8)

    cache.put("a", "first", size_bytes=4)
    cache.put("b", "second", size_bytes=4)
    assert cache.get("a") == "first"

    cache.put("c", "third", size_bytes=4)

    assert cache.get("b") is None
    assert cache.get("a") == "first"
    assert cache.get("c") == "third"
    assert cache.stats == {
        "entries": 2,
        "resident_bytes": 8,
        "hits": 3,
        "misses": 1,
        "evictions": 1,
    }


def test_byte_bounded_lru_cache_bypasses_oversized_entry_without_eviction():
    cache_module = importlib.import_module("relax.backends.vision.cache")
    cache = cache_module.ByteBoundedLRUCache(max_bytes=8)

    cache.put("oversized", "value", size_bytes=9)

    assert cache.stats["entries"] == 0
    assert cache.stats["resident_bytes"] == 0
    assert cache.stats["evictions"] == 0
    assert cache.get("oversized") is None
