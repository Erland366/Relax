# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib
import sys
import types
from argparse import Namespace
from unittest.mock import MagicMock

import pytest
import torch

from relax.backends.vision.qwen3_vl import Qwen3VLFrozenVisionFeatures
from tests.engine.rollout.test_precomputed_vision_live import _import_sglang_rollout


class _AwaitableValue:
    def __init__(self, value):
        self.value = value

    def __await__(self):
        yield
        return self.value


def _create_test_manager(monkeypatch, args):
    transfer_queue = types.ModuleType("transfer_queue")
    transfer_queue.init = lambda *args, **kwargs: None
    transfer_queue.get_client = lambda: None
    monkeypatch.setitem(sys.modules, "transfer_queue", transfer_queue)

    tracking_utils = types.ModuleType("relax.utils.tracking_utils")
    tracking_utils.init_tracking = lambda *args, **kwargs: None
    tracking_utils.flush_metrics = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "relax.utils.tracking_utils", tracking_utils)

    monkeypatch.delitem(sys.modules, "relax.distributed.ray.rollout", raising=False)
    rollout_module = importlib.import_module("relax.distributed.ray.rollout")
    metadata = getattr(rollout_module.RolloutManager, "__ray_metadata__", None)
    manager_class = getattr(metadata, "modified_class", None) or rollout_module.RolloutManager
    manager = object.__new__(manager_class)
    manager.args = args
    engine = MagicMock()
    engine.get_url.remote.return_value = _AwaitableValue("http://engine-a:31000")
    manager.servers = {"default": Namespace(engines=[engine])}
    return manager


def _features(value: float) -> Qwen3VLFrozenVisionFeatures:
    grid = torch.tensor([[1, 2, 2]], dtype=torch.int64)
    final = torch.full((1, 2), value, dtype=torch.bfloat16)
    deepstack = tuple(torch.full((1, 2), value + index, dtype=torch.bfloat16) for index in (1, 2, 3))
    return Qwen3VLFrozenVisionFeatures(
        image_grid_thw=grid,
        vision_embeds=final,
        deepstack_visual_embeds=deepstack,
        feature_id=f"feature-{value:g}",
        vision_revision="vision-revision-test",
    )


def _metrics_snapshot(*, requests, hits, misses, entries, resident_bytes, evictions=0):
    return {
        "entries": entries,
        "resident_bytes": resident_bytes,
        "hits": hits,
        "misses": misses,
        "evictions": evictions,
        "encode_requests_total": requests,
        "backend_encode_requests_total": misses,
        "backend_encoded_images_total": misses,
        "emitted_feature_bytes_total": resident_bytes,
        "backend_encode_seconds_total": float(misses),
        "process_cpu_seconds_total": float(requests),
        "snapshot_monotonic_seconds": float(requests + 1),
        "rss_bytes": 1_024,
    }


def _encoder_response(features, *, requests, hits, misses, entries=None, evictions=0):
    return Namespace(
        features=features,
        replica_id="vision-replica-a",
        feature_id=features.feature_id,
        feature_schema_version=features.feature_schema_version,
        cache_hit=hits > 0,
        backend_encode_seconds=0.0 if hits else 1.0,
        backend_batch_size=0 if hits else 1,
        metrics_snapshot=_metrics_snapshot(
            requests=requests,
            hits=hits,
            misses=misses,
            entries=misses if entries is None else entries,
            resident_bytes=features.nbytes,
            evictions=evictions,
        ),
    )


class _RemoteEncoder:
    def __init__(self, responses):
        self._responses = iter(responses)
        self.calls = []

    async def remote(self, **kwargs):
        self.calls.append(kwargs)
        return next(self._responses)


def _manager_and_state(monkeypatch, *, samples, encoder_responses):
    args = Namespace(
        preload_vision_features=True,
        vision_encoder_device="cpu",
        vision_encoder_cache_max_bytes=4_096,
        sglang_vision_feature_cache_max_bytes=4_096,
        use_streaming_dataset=False,
        rollout_global_dataset=True,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30_000,
        vision_encoder_num_replicas=1,
    )
    manager = _create_test_manager(monkeypatch, args)
    sglang_rollout_module = _import_sglang_rollout(monkeypatch)
    data_source = MagicMock()
    data_source.snapshot_dataset_samples.remote.return_value = _AwaitableValue(samples)
    dataset_state = {
        "sample_offset": 0,
        "epoch_id": 0,
        "sample_group_index": 0,
        "sample_index": 0,
        "dataset_size": len(samples),
        "dataset_fingerprint": "a" * 64,
    }
    data_source.snapshot_dataset_state.remote.side_effect = [
        _AwaitableValue(dataset_state),
        _AwaitableValue(dataset_state),
    ]
    data_source.get_samples.remote.side_effect = AssertionError("preload must not consume training samples")
    manager.data_source = data_source
    manager.generate_rollout = MagicMock(side_effect=AssertionError("preload must not generate rollout samples"))

    remote_encoder = _RemoteEncoder(encoder_responses)
    state = Namespace(
        args=args,
        processor=object(),
        vision_encoder=Namespace(encode=remote_encoder),
        vision_encoder_cumulative_replica_snapshots={},
    )
    monkeypatch.setattr(sglang_rollout_module, "GenerateState", lambda unused_args: state)

    processor_calls = []

    async def fake_run_image_processor(state_arg, args_arg, prompt, multimodal_inputs):
        processor_calls.append((state_arg, args_arg, prompt, multimodal_inputs))
        return (
            [7, 99, 8],
            {
                "pixel_values": multimodal_inputs["pixel_values"],
                "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.int64),
            },
            0.25,
        )

    monkeypatch.setattr(sglang_rollout_module, "_run_image_processor", fake_run_image_processor)
    return manager, state, data_source, remote_encoder, processor_calls, sglang_rollout_module


@pytest.mark.asyncio
async def test_preload_snapshots_without_consuming_and_publishes_unique_identity_metrics(monkeypatch):
    shared_pixels = torch.ones((4, 2), dtype=torch.float32)
    samples = [
        Namespace(prompt="prompt-a", multimodal_inputs={"images": ["a"], "pixel_values": shared_pixels}),
        Namespace(prompt="prompt-a-duplicate", multimodal_inputs={"images": ["a"], "pixel_values": shared_pixels}),
    ]
    features = _features(1.0)
    manager, state, data_source, remote_encoder, processor_calls, sglang_rollout_module = _manager_and_state(
        monkeypatch,
        samples=samples,
        encoder_responses=[
            _encoder_response(features, requests=1, hits=0, misses=1),
            _encoder_response(features, requests=2, hits=1, misses=1),
        ],
    )
    cache_calls = []

    async def fake_post(url, payload, **kwargs):
        assert url == "http://engine-a:31000/relax/vision-features/cache"
        assert payload["feature_id"] == features.feature_id
        cache_calls.append(payload)
        return {
            "feature_id": features.feature_id,
            "vision_revision": features.vision_revision,
            "feature_schema_version": features.feature_schema_version,
            "image_grid_thw": [[1, 2, 2]],
            "cached_bytes": features.nbytes,
            "entries": 1,
            "resident_bytes": features.nbytes,
            "evictions": 0,
        }

    monkeypatch.setattr(sglang_rollout_module, "post", fake_post)

    metrics = await manager.preload_vision_features()

    identity = (features.feature_id, features.vision_revision, features.feature_schema_version)
    assert data_source.snapshot_dataset_samples.remote.call_count == 1
    assert data_source.snapshot_dataset_state.remote.call_count == 2
    data_source.get_samples.remote.assert_not_called()
    manager.generate_rollout.assert_not_called()
    assert len(processor_calls) == 2
    assert len(remote_encoder.calls) == 2
    assert len(cache_calls) == 1
    manager.rollout_engines[0].get_url.remote.assert_called_once_with()
    assert sglang_rollout_module._published_sglang_vision_features(state) == {identity}
    assert metrics["schema_version"] == 2
    assert metrics["dataset_samples"] == 2
    assert metrics["unique_features"] == 1
    assert metrics["duplicate_features"] == 1
    assert metrics["feature_ids"] == [features.feature_id]
    assert metrics["features"] == [
        {
            "feature_id": features.feature_id,
            "vision_revision": features.vision_revision,
            "feature_schema_version": features.feature_schema_version,
            "image_grid_thw": [[1, 2, 2]],
            "image_tokens": 1,
            "feature_bytes": features.nbytes,
        }
    ]
    assert metrics["work_counters"] == {
        "generation_requests": 0,
        "generated_samples": 0,
        "training_samples": 0,
    }
    assert metrics["dataset_state"] == {
        "before": {
            "sample_offset": 0,
            "epoch_id": 0,
            "sample_group_index": 0,
            "sample_index": 0,
            "dataset_size": 2,
            "dataset_fingerprint": "a" * 64,
        },
        "after": {
            "sample_offset": 0,
            "epoch_id": 0,
            "sample_group_index": 0,
            "sample_index": 0,
            "dataset_size": 2,
            "dataset_fingerprint": "a" * 64,
        },
        "unchanged": True,
    }
    for timing_key in (
        "wall_seconds",
        "image_processor_seconds",
        "encode_round_trip_seconds",
        "serialization_seconds",
        "cache_request_seconds",
    ):
        assert metrics[timing_key] >= 0
    assert metrics["cached_feature_bytes"] == features.nbytes
    assert metrics["cpu_cache"]["entries"] == 1
    assert metrics["cpu_cache"]["resident_bytes"] == features.nbytes
    assert metrics["cpu_cache"]["evictions"] == 0
    assert metrics["sglang_cache"] == {
        "entries": 1,
        "resident_bytes": features.nbytes,
        "evictions": 0,
        "stores": 1,
    }

    post_preload = await sglang_rollout_module._collect_cpu_vision_metrics(state, phase="rollout_0")
    assert post_preload["vision_encoder/requests_total"] == 2
    assert post_preload["vision_encoder/requests_interval"] == 0
    assert post_preload["vision_encoder/cache/hits_interval"] == 0
    assert post_preload["vision_encoder/cache/misses_interval"] == 0


@pytest.mark.asyncio
async def test_preload_publishes_each_identity_only_after_successful_cache(monkeypatch):
    first_features = _features(1.0)
    second_features = _features(9.0)
    samples = [
        Namespace(prompt="prompt-a", multimodal_inputs={"images": ["a"], "pixel_values": torch.ones((4, 2))}),
        Namespace(prompt="prompt-b", multimodal_inputs={"images": ["b"], "pixel_values": torch.zeros((4, 2))}),
    ]
    manager, state, data_source, _remote_encoder, _processor_calls, sglang_rollout_module = _manager_and_state(
        monkeypatch,
        samples=samples,
        encoder_responses=[
            _encoder_response(first_features, requests=1, hits=0, misses=1),
            _encoder_response(second_features, requests=2, hits=0, misses=2),
        ],
    )
    cache_count = 0

    async def fake_post(url, payload, **kwargs):
        nonlocal cache_count
        assert url == "http://engine-a:31000/relax/vision-features/cache"
        cache_count += 1
        if cache_count == 2:
            raise RuntimeError("SGLang cache failed for second feature")
        return {
            "feature_id": first_features.feature_id,
            "vision_revision": first_features.vision_revision,
            "feature_schema_version": first_features.feature_schema_version,
            "image_grid_thw": [[1, 2, 2]],
            "cached_bytes": first_features.nbytes,
            "entries": 1,
            "resident_bytes": first_features.nbytes,
            "evictions": 0,
        }

    monkeypatch.setattr(sglang_rollout_module, "post", fake_post)

    with pytest.raises(RuntimeError, match="SGLang cache failed for second feature"):
        await manager.preload_vision_features()

    first_identity = (
        first_features.feature_id,
        first_features.vision_revision,
        first_features.feature_schema_version,
    )
    second_identity = (
        second_features.feature_id,
        second_features.vision_revision,
        second_features.feature_schema_version,
    )
    assert sglang_rollout_module._published_sglang_vision_features(state) == {first_identity}
    assert second_identity not in sglang_rollout_module._published_sglang_vision_features(state)
    data_source.get_samples.remote.assert_not_called()
    manager.generate_rollout.assert_not_called()


@pytest.mark.parametrize(
    ("cpu_cache_entries", "cpu_cache_evictions", "sglang_cache_entries", "sglang_cache_evictions", "expected_error"),
    [
        pytest.param(2, 1, 2, 0, "CPU vision feature cache.*eviction", id="cpu-cache-eviction"),
        pytest.param(2, 0, 2, 1, "SGLang vision feature cache.*eviction", id="sglang-cache-eviction"),
        pytest.param(1, 0, 2, 0, "CPU vision feature cache entries.*2", id="cpu-cache-missing-entry"),
        pytest.param(2, 0, 1, 0, "SGLang vision feature cache entries.*2", id="sglang-cache-missing-entry"),
    ],
)
@pytest.mark.asyncio
async def test_preload_aborts_when_either_cache_does_not_retain_every_unique_feature(
    monkeypatch,
    cpu_cache_entries,
    cpu_cache_evictions,
    sglang_cache_entries,
    sglang_cache_evictions,
    expected_error,
):
    first_features = _features(1.0)
    second_features = _features(9.0)
    samples = [
        Namespace(prompt="prompt-a", multimodal_inputs={"images": ["a"], "pixel_values": torch.ones((4, 2))}),
        Namespace(prompt="prompt-b", multimodal_inputs={"images": ["b"], "pixel_values": torch.zeros((4, 2))}),
    ]
    manager, _state, _data_source, _remote_encoder, _processor_calls, sglang_rollout_module = _manager_and_state(
        monkeypatch,
        samples=samples,
        encoder_responses=[
            _encoder_response(first_features, requests=1, hits=0, misses=1),
            _encoder_response(
                second_features,
                requests=2,
                hits=0,
                misses=2,
                entries=cpu_cache_entries,
                evictions=cpu_cache_evictions,
            ),
        ],
    )
    cache_count = 0

    async def fake_post(url, payload, **kwargs):
        nonlocal cache_count
        cache_count += 1
        features = first_features if cache_count == 1 else second_features
        return {
            "feature_id": features.feature_id,
            "vision_revision": features.vision_revision,
            "feature_schema_version": features.feature_schema_version,
            "image_grid_thw": [[1, 2, 2]],
            "cached_bytes": features.nbytes,
            "entries": 1 if cache_count == 1 else sglang_cache_entries,
            "resident_bytes": features.nbytes * (1 if cache_count == 1 else sglang_cache_entries),
            "evictions": 0 if cache_count == 1 else sglang_cache_evictions,
        }

    monkeypatch.setattr(sglang_rollout_module, "post", fake_post)

    with pytest.raises(RuntimeError, match=expected_error):
        await manager.preload_vision_features()


@pytest.mark.parametrize("engine_count", [0, 2])
@pytest.mark.asyncio
async def test_preload_requires_exactly_one_rollout_engine(monkeypatch, engine_count):
    manager, _state, _data_source, _remote_encoder, _processor_calls, _rollout_module = _manager_and_state(
        monkeypatch,
        samples=[],
        encoder_responses=[],
    )
    engines = []
    for index in range(engine_count):
        engine = MagicMock()
        engine.get_url.remote.return_value = _AwaitableValue(f"http://engine-{index}:31000")
        engines.append(engine)
    manager.servers = {"default": Namespace(engines=engines)}

    with pytest.raises(RuntimeError, match=rf"one rollout engine.*found {engine_count}"):
        await manager.preload_vision_features()


@pytest.mark.parametrize("engine_url", [None, ""])
@pytest.mark.asyncio
async def test_preload_requires_nonempty_rollout_engine_url(monkeypatch, engine_url):
    manager, _state, _data_source, _remote_encoder, _processor_calls, _rollout_module = _manager_and_state(
        monkeypatch,
        samples=[],
        encoder_responses=[],
    )
    manager.rollout_engines[0].get_url.remote.return_value = _AwaitableValue(engine_url)

    with pytest.raises(RuntimeError, match="rollout engine URL"):
        await manager.preload_vision_features()


@pytest.mark.parametrize(
    ("response_override", "expected_error"),
    [
        pytest.param({"feature_id": "wrong-feature"}, "feature_id", id="feature-id"),
        pytest.param({"vision_revision": "wrong-revision"}, "vision_revision", id="vision-revision"),
        pytest.param(
            {"feature_schema_version": "wrong-schema"},
            "feature_schema_version",
            id="schema-version",
        ),
        pytest.param({"image_grid_thw": [[1, 3, 2]]}, "image_grid_thw", id="image-grid"),
    ],
)
@pytest.mark.asyncio
async def test_preload_rejects_cache_response_identity_mismatch(
    monkeypatch,
    response_override,
    expected_error,
):
    features = _features(1.0)
    sample = Namespace(
        prompt="prompt-a",
        multimodal_inputs={"images": ["a"], "pixel_values": torch.ones((4, 2))},
    )
    manager, state, _data_source, _remote_encoder, _processor_calls, sglang_rollout_module = _manager_and_state(
        monkeypatch,
        samples=[sample],
        encoder_responses=[_encoder_response(features, requests=1, hits=0, misses=1)],
    )
    cache = {
        "feature_id": features.feature_id,
        "vision_revision": features.vision_revision,
        "feature_schema_version": features.feature_schema_version,
        "image_grid_thw": [[1, 2, 2]],
        "cached_bytes": features.nbytes,
        "entries": 1,
        "resident_bytes": features.nbytes,
        "evictions": 0,
        **response_override,
    }

    async def fake_post(url, payload, **kwargs):
        return cache

    monkeypatch.setattr(sglang_rollout_module, "post", fake_post)

    with pytest.raises(RuntimeError, match=rf"cache response.*{expected_error}"):
        await manager.preload_vision_features()

    assert sglang_rollout_module._published_sglang_vision_features(state) == set()


@pytest.mark.asyncio
async def test_preload_aborts_when_dataset_state_changes(monkeypatch):
    features = _features(1.0)
    sample = Namespace(
        prompt="prompt-a",
        multimodal_inputs={"images": ["a"], "pixel_values": torch.ones((4, 2))},
    )
    manager, _state, data_source, _remote_encoder, _processor_calls, sglang_rollout_module = _manager_and_state(
        monkeypatch,
        samples=[sample],
        encoder_responses=[_encoder_response(features, requests=1, hits=0, misses=1)],
    )
    before = {
        "sample_offset": 0,
        "epoch_id": 0,
        "sample_group_index": 0,
        "sample_index": 0,
        "dataset_size": 1,
        "dataset_fingerprint": "a" * 64,
    }
    after = {**before, "sample_offset": 1}
    data_source.snapshot_dataset_state.remote.side_effect = [
        _AwaitableValue(before),
        _AwaitableValue(after),
    ]

    async def fake_post(url, payload, **kwargs):
        return {
            "feature_id": features.feature_id,
            "vision_revision": features.vision_revision,
            "feature_schema_version": features.feature_schema_version,
            "image_grid_thw": [[1, 2, 2]],
            "cached_bytes": features.nbytes,
            "entries": 1,
            "resident_bytes": features.nbytes,
            "evictions": 0,
        }

    monkeypatch.setattr(sglang_rollout_module, "post", fake_post)

    with pytest.raises(RuntimeError, match="dataset state changed while preloading vision features"):
        await manager.preload_vision_features()
