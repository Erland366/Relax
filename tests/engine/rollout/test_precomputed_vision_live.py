# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib
import json
import sys
import types
from argparse import Namespace

import pytest
import torch


def _stub_module(monkeypatch, name: str, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _import_sglang_rollout(monkeypatch):
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    _stub_module(monkeypatch, "sglang_router", __version__="0.2.2")
    _stub_module(monkeypatch, "relax.distributed.ray.rollout", _log_rollout_data=lambda *args, **kwargs: None)
    _stub_module(
        monkeypatch,
        "relax.engine.filters.base_types",
        MetricGatherer=object,
        call_dynamic_filter=lambda *args, **kwargs: None,
    )
    _stub_module(
        monkeypatch,
        "relax.engine.rewards",
        async_rm=lambda *args, **kwargs: None,
        batched_async_rm=lambda *args, **kwargs: None,
    )
    _stub_module(
        monkeypatch,
        "relax.engine.rollout.base_types",
        RolloutFnEvalOutput=object,
        RolloutFnTrainOutput=object,
    )
    _stub_module(
        monkeypatch,
        "relax.utils.data.data",
        Dataset=object,
    )
    _stub_module(
        monkeypatch,
        "relax.utils.data.processing_utils",
        _ENCODE_EXECUTOR=object(),
        async_encode_audio_for_rollout_engine=lambda value: value,
        async_encode_image_for_rollout_engine=lambda value: value,
        async_encode_video_tensor_for_rollout_engine=lambda value: value,
        load_processor=lambda *args, **kwargs: None,
        load_tokenizer=lambda *args, **kwargs: None,
        sanitize_kimi_k25_response_tokens=lambda processor, tokens: tokens,
    )
    _stub_module(
        monkeypatch,
        "relax.utils.data.processor_pool",
        ProcessorPool=object,
        prepare_mm_inputs_for_ipc=lambda value: value,
        process_sample_in_worker=lambda *args, **kwargs: None,
    )
    _stub_module(
        monkeypatch,
        "relax.utils.http_utils",
        get=lambda *args, **kwargs: None,
        post=lambda *args, **kwargs: None,
        is_port_available=lambda *args, **kwargs: True,
    )
    _stub_module(
        monkeypatch,
        "relax.utils.profile_utils",
        start_sglang_profile=lambda *args, **kwargs: None,
        stop_sglang_profile=lambda *args, **kwargs: None,
    )
    _stub_module(monkeypatch, "relax.utils.timer", Timer=object)
    _stub_module(monkeypatch, "relax.utils.training.eval_config", EvalDatasetConfig=object)
    _stub_module(
        monkeypatch,
        "relax.utils.training.train_dump_utils",
        save_debug_rollout_data=lambda *args, **kwargs: None,
    )
    _stub_module(
        monkeypatch,
        "relax.utils.utils",
        CURRENT_ROLLOUT_BATCH=None,
        transfer_batch_to_data_system=lambda *args, **kwargs: None,
    )
    monkeypatch.delitem(sys.modules, "relax.engine.rollout.sglang_rollout", raising=False)
    return importlib.import_module("relax.engine.rollout.sglang_rollout")


@pytest.mark.asyncio
async def test_generate_uses_vision_handle_and_sends_json_precomputed_image_data(monkeypatch):
    rollout_module = _import_sglang_rollout(monkeypatch)
    vision_module = importlib.import_module("relax.backends.vision.qwen3_vl")
    grid = torch.tensor([[1, 4, 4]], dtype=torch.int64)
    final = torch.ones((4, 2), dtype=torch.bfloat16)
    deepstack = tuple(torch.full((4, 2), value, dtype=torch.bfloat16) for value in (2.0, 3.0, 4.0))
    features = vision_module.Qwen3VLFrozenVisionFeatures(
        image_grid_thw=grid,
        vision_embeds=final,
        deepstack_visual_embeds=deepstack,
    )
    encode_calls = []
    posted = {}

    class RemoteEncode:
        async def remote(self, **kwargs):
            encode_calls.append(kwargs)
            return features

    class VisionHandle:
        encode = RemoteEncode()

    class Tokenizer:
        image_token_id = 99
        pad_token_id = 0

        def encode(self, text, add_special_tokens=False):
            return [10, 11]

    state = Namespace(
        tokenizer=Tokenizer(),
        processor=object(),
        vision_encoder=VisionHandle(),
    )
    monkeypatch.setattr(rollout_module, "GenerateState", lambda args: state)

    async def fake_image_processor(state, args, prompt, multimodal_inputs):
        return [7, 99, 99, 99, 99, 8], {"pixel_values": torch.ones((16, 2)), "image_grid_thw": grid}, 0.01

    async def reject_raw_media(multimodal_inputs):
        raise AssertionError("raw media encoding must not run when the CPU vision encoder is enabled")

    async def fake_post(url, payload, headers=None):
        json.dumps(payload)
        posted.update(payload)
        return {
            "text": "",
            "meta_info": {
                "output_token_logprobs": [],
                "finish_reason": {"type": "stop"},
                "prompt_tokens": 6,
                "cached_tokens": 0,
            },
        }

    monkeypatch.setattr(rollout_module, "_run_image_processor", fake_image_processor)
    monkeypatch.setattr(rollout_module, "_encode_multimodal_inputs", reject_raw_media)
    monkeypatch.setattr(rollout_module, "post", fake_post)

    args = Namespace(
        ci_test=False,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        use_rollout_routing_replay=False,
        use_audio_in_video=False,
        sglang_router_policy="round_robin",
        use_slime_router=False,
        slime_router_middleware_paths=[],
        partial_rollout=False,
        mask_offpolicy_in_partial_rollout=False,
        sglang_speculative_algorithm=None,
        vision_encoder_backend="pytorch",
    )
    sample = rollout_module.Sample(
        prompt="look",
        multimodal_inputs={"images": ["raw-image"], "videos": [], "audio": []},
    )

    await rollout_module.generate(args, sample, {"max_new_tokens": 2})

    assert len(encode_calls) == 1
    assert encode_calls[0]["image_grid_thw"] is grid
    assert posted["input_ids"] == [7, 99, 99, 99, 99, 8]
    assert posted["image_data"][0]["format"] == "precomputed_embedding"
    assert posted["image_data"][0]["feature"] == torch.cat((final, *deepstack), dim=-1).tolist()
    assert posted["image_data"][0]["image_grid_thw"] == grid.tolist()
    assert sample.multimodal_train_inputs["vision_embeds"] is final
    assert sample.multimodal_train_inputs["deepstack_visual_embeds_2"] is deepstack[2]
    assert "pixel_values" not in sample.multimodal_train_inputs


@pytest.mark.asyncio
async def test_collect_cpu_vision_metrics_reports_cumulative_and_interval_counters(monkeypatch):
    rollout_module = _import_sglang_rollout(monkeypatch)
    snapshots = iter(
        (
            {
                "entries": 8,
                "resident_bytes": 4096,
                "hits": 6,
                "misses": 2,
                "evictions": 0,
                "encode_requests_total": 8,
                "backend_encode_requests_total": 2,
                "backend_encoded_images_total": 2,
                "emitted_feature_bytes_total": 1024,
                "backend_encode_seconds_total": 0.5,
            },
            {
                "entries": 12,
                "resident_bytes": 6144,
                "hits": 14,
                "misses": 4,
                "evictions": 1,
                "encode_requests_total": 18,
                "backend_encode_requests_total": 4,
                "backend_encoded_images_total": 4,
                "emitted_feature_bytes_total": 2048,
                "backend_encode_seconds_total": 1.0,
            },
        )
    )

    class RemoteMetrics:
        async def remote(self):
            return next(snapshots)

    state = Namespace(
        vision_encoder=Namespace(get_metrics=RemoteMetrics()),
        vision_encoder_metrics_previous=None,
    )

    first = await rollout_module._collect_cpu_vision_metrics(state, phase="baseline_eval")
    second = await rollout_module._collect_cpu_vision_metrics(state, phase="rollout_0")

    assert first["vision_encoder/cache/hits_total"] == 6
    assert first["vision_encoder/cache/hits_interval"] == 6
    assert first["vision_encoder/cache/hit_rate_interval"] == pytest.approx(0.75)
    assert first["vision_encoder/backend/images_per_second_interval"] == pytest.approx(4.0)
    assert second["vision_encoder/cache/hits_total"] == 14
    assert second["vision_encoder/cache/hits_interval"] == 8
    assert second["vision_encoder/cache/misses_interval"] == 2
    assert second["vision_encoder/cache/hit_rate_interval"] == pytest.approx(0.8)
    assert second["vision_encoder/backend/encode_seconds_interval"] == pytest.approx(0.5)
    assert second["vision_encoder/backend/images_per_second_interval"] == pytest.approx(4.0)
@pytest.mark.asyncio
async def test_collect_cpu_vision_metrics_is_empty_when_service_is_disabled(monkeypatch):
    rollout_module = _import_sglang_rollout(monkeypatch)

    assert await rollout_module._collect_cpu_vision_metrics(
        Namespace(vision_encoder=None),
        phase="rollout_0",
    ) == {}
