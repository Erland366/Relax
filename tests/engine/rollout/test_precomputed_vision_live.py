# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import base64
import importlib
import json
import sys
import types
from argparse import Namespace
from contextlib import nullcontext

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
    _stub_module(monkeypatch, "pybase64", b64decode=base64.b64decode)
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
            return _cpu_vision_telemetry_response(
                "vision-replica-a",
                features=features,
                feature_id=features.feature_id,
                encode_requests=1,
                backend_encodes=1,
            )

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

    async def fake_post(url, payload, headers=None, request_metrics=None):
        json.dumps(payload)
        posted.update(payload)
        request_metrics.update(
            {
                "attempts": 1,
                "request_body_bytes": 512,
                "response_body_bytes": 64,
                "request_build_seconds": 0.02,
                "response_wait_seconds": 0.03,
                "response_read_seconds": 0.004,
                "response_decode_seconds": 0.001,
            }
        )
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
        vision_encoder_device="cpu",
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
    assert posted["image_data"][0]["feature_dtype"] == "bfloat16"
    assert posted["image_data"][0]["feature_shape"] == [4, 8]
    packed_feature_bits = torch.frombuffer(
        bytearray(base64.b64decode(posted["image_data"][0]["feature_b64"], validate=True)),
        dtype=torch.uint16,
    ).reshape(4, 8)
    assert torch.equal(packed_feature_bits, torch.cat((final, *deepstack), dim=-1).view(torch.uint16))
    assert "feature" not in posted["image_data"][0]
    assert posted["image_data"][0]["image_grid_thw"] == grid.tolist()
    assert sample.multimodal_train_inputs["vision_embeds"] is final
    assert sample.multimodal_train_inputs["deepstack_visual_embeds_2"] is deepstack[2]
    assert "pixel_values" not in sample.multimodal_train_inputs
    assert sample.metadata["_timing"]["http_request_build"] == pytest.approx(0.02)
    assert sample.metadata["_timing"]["http_response_wait"] == pytest.approx(0.03)
    assert sample.metadata["_sizes"]["http_request_body_bytes"] == 512
    assert sample.metadata["_sizes"]["http_response_body_bytes"] == 64


@pytest.mark.asyncio
async def test_generate_republishes_inline_once_after_typed_sglang_feature_cache_miss(monkeypatch):
    rollout_module = _import_sglang_rollout(monkeypatch)
    sglang_vision_module = importlib.import_module("relax.backends.sglang.precomputed_vision")
    monkeypatch.setenv("SGLANG_VISION_FEATURE_CACHE_MAX_BYTES", "4096")
    grid = torch.tensor([[1, 4, 4]], dtype=torch.int64)
    vision_embeds = torch.ones((4, 2), dtype=torch.bfloat16)
    deepstack_visual_embeds = tuple(
        torch.full((4, 2), value, dtype=torch.bfloat16) for value in (2.0, 3.0, 4.0)
    )
    features = Namespace(
        image_grid_thw=grid,
        vision_embeds=vision_embeds,
        deepstack_visual_embeds=deepstack_visual_embeds,
        embedding_streams=(vision_embeds, *deepstack_visual_embeds),
        feature_id="feature-a",
        vision_revision="vision-revision-a",
        feature_schema_version="qwen3-vl-frozen-vision-v1",
        nbytes=grid.nbytes
        + vision_embeds.nbytes
        + sum(feature.nbytes for feature in deepstack_visual_embeds),
    )

    class RemoteEncode:
        async def remote(self, **kwargs):
            return _cpu_vision_telemetry_response(
                "vision-replica-a",
                features=features,
                feature_id=features.feature_id,
                encode_requests=1,
                backend_encodes=1,
            )

    class Tokenizer:
        image_token_id = 99
        pad_token_id = 0

        def encode(self, text, add_special_tokens=False):
            return [10, 11]

    state = Namespace(
        tokenizer=Tokenizer(),
        processor=object(),
        vision_encoder=Namespace(encode=RemoteEncode()),
    )
    monkeypatch.setattr(rollout_module, "GenerateState", lambda args: state)

    async def fake_image_processor(state, args, prompt, multimodal_inputs):
        return [7, 99, 99, 99, 99, 8], {"pixel_values": torch.ones((16, 2)), "image_grid_thw": grid}, 0.01

    posted_formats = []
    posted_headers = []

    async def fake_post(url, payload, headers=None, request_metrics=None):
        json.dumps(payload)
        image_data = payload["image_data"][0]
        posted_formats.append(image_data["format"])
        posted_headers.append(headers)
        request_metrics.update(
            {
                "attempts": 1,
                "request_body_bytes": 256,
                "response_body_bytes": 64,
                "request_build_seconds": 0.01,
                "response_wait_seconds": 0.02,
                "response_read_seconds": 0.003,
                "response_decode_seconds": 0.001,
            }
        )
        if image_data["format"] == "precomputed_embedding_id":
            raise sglang_vision_module.SGLangVisionFeatureCacheMiss(
                feature_id="feature-a",
                vision_revision="vision-revision-a",
                feature_schema_version="qwen3-vl-frozen-vision-v1",
            )
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
    monkeypatch.setattr(rollout_module, "post", fake_post)
    args = Namespace(
        ci_test=False,
        resource={"rollout": [1, 4]},
        rollout_num_gpus_per_engine=1,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        use_rollout_routing_replay=False,
        use_audio_in_video=False,
        sglang_router_policy="consistent_hashing",
        use_slime_router=False,
        slime_router_middleware_paths=[],
        partial_rollout=False,
        mask_offpolicy_in_partial_rollout=False,
        sglang_speculative_algorithm=None,
        vision_encoder_device="cpu",
    )
    first = rollout_module.Sample(
        prompt="look",
        multimodal_inputs={"images": ["raw-image"], "videos": [], "audio": []},
    )
    second = rollout_module.Sample(
        prompt="look",
        multimodal_inputs={"images": ["raw-image"], "videos": [], "audio": []},
    )

    await rollout_module.generate(args, first, {"max_new_tokens": 2})
    await rollout_module.generate(args, second, {"max_new_tokens": 2})

    assert posted_formats == [
        "precomputed_embedding",
        "precomputed_embedding_id",
        "precomputed_embedding",
    ]
    assert posted_headers == [{"X-SMG-Routing-Key": "feature-a"}] * 3
    assert second.metadata["_sizes"]["sglang_vision_cache_id_only_requests"] == 1
    assert second.metadata["_sizes"]["sglang_vision_cache_republish"] == 1


def test_sglang_feature_cache_rejects_multi_engine_round_robin_routing(monkeypatch):
    rollout_module = _import_sglang_rollout(monkeypatch)
    args = Namespace(
        resource={"rollout": [1, 4]},
        rollout_num_gpus_per_engine=1,
        sglang_router_policy="round_robin",
    )

    with pytest.raises(ValueError, match="feature.*consistent.*routing|feature-sticky"):
        rollout_module._validate_sglang_vision_feature_cache_routing(args, cache_enabled=True)

    args.sglang_router_policy = "consistent_hashing"
    rollout_module._validate_sglang_vision_feature_cache_routing(args, cache_enabled=True)


def test_aggregate_rollout_timing_reports_shared_stage_totals_and_percentiles(monkeypatch):
    rollout_module = _import_sglang_rollout(monkeypatch)
    samples = [
        Namespace(
            metadata={
                "_timing": {
                    "vision_service_round_trip": 0.4,
                    "precomputed_to_list": 0.1,
                    "generate": 1.0,
                },
                "_sizes": {
                    "precomputed_feature_bytes": 100,
                    "http_request_body_bytes": 1000,
                },
            }
        ),
        Namespace(
            metadata={
                "_timing": {"generate": 3.0},
                "_sizes": {"http_request_body_bytes": 1200},
            }
        ),
    ]

    metrics = rollout_module._aggregate_rollout_timing(samples, [])

    assert metrics["perf_detail/rollout/generate_time/count"] == 2
    assert metrics["perf_detail/rollout/generate_time/total"] == pytest.approx(4.0)
    assert metrics["perf_detail/rollout/generate_time/mean"] == pytest.approx(2.0)
    assert metrics["perf_detail/rollout/generate_time/p50"] == pytest.approx(1.0)
    assert metrics["perf_detail/rollout/generate_time/p95"] == pytest.approx(3.0)
    assert metrics["perf_detail/rollout/vision_service_round_trip_time/count"] == 1
    assert metrics["perf_detail/rollout/precomputed_to_list_time/total"] == pytest.approx(0.1)
    assert metrics["perf_detail/rollout/precomputed_feature_bytes/total"] == 100
    assert metrics["perf_detail/rollout/http_request_body_bytes/total"] == 2200
    assert metrics["perf_detail/rollout/http_request_body_bytes/max"] == 1200


@pytest.mark.asyncio
async def test_cpu_vision_encoder_rejects_legacy_raw_feature_response(monkeypatch):
    rollout_module = _import_sglang_rollout(monkeypatch)
    vision_module = importlib.import_module("relax.backends.vision.qwen3_vl")
    image_grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.int64)
    features = vision_module.Qwen3VLFrozenVisionFeatures(
        image_grid_thw=image_grid_thw,
        vision_embeds=torch.ones((1, 2), dtype=torch.bfloat16),
        deepstack_visual_embeds=(),
    )

    class RemoteEncode:
        async def remote(self, **kwargs):
            return features

    state = Namespace(vision_encoder=Namespace(encode=RemoteEncode()))
    multimodal_train_inputs = {
        "pixel_values": torch.ones((4, 2)),
        "image_grid_thw": image_grid_thw,
    }

    with pytest.raises(TypeError, match="VisionEncoderResponse"):
        await rollout_module._run_cpu_vision_encoder(state, multimodal_train_inputs)


@pytest.mark.asyncio
async def test_generate_group_uses_one_sglang_parallel_request_and_maps_outputs(monkeypatch):
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
    posted = []

    class RemoteEncode:
        async def remote(self, **kwargs):
            encode_calls.append(kwargs)
            return _cpu_vision_telemetry_response(
                "vision-replica-a",
                features=features,
                feature_id=features.feature_id,
                encode_requests=1,
                backend_encodes=1,
            )

    class Tokenizer:
        image_token_id = 99
        pad_token_id = 0

        def encode(self, text, add_special_tokens=False):
            return [10, 11]

    state = Namespace(
        tokenizer=Tokenizer(),
        processor=object(),
        vision_encoder=Namespace(encode=RemoteEncode()),
        semaphore=asyncio.Semaphore(8),
        aborted=False,
        dp_rank_context=lambda: nullcontext(),
    )
    monkeypatch.setattr(rollout_module, "GenerateState", lambda args: state)

    async def fake_image_processor(state, args, prompt, multimodal_inputs):
        return [7, 99, 99, 99, 99, 8], {"pixel_values": torch.ones((16, 2)), "image_grid_thw": grid}, 0.01

    async def fake_post(url, payload, headers=None, request_metrics=None):
        json.dumps(payload)
        posted.append(payload)
        request_metrics.update(
            {
                "attempts": 1,
                "request_body_bytes": 700,
                "response_body_bytes": 200,
                "request_build_seconds": 0.02,
                "response_wait_seconds": 0.03,
                "response_read_seconds": 0.004,
                "response_decode_seconds": 0.001,
            }
        )
        return [
            {
                "text": response,
                "meta_info": {
                    "output_token_logprobs": [[logprob, token_id, None]],
                    "finish_reason": {"type": "stop"},
                    "prompt_tokens": 6,
                    "cached_tokens": 0,
                },
            }
            for response, token_id, logprob in (("A", 21, -0.1), ("B", 22, -0.2))
        ]

    async def fake_reward(args, sample):
        return 1.0 if sample.response == "A" else 0.0

    monkeypatch.setattr(rollout_module, "_run_image_processor", fake_image_processor)
    monkeypatch.setattr(rollout_module, "post", fake_post)
    monkeypatch.setattr(rollout_module, "async_rm", fake_reward)

    args = Namespace(
        ci_test=False,
        custom_generate_function_path=None,
        group_rm=False,
        mask_offpolicy_in_partial_rollout=False,
        partial_rollout=False,
        sglang_enable_deterministic_inference=False,
        sglang_router_ip="127.0.0.1",
        sglang_router_policy="round_robin",
        sglang_router_port=30000,
        sglang_speculative_algorithm=None,
        slime_router_middleware_paths=[],
        use_audio_in_video=False,
        use_opd=False,
        use_rollout_routing_replay=False,
        use_slime_router=False,
        vision_encoder_device="cpu",
    )
    multimodal_inputs = {"images": ["raw-image"], "videos": [], "audio": []}
    group = [
        rollout_module.Sample(prompt="look", multimodal_inputs=multimodal_inputs),
        rollout_module.Sample(prompt="look", multimodal_inputs=multimodal_inputs),
    ]

    result = await rollout_module.generate_and_rm_group(
        args,
        group,
        sampling_params={"max_new_tokens": 2, "temperature": 1.0},
    )

    assert len(encode_calls) == 1
    assert len(posted) == 1
    assert posted[0]["sampling_params"]["n"] == 2
    assert posted[0]["input_ids"] == [7, 99, 99, 99, 99, 8]
    assert posted[0]["image_data"][0]["format"] == "precomputed_embedding"
    assert [sample.response for sample in result] == ["A", "B"]
    assert [sample.rollout_log_probs for sample in result] == [[-0.1], [-0.2]]
    assert [sample.reward for sample in result] == [1.0, 0.0]
    assert all(sample.status == rollout_module.Sample.Status.COMPLETED for sample in result)
    assert sum("http_request_body_bytes" in sample.metadata.get("_sizes", {}) for sample in result) == 1
    assert result[0].metadata["_sizes"]["http_request_body_bytes"] == 700
    assert result[0].metadata["_sizes"]["sglang_parallel_samples"] == 2


@pytest.mark.asyncio
async def test_generate_precomputed_group_reuses_cached_feature_and_republishes_once_on_miss(monkeypatch):
    rollout_module = _import_sglang_rollout(monkeypatch)
    sglang_vision_module = importlib.import_module("relax.backends.sglang.precomputed_vision")
    vision_module = importlib.import_module("relax.backends.vision.qwen3_vl")
    monkeypatch.setenv("SGLANG_VISION_FEATURE_CACHE_MAX_BYTES", "4096")
    monkeypatch.setenv("RELAX_SGLANG_VISION_FEATURE_CACHE_MAX_BYTES", "4096")
    grid = torch.tensor([[1, 4, 4]], dtype=torch.int64)
    final = torch.ones((4, 2), dtype=torch.bfloat16)
    deepstack = tuple(
        torch.full((4, 2), value, dtype=torch.bfloat16) for value in (2.0, 3.0, 4.0)
    )
    features = vision_module.Qwen3VLFrozenVisionFeatures(
        image_grid_thw=grid,
        vision_embeds=final,
        deepstack_visual_embeds=deepstack,
        feature_id="feature-a",
        vision_revision="vision-revision-a",
        feature_schema_version="qwen3-vl-frozen-vision-v1",
    )

    class RemoteEncode:
        async def remote(self, **kwargs):
            return _cpu_vision_telemetry_response(
                "vision-replica-a",
                features=features,
                feature_id=features.feature_id,
                encode_requests=1,
                backend_encodes=1,
            )

    class Tokenizer:
        image_token_id = 99
        pad_token_id = 0

        def encode(self, text, add_special_tokens=False):
            return [10, 11]

    state = Namespace(
        tokenizer=Tokenizer(),
        processor=object(),
        vision_encoder=Namespace(encode=RemoteEncode()),
        semaphore=asyncio.Semaphore(8),
        aborted=False,
        dp_rank_context=lambda: nullcontext(),
    )
    monkeypatch.setattr(rollout_module, "GenerateState", lambda args: state)

    async def fake_image_processor(state, args, prompt, multimodal_inputs):
        return [7, 99, 99, 99, 99, 8], {"pixel_values": torch.ones((16, 2)), "image_grid_thw": grid}, 0.01

    original_serializer = rollout_module.serialize_sglang_precomputed_image_data
    serializer_calls = []

    def counting_serializer(image_data):
        serializer_calls.append(image_data["feature_id"])
        return original_serializer(image_data)

    posted_formats = []
    posted_headers = []

    async def fake_post(url, payload, headers=None, request_metrics=None):
        json.dumps(payload)
        image_data = payload["image_data"][0]
        posted_formats.append(image_data["format"])
        posted_headers.append(headers)
        request_metrics.update(
            {
                "attempts": 1,
                "request_body_bytes": 256,
                "response_body_bytes": 128,
                "request_build_seconds": 0.01,
                "response_wait_seconds": 0.02,
                "response_read_seconds": 0.003,
                "response_decode_seconds": 0.001,
            }
        )
        if len(posted_formats) == 3 and image_data["format"] == "precomputed_embedding_id":
            raise sglang_vision_module.SGLangVisionFeatureCacheMiss(
                feature_id="feature-a",
                vision_revision="vision-revision-a",
                feature_schema_version="qwen3-vl-frozen-vision-v1",
            )
        return [
            {
                "text": response,
                "meta_info": {
                    "output_token_logprobs": [[logprob, token_id, None]],
                    "finish_reason": {"type": "stop"},
                    "prompt_tokens": 6,
                    "cached_tokens": 0,
                },
            }
            for response, token_id, logprob in (("A", 21, -0.1), ("B", 22, -0.2))
        ]

    async def fake_reward(args, sample):
        return 1.0

    monkeypatch.setattr(rollout_module, "_run_image_processor", fake_image_processor)
    monkeypatch.setattr(rollout_module, "serialize_sglang_precomputed_image_data", counting_serializer)
    monkeypatch.setattr(rollout_module, "post", fake_post)
    monkeypatch.setattr(rollout_module, "async_rm", fake_reward)
    args = Namespace(
        ci_test=False,
        custom_generate_function_path=None,
        group_rm=False,
        mask_offpolicy_in_partial_rollout=False,
        partial_rollout=False,
        resource={"rollout": [1, 1]},
        rollout_num_gpus_per_engine=1,
        sglang_enable_deterministic_inference=False,
        sglang_router_ip="127.0.0.1",
        sglang_router_policy="consistent_hashing",
        sglang_router_port=30000,
        sglang_speculative_algorithm=None,
        slime_router_middleware_paths=[],
        use_audio_in_video=False,
        use_opd=False,
        use_rollout_routing_replay=False,
        use_slime_router=False,
        vision_encoder_device="cpu",
    )

    def new_group():
        multimodal_inputs = {"images": ["raw-image"], "videos": [], "audio": []}
        return [
            rollout_module.Sample(prompt="look", multimodal_inputs=multimodal_inputs),
            rollout_module.Sample(prompt="look", multimodal_inputs=multimodal_inputs),
        ]

    first = await rollout_module.generate_and_rm_group(
        args,
        new_group(),
        sampling_params={"max_new_tokens": 2, "temperature": 1.0},
    )
    serializer_count_after_publish = len(serializer_calls)
    second = await rollout_module.generate_and_rm_group(
        args,
        new_group(),
        sampling_params={"max_new_tokens": 2, "temperature": 1.0},
    )
    serializer_count_after_hit = len(serializer_calls)
    third = await rollout_module.generate_and_rm_group(
        args,
        new_group(),
        sampling_params={"max_new_tokens": 2, "temperature": 1.0},
    )

    assert posted_formats == [
        "precomputed_embedding",
        "precomputed_embedding_id",
        "precomputed_embedding_id",
        "precomputed_embedding",
    ]
    assert posted_headers == [{"X-SMG-Routing-Key": "feature-a"}] * 4
    assert serializer_count_after_publish == 1
    assert serializer_count_after_hit == 1
    first_counters = {
        "sglang_vision_cache_inline_publish_requests": 1,
        "sglang_vision_cache_id_only_requests": 0,
        "sglang_vision_cache_id_only_hits": 0,
        "sglang_vision_cache_id_only_misses": 0,
        "sglang_vision_cache_republish": 0,
    }
    second_counters = {
        "sglang_vision_cache_inline_publish_requests": 0,
        "sglang_vision_cache_id_only_requests": 1,
        "sglang_vision_cache_id_only_hits": 1,
        "sglang_vision_cache_id_only_misses": 0,
        "sglang_vision_cache_republish": 0,
    }
    third_counters = {
        "sglang_vision_cache_inline_publish_requests": 1,
        "sglang_vision_cache_id_only_requests": 1,
        "sglang_vision_cache_id_only_hits": 0,
        "sglang_vision_cache_id_only_misses": 1,
        "sglang_vision_cache_republish": 1,
    }
    for key, value in first_counters.items():
        assert first[0].metadata["_sizes"][key] == value
    for key, value in second_counters.items():
        assert second[0].metadata["_sizes"][key] == value
    for key, value in third_counters.items():
        assert third[0].metadata["_sizes"][key] == value


@pytest.mark.asyncio
async def test_generate_native_image_group_uses_one_sglang_parallel_request_and_shared_processing(monkeypatch):
    rollout_module = _import_sglang_rollout(monkeypatch)
    processor_outputs = []
    encode_calls = []
    posted = []

    class Tokenizer:
        image_token_id = 99
        pad_token_id = 0

        def encode(self, text, add_special_tokens=False):
            return [10, 11]

    state = Namespace(
        tokenizer=Tokenizer(),
        processor=object(),
        vision_encoder=None,
        semaphore=asyncio.Semaphore(8),
        aborted=False,
        dp_rank_context=lambda: nullcontext(),
    )
    monkeypatch.setattr(rollout_module, "GenerateState", lambda args: state)

    async def fake_image_processor(state, args, prompt, multimodal_inputs):
        processor_output = {"pixel_values": torch.ones((16, 2))}
        processor_outputs.append(processor_output)
        return [7, 99, 99, 99, 99, 8], processor_output, 0.01

    async def fake_encode_multimodal_inputs(multimodal_inputs):
        encode_calls.append(multimodal_inputs)
        return {"image_data": ["encoded-image"]}, 0.02

    async def fake_post(url, payload, headers=None, request_metrics=None):
        json.dumps(payload)
        posted.append(payload)
        request_metrics.update(
            {
                "attempts": 1,
                "request_body_bytes": 300,
                "response_body_bytes": 200,
                "request_build_seconds": 0.01,
                "response_wait_seconds": 0.02,
                "response_read_seconds": 0.003,
                "response_decode_seconds": 0.001,
            }
        )
        responses = [
            {
                "text": response,
                "meta_info": {
                    "output_token_logprobs": [[logprob, token_id, None]],
                    "finish_reason": {"type": "stop"},
                    "prompt_tokens": 2,
                    "cached_tokens": 0,
                },
            }
            for response, token_id, logprob in (("A", 21, -0.1), ("B", 22, -0.2))
        ]
        if payload["sampling_params"].get("n") == 2:
            return responses
        return responses[len(posted) - 1]

    async def fake_reward(args, sample):
        return 1.0 if sample.response == "A" else 0.0

    monkeypatch.setattr(rollout_module, "_run_image_processor", fake_image_processor)
    monkeypatch.setattr(rollout_module, "_encode_multimodal_inputs", fake_encode_multimodal_inputs)
    monkeypatch.setattr(rollout_module, "post", fake_post)
    monkeypatch.setattr(rollout_module, "async_rm", fake_reward)

    args = Namespace(
        ci_test=False,
        custom_generate_function_path=None,
        group_rm=False,
        mask_offpolicy_in_partial_rollout=False,
        partial_rollout=False,
        sglang_enable_deterministic_inference=False,
        sglang_router_ip="127.0.0.1",
        sglang_router_policy="round_robin",
        sglang_router_port=30000,
        sglang_speculative_algorithm=None,
        slime_router_middleware_paths=[],
        use_audio_in_video=False,
        use_opd=False,
        use_rollout_routing_replay=False,
        use_slime_router=False,
        vision_encoder_device="gpu",
    )
    multimodal_inputs = {"images": ["raw-image"], "videos": [], "audio": []}
    group = [
        rollout_module.Sample(prompt="look", multimodal_inputs=multimodal_inputs),
        rollout_module.Sample(prompt="look", multimodal_inputs=multimodal_inputs),
    ]

    result = await rollout_module.generate_and_rm_group(
        args,
        group,
        sampling_params={"max_new_tokens": 2, "temperature": 1.0},
    )

    assert {
        "image_processor_calls": len(processor_outputs),
        "media_encoding_calls": len(encode_calls),
        "sglang_requests": len(posted),
        "parallel_samples": posted[0]["sampling_params"].get("n"),
        "responses": [sample.response for sample in result],
        "log_probs": [sample.rollout_log_probs for sample in result],
        "rewards": [sample.reward for sample in result],
        "shared_processor_result": all(sample.multimodal_train_inputs is processor_outputs[0] for sample in result),
    } == {
        "image_processor_calls": 1,
        "media_encoding_calls": 1,
        "sglang_requests": 1,
        "parallel_samples": 2,
        "responses": ["A", "B"],
        "log_probs": [[-0.1], [-0.2]],
        "rewards": [1.0, 0.0],
        "shared_processor_result": True,
    }
    assert posted[0]["input_ids"] == [10, 11]
    assert posted[0]["image_data"] == ["encoded-image"]
    assert all(sample.status == rollout_module.Sample.Status.COMPLETED for sample in result)


@pytest.mark.asyncio
async def test_generate_precomputed_group_rejects_wrong_sglang_output_count(monkeypatch):
    rollout_module = _import_sglang_rollout(monkeypatch)

    async def fake_post(url, payload, headers=None, request_metrics=None):
        request_metrics.update(
            {
                "attempts": 1,
                "request_body_bytes": 700,
                "response_body_bytes": 100,
                "request_build_seconds": 0.02,
                "response_wait_seconds": 0.03,
                "response_read_seconds": 0.004,
                "response_decode_seconds": 0.001,
            }
        )
        return [{}]

    monkeypatch.setattr(rollout_module, "post", fake_post)
    args = Namespace(sglang_router_ip="127.0.0.1", sglang_router_port=30000)
    state = Namespace(tokenizer=object(), processor=object())
    group = [rollout_module.Sample(prompt="look"), rollout_module.Sample(prompt="look")]

    with pytest.raises(ValueError, match="expected 2, got 1"):
        await rollout_module._generate_precomputed_group(
            args,
            state,
            group,
            {"max_new_tokens": 2, "temperature": 1.0},
            processor_prompt_ids=[1, 2],
            actor_inputs={"vision_embeds": torch.ones((1, 2))},
            sglang_image_data={"format": "precomputed_embedding"},
            image_processor_elapsed=0.01,
            shared_timing={},
            shared_sizes={},
            evaluation=False,
        )


@pytest.mark.parametrize(
    ("args_overrides", "sample_overrides", "expected"),
    [
        ({"sglang_enable_deterministic_inference": True}, {}, "deterministic inference"),
        ({"partial_rollout": True}, {}, "partial rollout"),
        ({"custom_generate_function_path": "pkg.generate"}, {}, "custom generate function"),
        ({}, {"response": "A"}, "fresh samples"),
    ],
)
def test_sglang_parallel_sampling_skip_reason_is_explicit(
    monkeypatch,
    args_overrides,
    sample_overrides,
    expected,
):
    rollout_module = _import_sglang_rollout(monkeypatch)
    args = Namespace(
        custom_generate_function_path=None,
        partial_rollout=False,
        sglang_enable_deterministic_inference=False,
        sglang_router_policy="round_robin",
        use_rollout_routing_replay=False,
        use_slime_router=False,
    )
    for name, value in args_overrides.items():
        setattr(args, name, value)
    multimodal_inputs = {"images": ["raw-image"], "videos": [], "audio": []}
    group = [
        rollout_module.Sample(prompt="look", multimodal_inputs=multimodal_inputs),
        rollout_module.Sample(prompt="look", multimodal_inputs=multimodal_inputs),
    ]
    for sample in group:
        for name, value in sample_overrides.items():
            setattr(sample, name, value)

    reason = rollout_module._sglang_parallel_sampling_skip_reason(
        args,
        group,
        shared_multimodal_input=True,
    )

    assert expected in reason


def test_sglang_parallel_sampling_allows_single_engine_cache_aware_router(monkeypatch):
    rollout_module = _import_sglang_rollout(monkeypatch)
    args = Namespace(
        custom_generate_function_path=None,
        num_gpus_per_node=4,
        partial_rollout=False,
        resource={"rollout": [1, 1]},
        rollout_num_gpus_per_engine=1,
        sglang_enable_deterministic_inference=False,
        sglang_router_policy="cache_aware",
        use_rollout_routing_replay=False,
        use_slime_router=False,
    )
    multimodal_inputs = {"images": ["raw-image"], "videos": [], "audio": []}
    group = [
        rollout_module.Sample(prompt="look", multimodal_inputs=multimodal_inputs),
        rollout_module.Sample(prompt="look", multimodal_inputs=multimodal_inputs),
    ]

    reason = rollout_module._sglang_parallel_sampling_skip_reason(
        args,
        group,
        shared_multimodal_input=True,
    )

    assert reason is None


@pytest.mark.asyncio
async def test_eval_rollout_groups_branches_per_prompt_without_group_rm(monkeypatch):
    rollout_module = _import_sglang_rollout(monkeypatch)
    dataset_cfg = Namespace(
        cache_key=("eval",),
        custom_generate_function_path=None,
        inject_metadata=lambda metadata: dict(metadata or {}),
        max_response_len=16,
        n_samples_per_eval_prompt=3,
        name="eval",
        temperature=1.0,
        top_k=-1,
        top_p=1.0,
    )
    args = Namespace(
        apply_chat_template=False,
        eval_reward_key=None,
        group_rm=False,
        hf_checkpoint="checkpoint",
        reward_key=None,
        rollout_skip_special_tokens=False,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        sglang_enable_deterministic_inference=False,
    )
    cache_key = dataset_cfg.cache_key + (args.hf_checkpoint, args.apply_chat_template)
    rollout_module.EVAL_PROMPT_DATASET[cache_key] = Namespace(
        samples=[rollout_module.Sample(prompt="first"), rollout_module.Sample(prompt="second")]
    )
    grouped_calls = []
    individual_calls = []

    async def fake_generate_and_rm_group(args, group, sampling_params, evaluation):
        grouped_calls.append(group)
        for sample in group:
            sample.reward = float(sample.index)
            sample.status = rollout_module.Sample.Status.COMPLETED
        return group

    async def fake_generate_and_rm(args, sample, sampling_params, evaluation):
        individual_calls.append(sample)
        sample.reward = float(sample.index)
        sample.status = rollout_module.Sample.Status.COMPLETED
        return sample

    monkeypatch.setattr(rollout_module, "generate_and_rm_group", fake_generate_and_rm_group)
    monkeypatch.setattr(rollout_module, "generate_and_rm", fake_generate_and_rm)

    result = await rollout_module.eval_rollout_single_dataset(args, rollout_id=7, dataset_cfg=dataset_cfg)
    samples = result[dataset_cfg.name]["samples"]

    assert {
        "grouped": [[sample.index for sample in group] for group in grouped_calls],
        "individual": [sample.index for sample in individual_calls],
    } == {
        "grouped": [[0, 1, 2], [3, 4, 5]],
        "individual": [],
    }
    assert [sample.index for sample in samples] == list(range(6))
    assert len({id(sample) for sample in samples}) == 6


@pytest.mark.asyncio
async def test_eval_rollout_group_branches_share_multimodal_inputs_object(monkeypatch):
    rollout_module = _import_sglang_rollout(monkeypatch)
    dataset_cfg = Namespace(
        cache_key=("eval-shared-media",),
        custom_generate_function_path=None,
        inject_metadata=lambda metadata: dict(metadata or {}),
        max_response_len=16,
        n_samples_per_eval_prompt=2,
        name="eval-shared-media",
        temperature=1.0,
        top_k=-1,
        top_p=1.0,
    )
    args = Namespace(
        apply_chat_template=False,
        eval_reward_key=None,
        hf_checkpoint="checkpoint",
        reward_key=None,
        rollout_skip_special_tokens=False,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        sglang_enable_deterministic_inference=False,
    )
    cache_key = dataset_cfg.cache_key + (args.hf_checkpoint, args.apply_chat_template)
    rollout_module.EVAL_PROMPT_DATASET[cache_key] = Namespace(
        samples=[
            rollout_module.Sample(
                prompt="look",
                multimodal_inputs={"images": ["raw-image"], "videos": [], "audio": []},
            )
        ]
    )
    grouped_calls = []

    async def fake_generate_and_rm_group(args, group, sampling_params, evaluation):
        grouped_calls.append(group)
        for sample in group:
            sample.reward = 1.0
            sample.status = rollout_module.Sample.Status.COMPLETED
        return group

    monkeypatch.setattr(rollout_module, "generate_and_rm_group", fake_generate_and_rm_group)

    await rollout_module.eval_rollout_single_dataset(args, rollout_id=7, dataset_cfg=dataset_cfg)

    assert len(grouped_calls) == 1
    first, second = grouped_calls[0]
    assert second.multimodal_inputs is first.multimodal_inputs


def test_collect_cpu_vision_metrics_aggregates_latest_snapshot_from_every_replica(monkeypatch):
    rollout_module = _import_sglang_rollout(monkeypatch)

    def response(
        replica_id,
        feature_id,
        *,
        cache_hit,
        backend_batch_size,
        encode_requests,
        hits,
        misses,
        backend_encodes,
        encoded_images,
        process_cpu_seconds,
        rss_bytes,
    ):
        return Namespace(
            replica_id=replica_id,
            feature_id=feature_id,
            cache_hit=cache_hit,
            backend_batch_size=backend_batch_size,
            metrics_snapshot={
                "entries": misses,
                "resident_bytes": misses * 100,
                "hits": hits,
                "misses": misses,
                "evictions": 0,
                "encode_requests_total": encode_requests,
                "backend_encode_requests_total": backend_encodes,
                "backend_encoded_images_total": encoded_images,
                "emitted_feature_bytes_total": encoded_images * 100,
                "backend_encode_seconds_total": encoded_images / 2,
                "process_cpu_seconds_total": process_cpu_seconds,
                "rss_bytes": rss_bytes,
            },
        )

    state = Namespace(
        args=Namespace(vision_encoder_num_replicas=4),
        vision_encoder=object(),
        vision_encoder_metrics_previous=None,
    )
    idle_snapshot = response(
        "replica-idle",
        "",
        cache_hit=False,
        backend_batch_size=0,
        encode_requests=0,
        hits=0,
        misses=0,
        backend_encodes=0,
        encoded_images=0,
        process_cpu_seconds=0.0,
        rss_bytes=750,
    )
    replica_a_latest = response(
        "replica-a",
        "shared-feature",
        cache_hit=False,
        backend_batch_size=4,
        encode_requests=5,
        hits=4,
        misses=1,
        backend_encodes=1,
        encoded_images=4,
        process_cpu_seconds=3.0,
        rss_bytes=1_000,
    )
    replica_b = response(
        "replica-b",
        "shared-feature",
        cache_hit=False,
        backend_batch_size=1,
        encode_requests=1,
        hits=0,
        misses=1,
        backend_encodes=1,
        encoded_images=1,
        process_cpu_seconds=1.0,
        rss_bytes=2_000,
    )
    replica_a_stale = response(
        "replica-a",
        "cached-feature",
        cache_hit=True,
        backend_batch_size=0,
        encode_requests=4,
        hits=2,
        misses=2,
        backend_encodes=2,
        encoded_images=2,
        process_cpu_seconds=2.0,
        rss_bytes=900,
    )

    # Replica A's older request finishes last. Its feature was consumed, but its
    # stale cumulative snapshot must not replace the newer replica-A snapshot.
    for encode_response in (idle_snapshot, replica_a_latest, replica_b, replica_a_stale):
        rollout_module._record_cpu_vision_response(state, encode_response)

    metrics = asyncio.run(rollout_module._collect_cpu_vision_metrics(state, phase="rollout_0"))

    assert metrics["vision_encoder/replicas/expected"] == 4
    assert metrics["vision_encoder/replicas/observed"] == 3
    assert metrics["vision_encoder/replicas/active"] == 2
    assert metrics["vision_encoder/replicas/idle"] == 1
    assert metrics["vision_encoder/replicas/unobserved"] == 1
    assert metrics["vision_encoder/replica/replica-a/requests_total"] == 5
    assert metrics["vision_encoder/replica/replica-b/requests_total"] == 1
    assert metrics["vision_encoder/replica/replica-a/request_fraction_total"] == pytest.approx(5 / 6)
    assert metrics["vision_encoder/replica/replica-b/request_fraction_total"] == pytest.approx(1 / 6)
    assert metrics["vision_encoder/replica/replica-a/process_cpu_seconds_total"] == pytest.approx(3.0)
    assert metrics["vision_encoder/replica/replica-a/rss_bytes"] == 1_000
    assert metrics["vision_encoder/requests_total"] == 6
    assert metrics["vision_encoder/cache/hits_total"] == 4
    assert metrics["vision_encoder/cache/misses_total"] == 2
    assert metrics["vision_encoder/backend/encode_requests_total"] == 2
    assert metrics["vision_encoder/backend/encoded_images_total"] == 5
    assert metrics["vision_encoder/features/unique_total"] == 2
    assert metrics["vision_encoder/backend/duplicate_encode_ratio"] == pytest.approx(2.0)
    assert metrics["vision_encoder/backend/batch_size_mean"] == pytest.approx(2.5)
    assert metrics["vision_encoder/backend/batch_size_p95"] == pytest.approx(4.0)
    assert metrics["vision_encoder/backend/batch_size_max"] == 4


def _cpu_vision_telemetry_response(
    replica_id,
    *,
    features=None,
    feature_id="",
    cache_hit=False,
    backend_batch_size=0,
    encode_requests=0,
    backend_encodes=0,
    process_cpu_seconds=0.0,
    snapshot_monotonic_seconds=0.0,
):
    return Namespace(
        features=features,
        replica_id=replica_id,
        feature_id=feature_id,
        cache_hit=cache_hit,
        backend_batch_size=backend_batch_size,
        metrics_snapshot={
            "entries": 0,
            "resident_bytes": 0,
            "hits": 0,
            "misses": backend_encodes,
            "evictions": 0,
            "encode_requests_total": encode_requests,
            "backend_encode_requests_total": backend_encodes,
            "backend_encoded_images_total": backend_encodes,
            "emitted_feature_bytes_total": backend_encodes * 100,
            "backend_encode_seconds_total": backend_encodes / 2,
            "process_cpu_seconds_total": process_cpu_seconds,
            "snapshot_monotonic_seconds": snapshot_monotonic_seconds,
            "rss_bytes": 1_000,
        },
    )


def _cpu_vision_telemetry_state(*, replicas=2):
    return Namespace(
        args=Namespace(vision_encoder_num_replicas=replicas),
        vision_encoder=object(),
        vision_encoder_metrics_previous=None,
    )


def test_collect_cpu_vision_metrics_reports_per_replica_interval_request_distribution(monkeypatch):
    rollout_module = _import_sglang_rollout(monkeypatch)
    state = _cpu_vision_telemetry_state()

    rollout_module._record_cpu_vision_response(
        state,
        _cpu_vision_telemetry_response("replica-a", encode_requests=2),
    )
    rollout_module._record_cpu_vision_response(
        state,
        _cpu_vision_telemetry_response("replica-b", encode_requests=2),
    )
    asyncio.run(rollout_module._collect_cpu_vision_metrics(state, phase="rollout_0"))

    rollout_module._record_cpu_vision_response(
        state,
        _cpu_vision_telemetry_response("replica-a", encode_requests=5),
    )
    rollout_module._record_cpu_vision_response(
        state,
        _cpu_vision_telemetry_response("replica-b", encode_requests=3),
    )
    metrics = asyncio.run(rollout_module._collect_cpu_vision_metrics(state, phase="rollout_1"))

    assert metrics["vision_encoder/replica/replica-a/requests_interval"] == 3
    assert metrics["vision_encoder/replica/replica-b/requests_interval"] == 1
    assert metrics["vision_encoder/replica/replica-a/request_fraction_interval"] == pytest.approx(0.75)
    assert metrics["vision_encoder/replica/replica-b/request_fraction_interval"] == pytest.approx(0.25)


def test_collect_cpu_vision_metrics_derives_per_replica_process_cpu_utilization(monkeypatch):
    rollout_module = _import_sglang_rollout(monkeypatch)
    state = _cpu_vision_telemetry_state()

    rollout_module._record_cpu_vision_response(
        state,
        _cpu_vision_telemetry_response(
            "replica-a",
            encode_requests=1,
            process_cpu_seconds=1.0,
            snapshot_monotonic_seconds=10.0,
        ),
    )
    rollout_module._record_cpu_vision_response(
        state,
        _cpu_vision_telemetry_response(
            "replica-b",
            encode_requests=1,
            process_cpu_seconds=0.5,
            snapshot_monotonic_seconds=20.0,
        ),
    )
    asyncio.run(rollout_module._collect_cpu_vision_metrics(state, phase="rollout_0"))

    rollout_module._record_cpu_vision_response(
        state,
        _cpu_vision_telemetry_response(
            "replica-a",
            encode_requests=2,
            process_cpu_seconds=3.0,
            snapshot_monotonic_seconds=14.0,
        ),
    )
    rollout_module._record_cpu_vision_response(
        state,
        _cpu_vision_telemetry_response(
            "replica-b",
            encode_requests=2,
            process_cpu_seconds=1.0,
            snapshot_monotonic_seconds=22.0,
        ),
    )
    metrics = asyncio.run(rollout_module._collect_cpu_vision_metrics(state, phase="rollout_1"))

    assert metrics[
        "vision_encoder/replica/replica-a/process_cpu_utilization_percent_interval"
    ] == pytest.approx(50.0)
    assert metrics[
        "vision_encoder/replica/replica-b/process_cpu_utilization_percent_interval"
    ] == pytest.approx(25.0)


def test_collect_cpu_vision_metrics_counts_recurring_feature_ids_in_each_interval(monkeypatch):
    rollout_module = _import_sglang_rollout(monkeypatch)
    state = _cpu_vision_telemetry_state()

    for replica_id in ("replica-a", "replica-b"):
        rollout_module._record_cpu_vision_response(
            state,
            _cpu_vision_telemetry_response(
                replica_id,
                feature_id="shared-feature",
                encode_requests=1,
                backend_encodes=1,
            ),
        )
    first = asyncio.run(rollout_module._collect_cpu_vision_metrics(state, phase="rollout_0"))

    for replica_id in ("replica-a", "replica-b"):
        rollout_module._record_cpu_vision_response(
            state,
            _cpu_vision_telemetry_response(
                replica_id,
                feature_id="shared-feature",
                encode_requests=2,
                backend_encodes=2,
            ),
        )
    second = asyncio.run(rollout_module._collect_cpu_vision_metrics(state, phase="rollout_1"))

    assert first["vision_encoder/features/unique_interval"] == 1
    assert first["vision_encoder/backend/duplicate_encode_ratio_interval"] == pytest.approx(2.0)
    assert second["vision_encoder/features/unique_interval"] == 1
    assert second["vision_encoder/backend/duplicate_encode_ratio_interval"] == pytest.approx(2.0)


def test_collect_cpu_vision_metrics_reports_backend_batch_sizes_for_current_interval_only(monkeypatch):
    rollout_module = _import_sglang_rollout(monkeypatch)
    state = _cpu_vision_telemetry_state(replicas=1)

    for request_count, batch_size in enumerate((1, 16), start=1):
        rollout_module._record_cpu_vision_response(
            state,
            _cpu_vision_telemetry_response(
                "replica-a",
                backend_batch_size=batch_size,
                encode_requests=request_count,
                backend_encodes=request_count,
            ),
        )
    asyncio.run(rollout_module._collect_cpu_vision_metrics(state, phase="rollout_0"))

    for request_count, batch_size in enumerate((2, 4, 4, 8), start=3):
        rollout_module._record_cpu_vision_response(
            state,
            _cpu_vision_telemetry_response(
                "replica-a",
                backend_batch_size=batch_size,
                encode_requests=request_count,
                backend_encodes=request_count,
            ),
        )
    metrics = asyncio.run(rollout_module._collect_cpu_vision_metrics(state, phase="rollout_1"))

    assert metrics["vision_encoder/backend/batch_size_mean_interval"] == pytest.approx(4.5)
    assert metrics["vision_encoder/backend/batch_size_p95_interval"] == pytest.approx(8.0)
    assert metrics["vision_encoder/backend/batch_size_max_interval"] == 8


@pytest.mark.asyncio
async def test_collect_cpu_vision_metrics_is_empty_when_service_is_disabled(monkeypatch):
    rollout_module = _import_sglang_rollout(monkeypatch)

    assert (
        await rollout_module._collect_cpu_vision_metrics(
            Namespace(vision_encoder=None),
            phase="rollout_0",
        )
        == {}
    )
