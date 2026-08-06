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
            return features

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
        vision_encoder_backend="pytorch",
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
        vision_encoder_backend="disabled",
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

    assert (
        await rollout_module._collect_cpu_vision_metrics(
            Namespace(vision_encoder=None),
            phase="rollout_0",
        )
        == {}
    )
