import json
import math
import sys
from types import ModuleType, SimpleNamespace

import pytest

from examples.visual_xor import validate_sglang_cpu_vision_parity as parity_cli


def _response(*, generated_token_id: int, a_logprob: float, b_logprob: float) -> dict:
    return {
        "text": "A" if generated_token_id == 32 else "B",
        "meta_info": {
            "output_token_logprobs": [[-0.2, generated_token_id, None]],
            "output_token_ids_logprobs": [
                [
                    [a_logprob, 32, "A"],
                    [b_logprob, 33, "B"],
                ]
            ],
        },
    }


def test_extract_next_token_result_preserves_requested_action_logprobs():
    result = parity_cli.extract_next_token_result(
        _response(generated_token_id=32, a_logprob=-0.25, b_logprob=-1.5),
        action_a_token_id=32,
        action_b_token_id=33,
    )

    assert result["text"] == "A"
    assert result["generated_token_id"] == 32
    assert result["action_a_logprob"] == -0.25
    assert result["action_b_logprob"] == -1.5
    assert result["action_margin"] == 1.25
    assert result["action_a_probability"] == pytest.approx(1 / (1 + math.exp(-1.25)))


@pytest.mark.parametrize(
    ("response", "error_pattern"),
    [
        ({"text": "A", "meta_info": {}}, "output_token_logprobs"),
        (
            {
                "text": "A",
                "meta_info": {
                    "output_token_logprobs": [[-0.2, 32, None]],
                    "output_token_ids_logprobs": [[[-0.25, 32, "A"]]],
                },
            },
            "missing requested action token IDs.*33",
        ),
    ],
)
def test_extract_next_token_result_rejects_incomplete_sglang_metadata(response, error_pattern):
    with pytest.raises(ValueError, match=error_pattern):
        parity_cli.extract_next_token_result(
            response,
            action_a_token_id=32,
            action_b_token_id=33,
        )


def test_compare_sglang_results_builds_passing_raw_artifact(tmp_path):
    output = tmp_path / "nested" / "sglang-parity.json"
    GPU = [
        parity_cli.extract_next_token_result(
            _response(generated_token_id=32, a_logprob=-0.25, b_logprob=-1.5),
            action_a_token_id=32,
            action_b_token_id=33,
        )
    ]
    precomputed = [
        parity_cli.extract_next_token_result(
            _response(generated_token_id=32, a_logprob=-0.251, b_logprob=-1.499),
            action_a_token_id=32,
            action_b_token_id=33,
        )
    ]

    artifact = parity_cli.write_sglang_parity_artifact(
        output,
        gpu_results=GPU,
        precomputed_results=precomputed,
        sample_ids=["sample-1"],
        feature_ids=["feature-1"],
        metadata={"checkpoint": "/models/qwen3-vl"},
    )

    assert artifact == json.loads(output.read_text())
    assert artifact["schema_version"] == 1
    assert artifact["passed"] is True
    assert artifact["summary"]["generated_token_match_rate"] == 1.0
    assert artifact["summary"]["max_action_logprob_absolute_delta"] == pytest.approx(0.001)
    assert artifact["summary"]["max_action_margin_absolute_delta"] == pytest.approx(0.002)
    assert artifact["samples"][0]["sample_id"] == "sample-1"
    assert artifact["samples"][0]["feature_id"] == "feature-1"
    assert artifact["samples"][0]["gpu"] == GPU[0]
    assert artifact["samples"][0]["precomputed"] == precomputed[0]


def test_compare_sglang_results_fails_on_generated_token_or_margin_divergence(tmp_path):
    GPU = [
        parity_cli.extract_next_token_result(
            _response(generated_token_id=32, a_logprob=-0.1, b_logprob=-2.0),
            action_a_token_id=32,
            action_b_token_id=33,
        )
    ]
    precomputed = [
        parity_cli.extract_next_token_result(
            _response(generated_token_id=33, a_logprob=-2.0, b_logprob=-0.1),
            action_a_token_id=32,
            action_b_token_id=33,
        )
    ]

    artifact = parity_cli.write_sglang_parity_artifact(
        tmp_path / "sglang-parity.json",
        gpu_results=GPU,
        precomputed_results=precomputed,
        sample_ids=["sample-1"],
        feature_ids=["feature-1"],
        metadata={},
    )

    assert artifact["passed"] is False
    assert artifact["summary"]["generated_token_match_rate"] == 0.0
    assert artifact["summary"]["max_action_margin_absolute_delta"] == pytest.approx(3.8)


def test_compare_sglang_results_rejects_misaligned_inputs(tmp_path):
    with pytest.raises(ValueError, match="same non-zero length"):
        parity_cli.write_sglang_parity_artifact(
            tmp_path / "sglang-parity.json",
            gpu_results=[{}],
            precomputed_results=[],
            sample_ids=["sample-1"],
            feature_ids=["feature-1"],
            metadata={},
        )


def test_summarize_parallel_sampling_requires_ordered_matching_outputs_and_one_body():
    scalar = [
        parity_cli.extract_next_token_result(
            _response(generated_token_id=32, a_logprob=-0.25, b_logprob=-1.5),
            action_a_token_id=32,
            action_b_token_id=33,
        )
    ]
    parallel = [[dict(scalar[0]) for _ in range(8)]]

    result = parity_cli.summarize_sglang_parallel_sampling(
        scalar_results=scalar,
        parallel_results=parallel,
        scalar_request_body_bytes=[2_900_000],
        parallel_request_body_bytes=[2_900_006],
        parallel_samples=8,
    )

    assert result["passed"] is True
    assert result["summary"]["generation_requests"] == 1
    assert result["summary"]["generated_outputs"] == 8
    assert result["summary"]["output_match_rate"] == 1.0
    assert result["summary"]["grouped_request_bytes"] == 2_900_006
    assert result["summary"]["scalar_equivalent_request_bytes"] == 23_200_000
    assert result["summary"]["transport_reduction_ratio"] == pytest.approx(0.8749997413793104)


def test_summarize_parallel_sampling_rejects_wrong_response_count():
    with pytest.raises(ValueError, match="response count mismatch"):
        parity_cli.summarize_sglang_parallel_sampling(
            scalar_results=[{"generated_token_id": 32}],
            parallel_results=[[{"generated_token_id": 32}]],
            scalar_request_body_bytes=[100],
            parallel_request_body_bytes=[100],
            parallel_samples=2,
        )


def test_live_probe_groups_and_reports_gpu_and_precomputed_parallel_sampling(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    dataset = tmp_path / "eval.parquet"
    dataset.touch()
    output = tmp_path / "sglang-parity.json"
    requests = []

    class FakeTensor:
        def detach(self):
            return self

        def cpu(self):
            return self

        def __getitem__(self, index):
            return self

        def tolist(self):
            return [101, 102]

    tokenizer = SimpleNamespace(encode=lambda prompt, add_special_tokens=False: [101, 102])
    processor = SimpleNamespace(tokenizer=tokenizer)
    features = SimpleNamespace(feature_id="feature-1")
    cpu_backend = SimpleNamespace(
        revision="vision-revision",
        encode=lambda **kwargs: features,
    )
    server_process = SimpleNamespace(pid=123, join=lambda timeout: None)

    torch_module = ModuleType("torch")
    torch_module.cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)
    transformers_module = ModuleType("transformers")
    transformers_module.AutoProcessor = SimpleNamespace(from_pretrained=lambda *args, **kwargs: processor)
    cpu_parity_module = ModuleType("examples.visual_xor.validate_cpu_vision_parity")
    cpu_parity_module._read_unique_images = lambda path, count: ([object()], ["sample-1"])
    cpu_parity_module.resolve_action_token_ids = lambda tokenizer: (32, 33)
    sglang_engine_module = ModuleType("relax.backends.sglang.sglang_engine")
    sglang_engine_module.launch_server_process = lambda server_args: server_process
    sglang_engine_module._kill_process_tree = lambda pid: None
    qwen3_vl_module = ModuleType("relax.backends.vision.qwen3_vl")
    qwen3_vl_module.build_qwen3_vl_cpu_vision_backend = lambda path: cpu_backend
    qwen3_vl_module.build_sglang_precomputed_image_data = lambda value: {"format": "precomputed"}
    precomputed_module = ModuleType("relax.engine.rollout.precomputed_vision")
    precomputed_module.serialize_sglang_precomputed_image_data = lambda value: {"format": "packed"}
    processing_module = ModuleType("relax.utils.data.processing_utils")
    processing_module.encode_image_for_rollout_engine = lambda image: {"format": "gpu"}
    for name, module in {
        "torch": torch_module,
        "transformers": transformers_module,
        "examples.visual_xor.validate_cpu_vision_parity": cpu_parity_module,
        "relax.backends.sglang.sglang_engine": sglang_engine_module,
        "relax.backends.vision.qwen3_vl": qwen3_vl_module,
        "relax.engine.rollout.precomputed_vision": precomputed_module,
        "relax.utils.data.processing_utils": processing_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.setattr(parity_cli, "_server_args", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        parity_cli,
        "_build_prompt_and_inputs",
        lambda processor, image: (
            "prompt",
            {
                "input_ids": FakeTensor(),
                "pixel_values": FakeTensor(),
                "image_grid_thw": FakeTensor(),
            },
        ),
    )
    monkeypatch.setattr(parity_cli, "_flush_cache", lambda url: None)

    def post_generate(url, payload):
        requests.append(payload)
        response = _response(generated_token_id=32, a_logprob=-0.25, b_logprob=-1.5)
        parallel_samples = payload["sampling_params"].get("n")
        return ([response] * parallel_samples if parallel_samples else response), 100

    monkeypatch.setattr(parity_cli, "_post_generate", post_generate)

    artifact = parity_cli.run_sglang_parity(
        checkpoint=str(checkpoint),
        dataset=str(dataset),
        output=str(output),
        num_images=1,
        host="127.0.0.1",
        port=31000,
        base_gpu_id=0,
        parallel_samples=3,
    )

    grouped_requests = [request for request in requests if request["sampling_params"].get("n") == 3]
    assert len(grouped_requests) == 2
    assert {request["image_data"][0]["format"] for request in grouped_requests} == {"gpu", "packed"}
    for artifact_key in ("gpu_parallel_sampling", "parallel_sampling"):
        comparison = artifact[artifact_key]
        assert comparison["passed"] is True
        assert comparison["summary"]["generation_requests"] == 1
        assert comparison["summary"]["generated_outputs"] == 3
        assert comparison["summary"]["output_match_rate"] == 1.0
        assert comparison["summary"]["text_match_rate"] == 1.0
        assert comparison["summary"]["max_action_logprob_absolute_delta"] == 0.0


def test_main_runs_injected_probe_and_returns_written_artifact(tmp_path):
    output = tmp_path / "sglang-parity.json"
    expected = {"schema_version": 1, "passed": True}
    calls = []

    def probe(**kwargs):
        calls.append(kwargs)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(expected))
        return expected

    artifact = parity_cli.main(
        [
            "--checkpoint",
            "/models/qwen3-vl",
            "--dataset",
            "/data/eval.parquet",
            "--output",
            str(output),
            "--num-images",
            "2",
            "--host",
            "127.0.0.1",
            "--port",
            "31000",
            "--base-gpu-id",
            "0",
            "--parallel-samples",
            "8",
        ],
        probe=probe,
    )

    assert artifact == expected
    assert calls == [
        {
            "checkpoint": "/models/qwen3-vl",
            "dataset": "/data/eval.parquet",
            "output": str(output),
            "num_images": 2,
            "host": "127.0.0.1",
            "port": 31000,
            "base_gpu_id": 0,
            "parallel_samples": 8,
        }
    ]
