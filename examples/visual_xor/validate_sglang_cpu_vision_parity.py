"""Compare live SGLang native and CPU-precomputed Qwen3-VL next-token scores."""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


SCHEMA_VERSION = 1
ACTION_LOGPROB_ABSOLUTE_DELTA_MAX = 0.01
ACTION_MARGIN_ABSOLUTE_DELTA_MAX = 0.01
ACTION_PROBABILITY_ABSOLUTE_DELTA_MAX = 0.01


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {parsed}")
    return parsed


def _port(value: str) -> int:
    parsed = int(value)
    if not 1024 <= parsed <= 65528:
        raise argparse.ArgumentTypeError(f"must be between 1024 and 65528, got {parsed}")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"must be nonnegative, got {parsed}")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the explicit inputs needed for a reproducible live SGLang probe."""
    parser = argparse.ArgumentParser(
        description="Compare native GPU vision and CPU-precomputed Qwen3-VL scores in one SGLang server."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-images", required=True, type=_positive_int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=31000, type=_port)
    parser.add_argument("--base-gpu-id", default=0, type=_nonnegative_int)
    parser.add_argument("--parallel-samples", default=1, type=_positive_int)
    return parser.parse_args(argv)


def _first_position(values: Any, *, field: str) -> list[Any]:
    if not isinstance(values, list) or not values or not isinstance(values[0], list):
        raise ValueError(f"SGLang response {field} must contain a non-empty first position")
    return values[0]


def extract_next_token_result(
    response: Mapping[str, Any],
    *,
    action_a_token_id: int,
    action_b_token_id: int,
) -> dict[str, Any]:
    """Extract the generated token and requested A/B log-probabilities."""
    meta_info = response.get("meta_info")
    if not isinstance(meta_info, Mapping):
        raise ValueError("SGLang response is missing meta_info")

    generated = _first_position(meta_info.get("output_token_logprobs"), field="output_token_logprobs")
    if len(generated) < 2:
        raise ValueError("SGLang output_token_logprobs entry must contain a token ID")
    generated_token_id = int(generated[1])

    requested = _first_position(
        meta_info.get("output_token_ids_logprobs"),
        field="output_token_ids_logprobs",
    )
    requested_by_id: dict[int, float] = {}
    for item in requested:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            raise ValueError("SGLang output_token_ids_logprobs entries must contain logprob and token ID")
        requested_by_id[int(item[1])] = float(item[0])

    missing = sorted({action_a_token_id, action_b_token_id}.difference(requested_by_id))
    if missing:
        raise ValueError(f"SGLang response is missing requested action token IDs: {missing}")

    action_a_logprob = requested_by_id[action_a_token_id]
    action_b_logprob = requested_by_id[action_b_token_id]
    action_margin = action_a_logprob - action_b_logprob
    if action_margin >= 0:
        action_a_probability = 1.0 / (1.0 + math.exp(-action_margin))
    else:
        exp_margin = math.exp(action_margin)
        action_a_probability = exp_margin / (1.0 + exp_margin)
    return {
        "text": str(response.get("text", "")),
        "generated_token_id": generated_token_id,
        "action_a_logprob": action_a_logprob,
        "action_b_logprob": action_b_logprob,
        "action_margin": action_margin,
        "action_a_probability": action_a_probability,
        "action_b_probability": 1.0 - action_a_probability,
    }


def _with_parallel_sampling(payload: Mapping[str, Any], parallel_samples: int) -> dict[str, Any]:
    parallel_payload = dict(payload)
    parallel_payload["sampling_params"] = {
        **payload["sampling_params"],
        "n": parallel_samples,
    }
    return parallel_payload


def _extract_next_token_results(
    responses: Sequence[Mapping[str, Any]],
    *,
    action_a_token_id: int,
    action_b_token_id: int,
) -> list[dict[str, Any]]:
    return [
        extract_next_token_result(
            response,
            action_a_token_id=action_a_token_id,
            action_b_token_id=action_b_token_id,
        )
        for response in responses
    ]


def write_sglang_parity_artifact(
    output: str | Path,
    *,
    native_results: Sequence[Mapping[str, Any]],
    precomputed_results: Sequence[Mapping[str, Any]],
    sample_ids: Sequence[str],
    feature_ids: Sequence[str],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Write raw per-sample scores plus explicit deterministic parity gates."""
    lengths = {
        len(native_results),
        len(precomputed_results),
        len(sample_ids),
        len(feature_ids),
    }
    if lengths == {0} or len(lengths) != 1:
        raise ValueError("native, precomputed, sample ID, and feature ID inputs must have the same non-zero length")

    samples = []
    generated_token_matches = 0
    text_matches = 0
    action_logprob_deltas = []
    action_margin_deltas = []
    action_probability_deltas = []
    for sample_id, feature_id, native, precomputed in zip(
        sample_ids,
        feature_ids,
        native_results,
        precomputed_results,
        strict=True,
    ):
        generated_token_match = native["generated_token_id"] == precomputed["generated_token_id"]
        text_match = native["text"] == precomputed["text"]
        generated_token_matches += int(generated_token_match)
        text_matches += int(text_match)
        per_action_deltas = {
            "action_a": abs(native["action_a_logprob"] - precomputed["action_a_logprob"]),
            "action_b": abs(native["action_b_logprob"] - precomputed["action_b_logprob"]),
        }
        action_logprob_deltas.extend(per_action_deltas.values())
        action_margin_delta = abs(native["action_margin"] - precomputed["action_margin"])
        action_probability_delta = abs(native["action_a_probability"] - precomputed["action_a_probability"])
        action_margin_deltas.append(action_margin_delta)
        action_probability_deltas.append(action_probability_delta)
        samples.append(
            {
                "sample_id": sample_id,
                "feature_id": feature_id,
                "generated_token_match": generated_token_match,
                "text_match": text_match,
                "action_logprob_absolute_delta": per_action_deltas,
                "action_margin_absolute_delta": action_margin_delta,
                "action_probability_absolute_delta": action_probability_delta,
                "native": dict(native),
                "precomputed": dict(precomputed),
            }
        )

    num_samples = len(samples)
    summary = {
        "num_samples": num_samples,
        "generated_token_match_rate": generated_token_matches / num_samples,
        "text_match_rate": text_matches / num_samples,
        "max_action_logprob_absolute_delta": max(action_logprob_deltas),
        "max_action_margin_absolute_delta": max(action_margin_deltas),
        "max_action_probability_absolute_delta": max(action_probability_deltas),
    }
    gates = {
        "generated_token_match_rate_min": 1.0,
        "text_match_rate_min": 1.0,
        "action_logprob_absolute_delta_max": ACTION_LOGPROB_ABSOLUTE_DELTA_MAX,
        "action_margin_absolute_delta_max": ACTION_MARGIN_ABSOLUTE_DELTA_MAX,
        "action_probability_absolute_delta_max": ACTION_PROBABILITY_ABSOLUTE_DELTA_MAX,
    }
    passed = (
        summary["generated_token_match_rate"] >= gates["generated_token_match_rate_min"]
        and summary["text_match_rate"] >= gates["text_match_rate_min"]
        and summary["max_action_logprob_absolute_delta"] <= gates["action_logprob_absolute_delta_max"]
        and summary["max_action_margin_absolute_delta"] <= gates["action_margin_absolute_delta_max"]
        and summary["max_action_probability_absolute_delta"] <= gates["action_probability_absolute_delta_max"]
    )
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "passed": passed,
        "gates": gates,
        "summary": summary,
        "samples": samples,
        "metadata": dict(metadata),
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


def summarize_sglang_parallel_sampling(
    *,
    scalar_results: Sequence[Mapping[str, Any]],
    parallel_results: Sequence[Sequence[Mapping[str, Any]]],
    scalar_request_body_bytes: Sequence[int],
    parallel_request_body_bytes: Sequence[int],
    parallel_samples: int,
) -> dict[str, Any]:
    """Summarize one-request parallel sampling against scalar requests."""
    lengths = {
        len(scalar_results),
        len(parallel_results),
        len(scalar_request_body_bytes),
        len(parallel_request_body_bytes),
    }
    if lengths == {0} or len(lengths) != 1:
        raise ValueError("parallel sampling inputs must have the same non-zero image count")
    for sample_index, results in enumerate(parallel_results):
        if len(results) != parallel_samples:
            raise ValueError(
                "SGLang parallel sampling response count mismatch for "
                f"sample {sample_index}: expected {parallel_samples}, got {len(results)}"
            )

    output_matches = 0
    text_matches = 0
    action_logprob_deltas = []
    action_margin_deltas = []
    action_probability_deltas = []
    samples = []
    for sample_index, (scalar, results, scalar_bytes, parallel_bytes) in enumerate(
        zip(
            scalar_results,
            parallel_results,
            scalar_request_body_bytes,
            parallel_request_body_bytes,
            strict=True,
        )
    ):
        branch_results = []
        for branch_index, result in enumerate(results):
            output_match = result["generated_token_id"] == scalar["generated_token_id"]
            text_match = result["text"] == scalar["text"]
            output_matches += int(output_match)
            text_matches += int(text_match)
            per_action_deltas = {
                "action_a": abs(result["action_a_logprob"] - scalar["action_a_logprob"]),
                "action_b": abs(result["action_b_logprob"] - scalar["action_b_logprob"]),
            }
            action_logprob_deltas.extend(per_action_deltas.values())
            action_margin_delta = abs(result["action_margin"] - scalar["action_margin"])
            action_probability_delta = abs(result["action_a_probability"] - scalar["action_a_probability"])
            action_margin_deltas.append(action_margin_delta)
            action_probability_deltas.append(action_probability_delta)
            branch_results.append(
                {
                    "branch_index": branch_index,
                    "output_match": output_match,
                    "text_match": text_match,
                    "action_logprob_absolute_delta": per_action_deltas,
                    "action_margin_absolute_delta": action_margin_delta,
                    "action_probability_absolute_delta": action_probability_delta,
                    "result": dict(result),
                }
            )
        samples.append(
            {
                "sample_index": sample_index,
                "scalar_request_body_bytes": scalar_bytes,
                "parallel_request_body_bytes": parallel_bytes,
                "scalar_result": dict(scalar),
                "branches": branch_results,
            }
        )

    generated_outputs = len(scalar_results) * parallel_samples
    grouped_request_bytes = sum(parallel_request_body_bytes)
    scalar_equivalent_request_bytes = sum(scalar_request_body_bytes) * parallel_samples
    summary = {
        "parallel_samples": parallel_samples,
        "generation_requests": len(parallel_results),
        "generated_outputs": generated_outputs,
        "output_match_rate": output_matches / generated_outputs,
        "text_match_rate": text_matches / generated_outputs,
        "max_action_logprob_absolute_delta": max(action_logprob_deltas),
        "max_action_margin_absolute_delta": max(action_margin_deltas),
        "max_action_probability_absolute_delta": max(action_probability_deltas),
        "grouped_request_bytes": grouped_request_bytes,
        "scalar_equivalent_request_bytes": scalar_equivalent_request_bytes,
        "transport_reduction_ratio": 1.0 - grouped_request_bytes / scalar_equivalent_request_bytes,
    }
    gates = {
        "output_match_rate_min": 1.0,
        "text_match_rate_min": 1.0,
        "action_logprob_absolute_delta_max": ACTION_LOGPROB_ABSOLUTE_DELTA_MAX,
        "action_margin_absolute_delta_max": ACTION_MARGIN_ABSOLUTE_DELTA_MAX,
        "action_probability_absolute_delta_max": ACTION_PROBABILITY_ABSOLUTE_DELTA_MAX,
        "parallel_to_scalar_request_body_ratio_max": 1.01,
    }
    passed = (
        summary["output_match_rate"] >= gates["output_match_rate_min"]
        and summary["text_match_rate"] >= gates["text_match_rate_min"]
        and summary["max_action_logprob_absolute_delta"] <= gates["action_logprob_absolute_delta_max"]
        and summary["max_action_margin_absolute_delta"] <= gates["action_margin_absolute_delta_max"]
        and summary["max_action_probability_absolute_delta"] <= gates["action_probability_absolute_delta_max"]
        and all(
            parallel_bytes / scalar_bytes <= gates["parallel_to_scalar_request_body_ratio_max"]
            for scalar_bytes, parallel_bytes in zip(
                scalar_request_body_bytes,
                parallel_request_body_bytes,
                strict=True,
            )
        )
    )
    return {"passed": passed, "gates": gates, "summary": summary, "samples": samples}


def _build_prompt_and_inputs(processor: Any, image: Any) -> tuple[str, dict[str, Any]]:
    from examples.visual_xor.task import SFT_SYSTEM_PROMPT, XOR_USER_PROMPT

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SFT_SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": XOR_USER_PROMPT},
            ],
        },
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    required = {"input_ids", "pixel_values", "image_grid_thw"}
    missing = sorted(required.difference(encoded))
    if missing:
        raise ValueError(f"Qwen3-VL processor output is missing required fields: {missing}")
    return text, dict(encoded)


def _post_generate(url: str, payload: Mapping[str, Any]) -> tuple[Any, int]:
    import requests

    response = requests.post(f"{url}/generate", json=dict(payload), timeout=120)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        exc.add_note(f"SGLang response body: {response.text}")
        raise
    result = response.json()
    request_body = response.request.body
    if isinstance(request_body, str):
        request_body = request_body.encode()
    if not isinstance(request_body, bytes):
        raise TypeError(f"SGLang request body must be bytes, got {type(request_body).__name__}")
    return result, len(request_body)


def _flush_cache(url: str) -> None:
    import requests

    response = requests.get(f"{url}/flush_cache", timeout=30)
    response.raise_for_status()


def _server_args(checkpoint: Path, *, host: str, port: int, base_gpu_id: int) -> Any:
    from relax.backends.sglang.sglang_engine import _get_server_args_cls

    return _get_server_args_cls()(
        model_path=str(checkpoint),
        tokenizer_path=str(checkpoint),
        trust_remote_code=True,
        model_impl="transformers",
        host=host,
        port=port,
        skip_server_warmup=True,
        mem_fraction_static=0.4,
        max_running_requests=2,
        max_total_tokens=8192,
        max_prefill_tokens=16384,
        chunked_prefill_size=8192,
        base_gpu_id=base_gpu_id,
        tp_size=1,
        pp_size=1,
        attention_backend="torch_native",
        sampling_backend="pytorch",
        disable_cuda_graph=True,
        disable_piecewise_cuda_graph=True,
        disable_overlap_schedule=True,
        disable_custom_all_reduce=True,
        random_seed=1234,
        nccl_port=port + 1,
        dist_init_addr=f"{host}:{port + 7}",
    )


def run_sglang_parity(
    *,
    checkpoint: str,
    dataset: str,
    output: str,
    num_images: int,
    host: str,
    port: int,
    base_gpu_id: int,
    parallel_samples: int,
) -> dict[str, Any]:
    """Run native and precomputed requests against the same live SGLang model."""
    import torch
    from transformers import AutoProcessor

    from examples.visual_xor.validate_cpu_vision_parity import (
        _read_unique_images,
        resolve_action_token_ids,
    )
    from relax.backends.sglang.sglang_engine import _kill_process_tree, launch_server_process
    from relax.backends.vision.qwen3_vl import (
        build_qwen3_vl_cpu_vision_backend,
        build_sglang_precomputed_image_data,
    )
    from relax.engine.rollout.precomputed_vision import serialize_sglang_precomputed_image_data
    from relax.utils.data.processing_utils import encode_image_for_rollout_engine

    checkpoint_path = Path(checkpoint)
    dataset_path = Path(dataset)
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"SGLang parity checkpoint does not exist: {checkpoint_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("SGLang parity requires one visible ROCm/CUDA GPU")
    if base_gpu_id >= torch.cuda.device_count():
        raise ValueError(f"base GPU ID {base_gpu_id} is not visible; device count is {torch.cuda.device_count()}")

    processor = AutoProcessor.from_pretrained(
        checkpoint_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    images, sample_ids = _read_unique_images(dataset_path, num_images)
    action_a_token_id, action_b_token_id = resolve_action_token_ids(processor.tokenizer)
    cpu_backend = build_qwen3_vl_cpu_vision_backend(checkpoint_path)

    os.environ["RELAX_SGLANG_QWEN3_VL_PRECOMPUTED_VISION"] = "1"
    os.environ.pop("RELAX_SGLANG_QWEN3_VL_OMIT_GPU_WEIGHTS", None)
    server_process = launch_server_process(
        _server_args(checkpoint_path, host=host, port=port, base_gpu_id=base_gpu_id)
    )
    server_url = f"http://{host}:{port}"
    native_results = []
    native_scalar_request_body_bytes = []
    native_parallel_results = []
    native_parallel_request_body_bytes = []
    precomputed_results = []
    scalar_request_body_bytes = []
    parallel_results = []
    parallel_request_body_bytes = []
    feature_ids = []
    try:
        for image in images:
            prompt, model_inputs = _build_prompt_and_inputs(processor, image)
            native_payload = {
                "input_ids": processor.tokenizer.encode(prompt, add_special_tokens=False),
                "image_data": [encode_image_for_rollout_engine(image)],
                "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
                "return_logprob": True,
                "token_ids_logprob": [action_a_token_id, action_b_token_id],
            }
            native_response, native_scalar_body_bytes = _post_generate(server_url, native_payload)
            if not isinstance(native_response, dict):
                raise TypeError(
                    f"SGLang scalar native request must return an object, got {type(native_response).__name__}"
                )
            native_results.append(
                extract_next_token_result(
                    native_response,
                    action_a_token_id=action_a_token_id,
                    action_b_token_id=action_b_token_id,
                )
            )
            native_scalar_request_body_bytes.append(native_scalar_body_bytes)
            _flush_cache(server_url)
            if parallel_samples > 1:
                native_parallel_payload = _with_parallel_sampling(native_payload, parallel_samples)
                native_parallel_response, native_parallel_body_bytes = _post_generate(
                    server_url, native_parallel_payload
                )
                if not isinstance(native_parallel_response, list):
                    raise TypeError(
                        "SGLang parallel native request must return a list, "
                        f"got {type(native_parallel_response).__name__}"
                    )
                if len(native_parallel_response) != parallel_samples:
                    raise ValueError(
                        "SGLang parallel native response count mismatch: "
                        f"expected {parallel_samples}, got {len(native_parallel_response)}"
                    )
                if any(not isinstance(branch, Mapping) for branch in native_parallel_response):
                    raise TypeError("SGLang parallel native response branches must be objects")
                native_parallel_results.append(
                    _extract_next_token_results(
                        native_parallel_response,
                        action_a_token_id=action_a_token_id,
                        action_b_token_id=action_b_token_id,
                    )
                )
                native_parallel_request_body_bytes.append(native_parallel_body_bytes)
                _flush_cache(server_url)

            features = cpu_backend.encode(
                pixel_values=model_inputs["pixel_values"].detach().cpu(),
                image_grid_thw=model_inputs["image_grid_thw"].detach().cpu(),
            )
            precomputed_image_data = serialize_sglang_precomputed_image_data(
                build_sglang_precomputed_image_data(features)
            )
            precomputed_payload = {
                "input_ids": model_inputs["input_ids"][0].tolist(),
                "image_data": [precomputed_image_data],
                "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
                "return_logprob": True,
                "token_ids_logprob": [action_a_token_id, action_b_token_id],
            }
            precomputed_response, scalar_body_bytes = _post_generate(server_url, precomputed_payload)
            if not isinstance(precomputed_response, dict):
                raise TypeError(
                    "SGLang scalar precomputed request must return an object, "
                    f"got {type(precomputed_response).__name__}"
                )
            precomputed_result = extract_next_token_result(
                precomputed_response,
                action_a_token_id=action_a_token_id,
                action_b_token_id=action_b_token_id,
            )
            precomputed_results.append(precomputed_result)
            scalar_request_body_bytes.append(scalar_body_bytes)
            feature_ids.append(features.feature_id)
            _flush_cache(server_url)
            if parallel_samples > 1:
                parallel_payload = _with_parallel_sampling(precomputed_payload, parallel_samples)
                parallel_response, parallel_body_bytes = _post_generate(server_url, parallel_payload)
                if not isinstance(parallel_response, list):
                    raise TypeError(
                        "SGLang parallel precomputed request must return a list, "
                        f"got {type(parallel_response).__name__}"
                    )
                parallel_results.append(
                    _extract_next_token_results(
                        parallel_response,
                        action_a_token_id=action_a_token_id,
                        action_b_token_id=action_b_token_id,
                    )
                )
                parallel_request_body_bytes.append(parallel_body_bytes)
                _flush_cache(server_url)
    finally:
        _kill_process_tree(server_process.pid)
        server_process.join(timeout=30)

    artifact = write_sglang_parity_artifact(
        output,
        native_results=native_results,
        precomputed_results=precomputed_results,
        sample_ids=sample_ids,
        feature_ids=feature_ids,
        metadata={
            "checkpoint": str(checkpoint_path.resolve()),
            "dataset": str(dataset_path.resolve()),
            "host": host,
            "port": port,
            "base_gpu_id": base_gpu_id,
            "num_images": num_images,
            "action_a_token_id": action_a_token_id,
            "action_b_token_id": action_b_token_id,
            "vision_revision": cpu_backend.revision,
            "parallel_samples": parallel_samples,
            "deterministic_parallel_sampling_supported": False,
        },
    )
    if parallel_samples > 1:
        native_parallel_sampling = summarize_sglang_parallel_sampling(
            scalar_results=native_results,
            parallel_results=native_parallel_results,
            scalar_request_body_bytes=native_scalar_request_body_bytes,
            parallel_request_body_bytes=native_parallel_request_body_bytes,
            parallel_samples=parallel_samples,
        )
        parallel_sampling = summarize_sglang_parallel_sampling(
            scalar_results=precomputed_results,
            parallel_results=parallel_results,
            scalar_request_body_bytes=scalar_request_body_bytes,
            parallel_request_body_bytes=parallel_request_body_bytes,
            parallel_samples=parallel_samples,
        )
        artifact["native_parallel_sampling"] = native_parallel_sampling
        artifact["parallel_sampling"] = parallel_sampling
        artifact["passed"] = artifact["passed"] and native_parallel_sampling["passed"] and parallel_sampling["passed"]
        Path(output).write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


ParityProbe = Callable[..., dict[str, Any]]


def main(
    argv: list[str] | None = None,
    *,
    probe: ParityProbe = run_sglang_parity,
) -> dict[str, Any]:
    """Run the live probe once, persist raw evidence, and fail on divergence."""
    args = parse_args(argv)
    artifact = probe(
        checkpoint=args.checkpoint,
        dataset=args.dataset,
        output=args.output,
        num_images=args.num_images,
        host=args.host,
        port=args.port,
        base_gpu_id=args.base_gpu_id,
        parallel_samples=args.parallel_samples,
    )
    if not artifact.get("passed", False):
        raise RuntimeError(f"Live SGLang CPU-vision parity failed; inspect {args.output}")
    return artifact


if __name__ == "__main__":
    main()
