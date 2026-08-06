# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Benchmark candidate transports for one packed Qwen3-VL vision feature.

This is a local serialization benchmark. It does not start SGLang, Ray, or a
vision encoder and it does not modify any environment.
"""

import argparse
import base64
import gc
import json
from statistics import median
from time import perf_counter
from typing import Any, Callable

import httpx
import torch


def _build_generate_body(image_data: dict[str, Any], *, parallel_samples: int) -> bytes:
    payload = {
        "input_ids": list(range(256)),
        "image_data": [image_data],
        "sampling_params": {"max_new_tokens": 8, "temperature": 1.0, "n": parallel_samples},
        "return_logprob": True,
    }
    with httpx.Client() as client:
        return client.build_request("POST", "http://127.0.0.1:30000/generate", json=payload).content


def _encode_nested_float_json(image_data: dict[str, Any]) -> dict[str, Any]:
    feature = image_data["feature"].detach().to(device="cpu", dtype=torch.float32)
    grid = image_data["image_grid_thw"].detach().to(device="cpu", dtype=torch.int64)
    return {
        "format": "precomputed_embedding",
        "feature": feature.tolist(),
        "image_grid_thw": grid.tolist(),
        "feature_id": image_data["feature_id"],
        "vision_revision": image_data["vision_revision"],
    }


def _encode_packed_bf16(image_data: dict[str, Any]) -> dict[str, Any]:
    feature = image_data["feature"].detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    feature_bytes = feature.view(torch.uint16).numpy().tobytes()
    grid = image_data["image_grid_thw"].detach().to(device="cpu", dtype=torch.int64).contiguous()
    return {
        "format": "precomputed_embedding",
        "feature_b64": base64.b64encode(feature_bytes).decode("ascii"),
        "feature_dtype": "bfloat16",
        "feature_shape": list(feature.shape),
        "image_grid_thw": grid.tolist(),
        "feature_id": image_data["feature_id"],
        "vision_revision": image_data["vision_revision"],
    }


def _decode_nested_float32(image_data: dict[str, Any]) -> torch.Tensor:
    return torch.as_tensor(image_data["feature"], dtype=torch.bfloat16)


def _decode_packed_bf16(image_data: dict[str, Any]) -> torch.Tensor:
    raw = bytearray(base64.b64decode(image_data["feature_b64"], validate=True))
    tensor = torch.frombuffer(raw, dtype=torch.uint16).view(torch.bfloat16)
    return tensor.reshape(image_data["feature_shape"]).clone()


def _measure(operation: Callable[[], Any], *, repetitions: int) -> tuple[Any, float]:
    durations = []
    result = None
    for _ in range(repetitions):
        gc.collect()
        started_at = perf_counter()
        result = operation()
        durations.append(perf_counter() - started_at)
    return result, median(durations)


def run_benchmark(*, repetitions: int) -> dict[str, Any]:
    feature = torch.linspace(-1.0, 1.0, steps=64 * 4096, dtype=torch.float32).reshape(64, 4096).to(torch.bfloat16)
    image_data = {
        "format": "precomputed_embedding",
        "feature": feature,
        "image_grid_thw": torch.tensor([[1, 8, 8]], dtype=torch.int64),
        "feature_id": "visual-xor-feature-00000000",
        "vision_revision": "qwen3-vl-visual-revision",
    }

    nested, nested_encode_seconds = _measure(
        lambda: _encode_nested_float_json(image_data),
        repetitions=repetitions,
    )
    nested_body, nested_body_seconds = _measure(
        lambda: _build_generate_body(nested, parallel_samples=4),
        repetitions=repetitions,
    )
    nested_decoded, nested_decode_seconds = _measure(
        lambda: _decode_nested_float32(nested),
        repetitions=repetitions,
    )

    packed, packed_encode_seconds = _measure(
        lambda: _encode_packed_bf16(image_data),
        repetitions=repetitions,
    )
    packed_body, packed_body_seconds = _measure(
        lambda: _build_generate_body(packed, parallel_samples=4),
        repetitions=repetitions,
    )
    packed_decoded, packed_decode_seconds = _measure(
        lambda: _decode_packed_bf16(packed),
        repetitions=repetitions,
    )
    torch.testing.assert_close(nested_decoded, feature, rtol=0, atol=0)
    torch.testing.assert_close(packed_decoded, feature, rtol=0, atol=0)

    id_image_data = {
        "format": "precomputed_embedding_id",
        "feature_id": image_data["feature_id"],
        "vision_revision": image_data["vision_revision"],
    }
    id_body = _build_generate_body(id_image_data, parallel_samples=4)
    upload_metadata = json.dumps(
        {
            "feature_id": image_data["feature_id"],
            "vision_revision": image_data["vision_revision"],
            "feature_dtype": "bfloat16",
            "feature_shape": list(feature.shape),
            "image_grid_thw": image_data["image_grid_thw"].tolist(),
        },
        separators=(",", ":"),
    ).encode("utf-8")
    upload_bytes = feature.numel() * feature.element_size() + len(upload_metadata)

    unique_features = 128
    grouped_eval_requests = 384
    nested_projected = grouped_eval_requests * len(nested_body)
    packed_projected = grouped_eval_requests * len(packed_body)
    registry_projected = unique_features * upload_bytes + grouped_eval_requests * len(id_body)

    return {
        "feature": {
            "shape": list(feature.shape),
            "dtype": str(feature.dtype),
            "raw_bytes": feature.numel() * feature.element_size() + image_data["image_grid_thw"].numel() * 8,
        },
        "repetitions": repetitions,
        "per_request": {
            "nested_float_json": {
                "body_bytes": len(nested_body),
                "feature_encode_seconds_median": nested_encode_seconds,
                "http_body_build_seconds_median": nested_body_seconds,
                "feature_decode_seconds_median": nested_decode_seconds,
            },
            "inline_bf16_base64_json": {
                "body_bytes": len(packed_body),
                "feature_encode_seconds_median": packed_encode_seconds,
                "http_body_build_seconds_median": packed_body_seconds,
                "feature_decode_seconds_median": packed_decode_seconds,
            },
            "registry_id_json": {"body_bytes": len(id_body)},
            "registry_binary_upload": {"body_bytes": upload_bytes},
        },
        "projected_grouped_evaluation": {
            "unique_features": unique_features,
            "generation_requests": grouped_eval_requests,
            "nested_float_json_bytes": nested_projected,
            "inline_bf16_base64_json_bytes": packed_projected,
            "binary_upload_plus_id_bytes": registry_projected,
            "inline_reduction_fraction": 1.0 - packed_projected / nested_projected,
            "registry_reduction_fraction": 1.0 - registry_projected / nested_projected,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    result = run_benchmark(repetitions=args.repetitions)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            output_file.write(f"{rendered}\n")
    print(rendered)


if __name__ == "__main__":
    main()
