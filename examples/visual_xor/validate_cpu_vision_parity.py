"""Validate local Qwen3-VL CPU vision features against the native GPU path."""

import argparse
import io
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from examples.visual_xor.cpu_vision_parity import (
    FEATURE_STREAM_NAMES,
    compare_feature_streams,
    compare_response_logits,
    write_parity_artifact,
)
from examples.visual_xor.task import SFT_SYSTEM_PROMPT, XOR_USER_PROMPT
from relax.backends.vision.qwen3_vl import Qwen3VLFrozenVisionFeatures


def resolve_action_token_ids(tokenizer: Any) -> tuple[int, int]:
    """Resolve the visual XOR actions, requiring one token per action."""
    token_ids: list[int] = []
    for action in ("A", "B"):
        encoded = tokenizer.encode(action, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"Visual XOR action {action!r} must encode to exactly one token, got {encoded}"
            )
        token_ids.append(encoded[0])
    return token_ids[0], token_ids[1]


def forward_with_precomputed_qwen3_vl_features(
    model: Any,
    model_inputs: Mapping[str, torch.Tensor],
    features: Qwen3VLFrozenVisionFeatures,
) -> torch.Tensor:
    """Run the Qwen3-VL language path with already-projected image features."""
    if model_inputs.get("pixel_values") is not None:
        raise ValueError("pixel_values cannot be supplied with precomputed vision features")
    if (
        model_inputs.get("pixel_values_videos") is not None
        or model_inputs.get("video_grid_thw") is not None
    ):
        raise NotImplementedError("video inputs are not supported by the precomputed vision parity path")

    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    image_grid_thw = model_inputs["image_grid_thw"]
    backbone = model.model

    if image_grid_thw.shape != features.image_grid_thw.shape or not torch.equal(
        image_grid_thw, features.image_grid_thw.to(device=image_grid_thw.device)
    ):
        raise ValueError("model input image_grid_thw must match the precomputed feature image_grid_thw")

    image_mask = input_ids == backbone.config.image_token_id
    image_token_count = int(image_mask.sum())
    features.validate(
        spatial_merge_size=backbone.config.vision_config.spatial_merge_size,
        image_token_count=image_token_count,
        hidden_size=backbone.config.text_config.hidden_size,
        expected_deepstack_count=len(backbone.config.vision_config.deepstack_visual_indexes),
    )

    inputs_embeds = backbone.get_input_embeddings()(input_ids).clone()
    embedding_device = inputs_embeds.device
    embedding_dtype = inputs_embeds.dtype
    vision_embeds = features.vision_embeds.to(
        device=embedding_device,
        dtype=embedding_dtype,
    )
    inputs_embeds[image_mask] = vision_embeds
    deepstack_visual_embeds = tuple(
        feature.to(device=embedding_device, dtype=embedding_dtype)
        for feature in features.deepstack_visual_embeds
    )

    if hasattr(backbone, "get_rope_index"):
        position_ids, _ = backbone.get_rope_index(
            input_ids=input_ids,
            mm_token_type_ids=model_inputs.get("mm_token_type_ids"),
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            attention_mask=attention_mask,
        )
    else:
        position_ids = backbone.compute_3d_position_ids(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            attention_mask=attention_mask,
            past_key_values=None,
            mm_token_type_ids=model_inputs.get("mm_token_type_ids"),
        )
    language_output = backbone.language_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        visual_pos_masks=image_mask,
        deepstack_visual_embeds=deepstack_visual_embeds,
        use_cache=False,
    )
    return model.lm_head(language_output.last_hidden_state)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {parsed}")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the explicit inputs needed for a reproducible local parity run."""
    parser = argparse.ArgumentParser(
        description="Compare Qwen3-VL native GPU vision with frozen CPU precomputed features."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--num-images", required=True, type=_positive_int)
    return parser.parse_args(argv)


def _unwrap_image_bytes(value: Any) -> bytes:
    if isinstance(value, (list, tuple)) or getattr(value, "ndim", 0) == 1:
        if len(value) != 1:
            raise ValueError(f"parity dataset rows must contain exactly one image, got {len(value)}")
        value = value[0]
    if isinstance(value, dict):
        value = value.get("bytes")
    if not isinstance(value, bytes):
        raise TypeError(f"parity dataset image must resolve to bytes, got {type(value).__name__}")
    return value


def _read_unique_images(dataset: Path, num_images: int) -> tuple[list[Any], list[str]]:
    import pandas as pd
    from PIL import Image

    if not dataset.is_file():
        raise FileNotFoundError(f"CPU vision parity dataset does not exist: {dataset}")
    if dataset.suffix == ".parquet":
        frame = pd.read_parquet(dataset)
    elif dataset.suffix in {".json", ".jsonl"}:
        frame = pd.read_json(dataset, lines=dataset.suffix == ".jsonl")
    else:
        raise ValueError(
            "CPU vision parity dataset must be a .parquet, .json, or .jsonl file, "
            f"got {dataset}"
        )
    if "image" not in frame:
        raise ValueError(f"CPU vision parity dataset has no 'image' column: {dataset}")

    images = []
    sample_ids = []
    seen_images: set[bytes] = set()
    for row_index, row in frame.iterrows():
        image_bytes = _unwrap_image_bytes(row["image"])
        if image_bytes in seen_images:
            continue
        seen_images.add(image_bytes)
        with Image.open(io.BytesIO(image_bytes)) as image:
            images.append(image.convert("RGB").copy())
        metadata = row.get("metadata")
        sample_id = metadata.get("sample_id") if isinstance(metadata, dict) else row.get("sample_id")
        sample_ids.append(str(sample_id) if sample_id is not None else str(row_index))
        if len(images) == num_images:
            break

    if len(images) != num_images:
        raise ValueError(
            f"requested {num_images} unique images, but {dataset} contains only "
            f"{len(images)} unique decodable images"
        )
    return images, sample_ids


def _prepare_model_inputs(processor: Any, image: Any) -> dict[str, torch.Tensor]:
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
    required = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
    missing = sorted(required.difference(encoded))
    if missing:
        raise ValueError(f"Qwen3-VL processor output is missing required fields: {missing}")
    if encoded.get("pixel_values_videos") is not None or encoded.get("video_grid_thw") is not None:
        raise NotImplementedError("video inputs are not supported by the CPU vision parity diagnostic")
    return dict(encoded)


def _feature_streams(features: Qwen3VLFrozenVisionFeatures) -> dict[str, torch.Tensor]:
    streams = {"vision_embeds": features.vision_embeds.detach().to(device="cpu", dtype=torch.float64)}
    streams.update(
        {
            f"deepstack_visual_embeds_{index}": feature.detach().to(device="cpu", dtype=torch.float64)
            for index, feature in enumerate(features.deepstack_visual_embeds)
        }
    )
    if tuple(streams) != FEATURE_STREAM_NAMES:
        raise ValueError(
            f"Qwen3-VL visual output streams must be exactly {FEATURE_STREAM_NAMES}, got {tuple(streams)}"
        )
    return streams


def _native_gpu_features(
    model: Any,
    *,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
) -> Qwen3VLFrozenVisionFeatures:
    visual_model = model.model.visual
    with torch.inference_mode():
        output = visual_model(
            pixel_values.to(device=visual_model.device, dtype=visual_model.dtype),
            grid_thw=image_grid_thw.to(device=visual_model.device),
            return_dict=True,
        )
    return Qwen3VLFrozenVisionFeatures(
        image_grid_thw=image_grid_thw.detach().cpu(),
        vision_embeds=output.pooler_output,
        deepstack_visual_embeds=tuple(output.deepstack_features),
    )


def _to_device_inputs(model_inputs: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: value.to(device)
        for name, value in model_inputs.items()
        if isinstance(value, torch.Tensor)
    }


def run_local_hf_parity(
    checkpoint: str,
    dataset: str,
    device: str,
    num_images: int,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Run deterministic native-GPU versus frozen-CPU Qwen3-VL parity."""
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    checkpoint_path = Path(checkpoint)
    dataset_path = Path(dataset)
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"CPU vision parity checkpoint does not exist: {checkpoint_path}")

    requested_device = torch.device(device)
    if requested_device.type != "cuda":
        raise ValueError(f"CPU vision parity requires a CUDA device, got {device!r}")
    if not torch.cuda.is_available():
        raise RuntimeError(f"CPU vision parity requested {device!r}, but CUDA is unavailable")
    device_index = requested_device.index if requested_device.index is not None else torch.cuda.current_device()
    if device_index >= torch.cuda.device_count():
        raise ValueError(
            f"CPU vision parity requested {device!r}, but only {torch.cuda.device_count()} CUDA devices are visible"
        )

    from relax.backends.vision.qwen3_vl import build_qwen3_vl_cpu_vision_backend

    processor = AutoProcessor.from_pretrained(
        checkpoint_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        checkpoint_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to(requested_device)
    model.eval()
    cpu_backend = build_qwen3_vl_cpu_vision_backend(checkpoint_path)
    images, sample_ids = _read_unique_images(dataset_path, num_images)
    action_a_token_id, action_b_token_id = resolve_action_token_ids(processor.tokenizer)

    reference_stream_parts = {name: [] for name in FEATURE_STREAM_NAMES}
    candidate_stream_parts = {name: [] for name in FEATURE_STREAM_NAMES}
    reference_logits = []
    candidate_logits = []
    feature_ids = []

    for image in images:
        cpu_inputs = _prepare_model_inputs(processor, image)
        pixel_values = cpu_inputs["pixel_values"].detach().cpu()
        image_grid_thw = cpu_inputs["image_grid_thw"].detach().cpu()
        cpu_features = cpu_backend.encode(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        gpu_features = _native_gpu_features(
            model,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        for name, tensor in _feature_streams(gpu_features).items():
            reference_stream_parts[name].append(tensor)
        for name, tensor in _feature_streams(cpu_features).items():
            candidate_stream_parts[name].append(tensor)

        gpu_inputs = _to_device_inputs(cpu_inputs, requested_device)
        with torch.inference_mode():
            native_output = model(**gpu_inputs, use_cache=False, logits_to_keep=1)
            precomputed_output = forward_with_precomputed_qwen3_vl_features(
                model,
                {
                    name: value
                    for name, value in gpu_inputs.items()
                    if name not in {"pixel_values", "pixel_values_videos", "video_grid_thw"}
                },
                cpu_features,
            )
        reference_logits.append(native_output.logits[:, -1, :].detach().to(device="cpu", dtype=torch.float64))
        candidate_logits.append(precomputed_output[:, -1, :].detach().to(device="cpu", dtype=torch.float64))
        feature_ids.append(cpu_features.feature_id)

    feature_metrics = compare_feature_streams(
        {name: torch.cat(parts, dim=0) for name, parts in reference_stream_parts.items()},
        {name: torch.cat(parts, dim=0) for name, parts in candidate_stream_parts.items()},
    )
    response_metrics = compare_response_logits(
        torch.cat(reference_logits, dim=0),
        torch.cat(candidate_logits, dim=0),
        action_a_token_id=action_a_token_id,
        action_b_token_id=action_b_token_id,
    )
    metadata = {
        "checkpoint": str(checkpoint_path.resolve()),
        "dataset": str(dataset_path.resolve()),
        "device": str(requested_device),
        "num_images": num_images,
        "sample_ids": sample_ids,
        "feature_ids": feature_ids,
        "vision_revision": cpu_backend.revision,
        "action_a_token_id": action_a_token_id,
        "action_b_token_id": action_b_token_id,
    }
    return feature_metrics, response_metrics, metadata


ParityRunner = Callable[
    [str, str, str, int],
    tuple[dict[str, object], dict[str, object], dict[str, object]],
]


def main(
    argv: list[str] | None = None,
    *,
    parity_runner: ParityRunner = run_local_hf_parity,
) -> dict[str, object]:
    """Run parity once and write a versioned JSON artifact."""
    args = parse_args(argv)
    feature_metrics, response_metrics, metadata = parity_runner(
        args.checkpoint,
        args.dataset,
        args.device,
        args.num_images,
    )
    return write_parity_artifact(
        args.output,
        feature_metrics=feature_metrics,
        response_metrics=response_metrics,
        metadata=metadata,
    )


if __name__ == "__main__":
    main()
