import pytest
import torch

from examples.visual_xor.train_sft import (
    EXPECTED_PARAMETER_COUNT,
    METADATA_ALLOW_PATTERNS,
    build_compact_qwen3_vl_config,
    count_parameters_on_meta,
    ensure_loadable_model_was_saved,
    load_training_model,
    load_sft_dataset,
    validate_metadata_allowlist,
    validate_metadata_only_directory,
)
from examples.visual_xor.task import write_dataset_bundle


def test_compact_qwen3_vl_config_matches_the_relax_bridge_shape():
    config = build_compact_qwen3_vl_config()

    assert config.model_type == "qwen3_vl"
    assert config.tie_word_embeddings is True
    assert config.text_config.num_hidden_layers == 12
    assert config.text_config.hidden_size == 1024
    assert config.text_config.intermediate_size == 3072
    assert config.text_config.num_attention_heads == 16
    assert config.text_config.num_key_value_heads == 8
    assert config.text_config.head_dim == 128
    assert config.text_config.rope_parameters["mrope_section"] == [24, 20, 20]
    assert config.vision_config.depth == 6
    assert config.vision_config.hidden_size == 384
    assert config.vision_config.deepstack_visual_indexes == [1, 3, 5]
    assert config.vision_config.out_hidden_size == config.text_config.hidden_size
    assert count_parameters_on_meta(config) == EXPECTED_PARAMETER_COUNT


def test_compact_qwen3_vl_model_has_tied_embeddings_on_meta():
    from transformers import Qwen3VLForConditionalGeneration

    with torch.device("meta"):
        model = Qwen3VLForConditionalGeneration(build_compact_qwen3_vl_config())

    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight


def test_metadata_allowlist_cannot_download_model_weights():
    validate_metadata_allowlist(METADATA_ALLOW_PATTERNS)

    assert all("safetensors" not in pattern for pattern in METADATA_ALLOW_PATTERNS)
    assert all("pytorch_model" not in pattern for pattern in METADATA_ALLOW_PATTERNS)


def test_metadata_directory_rejects_model_weights(tmp_path):
    for filename in ("config.json", "preprocessor_config.json", "tokenizer_config.json"):
        (tmp_path / filename).write_text("{}")
    (tmp_path / "model.safetensors").touch()

    with pytest.raises(RuntimeError, match="model weights"):
        validate_metadata_only_directory(tmp_path)


def test_saved_visual_sft_accepts_processor_config_filename(tmp_path):
    for filename in ("model.safetensors", "config.json", "processor_config.json", "tokenizer_config.json"):
        (tmp_path / filename).touch()

    ensure_loadable_model_was_saved(tmp_path)


def test_initial_checkpoint_must_be_a_complete_local_visual_sft(tmp_path):
    with pytest.raises(RuntimeError, match="did not produce Hugging Face weights"):
        load_training_model(build_compact_qwen3_vl_config(), seed=42, initial_checkpoint=str(tmp_path))


def test_sft_parquet_loads_as_multimodal_prompt_completion_dataset(tmp_path):
    paths = write_dataset_bundle(
        tmp_path,
        sft_train_examples=8,
        sft_eval_examples=8,
        xor_train_examples=8,
        xor_eval_examples=8,
    )

    dataset = load_sft_dataset(paths["sft_train"])
    example = dataset[0]

    assert example["prompt"][0]["role"] == "system"
    assert example["prompt"][1]["role"] == "user"
    assert example["completion"][0]["content"] in {"A", "B"}
    assert example["image"].size == (256, 256)
