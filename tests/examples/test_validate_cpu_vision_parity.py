import ast
import builtins
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from examples.visual_xor import validate_cpu_vision_parity as parity_cli
from examples.visual_xor.validate_cpu_vision_parity import (
    forward_with_precomputed_qwen3_vl_features,
    resolve_action_token_ids,
)
from relax.backends.vision.qwen3_vl import Qwen3VLFrozenVisionFeatures


class _RecordingEmbedding(torch.nn.Module):
    def __init__(self, *, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.arange(vocab_size * hidden_size, dtype=torch.float32).reshape(vocab_size, hidden_size),
            requires_grad=False,
        )
        self.input_ids = None

    def forward(self, input_ids):
        self.input_ids = input_ids
        return torch.nn.functional.embedding(input_ids, self.weight)


class _ForbiddenVisualTower(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0

    def forward(self, *args, **kwargs):
        self.forward_calls += 1
        raise AssertionError("the parity path must never call the visual tower")


class _RecordingLanguageModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.kwargs = None

    def forward(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(last_hidden_state=kwargs["inputs_embeds"] + 0.5)


class _RecordingLMHead(torch.nn.Module):
    def __init__(self, *, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.arange(vocab_size * hidden_size, dtype=torch.float32).reshape(vocab_size, hidden_size),
            requires_grad=False,
        )
        self.hidden_states = None

    def forward(self, hidden_states):
        self.hidden_states = hidden_states
        return torch.nn.functional.linear(hidden_states, self.weight)


class _FakeQwen3VLBackbone:
    def __init__(self, *, hidden_size: int = 3, deepstack_count: int = 3) -> None:
        self.config = SimpleNamespace(
            image_token_id=99,
            vision_config=SimpleNamespace(
                spatial_merge_size=2,
                deepstack_visual_indexes=list(range(deepstack_count)),
            ),
            text_config=SimpleNamespace(hidden_size=hidden_size),
        )
        self.visual = _ForbiddenVisualTower()
        self.language_model = _RecordingLanguageModel()
        self.embedding = _RecordingEmbedding(vocab_size=128, hidden_size=hidden_size)
        self.rope_kwargs = None
        self.position_ids = torch.tensor(
            [
                [[0, 1, 2, 2, 2, 3]],
                [[0, 1, 1, 1, 2, 3]],
                [[0, 1, 1, 2, 1, 3]],
            ],
            dtype=torch.long,
        )

    def get_input_embeddings(self):
        return self.embedding

    def get_rope_index(self, **kwargs):
        self.rope_kwargs = kwargs
        return self.position_ids, torch.tensor([[0]], dtype=torch.long)


class _FakeQwen3VLForConditionalGeneration:
    def __init__(self, *, hidden_size: int = 3, deepstack_count: int = 3) -> None:
        self.model = _FakeQwen3VLBackbone(
            hidden_size=hidden_size,
            deepstack_count=deepstack_count,
        )
        self.lm_head = _RecordingLMHead(hidden_size=hidden_size, vocab_size=7)


def _model_inputs() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[10, 99, 99, 99, 99, 11]], dtype=torch.long),
        "attention_mask": torch.ones((1, 6), dtype=torch.long),
        "image_grid_thw": torch.tensor([[1, 4, 4]], dtype=torch.long),
    }


def _features(
    *,
    image_grid_thw: torch.Tensor | None = None,
    image_token_count: int = 4,
    hidden_size: int = 3,
    deepstack_count: int = 3,
) -> Qwen3VLFrozenVisionFeatures:
    grid = image_grid_thw if image_grid_thw is not None else torch.tensor([[1, 4, 4]], dtype=torch.long)
    return Qwen3VLFrozenVisionFeatures(
        image_grid_thw=grid,
        vision_embeds=torch.arange(
            image_token_count * hidden_size,
            dtype=torch.float32,
        ).reshape(image_token_count, hidden_size)
        + 1000,
        deepstack_visual_embeds=tuple(
            torch.full(
                (image_token_count, hidden_size),
                2000.0 + index,
                dtype=torch.float32,
            )
            for index in range(deepstack_count)
        ),
        feature_id="feature-1",
        vision_revision="vision-revision-1",
    )


def test_forward_with_precomputed_features_runs_language_only_qwen3_vl_path():
    model = _FakeQwen3VLForConditionalGeneration()
    model_inputs = _model_inputs()
    features = _features()

    logits = forward_with_precomputed_qwen3_vl_features(model, model_inputs, features)

    image_mask = model_inputs["input_ids"] == model.model.config.image_token_id
    expected_embeddings = model.model.embedding(model_inputs["input_ids"]).clone()
    expected_embeddings[image_mask] = features.vision_embeds

    assert model.model.visual.forward_calls == 0
    assert model.model.embedding.input_ids is model_inputs["input_ids"]
    assert model.model.rope_kwargs == {
        "input_ids": model_inputs["input_ids"],
        "mm_token_type_ids": None,
        "image_grid_thw": model_inputs["image_grid_thw"],
        "video_grid_thw": None,
        "attention_mask": model_inputs["attention_mask"],
    }
    language_kwargs = model.model.language_model.kwargs
    torch.testing.assert_close(language_kwargs["inputs_embeds"], expected_embeddings)
    assert language_kwargs["attention_mask"] is model_inputs["attention_mask"]
    assert language_kwargs["position_ids"] is model.model.position_ids
    assert torch.equal(language_kwargs["visual_pos_masks"], image_mask)
    assert language_kwargs["deepstack_visual_embeds"] == features.deepstack_visual_embeds
    assert language_kwargs["use_cache"] is False
    torch.testing.assert_close(model.lm_head.hidden_states, expected_embeddings + 0.5)
    torch.testing.assert_close(logits, torch.nn.functional.linear(expected_embeddings + 0.5, model.lm_head.weight))


@pytest.mark.parametrize(
    ("unexpected_inputs", "error_pattern"),
    [
        ({"pixel_values": torch.ones((16, 3))}, "pixel_values.*precomputed"),
        ({"pixel_values_videos": torch.ones((16, 3))}, "video"),
        ({"video_grid_thw": torch.tensor([[1, 4, 4]])}, "video"),
    ],
)
def test_forward_with_precomputed_features_rejects_pixel_and_video_inputs(
    unexpected_inputs,
    error_pattern,
):
    model = _FakeQwen3VLForConditionalGeneration()
    model_inputs = {**_model_inputs(), **unexpected_inputs}

    with pytest.raises((ValueError, NotImplementedError), match=error_pattern):
        forward_with_precomputed_qwen3_vl_features(model, model_inputs, _features())

    assert model.model.visual.forward_calls == 0
    assert model.model.language_model.kwargs is None


@pytest.mark.parametrize(
    ("model_inputs", "features", "error_pattern"),
    [
        (
            {
                **_model_inputs(),
                "input_ids": torch.tensor([[10, 99, 99, 99, 11]], dtype=torch.long),
                "attention_mask": torch.ones((1, 5), dtype=torch.long),
            },
            _features(),
            "image token count",
        ),
        (
            _model_inputs(),
            _features(image_grid_thw=torch.tensor([[1, 2, 8]], dtype=torch.long)),
            "image_grid_thw.*match",
        ),
        (
            _model_inputs(),
            _features(hidden_size=4),
            "vision_embeds.*shape",
        ),
        (
            _model_inputs(),
            _features(deepstack_count=2),
            "DeepStack feature count",
        ),
    ],
)
def test_forward_with_precomputed_features_validates_qwen3_vl_feature_contract(
    model_inputs,
    features,
    error_pattern,
):
    model = _FakeQwen3VLForConditionalGeneration()

    with pytest.raises(ValueError, match=error_pattern):
        forward_with_precomputed_qwen3_vl_features(model, model_inputs, features)

    assert model.model.visual.forward_calls == 0
    assert model.model.language_model.kwargs is None


class _FakeTokenizer:
    def __init__(self, encodings: dict[str, list[int]]) -> None:
        self.encodings = encodings
        self.calls = []

    def encode(self, text, *, add_special_tokens):
        self.calls.append((text, add_special_tokens))
        return self.encodings[text]


def test_resolve_action_token_ids_requires_single_token_a_and_b():
    tokenizer = _FakeTokenizer({"A": [17], "B": [23]})

    action_a_token_id, action_b_token_id = resolve_action_token_ids(tokenizer)

    assert (action_a_token_id, action_b_token_id) == (17, 23)
    assert tokenizer.calls == [("A", False), ("B", False)]


@pytest.mark.parametrize(
    ("encodings", "error_pattern"),
    [
        ({"A": [17, 18], "B": [23]}, r"A.*exactly one token"),
        ({"A": [17], "B": []}, r"B.*exactly one token"),
    ],
)
def test_resolve_action_token_ids_rejects_non_single_token_actions(encodings, error_pattern):
    tokenizer = _FakeTokenizer(encodings)

    with pytest.raises(ValueError, match=error_pattern):
        resolve_action_token_ids(tokenizer)


def _complete_cli_argv(output_path: str = "/artifacts/cpu-vision-parity.json") -> list[str]:
    return [
        "--checkpoint",
        "/models/qwen3-vl",
        "--dataset",
        "/data/visual-xor.parquet",
        "--output",
        output_path,
        "--device",
        "cuda:0",
        "--num-images",
        "8",
    ]


def test_parse_args_accepts_required_parity_inputs():
    args = parity_cli.parse_args(_complete_cli_argv())

    assert args.checkpoint == "/models/qwen3-vl"
    assert args.dataset == "/data/visual-xor.parquet"
    assert args.output == "/artifacts/cpu-vision-parity.json"
    assert args.device == "cuda:0"
    assert args.num_images == 8


@pytest.mark.parametrize(
    "required_option",
    [
        "--checkpoint",
        "--dataset",
        "--output",
        "--device",
        "--num-images",
    ],
)
def test_parse_args_requires_each_parity_input(required_option):
    argv = _complete_cli_argv()
    option_index = argv.index(required_option)
    del argv[option_index : option_index + 2]

    with pytest.raises(SystemExit):
        parity_cli.parse_args(argv)


@pytest.mark.parametrize("num_images", ["0", "-1"])
def test_parse_args_rejects_nonpositive_num_images(num_images):
    argv = _complete_cli_argv()
    argv[argv.index("--num-images") + 1] = num_images

    with pytest.raises(SystemExit):
        parity_cli.parse_args(argv)


def test_main_runs_injected_parity_runner_once_and_writes_metadata(tmp_path):
    output_path = tmp_path / "nested" / "cpu-vision-parity.json"
    runner_calls = []
    feature_metrics = {"vision_embeds": {"shape": [4, 8]}}
    response_metrics = {"mean_kl_divergence": 0.0, "per_sample": []}
    metadata = {
        "feature_ids": ["feature-1", "feature-2"],
        "vision_revision": "revision-1",
    }

    def parity_runner(checkpoint, dataset, device, num_images):
        runner_calls.append(
            {
                "checkpoint": checkpoint,
                "dataset": dataset,
                "device": device,
                "num_images": num_images,
            }
        )
        return feature_metrics, response_metrics, metadata

    artifact = parity_cli.main(
        _complete_cli_argv(str(output_path)),
        parity_runner=parity_runner,
    )

    assert runner_calls == [
        {
            "checkpoint": "/models/qwen3-vl",
            "dataset": "/data/visual-xor.parquet",
            "device": "cuda:0",
            "num_images": 8,
        }
    ]
    assert artifact == json.loads(output_path.read_text())
    assert artifact["feature_streams"] == feature_metrics
    assert artifact["response_logits"] == response_metrics
    assert artifact["metadata"] == metadata


def test_module_execution_delegates_to_main():
    module = ast.parse(Path(parity_cli.__file__).read_text())
    main_guards = [
        statement
        for statement in module.body
        if isinstance(statement, ast.If)
        and isinstance(statement.test, ast.Compare)
        and isinstance(statement.test.left, ast.Name)
        and statement.test.left.id == "__name__"
        and len(statement.test.ops) == 1
        and isinstance(statement.test.ops[0], ast.Eq)
        and len(statement.test.comparators) == 1
        and isinstance(statement.test.comparators[0], ast.Constant)
        and statement.test.comparators[0].value == "__main__"
    ]

    assert len(main_guards) == 1
    assert len(main_guards[0].body) == 1
    call_statement = main_guards[0].body[0]
    assert isinstance(call_statement, ast.Expr)
    assert isinstance(call_statement.value, ast.Call)
    assert isinstance(call_statement.value.func, ast.Name)
    assert call_statement.value.func.id == "main"


def test_parity_cli_module_does_not_import_model_libraries_eagerly(monkeypatch):
    module_path = Path(parity_cli.__file__)
    spec = importlib.util.spec_from_file_location("_cpu_vision_parity_import_test", module_path)
    module = importlib.util.module_from_spec(spec)
    original_import = builtins.__import__

    def reject_heavy_imports(name, *args, **kwargs):
        if name.split(".", maxsplit=1)[0] in {"pandas", "transformers"}:
            raise AssertionError(f"eager heavy import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_heavy_imports)
    spec.loader.exec_module(module)
