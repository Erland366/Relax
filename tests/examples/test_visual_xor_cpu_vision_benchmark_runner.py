import ast
import builtins
import importlib.util
import json
from pathlib import Path

import pytest

from examples.visual_xor import benchmark_cpu_vision_scaling as benchmark


def test_build_benchmark_matrix_returns_deterministic_cartesian_product():
    matrix = benchmark.build_benchmark_matrix(
        [1, 2],
        [2, 4],
        [8, 16],
    )

    assert matrix == [
        {"num_replicas": 1, "threads_per_replica": 2, "batch_size": 8},
        {"num_replicas": 1, "threads_per_replica": 2, "batch_size": 16},
        {"num_replicas": 1, "threads_per_replica": 4, "batch_size": 8},
        {"num_replicas": 1, "threads_per_replica": 4, "batch_size": 16},
        {"num_replicas": 2, "threads_per_replica": 2, "batch_size": 8},
        {"num_replicas": 2, "threads_per_replica": 2, "batch_size": 16},
        {"num_replicas": 2, "threads_per_replica": 4, "batch_size": 8},
        {"num_replicas": 2, "threads_per_replica": 4, "batch_size": 16},
    ]


@pytest.mark.parametrize(
    ("replica_counts", "thread_counts", "batch_sizes", "error_pattern"),
    [
        ([], [1], [1], "replica_counts.*non-empty"),
        ([1], [], [1], "thread_counts.*non-empty"),
        ([1], [1], [], "batch_sizes.*non-empty"),
        ([0], [1], [1], "replica_counts.*positive"),
        ([1], [-1], [1], "thread_counts.*positive"),
        ([1], [1], [0], "batch_sizes.*positive"),
        ([1, 1], [1], [1], "replica_counts.*duplicate"),
        ([1], [2, 2], [1], "thread_counts.*duplicate"),
        ([1], [1], [4, 4], "batch_sizes.*duplicate"),
    ],
)
def test_build_benchmark_matrix_rejects_invalid_dimensions(
    replica_counts,
    thread_counts,
    batch_sizes,
    error_pattern,
):
    with pytest.raises(ValueError, match=error_pattern):
        benchmark.build_benchmark_matrix(replica_counts, thread_counts, batch_sizes)


def test_run_scaling_matrix_measures_each_configuration_once_and_applies_capacity_gate():
    matrix = [
        {"num_replicas": 1, "threads_per_replica": 2, "batch_size": 8},
        {"num_replicas": 2, "threads_per_replica": 4, "batch_size": 16},
    ]
    measurements = iter(
        [
            {"completed_images": 100, "wall_seconds": 2.0},
            {"completed_images": 80, "wall_seconds": 2.0},
        ]
    )
    measured_configurations = []

    def measure_configuration(configuration):
        measured_configurations.append(dict(configuration))
        return next(measurements)

    summary = benchmark.run_scaling_matrix(
        matrix,
        measure_configuration,
        peak_unique_images_per_second=40.0,
    )

    assert measured_configurations == matrix
    assert summary["results"] == [
        {
            **matrix[0],
            "num_images": 100,
            "wall_time_seconds": 2.0,
            "total_reserved_cpus": 2,
            "images_per_second": 50.0,
            "capacity_ratio": 1.25,
            "gate_passed": True,
        },
        {
            **matrix[1],
            "num_images": 80,
            "wall_time_seconds": 2.0,
            "total_reserved_cpus": 8,
            "images_per_second": 40.0,
            "capacity_ratio": 1.0,
            "gate_passed": False,
        },
    ]
    assert summary["recommended_configuration"] == summary["results"][0]


def _complete_cli_argv() -> list[str]:
    return [
        "--checkpoint",
        "/models/qwen3-vl",
        "--dataset",
        "/data/visual-xor.jsonl",
        "--output",
        "/artifacts/cpu-vision-scaling.json",
        "--peak-unique-images-per-second",
        "40",
        "--replica-counts",
        "1,2",
        "--thread-counts",
        "2,4",
        "--batch-sizes",
        "8,16",
        "--num-images",
        "256",
        "--repeats",
        "3",
    ]


def test_parse_args_accepts_required_paths_and_comma_separated_matrix_dimensions():
    args = benchmark.parse_args(_complete_cli_argv())

    assert args.checkpoint == "/models/qwen3-vl"
    assert args.dataset == "/data/visual-xor.jsonl"
    assert args.output == "/artifacts/cpu-vision-scaling.json"
    assert args.peak_unique_images_per_second == 40.0
    assert args.replica_counts == [1, 2]
    assert args.thread_counts == [2, 4]
    assert args.batch_sizes == [8, 16]
    assert args.num_images == 256
    assert args.repeats == 3


@pytest.mark.parametrize(
    "required_option",
    [
        "--checkpoint",
        "--dataset",
        "--output",
        "--peak-unique-images-per-second",
    ],
)
def test_parse_args_requires_each_input_and_output_option(required_option):
    argv = _complete_cli_argv()
    option_index = argv.index(required_option)
    del argv[option_index : option_index + 2]

    with pytest.raises(SystemExit):
        benchmark.parse_args(argv)


@pytest.mark.parametrize("option", ["--num-images", "--repeats"])
def test_parse_args_rejects_nonpositive_work_counts(option):
    argv = _complete_cli_argv()
    argv[argv.index(option) + 1] = "0"

    with pytest.raises(SystemExit):
        benchmark.parse_args(argv)


def test_main_builds_and_measures_the_requested_matrix_with_one_factory(tmp_path):
    output_path = tmp_path / "cpu-vision-scaling.json"
    argv = [
        "--checkpoint",
        "/models/qwen3-vl",
        "--dataset",
        "/data/visual-xor.jsonl",
        "--output",
        str(output_path),
        "--peak-unique-images-per-second",
        "40",
        "--replica-counts",
        "1,2",
        "--thread-counts",
        "4",
        "--batch-sizes",
        "8,16",
        "--num-images",
        "256",
        "--repeats",
        "3",
    ]
    factory_calls = []
    measured_configurations = []

    def measurement_factory(checkpoint, dataset, num_images, repeats):
        factory_calls.append(
            {
                "checkpoint": checkpoint,
                "dataset": dataset,
                "num_images": num_images,
                "repeats": repeats,
            }
        )

        def measure_configuration(configuration):
            measured_configurations.append(dict(configuration))
            return {"completed_images": 256, "wall_seconds": 4.0}

        return measure_configuration

    artifact = benchmark.main(argv, measurement_factory=measurement_factory)

    assert factory_calls == [
        {
            "checkpoint": "/models/qwen3-vl",
            "dataset": "/data/visual-xor.jsonl",
            "num_images": 256,
            "repeats": 3,
        }
    ]
    assert measured_configurations == benchmark.build_benchmark_matrix([1, 2], [4], [8, 16])
    assert artifact == json.loads(output_path.read_text())
    assert artifact["schema_version"] == 1
    assert artifact["passed"] is True


def test_module_execution_delegates_to_main():
    module = ast.parse(Path(benchmark.__file__).read_text())
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


def test_benchmark_module_does_not_import_model_or_dataset_dependencies_eagerly(monkeypatch):
    module_path = Path(benchmark.__file__)
    spec = importlib.util.spec_from_file_location("_benchmark_cpu_vision_scaling_import_test", module_path)
    module = importlib.util.module_from_spec(spec)
    original_import = builtins.__import__

    def reject_heavy_imports(name, *args, **kwargs):
        if name.split(".", maxsplit=1)[0] in {"pandas", "transformers"}:
            raise AssertionError(f"eager heavy import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_heavy_imports)
    spec.loader.exec_module(module)
