"""Benchmark and summarize frozen CPU vision scaling measurements."""

import argparse
import io
import json
import math
import multiprocessing
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


CAPACITY_RATIO_MIN = 1.25
_CONFIGURATION_FIELDS = ("num_replicas", "threads_per_replica", "batch_size")
_POSITIVE_FIELDS = (*_CONFIGURATION_FIELDS, "num_images", "wall_time_seconds")
_WORKER_BACKEND: Any = None
_WORKER_INPUTS: tuple[tuple[Any, Any], ...] = ()


@dataclass(frozen=True)
class _PreparedVisionInputs:
    pixel_values: Any
    image_grid_thw: Any


def _validate_positive_number(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def _validate_matrix_dimension(name: str, values: Sequence[int]) -> None:
    if not values:
        raise ValueError(f"{name} must be non-empty")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError(f"{name} values must be positive integers")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicate values")


def build_benchmark_matrix(
    replica_counts: Sequence[int],
    thread_counts: Sequence[int],
    batch_sizes: Sequence[int],
) -> list[dict[str, int]]:
    """Build a deterministic Cartesian product of validated scaling dimensions."""
    _validate_matrix_dimension("replica_counts", replica_counts)
    _validate_matrix_dimension("thread_counts", thread_counts)
    _validate_matrix_dimension("batch_sizes", batch_sizes)

    return [
        {
            "num_replicas": num_replicas,
            "threads_per_replica": threads_per_replica,
            "batch_size": batch_size,
        }
        for num_replicas in replica_counts
        for threads_per_replica in thread_counts
        for batch_size in batch_sizes
    ]


def run_scaling_matrix(
    matrix: Sequence[Mapping[str, int]],
    measure_configuration: Callable[[Mapping[str, int]], Mapping[str, int | float]],
    *,
    peak_unique_images_per_second: float,
) -> dict[str, object]:
    """Measure each matrix entry exactly once and summarize its capacity."""
    records = []
    for configuration in matrix:
        measurement = dict(measure_configuration(configuration))
        extra_measurements = {
            key: value
            for key, value in measurement.items()
            if key not in {"completed_images", "wall_seconds"}
        }
        records.append(
            {
                **extra_measurements,
                **configuration,
                "num_images": measurement["completed_images"],
                "wall_time_seconds": measurement["wall_seconds"],
            }
        )
    return summarize_scaling_results(
        records,
        peak_unique_images_per_second=peak_unique_images_per_second,
    )


def _parse_positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _parse_positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def _parse_int_list(value: str) -> list[int]:
    try:
        parsed = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a comma-separated list of integers") from error
    try:
        _validate_matrix_dimension("matrix dimension", parsed)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the import-light CPU vision scaling benchmark CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--peak-unique-images-per-second", required=True, type=_parse_positive_float)
    parser.add_argument("--replica-counts", required=True, type=_parse_int_list)
    parser.add_argument("--thread-counts", required=True, type=_parse_int_list)
    parser.add_argument("--batch-sizes", required=True, type=_parse_int_list)
    parser.add_argument("--num-images", required=True, type=_parse_positive_int)
    parser.add_argument("--repeats", required=True, type=_parse_positive_int)
    return parser.parse_args(argv)


def summarize_scaling_results(
    records: Sequence[Mapping[str, int | float]],
    *,
    peak_unique_images_per_second: float,
) -> dict[str, object]:
    """Derive capacity metrics and choose the smallest passing configuration."""
    _validate_positive_number("peak_unique_images_per_second", peak_unique_images_per_second)

    results = []
    seen_configurations: set[tuple[int | float, int | float, int | float]] = set()
    for record in records:
        for field in _POSITIVE_FIELDS:
            if field not in record:
                raise ValueError(f"scaling record is missing required field {field}")
            _validate_positive_number(field, record[field])

        configuration = tuple(record[field] for field in _CONFIGURATION_FIELDS)
        if configuration in seen_configurations:
            replicas, threads, batch_size = configuration
            raise ValueError(
                "duplicate scaling configuration: "
                f"replicas={replicas}, threads={threads}, batch_size={batch_size}"
            )
        seen_configurations.add(configuration)

        result = dict(record)
        result["total_reserved_cpus"] = record["num_replicas"] * record["threads_per_replica"]
        result["images_per_second"] = record["num_images"] / record["wall_time_seconds"]
        result["capacity_ratio"] = result["images_per_second"] / peak_unique_images_per_second
        result["gate_passed"] = result["capacity_ratio"] >= CAPACITY_RATIO_MIN
        results.append(result)

    passing_results = [result for result in results if result["gate_passed"]]
    recommended_configuration = (
        min(
            passing_results,
            key=lambda result: (
                result["total_reserved_cpus"],
                -result["images_per_second"],
                result["num_replicas"],
                result["threads_per_replica"],
                result["batch_size"],
            ),
        )
        if passing_results
        else None
    )

    return {
        "peak_unique_images_per_second": peak_unique_images_per_second,
        "results": results,
        "recommended_configuration": recommended_configuration,
    }


def write_scaling_artifact(
    output_path: str | Path,
    summary: Mapping[str, object],
) -> dict[str, object]:
    """Write a versioned JSON artifact containing scaling measurements."""
    artifact = {
        "schema_version": 1,
        "passed": summary.get("recommended_configuration") is not None,
        "gates": {
            "capacity_ratio_min": CAPACITY_RATIO_MIN,
            "demand_metric": "peak_unique_images_per_second",
        },
        **summary,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2) + "\n")
    return artifact


def _unwrap_image_bytes(value: object) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, Mapping):
        if value.get("bytes") is not None:
            return _unwrap_image_bytes(value["bytes"])
        raise TypeError("image mappings must contain a non-null 'bytes' value")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(
                "the CPU vision scaling benchmark requires exactly one image per dataset row"
            )
        return _unwrap_image_bytes(value[0])
    raise TypeError(f"unsupported image value type: {type(value).__name__}")


def _read_unique_images(dataset: str, num_images: int) -> list[Any]:
    import pandas as pd
    from PIL import Image

    dataset_path = Path(dataset)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"CPU vision benchmark dataset does not exist: {dataset_path}")
    if dataset_path.suffix == ".parquet":
        frame = pd.read_parquet(dataset_path, columns=["image"])
    elif dataset_path.suffix in {".json", ".jsonl"}:
        frame = pd.read_json(dataset_path, lines=dataset_path.suffix == ".jsonl")
        if "image" not in frame:
            raise ValueError(f"CPU vision benchmark dataset has no 'image' column: {dataset_path}")
        frame = frame[["image"]]
    else:
        raise ValueError(
            "CPU vision benchmark dataset must be a .parquet, .json, or .jsonl file, "
            f"got {dataset_path}"
        )

    unique_images = []
    seen_images: set[bytes] = set()
    for value in frame["image"]:
        image_bytes = _unwrap_image_bytes(value)
        if image_bytes in seen_images:
            continue
        seen_images.add(image_bytes)
        with Image.open(io.BytesIO(image_bytes)) as image:
            unique_images.append(image.convert("RGB").copy())
        if len(unique_images) == num_images:
            break
    if len(unique_images) != num_images:
        raise ValueError(
            f"requested {num_images} unique images, but {dataset_path} contains only "
            f"{len(unique_images)} unique decodable images"
        )
    return unique_images


def _prepare_vision_inputs(checkpoint: str, dataset: str, num_images: int) -> tuple[_PreparedVisionInputs, ...]:
    from transformers import AutoProcessor

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"CPU vision benchmark checkpoint does not exist: {checkpoint_path}")

    processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
    images = _read_unique_images(dataset, num_images)
    prepared_inputs = []
    for image in images:
        encoded = processor.image_processor(images=[image], return_tensors="pt")
        if "pixel_values" not in encoded or "image_grid_thw" not in encoded:
            raise ValueError(
                "Qwen3-VL image processor must return both pixel_values and image_grid_thw"
            )
        prepared_inputs.append(
            _PreparedVisionInputs(
                pixel_values=encoded["pixel_values"].detach().cpu(),
                image_grid_thw=encoded["image_grid_thw"].detach().cpu(),
            )
        )
    return tuple(prepared_inputs)


def _initialize_measurement_worker(
    checkpoint: str,
    prepared_inputs: tuple[_PreparedVisionInputs, ...],
    threads_per_replica: int,
    ready_queue: Any,
) -> None:
    global _WORKER_BACKEND, _WORKER_INPUTS

    try:
        import torch

        from relax.backends.vision.qwen3_vl import build_qwen3_vl_cpu_vision_backend

        torch.set_num_threads(threads_per_replica)
        _WORKER_INPUTS = tuple(
            (prepared.pixel_values, prepared.image_grid_thw) for prepared in prepared_inputs
        )
        _WORKER_BACKEND = build_qwen3_vl_cpu_vision_backend(checkpoint)
        pixel_values, image_grid_thw = _WORKER_INPUTS[0]
        _WORKER_BACKEND.encode(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        ready_queue.put({"pid": os.getpid(), "error": None})
    except BaseException as error:
        ready_queue.put(
            {
                "pid": os.getpid(),
                "error": f"{type(error).__name__}: {error}",
            }
        )
        raise


def _encode_prepared_batch(indices: tuple[int, ...]) -> dict[str, int | float]:
    if _WORKER_BACKEND is None or not _WORKER_INPUTS:
        raise RuntimeError("CPU vision benchmark worker was not initialized")

    import torch

    pixel_values = torch.cat([_WORKER_INPUTS[index][0] for index in indices], dim=0)
    image_grid_thw = torch.cat([_WORKER_INPUTS[index][1] for index in indices], dim=0)
    encode_started_at = time.perf_counter()
    features = _WORKER_BACKEND.encode(
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
    )
    return {
        "completed_images": len(indices),
        "backend_encode_seconds": time.perf_counter() - encode_started_at,
        "emitted_feature_bytes": features.nbytes,
    }


class _LocalCPUVisionMeasurement:
    def __init__(
        self,
        checkpoint: str,
        dataset: str,
        num_images: int,
        repeats: int,
    ) -> None:
        self.checkpoint = str(Path(checkpoint))
        self.dataset = dataset
        self.num_images = num_images
        self.repeats = repeats
        self._context = multiprocessing.get_context("spawn")
        self._prepared_inputs: tuple[_PreparedVisionInputs, ...] | None = None
        self._pools: dict[tuple[int, int], Any] = {}
        self._closed = False

    def _get_prepared_inputs(self) -> tuple[_PreparedVisionInputs, ...]:
        if self._closed:
            raise RuntimeError("CPU vision benchmark measurement has already been closed")
        if self._prepared_inputs is None:
            self._prepared_inputs = _prepare_vision_inputs(
                self.checkpoint,
                self.dataset,
                self.num_images,
            )
        return self._prepared_inputs

    def _create_pool(self, num_replicas: int, threads_per_replica: int) -> Any:
        ready_queue = self._context.Queue()
        pool = None
        try:
            pool = self._context.Pool(
                processes=num_replicas,
                initializer=_initialize_measurement_worker,
                initargs=(
                    self.checkpoint,
                    self._get_prepared_inputs(),
                    threads_per_replica,
                    ready_queue,
                ),
            )
            for _ in range(num_replicas):
                readiness = ready_queue.get()
                if readiness["error"] is not None:
                    raise RuntimeError(
                        f"CPU vision benchmark worker {readiness['pid']} failed to initialize: "
                        f"{readiness['error']}"
                    )
            return pool
        except BaseException:
            if pool is not None:
                pool.terminate()
                pool.join()
            raise
        finally:
            ready_queue.close()
            ready_queue.join_thread()

    def _get_pool(self, num_replicas: int, threads_per_replica: int) -> Any:
        if self._closed:
            raise RuntimeError("CPU vision benchmark measurement has already been closed")
        key = (num_replicas, threads_per_replica)
        if key not in self._pools:
            self._pools[key] = self._create_pool(num_replicas, threads_per_replica)
        return self._pools[key]

    def __call__(self, configuration: Mapping[str, int]) -> dict[str, int | float]:
        num_replicas = configuration["num_replicas"]
        threads_per_replica = configuration["threads_per_replica"]
        batch_size = configuration["batch_size"]
        pool = self._get_pool(num_replicas, threads_per_replica)
        batches = [
            tuple(range(start, min(start + batch_size, self.num_images)))
            for start in range(0, self.num_images, batch_size)
        ]

        encode_results = []
        measurement_started_at = time.perf_counter()
        for _ in range(self.repeats):
            encode_results.extend(pool.map(_encode_prepared_batch, batches, chunksize=1))
        wall_seconds = time.perf_counter() - measurement_started_at

        completed_images = sum(result["completed_images"] for result in encode_results)
        expected_images = self.num_images * self.repeats
        if completed_images != expected_images:
            raise RuntimeError(
                "CPU vision benchmark completed an unexpected image count: "
                f"expected {expected_images}, got {completed_images}"
            )
        return {
            "completed_images": completed_images,
            "wall_seconds": wall_seconds,
            "backend_encode_seconds": sum(
                result["backend_encode_seconds"] for result in encode_results
            ),
            "emitted_feature_bytes": sum(
                result["emitted_feature_bytes"] for result in encode_results
            ),
            "encode_batches": len(encode_results),
            "unique_images": self.num_images,
            "repeats": self.repeats,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._prepared_inputs = None
        pools = tuple(self._pools.values())
        self._pools.clear()
        for pool in pools:
            pool.close()
        for pool in pools:
            pool.join()


def _local_measurement_factory(
    checkpoint: str,
    dataset: str,
    num_images: int,
    repeats: int,
) -> _LocalCPUVisionMeasurement:
    return _LocalCPUVisionMeasurement(checkpoint, dataset, num_images, repeats)


def main(
    argv: Sequence[str] | None = None,
    *,
    measurement_factory: Callable[
        [str, str, int, int],
        Callable[[Mapping[str, int]], Mapping[str, int | float]],
    ] = _local_measurement_factory,
) -> dict[str, object]:
    """Run the configured CPU vision scaling matrix and write its JSON artifact."""
    args = parse_args(argv)
    matrix = build_benchmark_matrix(
        args.replica_counts,
        args.thread_counts,
        args.batch_sizes,
    )
    measure_configuration = measurement_factory(
        args.checkpoint,
        args.dataset,
        args.num_images,
        args.repeats,
    )
    try:
        summary = run_scaling_matrix(
            matrix,
            measure_configuration,
            peak_unique_images_per_second=args.peak_unique_images_per_second,
        )
        return write_scaling_artifact(args.output, summary)
    finally:
        close = getattr(measure_configuration, "close", None)
        if close is not None:
            close()


if __name__ == "__main__":
    main()
