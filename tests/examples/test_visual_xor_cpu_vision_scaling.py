import json

import pytest

from examples.visual_xor.benchmark_cpu_vision_scaling import (
    summarize_scaling_results,
    write_scaling_artifact,
)


def _record(
    *,
    num_replicas: int = 1,
    threads_per_replica: int = 4,
    batch_size: int = 8,
    num_images: int = 256,
    wall_time_seconds: float = 4.0,
) -> dict[str, int | float]:
    return {
        "num_replicas": num_replicas,
        "threads_per_replica": threads_per_replica,
        "batch_size": batch_size,
        "num_images": num_images,
        "wall_time_seconds": wall_time_seconds,
    }


def test_scaling_summary_derives_throughput_capacity_and_reserved_cpus():
    summary = summarize_scaling_results(
        [_record(num_replicas=2, threads_per_replica=3, num_images=300, wall_time_seconds=5.0)],
        peak_unique_images_per_second=40.0,
    )

    assert summary["peak_unique_images_per_second"] == 40.0
    assert len(summary["results"]) == 1
    result = summary["results"][0]
    assert result["total_reserved_cpus"] == 6
    assert result["images_per_second"] == pytest.approx(60.0)
    assert result["capacity_ratio"] == pytest.approx(1.5)
    assert result["gate_passed"] is True


@pytest.mark.parametrize(
    ("images_per_second", "expected_gate_passed"),
    [(49.999, False), (50.0, True)],
)
def test_scaling_capacity_gate_is_inclusive_at_one_point_two_five_times_demand(
    images_per_second,
    expected_gate_passed,
):
    summary = summarize_scaling_results(
        [_record(num_images=images_per_second * 10.0, wall_time_seconds=10.0)],
        peak_unique_images_per_second=40.0,
    )

    assert summary["results"][0]["gate_passed"] is expected_gate_passed


def test_scaling_summary_recommends_fewest_reserved_cpus_before_throughput():
    summary = summarize_scaling_results(
        [
            _record(
                num_replicas=2,
                threads_per_replica=4,
                batch_size=16,
                num_images=800,
                wall_time_seconds=10.0,
            ),
            _record(
                num_replicas=1,
                threads_per_replica=4,
                batch_size=8,
                num_images=600,
                wall_time_seconds=10.0,
            ),
        ],
        peak_unique_images_per_second=40.0,
    )

    recommendation = summary["recommended_configuration"]
    assert recommendation["num_replicas"] == 1
    assert recommendation["threads_per_replica"] == 4
    assert recommendation["batch_size"] == 8
    assert recommendation["images_per_second"] == pytest.approx(60.0)


def test_scaling_summary_uses_highest_throughput_to_break_equal_cpu_ties():
    summary = summarize_scaling_results(
        [
            _record(
                num_replicas=1,
                threads_per_replica=4,
                batch_size=8,
                num_images=550,
                wall_time_seconds=10.0,
            ),
            _record(
                num_replicas=2,
                threads_per_replica=2,
                batch_size=16,
                num_images=650,
                wall_time_seconds=10.0,
            ),
        ],
        peak_unique_images_per_second=40.0,
    )

    recommendation = summary["recommended_configuration"]
    assert recommendation["num_replicas"] == 2
    assert recommendation["threads_per_replica"] == 2
    assert recommendation["batch_size"] == 16
    assert recommendation["images_per_second"] == pytest.approx(65.0)


def test_scaling_summary_has_no_recommendation_when_capacity_gate_does_not_pass():
    summary = summarize_scaling_results(
        [_record(num_images=499, wall_time_seconds=10.0)],
        peak_unique_images_per_second=40.0,
    )

    assert summary["results"][0]["gate_passed"] is False
    assert summary["recommended_configuration"] is None


@pytest.mark.parametrize(
    ("record_override", "peak_unique_images_per_second", "error_pattern"),
    [
        ({}, 0.0, "peak_unique_images_per_second.*positive"),
        ({"num_replicas": 0}, 40.0, "num_replicas.*positive"),
        ({"threads_per_replica": 0}, 40.0, "threads_per_replica.*positive"),
        ({"batch_size": 0}, 40.0, "batch_size.*positive"),
        ({"num_images": 0}, 40.0, "num_images.*positive"),
        ({"wall_time_seconds": 0.0}, 40.0, "wall_time_seconds.*positive"),
    ],
)
def test_scaling_summary_rejects_nonpositive_measurements(
    record_override,
    peak_unique_images_per_second,
    error_pattern,
):
    record = _record()
    record.update(record_override)

    with pytest.raises(ValueError, match=error_pattern):
        summarize_scaling_results(
            [record],
            peak_unique_images_per_second=peak_unique_images_per_second,
        )


def test_scaling_summary_rejects_duplicate_replica_thread_batch_configurations():
    records = [
        _record(num_images=256, wall_time_seconds=4.0),
        _record(num_images=512, wall_time_seconds=7.0),
    ]

    with pytest.raises(ValueError, match="duplicate.*configuration.*replicas=1.*threads=4.*batch_size=8"):
        summarize_scaling_results(records, peak_unique_images_per_second=40.0)


def test_scaling_artifact_is_versioned_json_with_exact_capacity_gate_metadata(tmp_path):
    summary = summarize_scaling_results(
        [_record(num_images=600, wall_time_seconds=10.0)],
        peak_unique_images_per_second=40.0,
    )
    output_path = tmp_path / "nested" / "cpu_vision_scaling.json"

    artifact = write_scaling_artifact(output_path, summary)

    assert json.loads(output_path.read_text()) == artifact
    assert artifact["schema_version"] == 1
    assert artifact["passed"] is True
    assert artifact["gates"] == {
        "capacity_ratio_min": 1.25,
        "demand_metric": "peak_unique_images_per_second",
    }
    assert artifact["peak_unique_images_per_second"] == 40.0
    assert artifact["results"][0]["images_per_second"] == pytest.approx(60.0)
    assert artifact["recommended_configuration"]["total_reserved_cpus"] == 4
