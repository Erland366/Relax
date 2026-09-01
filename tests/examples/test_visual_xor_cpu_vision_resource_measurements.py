import sys
from types import SimpleNamespace

import pytest

from examples.visual_xor import measure_cpu_vision_scaling as scaling


def test_encode_prepared_batch_records_worker_resource_evidence(monkeypatch):
    class FakeBackend:
        def encode(self, *, pixel_values, image_grid_thw):
            assert pixel_values == "concatenated-pixels"
            assert image_grid_thw == "concatenated-grid"
            return SimpleNamespace(nbytes=4_096)

    fake_torch = SimpleNamespace(
        cat=lambda values, dim: (
            "concatenated-pixels" if values == ["pixels-0", "pixels-1"] else "concatenated-grid"
        ),
        get_num_threads=lambda: 4,
    )
    fake_resource = SimpleNamespace(
        RUSAGE_SELF=0,
        getrusage=lambda _who: SimpleNamespace(ru_maxrss=321),
    )
    wall_times = iter([10.0, 10.25])
    process_times = iter([2.0, 2.125])
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "resource", fake_resource)
    monkeypatch.setattr(scaling, "_WORKER_BACKEND", FakeBackend())
    monkeypatch.setattr(
        scaling,
        "_WORKER_INPUTS",
        (("pixels-0", "grid-0"), ("pixels-1", "grid-1")),
    )
    monkeypatch.setattr(scaling.time, "perf_counter", lambda: next(wall_times))
    monkeypatch.setattr(scaling.time, "process_time", lambda: next(process_times))
    monkeypatch.setattr(scaling.os, "getpid", lambda: 4_321)

    result = scaling._encode_prepared_batch((0, 1))

    assert result == {
        "completed_images": 2,
        "backend_encode_seconds": pytest.approx(0.25),
        "backend_process_cpu_seconds": pytest.approx(0.125),
        "emitted_feature_bytes": 4_096,
        "worker_pid": 4_321,
        "num_threads": 4,
        "rss_bytes": 321 * 1_024,
    }


def test_local_measurement_sums_backend_times_and_reports_peak_rss_per_replica(monkeypatch):
    results_by_repeat = iter(
        [
            [
                {
                    "completed_images": 2,
                    "backend_encode_seconds": 0.10,
                    "backend_process_cpu_seconds": 0.30,
                    "emitted_feature_bytes": 1_000,
                    "worker_pid": 101,
                    "num_threads": 2,
                    "rss_bytes": 10_000,
                },
                {
                    "completed_images": 2,
                    "backend_encode_seconds": 0.20,
                    "backend_process_cpu_seconds": 0.40,
                    "emitted_feature_bytes": 2_000,
                    "worker_pid": 202,
                    "num_threads": 2,
                    "rss_bytes": 20_000,
                },
            ],
            [
                {
                    "completed_images": 2,
                    "backend_encode_seconds": 0.30,
                    "backend_process_cpu_seconds": 0.50,
                    "emitted_feature_bytes": 3_000,
                    "worker_pid": 101,
                    "num_threads": 2,
                    "rss_bytes": 15_000,
                },
                {
                    "completed_images": 2,
                    "backend_encode_seconds": 0.40,
                    "backend_process_cpu_seconds": 0.60,
                    "emitted_feature_bytes": 4_000,
                    "worker_pid": 202,
                    "num_threads": 2,
                    "rss_bytes": 19_000,
                },
            ],
        ]
    )

    class FakePool:
        def map(self, _function, batches, chunksize):
            assert tuple(batches) == ((0, 1), (2, 3))
            assert chunksize == 1
            return next(results_by_repeat)

        def close(self):
            pass

        def join(self):
            pass

    measurement = scaling._LocalCPUVisionMeasurement(
        "/models/qwen3-vl",
        "/data/visual-xor.jsonl",
        num_images=4,
        repeats=2,
    )
    monkeypatch.setattr(measurement, "_create_pool", lambda _replicas, _threads: FakePool())
    wall_times = iter([50.0, 51.5])
    monkeypatch.setattr(scaling.time, "perf_counter", lambda: next(wall_times))

    try:
        result = measurement(
            {"num_replicas": 2, "threads_per_replica": 2, "batch_size": 2}
        )
    finally:
        measurement.close()

    assert result["completed_images"] == 8
    assert result["wall_seconds"] == pytest.approx(1.5)
    assert result["backend_encode_seconds"] == pytest.approx(1.0)
    assert result["backend_process_cpu_seconds"] == pytest.approx(1.8)
    assert result["rss_bytes"] == 35_000
    assert result["replica_metrics"] == [
        {"worker_pid": 101, "num_threads": 2, "rss_bytes": 15_000},
        {"worker_pid": 202, "num_threads": 2, "rss_bytes": 20_000},
    ]
