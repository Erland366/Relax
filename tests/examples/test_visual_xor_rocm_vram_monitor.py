# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from examples.visual_xor.monitor_rocm_vram import parse_rocm_smi_vram_csv, summarize_vram_samples


def test_parse_rocm_smi_vram_csv_extracts_device_bytes():
    output = """device,VRAM Total Memory (B),VRAM Total Used Memory (B)
card0,68702699520,13135872
card1,68702699520,20135872
"""

    assert parse_rocm_smi_vram_csv(output) == {
        "card0": {"total_bytes": 68702699520, "used_bytes": 13135872},
        "card1": {"total_bytes": 68702699520, "used_bytes": 20135872},
    }


def test_summarize_vram_samples_preserves_per_device_and_simultaneous_peaks():
    samples = [
        {
            "devices": {
                "card0": {"total_bytes": 100, "used_bytes": 10},
                "card1": {"total_bytes": 100, "used_bytes": 20},
            }
        },
        {
            "devices": {
                "card0": {"total_bytes": 100, "used_bytes": 80},
                "card1": {"total_bytes": 100, "used_bytes": 30},
            }
        },
        {
            "devices": {
                "card0": {"total_bytes": 100, "used_bytes": 40},
                "card1": {"total_bytes": 100, "used_bytes": 90},
            }
        },
        {
            "devices": {
                "card0": {"total_bytes": 100, "used_bytes": 11},
                "card1": {"total_bytes": 100, "used_bytes": 21},
            }
        },
    ]

    summary = summarize_vram_samples(samples)

    assert summary["devices"]["card0"] == {
        "total_bytes": 100,
        "baseline_used_bytes": 10,
        "peak_used_bytes": 80,
        "peak_delta_bytes": 70,
        "final_used_bytes": 11,
    }
    assert summary["devices"]["card1"]["peak_used_bytes"] == 90
    assert summary["peak_simultaneous_total_used_bytes"] == 130
    assert summary["peak_simultaneous_total_delta_bytes"] == 100
