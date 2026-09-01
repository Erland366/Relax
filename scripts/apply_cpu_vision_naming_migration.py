#!/usr/bin/env python3

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Rename local CPU-vision experiment artifacts without changing measured values."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ROOT = Path("benchmark_results/cpu_vision")
DEFAULT_LEDGER = Path("training_reports/2026-08-28-cpu-vision-naming-migration.json")
TEXT_SUFFIXES = {".csv", ".json", ".jsonl", ".log", ".started", ".tsv", ".txt"}


@dataclass(frozen=True)
class Replacement:
    old: str
    new: str


CONTENT_REPLACEMENTS = (
    Replacement(
        "RELAX_SGLANG_QWEN3_VL_OMIT_GPU_WEIGHTS",
        "RELAX_SGLANG_QWEN3_VL_SKIP_GPU_VISION_ENCODER",
    ),
    Replacement("VISION_ENCODER_OMIT_GPU_WEIGHTS", "SKIP_GPU_VISION_ENCODER"),
    Replacement("cache_oracle", "full_dataset_preload"),
    Replacement("CACHE_ORACLE", "FULL_DATASET_PRELOAD"),
    Replacement("cache-oracle", "full-dataset preload"),
    Replacement("cache oracle", "full-dataset preload"),
    Replacement("Cache oracle", "Full-dataset preload"),
    Replacement("prewarm_inclusive_actor_cycle_seconds", "training_cycle_seconds_including_preload"),
    Replacement("measured_actor_seconds", "measured_training_seconds"),
    Replacement("actor_cycle_seconds", "training_cycle_seconds"),
    Replacement("actor_cycles_per_hour", "training_cycles_per_hour"),
    Replacement("rollout_wall_seconds", "rollout_seconds"),
    Replacement("mean_rollout_wall_seconds", "mean_rollout_seconds"),
    Replacement("rollout_wall_reduction", "rollout_time_reduction"),
    Replacement("prewarm_seconds", "preload_seconds"),
    Replacement("prewarm_enabled", "preload_enabled"),
    Replacement("prewarm_metrics", "preload_artifact"),
    Replacement("prewarm", "preload"),
    Replacement("PREWARM", "PRELOAD"),
    Replacement("Prewarm", "Preload"),
    Replacement("tier1_only", "cpu_cache_only"),
    Replacement("tier1_tier2", "cpu_and_sglang_cache"),
    Replacement("tier1", "cpu_cache"),
    Replacement("tier2", "sglang_cache"),
    Replacement("Tier-1", "CPU cache"),
    Replacement("Tier-2", "SGLang cache"),
    Replacement("admissions", "stores"),
    Replacement("admitted_feature_bytes", "cached_feature_bytes"),
    Replacement("oracle_effect", "training_only_speedup"),
    Replacement("amortized_effect_22", "speedup_including_preload"),
    Replacement("break_even_cycles", "cycles_to_recover_preload_time"),
    Replacement("block_effects", "repeat_speedups"),
    Replacement("num_blocks", "repeat_count"),
    Replacement("matched cache-oracle block", "matched repeated run"),
    Replacement("native_parallel_sampling", "gpu_parallel_sampling"),
    Replacement("native_gpu", "gpu"),
    Replacement("cpu_resident", "cpu"),
    Replacement("cpu_omitted", "cpu_skip_gpu_encoder"),
    Replacement("cpu-resident", "cpu"),
    Replacement("cpu-omitted", "cpu-skip-gpu-encoder"),
    Replacement("E1\t", "cpu_vision_demand\t"),
    Replacement("E3\t", "cpu_worker_layouts\t"),
    Replacement("vision_placement_trace", "vision_workload"),
    Replacement("vision_placement_resolution", "vision_resolution"),
    Replacement("vision_placement", "vision_device"),
    Replacement("predicted_exposed_cpu_seconds", "predicted_cpu_seconds_after_overlap"),
    Replacement("predicted_cpu_time_reduction", "predicted_gap"),
    Replacement("cycles_with_gap_above_minimum_gain", "cycles_over_minimum_gap"),
    Replacement("fraction_with_gap_above_minimum_gain", "cycle_share_over_minimum_gap"),
    Replacement("faster_device_matches", "device_matches"),
    Replacement("mean_extra_time_fraction", "mean_extra_time"),
    Replacement("measured_cpu_gpu_gap", "device_time_gap"),
    Replacement("gap_exceeds_minimum_gain", "over_minimum_gap"),
    Replacement("matches_faster_device", "device_match"),
    Replacement("extra_time_fraction", "extra_time"),
    Replacement("minimum_gain", "minimum_gap"),
    Replacement("adaptive", "automatic"),
    Replacement("Adaptive", "Automatic"),
)


SEGMENT_REPLACEMENTS = (
    Replacement("20260731_three_mode_two_cycle", "20260731_gpu_cpu_vision_two_cycle"),
    Replacement("20260801_corrected_three_mode", "20260801_gpu_cpu_vision_performance"),
    Replacement("20260806_native_grouped_n8", "20260806_gpu_grouped_n8"),
    Replacement("20260806_overlap_cpu_omitted_nocache", "20260806_overlap_cpu_skip_gpu_encoder_no_cache"),
    Replacement("20260806_overlap_native_grouped", "20260806_overlap_gpu_grouped"),
    Replacement("slurm_141944_e0_e1", "slurm_141944_cpu_vision_demand"),
    Replacement("slurm_142800_e2", "slurm_142800_cpu_vision_scaling"),
    Replacement("slurm_142801_e2", "slurm_142801_cpu_vision_scaling"),
    Replacement("slurm_142802_e3", "slurm_142802_cpu_worker_layouts"),
    Replacement("slurm_142804_e3", "slurm_142804_cpu_worker_layouts"),
    Replacement("slurm_142805_e4_cache", "slurm_142805_vision_cache_reuse"),
    Replacement("slurm_142809_e4_cache", "slurm_142809_vision_cache_reuse"),
    Replacement("slurm_142817_e4_cache", "slurm_142817_vision_cache_reuse"),
    Replacement("slurm_154079_matched", "slurm_154079_vision_settings"),
    Replacement("trace_builder_validation", "vision_workload_validation"),
    Replacement("e1_", "cpu_vision_demand_"),
    Replacement("full_dataset_preload_full_retry2_", "full_dataset_preload_retry2_"),
    Replacement("full_dataset_preload_full_retry_", "full_dataset_preload_retry_"),
    Replacement("full_dataset_preload_full_", "full_dataset_preload_"),
    Replacement("matched_analysis", "vision_settings"),
    Replacement("hf_parity", "hf_gpu_cpu_parity"),
    Replacement("sglang_parity", "sglang_gpu_cpu_parity"),
)


LABEL_REPLACEMENTS = (
    Replacement("B0", "repeat_1"),
    Replacement("B1", "repeat_2"),
    Replacement("B2", "repeat_3"),
    Replacement("B3", "repeat_4"),
    Replacement("M0", "gpu"),
    Replacement("M1", "cpu_no_caches"),
    Replacement("M2", "cpu_cache"),
    Replacement("M3", "cpu_and_sglang_cache"),
    Replacement("D2_1x4", "replicas1_threads4"),
    Replacement("D2_2x2", "replicas2_threads2"),
    Replacement("D2_4x1", "replicas4_threads1"),
    Replacement("d2_1x4", "replicas1_threads4"),
    Replacement("d2_2x2", "replicas2_threads2"),
    Replacement("d2_4x1", "replicas4_threads1"),
    Replacement("D0", "prompts8_samples8"),
    Replacement("D1", "prompts16_samples4"),
    Replacement("D2", "prompts32_samples2"),
    Replacement("D3", "prompts64_samples1"),
    Replacement("d0", "prompts8_samples8"),
    Replacement("d1", "prompts16_samples4"),
    Replacement("d2", "prompts32_samples2"),
    Replacement("d3", "prompts64_samples1"),
    Replacement("e0_hf_parity", "hf_gpu_cpu_parity"),
    Replacement("e0_sglang_parity", "sglang_gpu_cpu_parity"),
    Replacement("e1_demand", "cpu_vision_demand"),
    Replacement("e2_scaling", "cpu_vision_scaling"),
    Replacement("e4_cache", "vision_cache_reuse"),
)


_CORRUPTED_HEXADECIMAL_LABELS = (
    ("prompts8_samples8", "d0"),
    ("prompts16_samples4", "d1"),
    ("prompts32_samples2", "d2"),
    ("prompts64_samples1", "d3"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replace_text(value: str, replacements: Iterable[Replacement]) -> tuple[str, list[str]]:
    changed_by = []
    for inserted_label, original_hex in _CORRUPTED_HEXADECIMAL_LABELS:
        pattern = rf"(?:(?<=[0-9a-fA-F]){re.escape(inserted_label)}|{re.escape(inserted_label)}(?=[0-9a-fA-F]))"
        value, count = re.subn(pattern, original_hex, value)
        if count:
            changed_by.append(f"repair {inserted_label} inside hexadecimal identifier->{original_hex}")
    for replacement in replacements:
        if replacement in LABEL_REPLACEMENTS:
            pattern = rf"(?<![A-Za-z0-9]){re.escape(replacement.old)}(?![A-Za-z0-9])"
            value, count = re.subn(pattern, replacement.new, value)
            if count:
                changed_by.append(f"{replacement.old}->{replacement.new}")
        elif replacement.old in value:
            value = value.replace(replacement.old, replacement.new)
            changed_by.append(f"{replacement.old}->{replacement.new}")
    value, count = re.subn(r"(?<![\w])native(?![\w])", "gpu", value)
    if count:
        changed_by.append("native->gpu")
    value, count = re.subn(r"(?<![\w])Native(?![\w])", "GPU", value)
    if count:
        changed_by.append("Native->GPU")
    return value, changed_by


def _masked_for_numeric_check(value: str) -> str:
    replacements = (*SEGMENT_REPLACEMENTS, *CONTENT_REPLACEMENTS, *LABEL_REPLACEMENTS)
    for replacement in replacements:
        value = value.replace(replacement.old, "<NAME>").replace(replacement.new, "<NAME>")
    value = re.sub(r"(?<![\w])(?:native|Native|gpu|GPU)(?![\w])", "<NAME>", value)
    return value


def _numeric_signature(value: str) -> list[str]:
    masked = _masked_for_numeric_check(value)
    return re.findall(r"(?<![\w.])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?(?![\w.])", masked)


def _path_after_migration(relative_path: Path) -> Path:
    value = relative_path.as_posix()
    value, _ = _replace_text(value, (*SEGMENT_REPLACEMENTS, *LABEL_REPLACEMENTS, *CONTENT_REPLACEMENTS))
    value = re.sub(r"repeat_([1-4])_U(?=[._])", r"repeat_\1_none", value)
    value = re.sub(r"repeat_([1-4])_O(?=[._])", r"repeat_\1_full_dataset", value)
    return Path(value)


def _migrate_parquet(source: Path, destination: Path, *, apply: bool) -> tuple[str, bool, list[str]]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(source)
    metadata_index = table.schema.get_field_index("metadata")
    if metadata_index < 0:
        return _sha256(source), True, []
    old_metadata = table.column(metadata_index).to_pylist()
    new_metadata = []
    for record in old_metadata:
        migrated = dict(record)
        if "vision_placement_trace" in migrated:
            migrated["vision_workload"] = migrated.pop("vision_placement_trace")
        if "vision_placement_resolution" in migrated:
            migrated["vision_resolution"] = migrated.pop("vision_placement_resolution")
        new_metadata.append(migrated)
    numeric_before = _numbers_in_object(old_metadata)
    numeric_after = _numbers_in_object(new_metadata)
    if numeric_before != numeric_after:
        raise RuntimeError(f"Parquet measured values changed during migration: {source}")
    columns = [table.column(index) for index in range(table.num_columns)]
    columns[metadata_index] = pa.array(new_metadata)
    migrated_table = pa.Table.from_arrays(columns, names=table.column_names)
    if apply:
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(migrated_table, destination)
        return _sha256(destination), True, ["vision placement metadata->vision workload metadata"]
    return _sha256(source), True, ["vision placement metadata->vision workload metadata"]


def _numbers_in_object(value: Any) -> list[float | int]:
    numbers: list[float | int] = []
    if isinstance(value, bool) or value is None:
        return numbers
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"non-finite measured value in artifact: {value}")
        return [value]
    if isinstance(value, dict):
        for key in sorted(value):
            numbers.extend(_numbers_in_object(value[key]))
    elif isinstance(value, (list, tuple)):
        for item in value:
            numbers.extend(_numbers_in_object(item))
    return numbers


def migrate(root: Path, ledger_path: Path, *, apply: bool) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(f"CPU-vision artifact root does not exist: {root}")
    source_files = sorted(path for path in root.rglob("*") if path.is_file())
    records = []
    planned_destinations: set[Path] = set()

    for source in source_files:
        relative_source = source.relative_to(root)
        relative_destination = _path_after_migration(relative_source)
        destination = root / relative_destination
        if destination in planned_destinations:
            raise RuntimeError(f"Naming migration collision at {destination}")
        planned_destinations.add(destination)
        old_sha = _sha256(source)
        replacements = []
        numeric_values_preserved = True

        if source.suffix == ".parquet":
            new_sha, numeric_values_preserved, replacements = _migrate_parquet(
                source, destination, apply=apply
            )
        elif source.suffix in TEXT_SUFFIXES or source.name in {"manifest", "driver"}:
            original = source.read_text(errors="surrogateescape")
            migrated, replacements = _replace_text(
                original,
                (*SEGMENT_REPLACEMENTS, *CONTENT_REPLACEMENTS, *LABEL_REPLACEMENTS),
            )
            numeric_values_preserved = _numeric_signature(original) == _numeric_signature(migrated)
            if not numeric_values_preserved:
                raise RuntimeError(f"Measured numeric tokens changed during migration: {source}")
            if apply:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(migrated, errors="surrogateescape")
                new_sha = _sha256(destination)
            else:
                new_sha = hashlib.sha256(migrated.encode(errors="surrogateescape")).hexdigest()
        else:
            new_sha = old_sha
            if apply and source != destination:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.read_bytes())

        records.append(
            {
                "old_path": relative_source.as_posix(),
                "new_path": relative_destination.as_posix(),
                "old_sha256": old_sha,
                "new_sha256": new_sha,
                "numeric_values_preserved": numeric_values_preserved,
                "content_replacements": sorted(set(replacements)),
            }
        )

    if apply:
        destination_paths = {root / Path(record["new_path"]) for record in records}
        for source in reversed(source_files):
            if source not in destination_paths and source.exists():
                source.unlink()
        for directory in sorted((path for path in root.rglob("*") if path.is_dir()), reverse=True):
            if not any(directory.iterdir()):
                directory.rmdir()

    ledger = {
        "schema_version": 1,
        "migration": "cpu-vision-direct-names-2026-08-28",
        "applied": apply,
        "artifact_root": str(root.resolve()),
        "file_count": len(records),
        "all_numeric_values_preserved": all(record["numeric_values_preserved"] for record in records),
        "files": records,
    }
    if apply:
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    return ledger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--apply", action="store_true", help="Rewrite artifacts. The default is a dry run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ledger = migrate(args.root, args.ledger, apply=args.apply)
    print(json.dumps({key: value for key, value in ledger.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
