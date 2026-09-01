# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from scripts.apply_cpu_vision_naming_migration import LABEL_REPLACEMENTS, _replace_text


def test_workload_labels_are_replaced_only_as_complete_tokens() -> None:
    migrated, replacements = _replace_text("feature_id=abcd0ef D0 d2", LABEL_REPLACEMENTS)

    assert migrated == "feature_id=abcd0ef prompts8_samples8 prompts32_samples2"
    assert "D0->prompts8_samples8" in replacements
    assert "d2->prompts32_samples2" in replacements


def test_corrupted_workload_label_inside_hexadecimal_identifier_is_repaired() -> None:
    migrated, replacements = _replace_text(
        "feature_id=abcprompts32_samples2def",
        LABEL_REPLACEMENTS,
    )

    assert migrated == "feature_id=abcd2def"
    assert "repair prompts32_samples2 inside hexadecimal identifier->d2" in replacements
