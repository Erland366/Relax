# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import torch

from relax.utils.training.tensor_backper import TensorBackuper


def test_tensor_backuper_respects_pin_memory_flag(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    weight = torch.ones(4)
    backuper = TensorBackuper.create(
        source_getter=lambda: [("weight", weight)],
        single_tag=None,
        pin_memory=False,
    )

    backuper.backup("actor")

    assert backuper.get("actor")["weight"].device.type == "cpu"
    assert backuper.get("actor")["weight"].is_pinned() is False
