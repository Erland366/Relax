# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace

from relax.utils import tracking_utils


def test_secondary_tracking_uses_only_metrics_service_adapter(monkeypatch):
    args = Namespace(use_metrics_service=True)
    calls = []

    monkeypatch.setattr(tracking_utils, "init_wandb_secondary", lambda *_args, **_kwargs: calls.append("wandb"))
    monkeypatch.setattr(
        tracking_utils,
        "init_metrics_service_adapter",
        lambda *_args, **_kwargs: calls.append("metrics_service"),
    )

    tracking_utils.init_tracking(args, primary=False)

    assert calls == ["metrics_service"]


def test_primary_tracking_still_initializes_wandb_before_metrics_service(monkeypatch):
    args = Namespace(use_metrics_service=True)
    calls = []

    monkeypatch.setattr(tracking_utils, "init_wandb_primary", lambda *_args, **_kwargs: calls.append("wandb"))
    monkeypatch.setattr(
        tracking_utils,
        "init_metrics_service_adapter",
        lambda *_args, **_kwargs: calls.append("metrics_service"),
    )

    tracking_utils.init_tracking(args, primary=True)

    assert calls == ["wandb", "metrics_service"]


def test_secondary_tracking_without_metrics_service_keeps_direct_wandb(monkeypatch):
    args = Namespace(use_metrics_service=False)
    calls = []

    monkeypatch.setattr(tracking_utils, "init_wandb_secondary", lambda *_args, **_kwargs: calls.append("wandb"))
    monkeypatch.setattr(
        tracking_utils,
        "init_metrics_service_adapter",
        lambda *_args, **_kwargs: calls.append("metrics_service"),
    )

    tracking_utils.init_tracking(args, primary=False)

    assert calls == ["wandb"]
