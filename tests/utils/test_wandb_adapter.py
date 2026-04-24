# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace

from relax.utils.metrics.adapters import wandb as wandb_adapter


def _build_args(**overrides):
    args = Namespace(
        wandb_run_id="run-123",
        wandb_mode="online",
        wandb_key=None,
        wandb_host=None,
        wandb_team="test-team",
        wandb_project="test-project",
        wandb_dir=None,
        sglang_enable_metrics=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_init_wandb_secondary_omits_config_payload(monkeypatch, tmp_path):
    init_calls = []
    settings_calls = []
    metric_names = []

    monkeypatch.setattr(
        wandb_adapter.wandb,
        "Settings",
        lambda **kwargs: settings_calls.append(kwargs) or kwargs,
    )
    monkeypatch.setattr(
        wandb_adapter.wandb,
        "init",
        lambda **kwargs: init_calls.append(kwargs),
    )
    monkeypatch.setattr(
        wandb_adapter.wandb,
        "define_metric",
        lambda name, **kwargs: metric_names.append((name, kwargs)),
    )

    args = _build_args(wandb_dir=str(tmp_path))
    wandb_adapter.init_wandb_secondary(args)

    assert len(init_calls) == 1
    assert "config" not in init_calls[0]
    assert init_calls[0]["id"] == "run-123"
    assert init_calls[0]["dir"] == str(tmp_path)
    assert settings_calls == [
        {
            "mode": "shared",
            "x_primary": False,
            "x_update_finish_state": False,
        }
    ]
    assert metric_names


def test_init_wandb_secondary_forwards_router_metrics(monkeypatch):
    settings_calls = []

    monkeypatch.setattr(
        wandb_adapter.wandb,
        "Settings",
        lambda **kwargs: settings_calls.append(kwargs) or kwargs,
    )
    monkeypatch.setattr(wandb_adapter.wandb, "init", lambda **kwargs: None)
    monkeypatch.setattr(wandb_adapter.wandb, "define_metric", lambda *args, **kwargs: None)

    args = _build_args(sglang_enable_metrics=True)
    wandb_adapter.init_wandb_secondary(args, router_addr="http://127.0.0.1:3662")

    assert settings_calls == [
        {
            "mode": "shared",
            "x_primary": False,
            "x_update_finish_state": False,
            "x_stats_open_metrics_endpoints": {
                "sgl_engine": "http://127.0.0.1:3662/engine_metrics",
            },
            "x_stats_open_metrics_filters": {
                "sgl_engine.*": {},
            },
        }
    ]
