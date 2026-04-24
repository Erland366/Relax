# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace

import relax.entrypoints.train as train_entry


def _build_args():
    return Namespace(
        use_wandb=True,
        use_metrics_service=False,
        wandb_mode="online",
        wandb_key=None,
        wandb_host=None,
        wandb_random_suffix=False,
        wandb_group="test-group",
        wandb_team="test-team",
        wandb_project="test-project",
        wandb_dir=None,
    )


def test_main_initializes_tracking_before_controller(monkeypatch):
    args = _build_args()
    call_order = []

    monkeypatch.setattr(train_entry.yaml, "safe_load", lambda _file: {"env_vars": {}})
    monkeypatch.setattr(train_entry, "post_process_env", lambda args, runtime_env: runtime_env)
    monkeypatch.setattr(train_entry, "init_tracking", lambda args, primary=True, **kwargs: call_order.append(("tracking", primary)))
    monkeypatch.setattr(train_entry.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(train_entry.atexit, "register", lambda fn: None)
    monkeypatch.setattr(train_entry.signal, "signal", lambda sig, handler: None)
    monkeypatch.setattr(train_entry, "_graceful_shutdown", lambda sig=None, frame=None: None)

    class DummyController:
        def __init__(self, config, runtime_env):
            call_order.append(("controller", getattr(config, "wandb_run_id", None)))

        def training_loop(self):
            call_order.append(("training_loop", None))

        def shutdown(self):
            call_order.append(("shutdown", None))

    monkeypatch.setattr(train_entry, "Controller", DummyController)

    train_entry.main(args)

    assert call_order[0] == ("tracking", True)
    assert call_order[1][0] == "controller"
    assert call_order[2] == ("training_loop", None)
