# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace

import relax.core.controller as controller_module


def test_build_data_source_actor_config_uses_module_builder():
    config = Namespace(prompt_data="/tmp/prompts.jsonl", extra_field=object())

    class DummyDataSource:
        __module__ = "relax.engine.rollout.data_source"

    result = controller_module.build_data_source_actor_config(config, DummyDataSource)

    assert isinstance(result, Namespace)
    assert result.prompt_data == "/tmp/prompts.jsonl"
    assert not hasattr(result, "extra_field")


def test_register_all_serve_passes_sanitized_data_source_config(monkeypatch):
    captured = {}

    class FakeRoles(list):
        rollout = controller_module.ROLES.rollout

    class FakeHealthManager:
        def __init__(self):
            self.status = {}

        def mark_healthy(self, role):
            return None

    class FakeRemoteActor:
        def remote(self, config):
            captured["config"] = config
            return "data-source"

    def fake_ray_remote(*args, **kwargs):
        def wrap(cls):
            captured["cls"] = cls
            return FakeRemoteActor()

        return wrap

    class DummyDataSource:
        pass

    def fake_builder(config, data_source_cls):
        captured["builder"] = (config, data_source_cls)
        return Namespace(sanitized=True)

    controller = controller_module.Controller.__new__(controller_module.Controller)
    controller.config = Namespace(
        advantage_estimator="test_algo",
        data_source_path="relax.engine.rollout.data_source.RolloutDataSourceWithBuffer",
        resource={controller_module.ROLES.rollout: (1, 1)},
        colocate=False,
        fully_async=False,
        extra_field=object(),
    )
    controller.serve_dict = {}
    controller._health_manager = FakeHealthManager()
    controller.runtime_env = None
    controller._validate_gpu_resources = lambda *args, **kwargs: None
    controller._create_service_task = lambda role, cls, num_gpus, data_source, actor_rollout_pgs: (
        role,
        "service",
        None,
    )

    monkeypatch.setattr(controller_module, "register_extra_roles", lambda config, algo: [])
    monkeypatch.setattr(controller_module, "process_role", lambda config: FakeRoles([controller_module.ROLES.rollout]))
    monkeypatch.setattr(controller_module, "load_function", lambda path: DummyDataSource)
    monkeypatch.setattr(controller_module, "build_data_source_actor_config", fake_builder)
    monkeypatch.setitem(controller_module.ALGOS, "test_algo", {controller_module.ROLES.rollout: object()})
    monkeypatch.setattr(controller_module.ray, "remote", fake_ray_remote)

    try:
        controller.register_all_serve()
    finally:
        controller_module.ALGOS.pop("test_algo", None)

    assert captured["builder"] == (controller.config, DummyDataSource)
    assert captured["cls"] is DummyDataSource
    assert captured["config"] == Namespace(sanitized=True)


def test_order_service_creation_moves_rollout_before_actor_for_serial_non_colocated():
    roles_to_create = [
        (controller_module.ROLES.actor, "actor_cls", 1, None),
        (controller_module.ROLES.rollout, "rollout_cls", 1, None),
        ("genrm", "genrm_cls", 1, None),
    ]

    ordered = controller_module.order_service_creation(roles_to_create, colocate=False, fully_async=False)

    assert [role for role, *_rest in ordered] == [
        controller_module.ROLES.rollout,
        controller_module.ROLES.actor,
        "genrm",
    ]


def test_order_service_creation_keeps_order_for_colocated_or_async():
    roles_to_create = [
        (controller_module.ROLES.actor, "actor_cls", 1, None),
        (controller_module.ROLES.rollout, "rollout_cls", 1, None),
    ]

    colocated = controller_module.order_service_creation(roles_to_create, colocate=True, fully_async=False)
    async_mode = controller_module.order_service_creation(roles_to_create, colocate=False, fully_async=True)

    assert colocated == roles_to_create
    assert async_mode == roles_to_create
