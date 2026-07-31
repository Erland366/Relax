# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib
import sys
import types
from argparse import Namespace


def _install_transfer_queue_stub(monkeypatch) -> None:
    transfer_queue = types.ModuleType("transfer_queue")
    transfer_queue.GRPOGroupNSampler = type("GRPOGroupNSampler", (), {})
    transfer_queue.SeqlenBalancedSampler = type("SeqlenBalancedSampler", (), {})
    transfer_queue.init = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "transfer_queue", transfer_queue)


def _install_lightweight_controller_stubs(monkeypatch) -> None:
    _install_transfer_queue_stub(monkeypatch)

    class GenRM:
        pass

    def register_genrm(config, algo):
        if getattr(config, "genrm_model_path", None) is None:
            return []
        algo["genrm"] = GenRM
        return ["genrm"]

    genrm_module = types.ModuleType("relax.components.genrm")
    genrm_module.GenRM = GenRM
    genrm_module.register_genrm = register_genrm
    monkeypatch.setitem(sys.modules, "relax.components.genrm", genrm_module)

    class Roles:
        actor = "actor"
        critic = "critic"
        rollout = "rollout"
        advantages = "advantages"
        reference = "reference"
        actor_fwd = "actor_fwd"

    registry_module = types.ModuleType("relax.core.registry")
    registry_module.ALGOS = {}
    registry_module.ROLES = Roles
    registry_module.process_role = lambda config: Roles
    monkeypatch.setitem(sys.modules, "relax.core.registry", registry_module)

    utils_module = types.ModuleType("relax.utils.utils")
    utils_module.compute_dp_size = lambda config: 1
    utils_module.get_serve_url = lambda route_prefix: f"http://localhost{route_prefix}"
    utils_module.recovery_load_path = lambda config: None
    monkeypatch.setitem(sys.modules, "relax.utils.utils", utils_module)


def test_controller_combines_genrm_and_enabled_vision_encoder_roles(monkeypatch):
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    _install_lightweight_controller_stubs(monkeypatch)
    controller_module = importlib.import_module("relax.core.controller")
    algo = {}
    config = Namespace(genrm_model_path="/models/judge", vision_encoder_backend="pytorch")

    roles = controller_module.register_extra_roles(config, algo)

    assert roles == ["genrm", "vision_encoder"]
    assert algo["genrm"].__name__ == "GenRM"
    assert algo["vision_encoder"].__name__ == "VisionEncoder"


def test_vision_encoder_deployment_configures_replicas_and_cpus_per_replica(monkeypatch):
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    _install_lightweight_controller_stubs(monkeypatch)
    service_module = importlib.import_module("relax.core.service")
    captured = {}

    class FakeDeployment:
        def options(self, **kwargs):
            captured["options"] = kwargs
            return self

        def bind(self, *args, **kwargs):
            captured["bind"] = (args, kwargs)
            return "bound-deployment"

    service = service_module.Service.__new__(service_module.Service)
    service.cls = FakeDeployment()
    service.role = "vision_encoder"
    service.healthy = object()
    service.num_gpus = 0
    service.config = Namespace(
        vision_encoder_num_replicas=3,
        vision_encoder_num_cpus=6,
        sglang_model_impl="",
    )
    service.data_source = None
    service.runtime_env = None
    monkeypatch.setattr(service_module.serve, "run", lambda *args, **kwargs: "handle")

    service._deploy(None)

    assert captured["options"]["num_replicas"] == 3
    assert captured["options"]["ray_actor_options"]["num_cpus"] == 6


def test_non_vision_deployment_does_not_apply_vision_replica_or_cpu_options(monkeypatch):
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    _install_lightweight_controller_stubs(monkeypatch)
    service_module = importlib.import_module("relax.core.service")
    captured = {}

    class FakeDeployment:
        def options(self, **kwargs):
            captured["options"] = kwargs
            return self

        def bind(self, *args, **kwargs):
            return "bound-deployment"

    service = service_module.Service.__new__(service_module.Service)
    service.cls = FakeDeployment()
    service.role = "actor"
    service.healthy = object()
    service.num_gpus = 1
    service.config = Namespace(
        vision_encoder_num_replicas=3,
        vision_encoder_num_cpus=6,
        sglang_model_impl="",
    )
    service.data_source = None
    service.runtime_env = None
    monkeypatch.setattr(service_module.serve, "run", lambda *args, **kwargs: "handle")

    service._deploy(None)

    assert "num_replicas" not in captured["options"]
    assert "num_cpus" not in captured["options"]["ray_actor_options"]
