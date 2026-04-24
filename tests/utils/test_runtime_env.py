# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace

from relax.utils.utils import post_process_env


def test_post_process_env_prefers_shell_over_yaml_defaults(monkeypatch):
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "bond0")
    monkeypatch.setenv("TP_SOCKET_IFNAME", "bond0")

    args = Namespace(rollout_batch_size=2, n_samples_per_prompt=8)
    runtime_env = {
        "env_vars": {
            "GLOO_SOCKET_IFNAME": "eth0",
            "TP_SOCKET_IFNAME": "eth0",
            "NCCL_DEBUG": "WARN",
        }
    }

    result = post_process_env(args, runtime_env)

    assert result["env_vars"]["GLOO_SOCKET_IFNAME"] == "bond0"
    assert result["env_vars"]["TP_SOCKET_IFNAME"] == "bond0"
    assert result["env_vars"]["NCCL_DEBUG"] == "WARN"
    assert result["env_vars"]["TQ_PRE_ALLOC_SAMPLE_NUM"] == "16"


def test_post_process_env_propagates_proxy_variables(monkeypatch):
    monkeypatch.setenv("http_proxy", "http://proxy.local:3128")
    monkeypatch.setenv("https_proxy", "http://proxy.local:3128")
    monkeypatch.setenv("no_proxy", ".example.internal")
    monkeypatch.setenv("MASTER_ADDR", "head-node")
    monkeypatch.setattr("relax.utils.utils.socket.gethostname", lambda: "worker-node")
    monkeypatch.setattr(
        "relax.utils.utils.socket.gethostbyname",
        lambda host: {
            "head-node": "172.27.112.25",
            "worker-node": "172.27.112.26",
        }[host],
    )

    args = Namespace(rollout_batch_size=2, n_samples_per_prompt=8)
    runtime_env = {"env_vars": {}}

    result = post_process_env(args, runtime_env)

    assert result["env_vars"]["http_proxy"] == "http://proxy.local:3128"
    assert result["env_vars"]["https_proxy"] == "http://proxy.local:3128"
    assert result["env_vars"]["no_proxy"] == (
        ".example.internal,127.0.0.1,localhost,::1,head-node,172.27.112.25,worker-node,172.27.112.26"
    )
    assert result["env_vars"]["NO_PROXY"] == "127.0.0.1,localhost,::1,head-node,172.27.112.25,worker-node,172.27.112.26"


def test_post_process_env_propagates_ray_runtime_tuning(monkeypatch):
    monkeypatch.setenv("RAY_task_events_report_interval_ms", "0")
    monkeypatch.setenv("RAY_grpc_client_keepalive_time_ms", "600000")
    monkeypatch.setenv("RAY_grpc_client_keepalive_timeout_ms", "300000")

    args = Namespace(rollout_batch_size=2, n_samples_per_prompt=8)
    runtime_env = {"env_vars": {}}

    result = post_process_env(args, runtime_env)

    assert result["env_vars"]["RAY_task_events_report_interval_ms"] == "0"
    assert result["env_vars"]["RAY_grpc_client_keepalive_time_ms"] == "600000"
    assert result["env_vars"]["RAY_grpc_client_keepalive_timeout_ms"] == "300000"
