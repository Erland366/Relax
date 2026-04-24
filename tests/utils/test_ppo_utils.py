# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib

import pytest
import torch

import relax.utils.training.ppo_utils as ppo_utils


def test_mul_reduce_matches_eager_product_sum():
    a = torch.randn(4, 7)
    b = torch.randn(4, 7)

    result = ppo_utils.mul_reduce(a, b)

    assert torch.allclose(result, (a * b).sum(dim=-1, keepdim=True))


def test_mul_reduce_uses_eager_impl_on_rocm():
    def fake_compile(fn=None, *args, **kwargs):
        def decorate(inner):
            def wrapped(*inner_args, **inner_kwargs):
                return inner(*inner_args, **inner_kwargs)

            wrapped._compiled_marker = True
            wrapped._orig = inner
            return wrapped

        if fn is None:
            return decorate
        return decorate(fn)

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(torch, "compile", fake_compile)
        monkeypatch.setattr(torch.version, "hip", "6.3.0", raising=False)

        reloaded = importlib.reload(ppo_utils)

        assert reloaded.mul_reduce is reloaded._mul_reduce_impl
        assert reloaded.compute_approx_kl.__name__ == "compute_approx_kl"
        assert reloaded.compute_sapo_loss.__name__ == "compute_sapo_loss"
        assert reloaded.compute_policy_loss.__name__ == "compute_policy_loss"
    finally:
        monkeypatch.undo()
        importlib.reload(ppo_utils)
