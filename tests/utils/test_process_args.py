# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace

import pytest

from relax.utils.utils import process_args


def test_process_args_accepts_missing_ref_actor_config_for_actor_fwd():
    args = Namespace(ref_actor_config=None, only_load_weight=False, load="actor", ref_load="reference")

    process_args(args, "actor_fwd")

    assert args.only_load_weight is True
    assert args.load == "actor"


def test_process_args_applies_ref_actor_config_and_reference_load():
    args = Namespace(
        ref_actor_config={"max_tokens_per_gpu": 8192},
        only_load_weight=False,
        load="actor",
        ref_load="reference",
    )

    process_args(args, "reference")

    assert args.only_load_weight is True
    assert args.max_tokens_per_gpu == 8192
    assert args.load == "reference"


def test_process_args_rejects_non_mapping_ref_actor_config():
    args = Namespace(ref_actor_config=["bad"], only_load_weight=False, load="actor", ref_load="reference")

    with pytest.raises(TypeError, match="ref_actor_config must be a dict"):
        process_args(args, "actor_fwd")
