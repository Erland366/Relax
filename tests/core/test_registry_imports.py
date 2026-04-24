# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import subprocess
import sys


def test_importing_core_registry_does_not_import_megatron_or_sglang():
    script = """
import sys

import relax.core.registry

seen = sorted(
    name for name in sys.modules if name == "megatron" or name.startswith(("megatron.", "sglang", "sglang."))
)
print("\\n".join(seen))
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)

    assert result.stdout.strip() == ""
