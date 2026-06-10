# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib
import queue
import sys
import types
from dataclasses import dataclass

import torch


def _install_fake_megatron_checkpoint_modules(monkeypatch):
    class FakeFileSystemWriterAsync:
        @staticmethod
        def preload_tensors(*args, **kwargs):
            return args, kwargs

    megatron = types.ModuleType("megatron")
    megatron.__path__ = []
    core = types.ModuleType("megatron.core")
    core.__path__ = []
    dist_checkpointing = types.ModuleType("megatron.core.dist_checkpointing")
    dist_checkpointing.__path__ = []
    strategies = types.ModuleType("megatron.core.dist_checkpointing.strategies")
    strategies.__path__ = []
    optimizer_pkg = types.ModuleType("megatron.core.optimizer")
    optimizer_pkg.__path__ = []
    distrib_optimizer = types.ModuleType("megatron.core.optimizer.distrib_optimizer")
    training = types.ModuleType("megatron.training")
    training.__path__ = []
    checkpointing = types.ModuleType("megatron.training.checkpointing")
    checkpointing.get_rng_state = lambda ckpt_format: ("original-rng-state", ckpt_format)
    global_vars = types.ModuleType("megatron.training.global_vars")
    global_vars._args = types.SimpleNamespace(no_save_rng=False)
    global_vars.get_args = lambda: global_vars._args

    class FakeDistributedOptimizer:
        def get_parameter_state_dp_reshardable(self):
            return {
                "per_bucket_numel": [4],
                "per_bucket_numel_unpadded": [4],
                0: {
                    "torch.bfloat16": [
                        [
                            {
                                "step": torch.tensor(1.0),
                                "exp_avg": torch.zeros(4),
                                "gbuf_local_start": 0,
                                "gbuf_local_end": 4,
                            }
                        ]
                    ]
                },
            }

    distrib_optimizer.DistributedOptimizer = FakeDistributedOptimizer
    filesystem_async = types.ModuleType("megatron.core.dist_checkpointing.strategies.filesystem_async")
    filesystem_async.FileSystemWriterAsync = FakeFileSystemWriterAsync
    filesystem_async._get_write_results_queue = lambda: "original-queue"
    filesystem_async.DEFAULT_SUFFIX = ".distcp"
    filesystem_async._split_by_size_and_type = lambda bins, items: [items]
    filesystem_async._split_by_separation_hint = lambda item_buckets, separation_hint: {"": item_buckets}
    checkpointable = types.ModuleType("megatron.core.dist_checkpointing.strategies.checkpointable")
    common = types.ModuleType("megatron.core.dist_checkpointing.strategies.common")
    common.COMMON_STATE_FNAME = "common.pt"
    common.CheckpointingException = RuntimeError
    common.logger = types.SimpleNamespace(debug=lambda *args, **kwargs: None)

    class FakeMultiStorageClientFeature:
        @staticmethod
        def is_enabled():
            return False

    class FakeTorchCommonLoadStrategy:
        def load_common(self, checkpoint_dir):
            return ("legacy-common", checkpoint_dir)

    common.MultiStorageClientFeature = FakeMultiStorageClientFeature
    common.TorchCommonLoadStrategy = FakeTorchCommonLoadStrategy

    class FakeCheckpointableShardedTensor:
        def __init__(self, sharded_tensor):
            self._sh_ten = sharded_tensor

        def __eq__(self, other):
            return isinstance(other, tuple) and other == ("checkpointable", self._sh_ten)

        @classmethod
        def from_sh_ten(cls, sharded_tensor):
            return cls(sharded_tensor)

    class FakeLocalShardsContainer:
        def __init__(self, local_shards):
            self.local_shards = local_shards

    checkpointable.CheckpointableShardedTensor = FakeCheckpointableShardedTensor
    checkpointable.LocalShardsContainer = FakeLocalShardsContainer
    resharding = types.ModuleType("megatron.core.dist_checkpointing.strategies.resharding")
    resharding.nd_flattened_tensor_reformulated_global_shape = lambda sharded_tensor: ("flat",)
    torch_strategy = types.ModuleType("megatron.core.dist_checkpointing.strategies.torch")
    torch_strategy.FileSystemWriterAsync = object
    torch_strategy.sharded_tensor_to_torch_sharded_tensor = lambda sharded_tensors, *args, **kwargs: (
        "legacy",
        sharded_tensors,
        args,
        kwargs,
    )
    torch_strategy._unwrap_pyt_sharded_tensor = lambda sharded_tensor: ("legacy-unwrap", sharded_tensor)

    class FakeMCoreSavePlanner:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class FakeMCoreLoadPlanner:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    torch_strategy.MCoreSavePlanner = FakeMCoreSavePlanner
    torch_strategy.MCoreLoadPlanner = FakeMCoreLoadPlanner
    utils = types.ModuleType("megatron.core.utils")
    utils.is_torch_min_version = lambda version: True

    modules = {
        "megatron": megatron,
        "megatron.core": core,
        "megatron.core.utils": utils,
        "megatron.core.optimizer": optimizer_pkg,
        "megatron.core.optimizer.distrib_optimizer": distrib_optimizer,
        "megatron.core.dist_checkpointing": dist_checkpointing,
        "megatron.core.dist_checkpointing.strategies": strategies,
        "megatron.core.dist_checkpointing.strategies.common": common,
        "megatron.core.dist_checkpointing.strategies.filesystem_async": filesystem_async,
        "megatron.core.dist_checkpointing.strategies.checkpointable": checkpointable,
        "megatron.core.dist_checkpointing.strategies.resharding": resharding,
        "megatron.core.dist_checkpointing.strategies.torch": torch_strategy,
        "megatron.training": training,
        "megatron.training.checkpointing": checkpointing,
        "megatron.training.global_vars": global_vars,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    sys.modules.pop("relax.utils.rocm_checkpoint_writer", None)
    return importlib.import_module("relax.utils.rocm_checkpoint_writer"), filesystem_async, torch_strategy


def test_patch_rocm_checkpoint_writer_updates_filesystem_and_torch_aliases(monkeypatch):
    writer, filesystem_async, torch_strategy = _install_fake_megatron_checkpoint_modules(monkeypatch)
    original_converter = torch_strategy.sharded_tensor_to_torch_sharded_tensor

    writer.patch_rocm_checkpoint_writer()

    assert filesystem_async.FileSystemWriterAsync is writer.ROCmFileSystemWriterAsync
    assert torch_strategy.FileSystemWriterAsync is writer.ROCmFileSystemWriterAsync
    assert torch_strategy.sharded_tensor_to_torch_sharded_tensor is not original_converter


def test_patch_rocm_checkpoint_writer_uses_thread_queue_on_hip(monkeypatch):
    writer, filesystem_async, _ = _install_fake_megatron_checkpoint_modules(monkeypatch)
    monkeypatch.setattr(writer.torch.version, "hip", "6.3.0")

    writer.patch_rocm_checkpoint_writer()

    assert isinstance(filesystem_async._get_write_results_queue(), queue.Queue)


def test_patch_rocm_checkpoint_writer_preserves_original_queue_off_hip(monkeypatch):
    writer, filesystem_async, _ = _install_fake_megatron_checkpoint_modules(monkeypatch)
    monkeypatch.setattr(writer.torch.version, "hip", None)

    writer.patch_rocm_checkpoint_writer()

    assert filesystem_async._get_write_results_queue() == "original-queue"


def test_patch_rocm_checkpoint_writer_skips_rng_state_when_no_save_rng_on_hip(monkeypatch):
    writer, _, _ = _install_fake_megatron_checkpoint_modules(monkeypatch)
    checkpointing = sys.modules["megatron.training.checkpointing"]
    global_vars = sys.modules["megatron.training.global_vars"]
    global_vars._args.no_save_rng = True
    monkeypatch.setattr(writer.torch.version, "hip", "6.3.0")

    writer.patch_rocm_checkpoint_writer()

    assert checkpointing.get_rng_state("torch_dist") is None


def test_patch_rocm_checkpoint_writer_rejects_rng_state_without_no_save_rng_on_hip(monkeypatch):
    writer, _, _ = _install_fake_megatron_checkpoint_modules(monkeypatch)
    checkpointing = sys.modules["megatron.training.checkpointing"]
    monkeypatch.setattr(writer.torch.version, "hip", "6.3.0")

    writer.patch_rocm_checkpoint_writer()

    try:
        checkpointing.get_rng_state("torch_dist")
    except RuntimeError as exc:
        assert "Pass --no-save-rng" in str(exc)
    else:
        raise AssertionError("Expected ROCm checkpoint RNG collection to fail without --no-save-rng")


def test_patch_rocm_checkpoint_writer_allows_rng_state_escape_hatch(monkeypatch):
    writer, _, _ = _install_fake_megatron_checkpoint_modules(monkeypatch)
    checkpointing = sys.modules["megatron.training.checkpointing"]
    monkeypatch.setattr(writer.torch.version, "hip", "6.3.0")
    monkeypatch.setenv("RELAX_ROCM_ALLOW_CHECKPOINT_RNG_STATE", "1")

    writer.patch_rocm_checkpoint_writer()

    assert checkpointing.get_rng_state("torch_dist") == ("original-rng-state", "torch_dist")


def test_patch_rocm_checkpoint_writer_drops_distopt_scalar_step_on_hip(monkeypatch):
    writer, _, _ = _install_fake_megatron_checkpoint_modules(monkeypatch)
    distrib_optimizer = sys.modules["megatron.core.optimizer.distrib_optimizer"]
    monkeypatch.setattr(writer.torch.version, "hip", "6.3.0")

    writer.patch_rocm_checkpoint_writer()
    state = distrib_optimizer.DistributedOptimizer().get_parameter_state_dp_reshardable()

    bucket_tensors = state[0]["torch.bfloat16"][0][0]
    assert "step" not in bucket_tensors
    assert bucket_tensors["exp_avg"].shape == (4,)


def test_patch_rocm_checkpoint_writer_preserves_distopt_scalar_step_off_hip(monkeypatch):
    writer, _, _ = _install_fake_megatron_checkpoint_modules(monkeypatch)
    distrib_optimizer = sys.modules["megatron.core.optimizer.distrib_optimizer"]
    monkeypatch.setattr(writer.torch.version, "hip", None)

    writer.patch_rocm_checkpoint_writer()
    state = distrib_optimizer.DistributedOptimizer().get_parameter_state_dp_reshardable()

    bucket_tensors = state[0]["torch.bfloat16"][0][0]
    assert bucket_tensors["step"].shape == ()


def test_patch_rocm_checkpoint_writer_loads_common_state_with_weights_only_false(monkeypatch, tmp_path):
    writer, _, _ = _install_fake_megatron_checkpoint_modules(monkeypatch)
    common = sys.modules["megatron.core.dist_checkpointing.strategies.common"]
    load_calls = []

    def fake_torch_load(load_path, **kwargs):
        load_calls.append((load_path, kwargs))
        return "loaded-common"

    monkeypatch.setattr(writer.torch.version, "hip", "6.3.0")
    monkeypatch.setattr(writer.torch, "load", fake_torch_load)

    writer.patch_rocm_checkpoint_writer()
    result = common.TorchCommonLoadStrategy().load_common(tmp_path)

    assert result == "loaded-common"
    assert load_calls == [
        (
            str(tmp_path / "common.pt"),
            {"map_location": "cpu", "weights_only": False},
        )
    ]


def test_patch_rocm_checkpoint_writer_disables_sharded_tensor_flattening_on_hip(monkeypatch):
    writer, _, torch_strategy = _install_fake_megatron_checkpoint_modules(monkeypatch)
    monkeypatch.setattr(writer.torch.version, "hip", "6.3.0")

    writer.patch_rocm_checkpoint_writer()

    save_planner = torch_strategy.MCoreSavePlanner(flatten_state_dict=False)
    load_planner = torch_strategy.MCoreLoadPlanner()

    assert save_planner.kwargs["flatten_sharded_tensors"] is False
    assert load_planner.kwargs["flatten_sharded_tensors"] is False


def test_patch_rocm_checkpoint_writer_preserves_explicit_sharded_tensor_flattening(monkeypatch):
    writer, _, torch_strategy = _install_fake_megatron_checkpoint_modules(monkeypatch)
    monkeypatch.setattr(writer.torch.version, "hip", "6.3.0")

    writer.patch_rocm_checkpoint_writer()

    planner = torch_strategy.MCoreSavePlanner(flatten_sharded_tensors=True)

    assert planner.kwargs["flatten_sharded_tensors"] is True


def test_patch_rocm_checkpoint_writer_preserves_planner_defaults_off_hip(monkeypatch):
    writer, _, torch_strategy = _install_fake_megatron_checkpoint_modules(monkeypatch)
    monkeypatch.setattr(writer.torch.version, "hip", None)

    writer.patch_rocm_checkpoint_writer()

    planner = torch_strategy.MCoreSavePlanner()

    assert "flatten_sharded_tensors" not in planner.kwargs


def test_configure_rocm_torch_dist_checkpoint_args_disables_pre_mcore_path(monkeypatch):
    writer, _, _ = _install_fake_megatron_checkpoint_modules(monkeypatch)
    monkeypatch.setattr(writer.torch.version, "hip", "6.3.0")
    args = types.SimpleNamespace(ckpt_format="torch_dist", dist_ckpt_save_pre_mcore_014=True)

    changed = writer.configure_rocm_torch_dist_checkpoint_args(args)

    assert changed is True
    assert args.dist_ckpt_save_pre_mcore_014 is False


def test_configure_rocm_torch_dist_checkpoint_args_preserves_non_torch_dist(monkeypatch):
    writer, _, _ = _install_fake_megatron_checkpoint_modules(monkeypatch)
    monkeypatch.setattr(writer.torch.version, "hip", "6.3.0")
    args = types.SimpleNamespace(ckpt_format="torch", dist_ckpt_save_pre_mcore_014=True)

    changed = writer.configure_rocm_torch_dist_checkpoint_args(args)

    assert changed is False
    assert args.dist_ckpt_save_pre_mcore_014 is True


def test_rocm_writer_forces_blocking_preload_on_hip(monkeypatch):
    writer, _, _ = _install_fake_megatron_checkpoint_modules(monkeypatch)
    monkeypatch.setattr(writer.torch.version, "hip", "6.3.0")

    _, kwargs = writer.ROCmFileSystemWriterAsync.preload_tensors("state_dict", non_blocking=True)

    assert kwargs["non_blocking"] is False


def test_rocm_writer_uses_streaming_save_without_bulk_preload_on_hip(monkeypatch):
    writer, _, _ = _install_fake_megatron_checkpoint_modules(monkeypatch)
    monkeypatch.setattr(writer.torch.version, "hip", "6.3.0")
    monkeypatch.setattr(writer.torch.distributed, "get_rank", lambda: 3)

    instance = writer.ROCmFileSystemWriterAsync.__new__(writer.ROCmFileSystemWriterAsync)
    instance.write_buckets = [("file", "key", ([], []))]
    instance.results_queue = queue.Queue()
    instance.use_msc = False
    instance._relax_rocm_lazy_planner = "planner"

    save_fn, preload_fn, save_args = instance.get_save_function_and_args()

    assert save_fn.func is writer.ROCmFileSystemWriterAsync.write_data_streaming
    assert save_fn.args == ([], False)
    assert preload_fn is None
    assert save_args == [3, instance.write_buckets, instance.results_queue, "planner", False]


def test_rocm_writer_prepare_write_data_is_lazy_on_hip(monkeypatch):
    writer, filesystem_async, _ = _install_fake_megatron_checkpoint_modules(monkeypatch)
    monkeypatch.setattr(writer.torch.version, "hip", "6.3.0")

    class FakeStoragePlan:
        prefix = "rank_"

    class FakeItemType:
        BYTE_IO = "byte"

    class FakeWriteItem:
        def __init__(self, item_type):
            self.type = item_type

    class FakePlan:
        storage_data = FakeStoragePlan()
        items = [FakeWriteItem(FakeItemType.BYTE_IO), FakeWriteItem("tensor")]

    class FakePlanner:
        def __init__(self):
            self.resolved = 0

        def resolve_data(self, item):
            self.resolved += 1
            return item

    planner_module = types.ModuleType("torch.distributed.checkpoint.planner")
    planner_module.WriteItemType = FakeItemType
    monkeypatch.setitem(sys.modules, "torch.distributed.checkpoint.planner", planner_module)

    instance = writer.ROCmFileSystemWriterAsync.__new__(writer.ROCmFileSystemWriterAsync)
    instance.checkpoint_dir = "/tmp/checkpoint"
    instance.thread_count = 2
    instance.separation_hint = None
    instance.results_queue = None
    instance.use_msc = False
    planner = FakePlanner()

    writer.patch_rocm_checkpoint_writer()
    instance.prepare_write_data(FakePlan(), planner)

    assert planner.resolved == 0
    assert instance._relax_rocm_lazy_planner is planner
    assert instance._relax_rocm_lazy_write_items is True
    assert filesystem_async._get_write_results_queue() is instance.results_queue
    assert instance.write_buckets == [
        (
            "/tmp/checkpoint/rank_0.distcp",
            "rank_0.distcp",
            ([FakePlan.items[0]], [FakePlan.items[1]]),
        )
    ]


def test_rocm_converter_uses_checkpointable_sharded_tensors_on_hip(monkeypatch):
    writer, _, torch_strategy = _install_fake_megatron_checkpoint_modules(monkeypatch)
    monkeypatch.setattr(writer.torch.version, "hip", "6.3.0")

    @dataclass
    class FakeShardedTensor:
        data: object
        flattened_range: object = None
        prepend_axis_num: int = 0

        def without_data(self):
            return "without-data"

    class FakeTensor:
        def __init__(self):
            self.detached = False

        def detach(self):
            self.detached = True
            return self

    sharded_tensor = FakeShardedTensor(data=FakeTensor())

    writer.patch_rocm_checkpoint_writer()
    converted = torch_strategy.sharded_tensor_to_torch_sharded_tensor([sharded_tensor])

    assert converted == ("checkpointable", sharded_tensor)
    assert sharded_tensor.data.detached is True


def test_rocm_converter_wraps_flattened_tensors_with_adjusted_metadata(monkeypatch):
    writer, _, torch_strategy = _install_fake_megatron_checkpoint_modules(monkeypatch)
    monkeypatch.setattr(writer.torch.version, "hip", "6.3.0")

    @dataclass
    class FakeShardedTensor:
        data: object
        global_shape: tuple
        local_shape: tuple
        global_offset: tuple
        axis_fragmentations: tuple | None
        flattened_range: object
        prepend_axis_num: int = 0

        def local_chunk_offset_in_global(self):
            return (0,)

        def without_data(self):
            return "without-data"

    class FakeTensor:
        shape = (1,)

        def detach(self):
            return self

        def view(self, *shape):
            self.shape = tuple(dim for dim in shape if dim != ())
            return self

    sharded_tensor = FakeShardedTensor(
        data=FakeTensor(),
        global_shape=(2,),
        local_shape=(2,),
        global_offset=(0,),
        axis_fragmentations=(1,),
        flattened_range=slice(0, 1),
        prepend_axis_num=0,
    )

    writer.patch_rocm_checkpoint_writer()
    converted = torch_strategy.sharded_tensor_to_torch_sharded_tensor([sharded_tensor])

    assert converted == ("checkpointable", converted._sh_ten)
    assert converted._relax_original_mcore_sh_ten == "without-data"
    assert converted._sh_ten.flattened_range is None
    assert converted._sh_ten.prepend_axis_num == 0


def test_rocm_streaming_save_delegates_bucket_without_bulk_precopy(monkeypatch):
    writer, _, _ = _install_fake_megatron_checkpoint_modules(monkeypatch)

    class FakeTensor:
        def detach(self):
            raise AssertionError("write_data_streaming must not bulk-stage tensors")

    written_buckets = []

    def fake_write_streaming_bucket(
        transform_list,
        use_msc,
        bucket_index,
        write_bucket,
        planner,
        lazy_write_items,
    ):
        assert planner is None
        assert lazy_write_items is False
        written_buckets.append((transform_list, use_msc, bucket_index, write_bucket))
        return bucket_index, ["write-result"]

    monkeypatch.setattr(writer, "_write_streaming_bucket", fake_write_streaming_bucket)
    results_queue = queue.Queue()
    fake_tensor = FakeTensor()
    write_buckets = [("file", "key", ([], [("item", fake_tensor)]))]

    writer.ROCmFileSystemWriterAsync.write_data_streaming([], False, 7, write_buckets, results_queue)

    assert written_buckets == [([], False, 0, ("file", "key", ([], [("item", fake_tensor)])))]
    assert results_queue.get_nowait() == {0: ["write-result"]}


def test_rocm_streaming_save_handles_lazy_write_items_without_prelogging_unpack(monkeypatch):
    writer, _, _ = _install_fake_megatron_checkpoint_modules(monkeypatch)

    class FakePlanner:
        pass

    written_buckets = []

    def fake_write_streaming_bucket(
        transform_list,
        use_msc,
        bucket_index,
        write_bucket,
        planner,
        lazy_write_items,
    ):
        assert isinstance(planner, FakePlanner)
        assert lazy_write_items is True
        written_buckets.append((transform_list, use_msc, bucket_index, write_bucket))
        return bucket_index, ["write-result"]

    monkeypatch.setattr(writer, "_write_streaming_bucket", fake_write_streaming_bucket)
    results_queue = queue.Queue()
    byte_item = object()
    tensor_item = object()
    write_buckets = [("file", "key", ([byte_item], [tensor_item]))]

    writer.ROCmFileSystemWriterAsync.write_data_streaming(
        [],
        False,
        7,
        write_buckets,
        results_queue,
        FakePlanner(),
        True,
    )

    assert written_buckets == [([], False, 0, ("file", "key", ([byte_item], [tensor_item])))]
    assert results_queue.get_nowait() == {0: ["write-result"]}


def test_write_streaming_bucket_copies_each_tensor_immediately_before_write(monkeypatch, tmp_path):
    writer, _, _ = _install_fake_megatron_checkpoint_modules(monkeypatch)
    import torch.distributed.checkpoint.filesystem as filesystem

    events = []

    class FakeDevice:
        type = "cuda"

    class FakeCpuTensor:
        def __init__(self, name):
            self.name = name
            self.device = types.SimpleNamespace(type="cpu")

    class FakeTensor:
        def __init__(self, name):
            self.name = name
            self.device = FakeDevice()

        def detach(self):
            events.append(("detach", self.name))
            return self

        def to(self, device, non_blocking):
            events.append(("copy", self.name, device, non_blocking))
            return FakeCpuTensor(self.name)

    def fake_write_item(*args, **kwargs):
        data = args[1]
        write_item = args[2]
        events.append(("write", write_item, getattr(data, "name", data)))
        return f"write-result-{write_item}"

    monkeypatch.setattr(filesystem, "_write_item", fake_write_item)
    write_bucket = (
        tmp_path / "bucket.distcp",
        "bucket.distcp",
        ([("bytes-item", b"bytes")], [("tensor-0", FakeTensor("t0")), ("tensor-1", FakeTensor("t1"))]),
    )

    idx, results = writer._write_streaming_bucket([], False, 4, write_bucket)

    assert idx == 4
    assert results == ["write-result-bytes-item", "write-result-tensor-0", "write-result-tensor-1"]
    assert events == [
        ("write", "bytes-item", b"bytes"),
        ("detach", "t0"),
        ("copy", "t0", "cpu", False),
        ("write", "tensor-0", "t0"),
        ("detach", "t1"),
        ("copy", "t1", "cpu", False),
        ("write", "tensor-1", "t1"),
    ]


def test_write_streaming_bucket_resolves_lazy_items_one_at_a_time(monkeypatch, tmp_path):
    writer, _, _ = _install_fake_megatron_checkpoint_modules(monkeypatch)
    import torch.distributed.checkpoint.filesystem as filesystem

    events = []

    class FakeDevice:
        type = "cuda"

    class FakeCpuTensor:
        def __init__(self, name):
            self.name = name
            self.device = types.SimpleNamespace(type="cpu")

    class FakeTensor:
        def __init__(self, name):
            self.name = name
            self.device = FakeDevice()

        def detach(self):
            events.append(("detach", self.name))
            return self

        def to(self, device, non_blocking):
            events.append(("copy", self.name, device, non_blocking))
            return FakeCpuTensor(self.name)

    class FakePlanner:
        def resolve_data(self, item):
            events.append(("resolve", item))
            if item == "bytes-item":
                return b"bytes"
            return FakeTensor(item)

    def fake_write_item(*args, **kwargs):
        data = args[1]
        write_item = args[2]
        events.append(("write", write_item, getattr(data, "name", data)))
        return f"write-result-{write_item}"

    monkeypatch.setattr(filesystem, "_write_item", fake_write_item)
    write_bucket = (
        tmp_path / "bucket.distcp",
        "bucket.distcp",
        (["bytes-item"], ["tensor-0", "tensor-1"]),
    )

    idx, results = writer._write_streaming_bucket([], False, 4, write_bucket, FakePlanner(), True)

    assert idx == 4
    assert results == ["write-result-bytes-item", "write-result-tensor-0", "write-result-tensor-1"]
    assert events == [
        ("resolve", "bytes-item"),
        ("write", "bytes-item", b"bytes"),
        ("resolve", "tensor-0"),
        ("detach", "tensor-0"),
        ("copy", "tensor-0", "cpu", False),
        ("write", "tensor-0", "tensor-0"),
        ("resolve", "tensor-1"),
        ("detach", "tensor-1"),
        ("copy", "tensor-1", "cpu", False),
        ("write", "tensor-1", "tensor-1"),
    ]


def test_write_streaming_bucket_fails_loudly_for_lazy_items_without_planner(monkeypatch, tmp_path):
    writer, _, _ = _install_fake_megatron_checkpoint_modules(monkeypatch)
    write_bucket = (tmp_path / "bucket.distcp", "bucket.distcp", ([], ["tensor-0"]))

    try:
        writer._write_streaming_bucket([], False, 4, write_bucket, None, True)
    except RuntimeError as exc:
        assert "requires a planner" in str(exc)
    else:
        raise AssertionError("lazy checkpoint buckets must fail loudly without a planner")
