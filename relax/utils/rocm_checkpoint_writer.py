# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import inspect
import os
import queue
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from time import time
from typing import Any

import torch
from megatron.core.dist_checkpointing.strategies.filesystem_async import FileSystemWriterAsync

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

_rocm_results_queue: queue.Queue[Any] | None = None
_logged_checkpointable_converter = False
_logged_planner_flattening_patch = False
_logged_common_load_patch = False
_logged_distributed_optimizer_step_patch = False
_logged_rng_state_skip = False


def configure_rocm_torch_dist_checkpoint_args(args: Any) -> bool:
    """Use Megatron's current torch_dist metadata path on HIP when available."""
    if not torch.version.hip:
        return False
    if getattr(args, "ckpt_format", None) != "torch_dist":
        return False

    from megatron.core.utils import is_torch_min_version

    if not is_torch_min_version("2.6a0"):
        return False
    if not getattr(args, "dist_ckpt_save_pre_mcore_014", False):
        return False

    args.dist_ckpt_save_pre_mcore_014 = False
    logger.info(
        "HIP/ROCm detected: disabled dist_ckpt_save_pre_mcore_014 for torch_dist on PyTorch >= 2.6"
    )
    return True


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    """Best-effort tensor byte size for checkpoint logging."""
    try:
        return tensor.numel() * tensor.element_size()
    except (AttributeError, TypeError, RuntimeError):
        return 0


def _copy_tensor_to_cpu_for_write(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach()
    if tensor.device.type == "cpu":
        return tensor
    return tensor.to("cpu", non_blocking=False)


def _write_streaming_bucket(
    transform_list: list[Any],
    use_msc: bool,
    bucket_index: int,
    write_bucket: Any,
    planner: Any | None = None,
    lazy_write_items: bool = False,
) -> tuple[int, list[Any]]:
    """Write one DCP bucket while staging at most one tensor to CPU at a time."""
    from torch.distributed.checkpoint.filesystem import _write_item

    if lazy_write_items and planner is None:
        raise RuntimeError("ROCm lazy checkpoint bucket requires a planner to resolve WriteItem data")

    file_name, storage_key, (bytes_data, tensor_data) = write_bucket
    extra_kwargs = {}
    if "serialization_format" in inspect.signature(_write_item).parameters:
        from torch.distributed.checkpoint.filesystem import SerializationFormat

        extra_kwargs["serialization_format"] = SerializationFormat.TORCH_SAVE
    if use_msc:
        import multistorageclient as msc

        open_file = msc.open
    else:
        open_file = open

    local_results = []
    with open_file(file_name, "wb") as stream:
        for entry in bytes_data:
            if lazy_write_items:
                write_item = entry
                data = planner.resolve_data(write_item)
            else:
                write_item, data = entry
            local_results.append(
                _write_item(
                    *transform_list,
                    stream,
                    data,
                    write_item,
                    storage_key,
                    **extra_kwargs,
                )
            )
        for entry in tensor_data:
            if lazy_write_items:
                write_item = entry
                tensor = planner.resolve_data(write_item)
            else:
                write_item, tensor = entry
            cpu_tensor = _copy_tensor_to_cpu_for_write(tensor)
            try:
                local_results.append(
                    _write_item(
                        *transform_list,
                        stream,
                        cpu_tensor,
                        write_item,
                        storage_key,
                        **extra_kwargs,
                    )
                )
            finally:
                del cpu_tensor
                del tensor
        if not use_msc:
            os.fsync(stream.fileno())
        else:
            stream.fsync()
    return bucket_index, local_results


def _rocm_write_results_queue(original_getter: Callable[[], Any]) -> Any:
    """Return a thread-local results queue on HIP to avoid mp.Manager spawn."""
    if not torch.version.hip:
        return original_getter()

    global _rocm_results_queue
    if _rocm_results_queue is None:
        _rocm_results_queue = queue.Queue()
        logger.info("HIP/ROCm detected: using thread-local checkpoint results queue")
    return _rocm_results_queue


def _rocm_sharded_tensor_to_checkpointable(
    original_converter: Callable[..., Any],
    sharded_tensors: list[Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Avoid PyTorch's legacy ShardedTensor construction on HIP."""
    if not torch.version.hip:
        return original_converter(sharded_tensors, *args, **kwargs)

    from megatron.core.dist_checkpointing.strategies.checkpointable import (
        CheckpointableShardedTensor,
        LocalShardsContainer,
    )
    from megatron.core.dist_checkpointing.strategies.resharding import (
        nd_flattened_tensor_reformulated_global_shape,
    )
    from megatron.core.utils import is_torch_min_version

    if not is_torch_min_version("2.6a0"):
        return original_converter(sharded_tensors, *args, **kwargs)

    global _logged_checkpointable_converter
    if not _logged_checkpointable_converter:
        _logged_checkpointable_converter = True
        logger.info("HIP/ROCm detected: using DCP checkpointable sharded tensors for torch_dist save")

    adjusted_tensors = []
    for sharded_tensor in sharded_tensors:
        data = getattr(sharded_tensor, "data", None)
        if data is None:
            adjusted_tensors.append(sharded_tensor)
            continue
        data = data.detach()
        flattened_range = getattr(sharded_tensor, "flattened_range", None)
        prepend_axis_num = getattr(sharded_tensor, "prepend_axis_num", 0)
        if flattened_range is not None:
            data = data.view((1,) * len(sharded_tensor.global_shape) + (-1,))
            adjusted_tensors.append(
                replace(
                    sharded_tensor,
                    data=data,
                    local_shape=tuple(data.shape),
                    global_shape=nd_flattened_tensor_reformulated_global_shape(sharded_tensor),
                    global_offset=sharded_tensor.local_chunk_offset_in_global() + (flattened_range.start,),
                    axis_fragmentations=None,
                    prepend_axis_num=0,
                    flattened_range=None,
                )
            )
        elif prepend_axis_num:
            data = data.view((1,) * prepend_axis_num + tuple(sharded_tensor.local_shape))
            adjusted_tensors.append(
                replace(
                    sharded_tensor,
                    data=data,
                    local_shape=tuple(data.shape),
                    prepend_axis_num=0,
                )
            )
        else:
            adjusted_tensors.append(replace(sharded_tensor, data=data))

    checkpointable_shards = []
    for original_tensor, adjusted_tensor in zip(sharded_tensors, adjusted_tensors, strict=True):
        checkpointable_shard = CheckpointableShardedTensor.from_sh_ten(adjusted_tensor)
        checkpointable_shard._relax_original_mcore_sh_ten = original_tensor.without_data()
        checkpointable_shards.append(checkpointable_shard)
    if len(checkpointable_shards) == 1:
        return checkpointable_shards[0]
    return LocalShardsContainer(checkpointable_shards)


def _rocm_unwrap_checkpointable_sharded_tensor(
    original_unwrap: Callable[[Any], Any],
    sharded_tensor: Any,
) -> Any:
    """Restore original MCore tensor shapes after HIP checkpointable loading."""
    if not torch.version.hip:
        return original_unwrap(sharded_tensor)

    from megatron.core.dist_checkpointing.strategies.checkpointable import (
        CheckpointableShardedTensor,
        LocalShardsContainer,
    )

    def unwrap_one(checkpointable_shard: Any) -> Any:
        original = getattr(checkpointable_shard, "_relax_original_mcore_sh_ten", None)
        if original is None:
            return None
        tensor = checkpointable_shard._sh_ten.data
        if original.flattened_range is not None:
            return tensor.view(-1)
        for _ in range(original.prepend_axis_num):
            tensor = tensor[0]
        return tensor

    if isinstance(sharded_tensor, CheckpointableShardedTensor):
        tensor = unwrap_one(sharded_tensor)
        if tensor is not None:
            return [tensor]
    if isinstance(sharded_tensor, LocalShardsContainer):
        tensors = [unwrap_one(local_shard) for local_shard in sharded_tensor._local_shards]
        if all(tensor is not None for tensor in tensors):
            return tensors
    return original_unwrap(sharded_tensor)


def _rocm_mcore_planner_init(
    original_init: Callable[..., None],
    planner_name: str,
    self: Any,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Match current Megatron's DCP planner defaults on HIP."""
    if torch.version.hip and "flatten_sharded_tensors" not in kwargs:
        from torch.distributed.checkpoint import DefaultSavePlanner

        if "flatten_sharded_tensors" in inspect.signature(DefaultSavePlanner.__init__).parameters:
            kwargs["flatten_sharded_tensors"] = False
            global _logged_planner_flattening_patch
            if not _logged_planner_flattening_patch:
                _logged_planner_flattening_patch = True
                logger.info(
                    "HIP/ROCm detected: setting flatten_sharded_tensors=False for Megatron "
                    "torch_dist checkpoint planners"
                )
    return original_init(self, *args, **kwargs)


def _patch_rocm_mcore_planner_init(torch_strategy_module: Any, planner_name: str) -> None:
    planner_cls = getattr(torch_strategy_module, planner_name, None)
    if planner_cls is None:
        return

    original_init = getattr(planner_cls, "_relax_original_init", planner_cls.__init__)
    planner_cls._relax_original_init = original_init

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        return _rocm_mcore_planner_init(original_init, planner_name, self, *args, **kwargs)

    planner_cls.__init__ = patched_init


def _torch_load_trusted_checkpoint(load_fn: Callable[..., Any], load_path: Any, **kwargs: Any) -> Any:
    """Load local Megatron checkpoint files that may contain non-tensor metadata."""
    return load_fn(load_path, **kwargs, weights_only=False)


def _rocm_torch_common_load(
    original_load_common: Callable[..., Any],
    common_module: Any,
    self: Any,
    checkpoint_dir: Any,
) -> Any:
    """Load Megatron common.pt with PyTorch 2.6's trusted-checkpoint flag."""
    if not torch.version.hip:
        return original_load_common(self, checkpoint_dir)

    load_path = os.path.join(checkpoint_dir, common_module.COMMON_STATE_FNAME)
    try:
        if common_module.MultiStorageClientFeature.is_enabled():
            msc = common_module.MultiStorageClientFeature.import_package()
            return _torch_load_trusted_checkpoint(msc.torch.load, load_path, map_location="cpu")
        return _torch_load_trusted_checkpoint(torch.load, load_path, map_location="cpu")
    except FileNotFoundError as exc:
        err_msg = f"Common file {load_path} does not exist"
        if common_module.MultiStorageClientFeature.is_enabled():
            msc = common_module.MultiStorageClientFeature.import_package()
            ckpt_files = [path.name for path in msc.Path(checkpoint_dir).iterdir()]
        else:
            ckpt_files = [path.name for path in os.scandir(checkpoint_dir)]
        common_module.logger.debug("%s. Checkpoint directory content: %s", err_msg, ckpt_files)
        raise common_module.CheckpointingException(err_msg) from exc


def _patch_rocm_common_load_strategy(common_module: Any) -> None:
    strategy_cls = getattr(common_module, "TorchCommonLoadStrategy", None)
    if strategy_cls is None:
        return

    original_load_common = getattr(
        strategy_cls,
        "_relax_original_load_common",
        strategy_cls.load_common,
    )
    strategy_cls._relax_original_load_common = original_load_common

    def patched_load_common(self: Any, checkpoint_dir: Any) -> Any:
        return _rocm_torch_common_load(original_load_common, common_module, self, checkpoint_dir)

    strategy_cls.load_common = patched_load_common

    global _logged_common_load_patch
    if torch.version.hip and not _logged_common_load_patch:
        _logged_common_load_patch = True
        logger.info("HIP/ROCm detected: loading Megatron common.pt with weights_only=False")


def _drop_scalar_steps_from_dp_reshardable_state(state: Any) -> Any:
    """Remove scalar Adam step tensors from DistOpt bucket state on HIP."""
    if not torch.version.hip:
        return state

    removed = 0
    for state_key, dtype_state in state.items():
        if not isinstance(state_key, int) or not isinstance(dtype_state, dict):
            continue
        for buckets_state in dtype_state.values():
            for bucket_state in buckets_state:
                for tensors in bucket_state:
                    step = tensors.get("step")
                    if isinstance(step, torch.Tensor) and step.ndim == 0:
                        del tensors["step"]
                        removed += 1

    global _logged_distributed_optimizer_step_patch
    if removed and not _logged_distributed_optimizer_step_patch:
        _logged_distributed_optimizer_step_patch = True
        logger.info(
            "HIP/ROCm detected: removed scalar optimizer step tensors from "
            "DistributedOptimizer dp_reshardable bucket state; step is saved in param_groups"
        )
    return state


def _patch_rocm_distributed_optimizer_step_state(distrib_optimizer_module: Any) -> None:
    optimizer_cls = getattr(distrib_optimizer_module, "DistributedOptimizer", None)
    if optimizer_cls is None:
        return

    original_get_state = getattr(
        optimizer_cls,
        "_relax_original_get_parameter_state_dp_reshardable",
        optimizer_cls.get_parameter_state_dp_reshardable,
    )
    optimizer_cls._relax_original_get_parameter_state_dp_reshardable = original_get_state

    def patched_get_parameter_state_dp_reshardable(self: Any) -> Any:
        state = original_get_state(self)
        return _drop_scalar_steps_from_dp_reshardable_state(state)

    optimizer_cls.get_parameter_state_dp_reshardable = patched_get_parameter_state_dp_reshardable


def patch_rocm_checkpoint_rng_state() -> None:
    """Avoid HIP RNG-state collection when checkpoint RNG saving is disabled."""
    if not torch.version.hip:
        return

    import megatron.training.checkpointing as checkpointing_module
    from megatron.training.global_vars import get_args

    if getattr(checkpointing_module.get_rng_state, "_relax_rocm_no_save_rng_patch", False):
        return

    original_get_rng_state = checkpointing_module.get_rng_state

    def _relax_rocm_get_rng_state(ckpt_format: str) -> Any | None:
        if os.environ.get("RELAX_ROCM_ALLOW_CHECKPOINT_RNG_STATE", "0") == "1":
            return original_get_rng_state(ckpt_format)

        args = get_args()
        if getattr(args, "no_save_rng", False):
            global _logged_rng_state_skip
            if not _logged_rng_state_skip:
                _logged_rng_state_skip = True
                logger.info("HIP/ROCm detected: skipping checkpoint RNG-state collection because --no-save-rng is set")
            return None

        raise RuntimeError(
            "ROCm checkpoint RNG-state collection is disabled because torch.cuda.get_rng_state() can "
            "raise CUDA driver error 700 after training. Pass --no-save-rng for this ROCm path, "
            "or set RELAX_ROCM_ALLOW_CHECKPOINT_RNG_STATE=1 to use Megatron's original behavior."
        )

    _relax_rocm_get_rng_state._relax_rocm_no_save_rng_patch = True
    _relax_rocm_get_rng_state._relax_rocm_original_get_rng_state = original_get_rng_state
    checkpointing_module.get_rng_state = _relax_rocm_get_rng_state


class ROCmFileSystemWriterAsync(FileSystemWriterAsync):
    """FileSystemWriterAsync wrapper for ROCm compatibility.

    On ROCm/HIP, using non_blocking=True causes tensors to be stored in pinned
    memory, which triggers segmentation faults when forking subprocesses
    afterward.
    """

    @staticmethod
    def preload_tensors(*args, **kwargs):
        # Change argument non_blocking to False on HIP platform
        # The tensors will be stored in pinned memory if non_blocking=True
        # Currently on the ROCm platform, forking a subprocess afterward
        # with pinned_memory=True will trigger segmentation fault
        if torch.version.hip:
            logger.info("HIP/ROCm detected: setting non_blocking=False in preload_tensors")
            if "non_blocking" in kwargs:
                kwargs["non_blocking"] = False
            elif len(args) > 1 and isinstance(args[-1], bool):
                # non_blocking is typically the last argument
                args = args[:-1] + (False,)

        return FileSystemWriterAsync.preload_tensors(*args, **kwargs)

    def prepare_write_data(self, plan: Any, planner: Any) -> None:
        """Build a lazy write plan on HIP without resolving every tensor up front."""
        if not torch.version.hip:
            return super().prepare_write_data(plan, planner)

        import megatron.core.dist_checkpointing.strategies.filesystem_async as filesystem_async_module
        from torch.distributed.checkpoint.planner import WriteItemType

        storage_plan = plan.storage_data
        if self.separation_hint:
            assert self.thread_count > 1, "thread_count must be at least 2 if separation_hint is provided"
        bins = self.thread_count // 2 if self.separation_hint is not None else self.thread_count
        item_buckets = filesystem_async_module._split_by_size_and_type(bins, plan.items)

        file_count = 0

        def gen_file(prefix: str = "") -> str:
            nonlocal file_count
            file_name = f"{prefix}{storage_plan.prefix}{file_count}{filesystem_async_module.DEFAULT_SUFFIX}"
            file_count += 1
            return file_name

        self.write_buckets = []
        for group_name, group_buckets in filesystem_async_module._split_by_separation_hint(
            item_buckets, self.separation_hint
        ).items():
            for bucket in group_buckets:
                bytes_items = [item for item in bucket if item.type == WriteItemType.BYTE_IO]
                tensor_items = [item for item in bucket if item.type != WriteItemType.BYTE_IO]
                if len(bytes_items) > 0 or len(tensor_items) > 0:
                    file_name = gen_file(prefix=group_name)
                    self.write_buckets.append(
                        (
                            os.path.join(self.checkpoint_dir, file_name),
                            file_name,
                            (bytes_items, tensor_items),
                        )
                    )

        self._relax_rocm_lazy_planner = planner
        self._relax_rocm_lazy_write_items = True
        if len(self.write_buckets) > 0:
            assert len(self.write_buckets) <= self.thread_count, (
                len(self.write_buckets),
                self.thread_count,
            )
            self.results_queue = filesystem_async_module._get_write_results_queue()
        else:
            self.results_queue = None
        logger.info(
            "ROCm lazy checkpoint prepare: write_items=%s, buckets=%s",
            len(plan.items),
            len(self.write_buckets),
        )

    def get_save_function_and_args(self) -> tuple[Callable[..., Any] | None, Callable[..., Any] | None, list[Any]]:
        """Use a streaming sync write path on HIP instead of bulk preloading."""
        if not torch.version.hip:
            return super().get_save_function_and_args()

        if not self.write_buckets:
            return None, None, []

        transform_list = [self.transforms] if hasattr(self, "transforms") else []
        return (
            partial(
                self.write_data_streaming,
                transform_list,
                self.use_msc,
            ),
            None,
            [
                torch.distributed.get_rank(),
                self.write_buckets,
                self.results_queue,
                getattr(self, "_relax_rocm_lazy_planner", None),
                getattr(self, "_relax_rocm_lazy_write_items", False),
            ],
        )

    @staticmethod
    def write_data_streaming(
        transform_list: list[Any],
        use_msc: bool,
        rank: int,
        write_buckets: list[Any],
        global_results_queue: Any,
        planner: Any | None = None,
        lazy_write_items: bool = False,
    ) -> None:
        """Copy and write checkpoint tensors one at a time on HIP."""
        start = time()
        write_results = {}
        try:
            for bucket_index, write_bucket in enumerate(write_buckets):
                _, _, (bytes_data, tensor_data) = write_bucket
                if lazy_write_items:
                    tensor_bytes_gib = "unresolved"
                else:
                    tensor_bytes_gib = f"{sum(_tensor_nbytes(tensor) for _, tensor in tensor_data) / (1024**3):.2f}"
                logger.info(
                    "ROCm streaming checkpoint write rank %s bucket %s/%s: tensors=%s, bytes_items=%s, "
                    "tensor_bytes_gib=%s",
                    rank,
                    bucket_index + 1,
                    len(write_buckets),
                    len(tensor_data),
                    len(bytes_data),
                    tensor_bytes_gib,
                )
                idx, results = _write_streaming_bucket(
                    transform_list,
                    use_msc,
                    bucket_index,
                    write_bucket,
                    planner,
                    lazy_write_items,
                )
                write_results[idx] = results
        except Exception as exc:
            logger.exception("ROCm streaming checkpoint write failed on rank %s", rank)
            global_results_queue.put(RuntimeError(f"ROCm streaming checkpoint write failed: {exc}"))
            return

        global_results_queue.put(write_results)
        logger.info(
            "ROCm streaming checkpoint write finished on rank %s: buckets=%s, elapsed=%.2fs",
            rank,
            len(write_buckets),
            time() - start,
        )


def patch_rocm_checkpoint_writer() -> None:
    """Install the ROCm-safe checkpoint writer in all Megatron save aliases."""
    import megatron.core.optimizer.distrib_optimizer as distrib_optimizer_module
    import megatron.core.dist_checkpointing.strategies.filesystem_async as filesystem_async_module
    import megatron.core.dist_checkpointing.strategies.common as common_strategy_module
    import megatron.core.dist_checkpointing.strategies.torch as torch_strategy_module

    original_getter = getattr(
        filesystem_async_module,
        "_relax_original_get_write_results_queue",
        filesystem_async_module._get_write_results_queue,
    )
    filesystem_async_module._relax_original_get_write_results_queue = original_getter
    filesystem_async_module._get_write_results_queue = lambda: _rocm_write_results_queue(original_getter)
    filesystem_async_module.FileSystemWriterAsync = ROCmFileSystemWriterAsync
    torch_strategy_module.FileSystemWriterAsync = ROCmFileSystemWriterAsync
    _patch_rocm_mcore_planner_init(torch_strategy_module, "MCoreSavePlanner")
    _patch_rocm_mcore_planner_init(torch_strategy_module, "MCoreLoadPlanner")
    _patch_rocm_common_load_strategy(common_strategy_module)
    _patch_rocm_distributed_optimizer_step_state(distrib_optimizer_module)
    patch_rocm_checkpoint_rng_state()

    original_converter = getattr(
        torch_strategy_module,
        "_relax_original_sharded_tensor_to_torch_sharded_tensor",
        torch_strategy_module.sharded_tensor_to_torch_sharded_tensor,
    )
    torch_strategy_module._relax_original_sharded_tensor_to_torch_sharded_tensor = original_converter
    torch_strategy_module.sharded_tensor_to_torch_sharded_tensor = partial(
        _rocm_sharded_tensor_to_checkpointable,
        original_converter,
    )

    original_unwrap = getattr(
        torch_strategy_module,
        "_relax_original_unwrap_pyt_sharded_tensor",
        torch_strategy_module._unwrap_pyt_sharded_tensor,
    )
    torch_strategy_module._relax_original_unwrap_pyt_sharded_tensor = original_unwrap
    torch_strategy_module._unwrap_pyt_sharded_tensor = partial(
        _rocm_unwrap_checkpointable_sharded_tensor,
        original_unwrap,
    )
