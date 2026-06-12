# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import abc
import os
import random
import time
from datetime import timedelta

import ray
import torch
import torch.distributed as dist

import relax.utils.training.eval_config
from relax.distributed.ray.ray_actor import RayActor
from relax.utils.distributed_utils import init_gloo_group
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def should_sleep_train_actor_after_init(args) -> bool:
    """Return whether the train actor should offload itself after init.

    In fully async mode, the actor must sleep after init because later weight
    sync paths expect the train backend to wake on demand. In sync colocated
    mode, the actor and rollout share the same GPUs, so keeping the actor
    resident through rollout startup can destabilize backend bring-up and waste
    GPU memory. Non-colocated sync mode can keep the actor resident to avoid an
    unnecessary wake-up before rollout-manager hookup.
    """

    if not getattr(args, "offload_train", False):
        return False

    if getattr(args, "fully_async", False):
        return True

    return bool(getattr(args, "colocate", False))


def get_local_gpu_id():
    from relax.utils import device as device_utils

    cvd = os.environ.get(device_utils.get_visible_devices_env_var(), None)
    if cvd is None:
        return ray.get_gpu_ids()[0]
    else:
        return cvd.split(",").index(str(ray.get_gpu_ids()[0]))


class TrainRayActor(RayActor):
    def __init__(self, world_size, rank, master_addr, master_port, lock):
        self._world_size = world_size
        self._rank = rank
        self.lock = lock
        if master_addr:
            self.master_addr, self.master_port = master_addr, master_port
        else:
            self.master_addr, self.master_port = self._get_current_node_ip_and_free_port(
                start_port=random.randint(20000, 21000)
            )

        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        # TODO: currently this doesn't work as ray has already set torch.cuda.device_count().
        # os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        # os.environ["LOCAL_RANK"] = str(ray.get_gpu_ids()[0])
        os.environ["LOCAL_RANK"] = str(get_local_gpu_id())

    def init(self, args, role, with_ref=False, with_opd_teacher=False):
        from relax.utils import device as device_utils

        self.args = args
        self.role = role
        self.with_ref = with_ref
        self.with_opd_teacher = with_opd_teacher

        torch.serialization.add_safe_globals([relax.utils.training.eval_config.EvalDatasetConfig])

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device_utils.set_device(f"{device_utils.get_device_name()}:{local_rank}")

        backend = args.distributed_backend

        dist.init_process_group(
            backend=backend,
            timeout=timedelta(minutes=args.distributed_timeout_minutes),
        )
        init_gloo_group()

        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()

        numa_local_rank = int(os.environ["RANK"]) % args.num_gpus_per_node
        device_utils.set_numa_affinity(numa_local_rank)

    def clear_memory(self):
        from relax.utils.memory_utils import clear_memory, print_memory

        print_memory("before TrainRayActor.clear_memory")
        clear_memory()
        print_memory("after TrainRayActor.clear_memory")

    @abc.abstractmethod
    def sleep(self, tags):
        raise NotImplementedError

    @abc.abstractmethod
    def wake_up(self, tags):
        raise NotImplementedError

    @abc.abstractmethod
    def train(self, rollout_id, rollout_data_ref):
        raise NotImplementedError

    @abc.abstractmethod
    def save_model(self, rollout_id, force_sync=False):
        raise NotImplementedError

    @abc.abstractmethod
    def update_weights(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _get_parallel_config(self):
        raise NotImplementedError

    def set_rollout_manager(self, rollout_manager):
        if self.args.offload_train and getattr(self, "_is_sleeping", False):
            logger.info("Waking actor before set_rollout_manager because offload_train is enabled")
            self.wake_up()

        self.rollout_manager = rollout_manager
        if not self.args.fully_async:
            return

        if not self.args.debug_rollout_only and self.args.rank == 0:
            ray.get(self.rollout_manager.set_train_parallel_config.remote(self.train_parallel_config))
        # Retrieve the distributed lock that serialises DCS weight sync with
        # P2P direct sync (_sync_weights_from_seed_engine on RolloutManager).
        self._weight_sync_lock = ray.get(self.rollout_manager.get_weight_sync_lock.remote())

    def set_genrm_manager(self, genrm_manager):
        """Set the genRM manager for coordinated offload/onload.

        In colocated mode, the genRM manager is used to offload genRM engines
        before training and onload them before rollout, since they share GPU
        resources.
        """
        self.genrm_manager = genrm_manager


class LazyMegatronTrainRayActor(TrainRayActor):
    """Thin Ray actor wrapper that imports the Megatron actor inside the worker.

    RayTrainGroup only needs a remote actor class while constructing the Serve
    replica. Importing ``relax.backends.megatron.actor`` there is expensive and
    delays replica startup. This wrapper keeps the Ray actor surface lightweight
    and creates the real Megatron actor inside the nested train worker on
    ``init``.
    """

    def __init__(self, world_size, rank, master_addr, master_port, lock):
        super().__init__(world_size, rank, master_addr, master_port, lock)
        self._impl = None

    def _require_impl(self):
        assert self._impl is not None, "Megatron train actor has not been initialized."
        return self._impl

    def init(self, args, role, with_ref=False, with_opd_teacher=False):
        phase_t0 = time.perf_counter()
        from relax.backends.megatron.actor import MegatronTrainRayActor

        logger.info(
            "launch_timing: lazy_megatron_train_actor.import_impl role=%s rank=%s elapsed=%.2fs",
            role,
            self._rank,
            time.perf_counter() - phase_t0,
        )
        self._impl = MegatronTrainRayActor(
            self._world_size,
            self._rank,
            self.master_addr,
            self.master_port,
            self.lock,
        )
        return self._impl.init(args, role, with_ref=with_ref, with_opd_teacher=with_opd_teacher)

    def clear_memory(self):
        return self._require_impl().clear_memory()

    def sleep(self, tags=None):
        return self._require_impl().sleep() if tags is None else self._require_impl().sleep(tags)

    def wake_up(self, tags=None):
        return self._require_impl().wake_up() if tags is None else self._require_impl().wake_up(tags)

    def train(self, *args, **kwargs):
        return self._require_impl().train(*args, **kwargs)

    def train_async(self, *args, **kwargs):
        return self._require_impl().train_async(*args, **kwargs)

    def train_hybrid(self, *args, **kwargs):
        return self._require_impl().train_hybrid(*args, **kwargs)

    def compute_ref_log_prob(self, *args, **kwargs):
        return self._require_impl().compute_ref_log_prob(*args, **kwargs)

    def compute_actor_log_prob(self, *args, **kwargs):
        return self._require_impl().compute_actor_log_prob(*args, **kwargs)

    def save_model(self, *args, **kwargs):
        return self._require_impl().save_model(*args, **kwargs)

    def update_weights(self, *args, **kwargs):
        return self._require_impl().update_weights(*args, **kwargs)

    def update_weights_fully_async(self, *args, **kwargs):
        return self._require_impl().update_weights_fully_async(*args, **kwargs)

    def recv_weight_fully_async(self, *args, **kwargs):
        return self._require_impl().recv_weight_fully_async(*args, **kwargs)

    def set_rollout_manager(self, *args, **kwargs):
        return self._require_impl().set_rollout_manager(*args, **kwargs)

    def set_genrm_manager(self, *args, **kwargs):
        return self._require_impl().set_genrm_manager(*args, **kwargs)

    def load_other_checkpoint(self, *args, **kwargs):
        return self._require_impl().load_other_checkpoint(*args, **kwargs)

    def _get_parallel_config(self):
        return self._require_impl()._get_parallel_config()
