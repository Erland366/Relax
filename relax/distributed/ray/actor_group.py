# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import os
import random
import time
from typing import Any

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from relax.distributed.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST, Lock
from relax.utils.http_utils import get_host_info
from relax.utils.logging_utils import get_logger
from relax.utils.misc import get_free_port


logger = get_logger(__name__)


class RayTrainGroup:
    """A group of ray actors Functions start with 'async' should return list of
    object refs.

    Args:
        args (Namespace): Arguments for the actor group.
        num_nodes (int): Number of nodes for this actor group.
        num_gpus_per_node (int): Number of gpus for this actor group.
        pg (PlacementGroup, optional): Placement group to schedule actor on.
            If none, create new placement group automatically. Defaults to None.
        num_gpus_per_actor (float, optional): Number of gpus allocated for each actor.
            If < 1.0, multiple models can share same gpu. Defaults to 1.
        resources (Dict[str, float], optional): Custom resources to allocate for each actor.
            See https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
        num_resources_per_node (int, optional): Number of custom resources to allocate for each node.
            See https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
    """

    def __init__(
        self,
        args,
        num_gpus,
        pg: tuple[PlacementGroup, list[int], list[int]],
        num_gpus_per_actor: float = 1,
        role: str = "actor",
        runtime_env: dict = None,
    ) -> None:
        self.args = args
        self._num_gpus = num_gpus
        self.role = role
        self.runtime_env = runtime_env
        # Allocate the GPUs for actors w/o instantiating them
        init_t0 = time.perf_counter()
        self._allocate_gpus_for_actor(pg, num_gpus_per_actor)
        logger.info(
            "launch_timing: ray_train_group.%s.__init__ num_gpus=%s elapsed=%.2fs",
            self.role,
            num_gpus,
            time.perf_counter() - init_t0,
        )

    def _allocate_gpus_for_actor(self, pg, num_gpus_per_actor):
        alloc_t0 = time.perf_counter()
        world_size = self._num_gpus

        # Use placement group to lock resources for models of same type
        assert pg is not None
        if len(pg) == 4:
            pg, reordered_bundle_indices, _reordered_gpu_ids, reordered_node_ips = pg
        else:
            pg, reordered_bundle_indices, _reordered_gpu_ids = pg
            reordered_node_ips = None

        env_vars = {
            # because sglang will always set NCCL_CUMEM_ENABLE to 0
            # we need also set it to 0 to prevent nccl error.
            "NCCL_CUMEM_ENABLE": os.environ.get("NCCL_CUMEM_ENABLE", "0"),
            "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": os.environ.get("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1"),
            **{name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST},
            **self.runtime_env.get("env_vars", {}),
            **self.args.train_env_vars,
        }

        phase_t0 = time.perf_counter()
        if self.args.offload_train and self.args.train_backend == "megatron":
            import torch_memory_saver

            dynlib_path = os.path.join(
                os.path.dirname(os.path.dirname(torch_memory_saver.__file__)),
                "torch_memory_saver_hook_mode_preload.abi3.so",
            )
            assert os.path.exists(dynlib_path), f"LD_PRELOAD so file {dynlib_path} does not exist."

            env_vars["LD_PRELOAD"] = dynlib_path
            env_vars["TMS_INIT_ENABLE"] = "1"
            env_vars["TMS_INIT_ENABLE_CPU_BACKUP"] = "1"

        # We cannot do routing replay for critic.
        if self.args.use_routing_replay and self.role == "actor":
            env_vars["ENABLE_ROUTING_REPLAY"] = "1"
        logger.info(
            "launch_timing: ray_train_group.%s.env_build elapsed=%.2fs",
            self.role,
            time.perf_counter() - phase_t0,
        )

        phase_t0 = time.perf_counter()
        from relax.distributed.ray.train_actor import LazyMegatronTrainRayActor
        logger.info(
            "launch_timing: ray_train_group.%s.import_train_actor elapsed=%.2fs",
            self.role,
            time.perf_counter() - phase_t0,
        )

        actor_impl = LazyMegatronTrainRayActor

        phase_t0 = time.perf_counter()
        TrainRayActor = ray.remote(
            num_gpus=1,
            runtime_env={"env_vars": env_vars},
            enable_task_events=False,
        )(actor_impl)
        lock = Lock.options(num_cpus=0, num_gpus=0).remote()
        logger.info(
            "launch_timing: ray_train_group.%s.remote_class_and_lock elapsed=%.2fs",
            self.role,
            time.perf_counter() - phase_t0,
        )

        # Create worker actors
        self._actor_handlers = []
        phase_t0 = time.perf_counter()
        master_addr, master_port = self._preallocate_rank0_master_addr(reordered_node_ips)
        preallocated_master = master_addr is not None
        logger.info(
            "launch_timing: ray_train_group.%s.preallocate_master_addr elapsed=%.2fs preallocated=%s",
            self.role,
            time.perf_counter() - phase_t0,
            preallocated_master,
        )

        phase_t0 = time.perf_counter()
        for rank in range(world_size):
            rank_t0 = time.perf_counter()
            actor = TrainRayActor.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=reordered_bundle_indices[rank],
                ),
            ).remote(world_size, rank, master_addr, master_port, lock)
            if rank == 0 and not preallocated_master:
                wait_t0 = time.perf_counter()
                master_addr, master_port = ray.get(actor.get_master_addr_and_port.remote())
                logger.info(
                    "launch_timing: ray_train_group.%s.rank0_master_addr elapsed=%.2fs",
                    self.role,
                    time.perf_counter() - wait_t0,
                )
            self._actor_handlers.append(actor)
            logger.info(
                "launch_timing: ray_train_group.%s.create_actor_rank rank=%s elapsed=%.2fs",
                self.role,
                rank,
                time.perf_counter() - rank_t0,
            )
        logger.info(
            "launch_timing: ray_train_group.%s.create_all_actors elapsed=%.2fs",
            self.role,
            time.perf_counter() - phase_t0,
        )
        logger.info(
            "launch_timing: ray_train_group.%s.allocate_total elapsed=%.2fs",
            self.role,
            time.perf_counter() - alloc_t0,
        )

    def _preallocate_rank0_master_addr(self, reordered_node_ips: list[str] | None) -> tuple[str | None, int | None]:
        if not reordered_node_ips:
            return None, None

        rank0_node_ip = reordered_node_ips[0]
        current_node_ip = get_host_info()[1]
        if rank0_node_ip != current_node_ip:
            return None, None

        return rank0_node_ip, get_free_port(start_port=random.randint(20000, 21000))

    def async_init(self, args, role, with_ref=False, with_opd_teacher=False):
        """Allocate GPU resourced and initialize model, optimzier, local ckpt,
        etc."""
        self.args = args
        return [
            actor.init.remote(args, role, with_ref=with_ref, with_opd_teacher=with_opd_teacher)
            for actor in self._actor_handlers
        ]

    def async_train(self, rollout_id):
        """Do one rollout training."""
        return [actor.train.remote(rollout_id) for actor in self._actor_handlers]

    def async_compute_ref_log_prob(self, rollout_id):
        """Compute reference log prob for routing replay."""
        return [actor.compute_ref_log_prob.remote(rollout_id) for actor in self._actor_handlers]

    def async_compute_actor_log_prob(self, rollout_id):
        """Compute actor log prob for routing replay."""
        return [actor.compute_actor_log_prob.remote(rollout_id) for actor in self._actor_handlers]

    def train_fully_async(self, rollout_id):
        """Do one rollout training without ref log prob computation."""
        return [actor.train_async.remote(rollout_id) for actor in self._actor_handlers]

    def train_hybrid(self, rollout_id):
        """Hybrid mode: actor handles ref/actor_fwd/adv internally."""
        return [actor.train_hybrid.remote(rollout_id) for actor in self._actor_handlers]

    def save_model(self, rollout_id, force_sync=False):
        """Save actor model."""
        ray.get([actor.save_model.remote(rollout_id, force_sync=force_sync) for actor in self._actor_handlers])

    def update_weights(self):
        """Broadcast weights from rank 0 to all other ranks."""
        ray.get([actor.update_weights.remote() for actor in self._actor_handlers])

    def update_weights_fully_async(self, rollout_id, rollout_only=False, actor_fwd_only=False) -> None:
        """Update weights in fully async mode (sends to rollout and
        actor_fwd)."""
        ray.get(
            [
                actor.update_weights_fully_async.remote(
                    rollout_id, rollout_only=rollout_only, actor_fwd_only=actor_fwd_only
                )
                for actor in self._actor_handlers
            ]
        )

    def recv_weight_fully_async(self, rollout_id) -> None:
        ray.get([actor.recv_weight_fully_async.remote(rollout_id) for actor in self._actor_handlers])

    def onload(self):
        ray.get([actor.wake_up.remote() for actor in self._actor_handlers])

    def offload(self):
        ray.get([actor.sleep.remote() for actor in self._actor_handlers])

    def clear_memory(self):
        ray.get([actor.clear_memory.remote() for actor in self._actor_handlers])

    def set_rollout_manager(self, rollout_manager: Any):
        ray.get([actor.set_rollout_manager.remote(rollout_manager) for actor in self._actor_handlers])

    def set_genrm_manager(self, genrm_manager: Any):
        """Set the genRM manager for coordinated offload/onload.

        In colocated mode, the genRM manager is used to offload genRM engines
        before training and onload them before rollout, since they share GPU
        resources.
        """
        ray.get([actor.set_genrm_manager.remote(genrm_manager) for actor in self._actor_handlers])
