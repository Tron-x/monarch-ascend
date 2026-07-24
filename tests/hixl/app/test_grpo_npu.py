#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
GRPO on NPU — Two-mesh test (Learner mesh + Generator mesh)
============================================================
Adapted from docs/source/examples/grpo_actor.py for Ascend NPU.

Changes from GPU version:
  - "cuda" → "npu", per_host={"gpus": N} → per_host={"npus": N}
  - Replaced torch.distributions.Categorical with manual softmax+multinomial
    (Categorical may not be fully supported on NPU)
  - Removed kl_divergence import (replaced with manual computation)
  - HCCS requires 2MB-aligned buffers — uses alloc_aligned_tensor() + flat buffer

Architecture:
  learner_mesh (1 NPU): Learner + Scorer + TrajectoryQueue + ReplayBuffer
  gen_mesh     (1 NPU): Generator ×1

  Mesh间通信: Generator.update() reads weights from Learner via XDMABuffer (HIXL)

Device management:
  Uses npu_device(N) bootstrap to set ASCEND_RT_VISIBLE_DEVICES=N,
  isolating each process to a single NPU card (logical device 0).
  Actor constructors don't need manual device management.
"""

import os
import sys
# Transport: defaults to HCCS (intra-supernode).
# HCCS requires 2MB-aligned device memory — use alloc_aligned_tensor().
# Set MONARCH_HIXL_TRANSPORT=roce to force RoCE if HCCS is unavailable.
os.environ["PYTHONPATH"] = os.pathsep.join(sys.path)
# Exercise the deployed CANN 9.1 AICPU kernel and avoid legacy HCCL port
# contention between the two local worker processes.
os.environ.setdefault("MONARCH_HIXL_USE_LOCAL_COMM_RES", "1")

_hixl_lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build")
if os.path.isdir(_hixl_lib_dir):
    os.environ["LD_LIBRARY_PATH"] = _hixl_lib_dir + ":" + os.environ.get("LD_LIBRARY_PATH", "")

import asyncio
import copy
import random
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

try:
    import torch_npu
except ImportError:
    print("ERROR: torch_npu not available")
    sys.exit(1)

from monarch.actor import Actor, endpoint, this_host
from monarch._src.actor.host_mesh import default_bootstrap_cmd
from monarch._src.rdma.xdma import XDMABuffer as RDMABuffer
from monarch._src.rdma.xdma import alloc_aligned_tensor

G = 8  # group size for GRPO
STATE_DIM = 4
ACTION_DIM = 4
DEVICE = "npu"


def npu_device(dev_id: int):
    """Bootstrap function: bind this process to a specific NPU."""
    def _bootstrap():
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(dev_id)
        os.environ["MONARCH_NPU_DEVICE"] = "0"
        import torch
        import torch_npu  # noqa: F401
        torch.npu.set_device(0)
    return _bootstrap


def npu_bootstrap_command(dev_id: int):
    return default_bootstrap_cmd().with_env(
        {
            "ASCEND_RT_VISIBLE_DEVICES": str(dev_id),
            "MONARCH_HIXL_USE_LOCAL_COMM_RES": "1",
            "MONARCH_NPU_DEVICE": "0",
        }
    )


@dataclass
class TrajectorySlice:
    policy_version: int
    state: torch.Tensor
    actions: torch.Tensor
    old_logps: torch.Tensor
    rewards: torch.Tensor


@dataclass
class TrainingBatch:
    states: torch.Tensor
    actions: torch.Tensor
    old_logps: torch.Tensor
    rewards: torch.Tensor
    policy_versions: List[int]


class TrajectoryQueue(Actor):
    def __init__(self):
        self.queue: asyncio.Queue[TrajectorySlice] = asyncio.Queue()

    @endpoint
    async def put(self, slice: TrajectorySlice) -> None:
        await self.queue.put(slice)

    @endpoint
    async def get(self) -> TrajectorySlice:
        return await self.queue.get()

    @endpoint
    async def get_timeout(self, timeout: float) -> Optional[TrajectorySlice]:
        try:
            return await asyncio.wait_for(self.queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None


class ReplayBuffer(Actor):
    def __init__(self):
        self.storage: List[Tuple[int, TrajectorySlice]] = []
        self.storage_event = asyncio.Event()

    @endpoint
    async def put(self, slice: TrajectorySlice) -> None:
        self.storage.append((slice.policy_version, slice))
        self.storage_event.set()

    async def _wait_for_storage(self):
        if not self.storage:
            await self.storage_event.wait()

    @endpoint
    async def sample_from(self, k: int) -> List[TrajectorySlice]:
        try:
            await asyncio.wait_for(self._wait_for_storage(), timeout=10.0)
        except asyncio.TimeoutError:
            raise RuntimeError("Timeout waiting for ReplayBuffer to be populated")

        policy_versions = [version + 1 for version, _ in self.storage]
        total = sum(policy_versions)
        probs = [v / total for v in policy_versions]
        indices = list(range(len(self.storage)))
        chosen_indices = random.choices(indices, weights=probs, k=k)
        return [self.storage[i][1] for i in chosen_indices]


class Scorer(Actor):
    def __init__(self, trajectory_queue: Any, replay_buffer: Any):
        self.trajectory_queue = trajectory_queue
        self.replay_buffer = replay_buffer
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + 1, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
        ).to(DEVICE)
    async def _score_slice(self, slice: TrajectorySlice) -> None:
        s = slice.state.to(DEVICE).unsqueeze(0).repeat(G, 1)
        a = slice.actions.to(DEVICE).float().unsqueeze(-1)
        rewards = self.net(torch.cat([s, a], dim=-1)).squeeze(-1)

        scored = TrajectorySlice(
            policy_version=slice.policy_version,
            state=slice.state,
            actions=slice.actions,
            old_logps=slice.old_logps,
            rewards=rewards,
        )
        await self.replay_buffer.put.call(scored)

    @endpoint
    async def score_one(self) -> None:
        slice_ = await self.trajectory_queue.get.call_one()
        await self._score_slice(slice_)


WEIGHT_BUF_SIZE = 2 * 1024 * 1024  # 2MB — HCCS requires 2MB-aligned buffers


class Learner(Actor):
    def __init__(self, replay_buffer: Any):
        print(f"[Learner.__init__] PID={os.getpid()} "
              f"ASCEND_RT_VISIBLE_DEVICES={os.environ.get('ASCEND_RT_VISIBLE_DEVICES', 'NOT_SET')}",
              flush=True)

        self.model = nn.Sequential(
            nn.Linear(STATE_DIM, 16), nn.Tanh(), nn.Linear(16, ACTION_DIM)
        ).to(DEVICE)
        self.ref_model = copy.deepcopy(self.model)
        for p in self.ref_model.parameters():
            p.requires_grad = False
        self.ref_model.eval()

        self.optim = optim.Adam(self.model.parameters(), lr=1e-3, eps=1e-5)
        self.eps = 0.2
        self.kl_coeff = 0.1
        self.policy_version = 0
        self.replay_buffer = replay_buffer
        self.batch_size = 2
        self.generators: Optional[Any] = None
        self._flat_weights, self._flat_weights_backing = alloc_aligned_tensor(
            WEIGHT_BUF_SIZE, dtype=torch.uint8, device=DEVICE
        )
        torch.npu.synchronize()
        print(
            f"[Learner] Weight buffer: addr={hex(self._flat_weights.data_ptr())}, "
            f"size={WEIGHT_BUF_SIZE}, aligned={self._flat_weights.data_ptr() % (2*1024*1024) == 0}",
            flush=True,
        )
        self._flat_buf: Optional[RDMABuffer] = None
        self._weight_metadata: List[Tuple[str, torch.Size, torch.dtype, int, int]] = []

    @endpoint
    async def init_generators(self, generators: Any) -> None:
        self.generators = generators

    def _pack_weights(self) -> None:
        """Copy model weights into the pre-allocated flat buffer."""
        sd = self.model.state_dict()
        metadata = []
        offset = 0
        for k, v in sd.items():
            flat = v.detach().view(torch.uint8).flatten()
            metadata.append((k, v.shape, v.dtype, offset, flat.numel()))
            self._flat_weights[offset:offset + flat.numel()] = flat
            offset += flat.numel()
        assert offset <= WEIGHT_BUF_SIZE, f"Model weights ({offset}B) exceed buffer ({WEIGHT_BUF_SIZE}B)"
        self._weight_metadata = metadata
        torch.npu.synchronize()

    @endpoint
    async def weights_handle(self) -> Tuple[RDMABuffer, List[Tuple[str, torch.Size, torch.dtype, int, int]]]:
        self._pack_weights()
        self._flat_buf = RDMABuffer(self._flat_weights)
        print(f"[Learner] Flat weight buffer: {self._flat_weights.numel()} bytes", flush=True)
        return self._flat_buf, self._weight_metadata

    def refresh_weights(self) -> None:
        """Copy latest model weights into the existing flat buffer (same address)."""
        sd = self.model.state_dict()
        offset = 0
        for k, v in sd.items():
            flat = v.detach().view(torch.uint8).flatten()
            self._flat_weights[offset:offset + flat.numel()] = flat
            offset += flat.numel()
        torch.npu.synchronize()

    def _compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        batch_size = rewards.shape[0] // G
        rewards_reshaped = rewards.view(batch_size, G)
        baselines = rewards_reshaped.mean(dim=1, keepdim=True)
        advantages = (rewards_reshaped - baselines).reshape(-1)
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    def _apply_policy_update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_logps: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.model(states)
        log_probs = torch.log_softmax(logits, dim=-1)
        new_logps = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        ratio = (new_logps - old_logps).exp()
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantages
        ppo_loss = -torch.min(unclipped, clipped).mean()

        with torch.no_grad():
            ref_logits = self.ref_model(states)
        ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        kl = (probs * (log_probs - ref_log_probs)).sum(dim=-1).mean()

        loss = ppo_loss + self.kl_coeff * kl
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optim.step()
        self.policy_version += 1
        return loss.detach()

    @endpoint
    async def update_generators(self) -> None:
        if self.generators:
            self.refresh_weights()
            await self.generators.update.call(self.policy_version)

    @endpoint
    async def step(self) -> torch.Tensor:
        slices = await self.replay_buffer.sample_from.call_one(self.batch_size)
        raw_states = torch.stack([s.state for s in slices])
        actions = torch.cat([s.actions for s in slices])
        old_logps = torch.cat([s.old_logps for s in slices])
        rewards = torch.cat([s.rewards for s in slices])

        states = raw_states.repeat_interleave(G, 0).to(DEVICE)
        actions, old_logps, rewards = [
            x.to(DEVICE) for x in (actions, old_logps, rewards)
        ]
        advs = self._compute_advantages(rewards)
        return self._apply_policy_update(states, actions, old_logps, advs)


class GeneratorState:
    READY_TO_GENERATE = "READY_TO_GENERATE"
    READY_TO_UPDATE = "READY_TO_UPDATE"


class Generator(Actor):
    def __init__(self, weight_buf, weight_metadata, trajectory_queue):
        print(f"[Generator.__init__] PID={os.getpid()} "
              f"ASCEND_RT_VISIBLE_DEVICES={os.environ.get('ASCEND_RT_VISIBLE_DEVICES', 'NOT_SET')}",
              flush=True)

        self._local_flat: Optional[torch.Tensor] = None

        self.model = nn.Sequential(
            nn.Linear(STATE_DIM, 16), nn.Tanh(), nn.Linear(16, ACTION_DIM)
        ).to(DEVICE)
        self.weight_buf: RDMABuffer = weight_buf
        self.weight_metadata = weight_metadata
        self.trajectory_queue = trajectory_queue
        self.state = GeneratorState.READY_TO_GENERATE
        self.cond = asyncio.Condition()
        self.policy_version = 0

    @endpoint
    async def generate(self, state: torch.Tensor) -> None:
        async with self.cond:
            await self.cond.wait_for(
                lambda: self.state == GeneratorState.READY_TO_GENERATE
            )

            x = state.to(DEVICE).unsqueeze(0).repeat(G, 1)
            with torch.no_grad():
                logits = self.model(x)
            probs = torch.softmax(logits, dim=-1)
            acts = torch.multinomial(probs, num_samples=1).squeeze(-1)
            logps = torch.log_softmax(logits, dim=-1).gather(
                1, acts.unsqueeze(-1)
            ).squeeze(-1)

            slice_ = TrajectorySlice(
                self.policy_version,
                state,
                acts,
                logps,
                torch.zeros(G),
            )

        await self.trajectory_queue.put.call(slice_)

        async with self.cond:
            self.state = GeneratorState.READY_TO_UPDATE
            self.cond.notify_all()

    def _ensure_local_flat(self) -> torch.Tensor:
        if self._local_flat is None:
            self._local_flat, self._local_flat_backing = alloc_aligned_tensor(
                WEIGHT_BUF_SIZE, dtype=torch.uint8, device=DEVICE
            )
            torch.npu.synchronize()
        return self._local_flat

    @endpoint
    async def update(self, version: int) -> None:
        async with self.cond:
            local_buf = self._ensure_local_flat()
            await self.weight_buf.read_into(local_buf)
            torch.npu.synchronize()
            sd = self.model.state_dict()
            for name, shape, dtype, offset, nbytes in self.weight_metadata:
                chunk = self._local_flat[offset:offset + nbytes].clone()
                sd[name] = chunk.view(dtype).reshape(shape)
            self.model.load_state_dict(sd)
            self.policy_version = version
            self.state = GeneratorState.READY_TO_GENERATE
            self.cond.notify_all()


async def main():
    print("=" * 60)
    print("GRPO on NPU — Two-mesh, two-card test")
    print("  learner_mesh: NPU 0 (Learner + Scorer + Queues)")
    print("  gen_mesh:     NPU 1 (Generator x1)")
    print("  Inter-mesh weight sync via RDMA (HIXL/HCCS)")
    print("=" * 60)

    # Two meshes, each on a different physical NPU.
    # npu_device(N) sets ASCEND_RT_VISIBLE_DEVICES=N so each process
    # only sees one NPU card as logical device 0.
    learner_mesh = this_host().spawn_procs(
        per_host={"npus": 1},
        bootstrap=npu_device(0),
        bootstrap_command=npu_bootstrap_command(0),
    )
    gen_mesh = this_host().spawn_procs(
        per_host={"npus": 1},
        bootstrap=npu_device(1),
        bootstrap_command=npu_bootstrap_command(1),
    )

    print("[1/6] Spawning actors on learner_mesh (NPU 0)...", flush=True)
    traj_q = learner_mesh.spawn("traj", TrajectoryQueue)
    replay_buf = learner_mesh.spawn("rb", ReplayBuffer)
    learner = learner_mesh.spawn("learner", Learner, replay_buf)
    scorer = learner_mesh.spawn("scorer", Scorer, traj_q, replay_buf)

    print("[2/6] Getting weight handles and spawning generators (NPU 1)...", flush=True)
    flat_buf, weight_meta = await asyncio.wait_for(
        learner.weights_handle.call_one(),
        timeout=60.0,
    )
    print("      weight handle received", flush=True)
    generators = gen_mesh.spawn("generator", Generator, flat_buf, weight_meta, traj_q)
    await asyncio.wait_for(learner.init_generators.call(generators), timeout=60.0)
    print("      generator reference installed", flush=True)

    print("[3/6] Initial generation...", flush=True)
    await asyncio.wait_for(
        generators.generate.call(torch.randn(STATE_DIM)),
        timeout=60.0,
    )

    print("[4/6] Scoring initial trajectory...", flush=True)
    await asyncio.wait_for(scorer.score_one.call_one(), timeout=60.0)

    print("[5/6] Training loop (5 steps)...", flush=True)
    for step in range(5):
        state = torch.randn(STATE_DIM)
        await asyncio.wait_for(learner.update_generators.call_one(), timeout=60.0)
        await asyncio.wait_for(generators.generate.call(state), timeout=60.0)
        await asyncio.wait_for(scorer.score_one.call_one(), timeout=60.0)
        loss = await asyncio.wait_for(
            learner.step.call_one(),
            timeout=60.0,
        )
        print(f"  [Step {step:02d}] loss={loss:.4f}", flush=True)

    print("[6/6] All trajectories scored.", flush=True)

    print("GRPO training complete!", flush=True)


if __name__ == "__main__":
    from monarch._src.actor.actor_mesh import context, shutdown_context

    context()
    try:
        asyncio.run(main())
    finally:
        shutdown_context().get(timeout=75.0)
