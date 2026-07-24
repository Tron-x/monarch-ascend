#!/usr/bin/env python3
"""
Monarch Actor API: Ping Pong on NPU
====================================
Adapted from docs/source/examples/ping_pong.py for Ascend NPU.

Demonstrates:
  - Creating and spawning actors in process meshes
  - Calling endpoints on actors (broadcast and per-rank)
  - Actor-to-actor communication with a ping-pong example
  - NPU tensor creation and cross-mesh transfer

Changes from GPU version:
  - per_host={"gpus": N} → per_host={"npus": N} with npu_device() bootstrap
  - Added NPU tensor round-trip to verify device compute works inside actors
"""

import asyncio
import os
import sys

os.environ["PYTHONPATH"] = os.pathsep.join(sys.path)

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    print("ERROR: torch_npu not available")
    sys.exit(1)

from monarch.actor import Actor, current_rank, endpoint, this_host

NUM_ACTORS = 2  # 2 NPU cards


def npu_device(dev_id: int):
    """Bootstrap: isolate this process to a single NPU."""
    def _bootstrap():
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(dev_id)
        import torch
        import torch_npu  # noqa: F401
        torch.npu.set_device(0)
    return _bootstrap


# ---------------------------------------------------------------------------
# Part 1: Hello World — basic actor spawn and endpoint calls
# ---------------------------------------------------------------------------

class ToyActor(Actor):
    def __init__(self):
        self.rank = current_rank().rank

    @endpoint
    def hello_world(self, msg) -> str:
        line = f"[ToyActor rank={self.rank}] {msg}"
        print(line, flush=True)
        return line

    @endpoint
    def npu_compute(self) -> str:
        t = torch.arange(4, dtype=torch.float32, device="npu")
        result = t.sum().item()
        return f"[ToyActor rank={self.rank}] NPU sum([0,1,2,3]) = {result}"


# ---------------------------------------------------------------------------
# Part 2: Ping Pong — actor-to-actor communication across meshes
# ---------------------------------------------------------------------------

class PingPongActor(Actor):
    def __init__(self, actor_name: str):
        self.actor_name = actor_name
        self.identity = current_rank().rank
        self.other = None
        self.received = []

    @endpoint
    def init(self, other_actor) -> None:
        self.other = other_actor
        self.other_pair = other_actor.slice(**current_rank())

    @endpoint
    def send(self, msg: str) -> str:
        out = f"{self.actor_name}:{self.identity} → {msg}"
        self.other_pair.recv.call(out).get()
        return out

    @endpoint
    def recv(self, msg: str) -> None:
        line = f"Pong! {self.actor_name}:{self.identity} received: {msg}"
        print(line, flush=True)
        self.received.append(msg)

    @endpoint
    def npu_ping(self) -> str:
        t = torch.ones(4, dtype=torch.float32, device="npu") * (self.identity + 1)
        val = t.sum().item()
        return f"{self.actor_name}:{self.identity} NPU tensor sum = {val}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    print("=" * 60)
    print("Ping Pong on NPU")
    print("=" * 60)

    # --- Part 1: Hello World ---
    print("\n--- Part 1: Hello World ---")
    mesh = this_host().spawn_procs(
        per_host={"npus": NUM_ACTORS},
        bootstrap=npu_device(0),
    )
    toy = mesh.spawn("toy", ToyActor)

    print("[broadcast] Calling hello_world on all actors...")
    await toy.hello_world.call("hey from main!")

    print("[per-rank] Calling each actor individually...")
    for idx in range(NUM_ACTORS):
        result = await toy.slice(npus=idx).hello_world.call_one(f"unique msg for rank {idx}")
        print(f"  got: {result}")

    print("[npu_compute] Verifying NPU tensor ops...")
    for idx in range(NUM_ACTORS):
        result = await toy.slice(npus=idx).npu_compute.call_one()
        print(f"  {result}")

    # --- Part 2: Ping Pong ---
    print("\n--- Part 2: Ping Pong (cross-mesh) ---")
    mesh_a = this_host().spawn_procs(per_host={"npus": 1}, bootstrap=npu_device(0))
    mesh_b = this_host().spawn_procs(per_host={"npus": 1}, bootstrap=npu_device(1))

    actor_a = mesh_a.spawn("actor_a", PingPongActor, "A")
    actor_b = mesh_b.spawn("actor_b", PingPongActor, "B")

    await actor_a.init.call(actor_b)
    await actor_b.init.call(actor_a)

    print("[ping] A → B")
    r1 = await actor_a.send.call_one("Ping!")
    print(f"  sent: {r1}")

    print("[ping] B → A")
    r2 = await actor_b.send.call_one("Pong back!")
    print(f"  sent: {r2}")

    print("[npu_ping] Verify NPU works in both meshes...")
    print(f"  {await actor_a.npu_ping.call_one()}")
    print(f"  {await actor_b.npu_ping.call_one()}")

    print("\n" + "=" * 60)
    print("Ping Pong on NPU — PASSED")
    print("=" * 60)


if __name__ == "__main__":
    from monarch._src.actor.actor_mesh import context, shutdown_context

    context()
    try:
        asyncio.run(main())
    finally:
        shutdown_context().get(timeout=45.0)
