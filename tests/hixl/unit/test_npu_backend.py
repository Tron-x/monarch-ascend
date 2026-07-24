#!/usr/bin/env python3
"""
NPU backend validation tests for monarch ascend_engine.

Run: source /usr/local/Ascend/ascend-toolkit/set_env.sh && python test_npu_backend.py
"""

import sys
import traceback

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def test_pass(name):
    print(f"  [PASS] {name}")

def test_fail(name, err):
    print(f"  [FAIL] {name}: {err}")

# ============================================================
# Level 1: Basic imports and environment
# ============================================================
print_section("Level 1: Basic Environment")

try:
    import torch
    test_pass(f"torch {torch.__version__}")
except Exception as e:
    test_fail("import torch", e)
    sys.exit(1)

try:
    import torch_npu
    test_pass(f"torch_npu {torch_npu.__version__}")
except Exception as e:
    test_fail("import torch_npu", e)
    sys.exit(1)

try:
    assert torch.npu.is_available(), "NPU not available"
    npu_count = torch.npu.device_count()
    test_pass(f"NPU available, count={npu_count}")
except Exception as e:
    test_fail("NPU availability", e)
    sys.exit(1)

# ============================================================
# Level 2: NPU compute basics (torch_npu)
# ============================================================
print_section("Level 2: NPU Compute (torch_npu)")

try:
    t = torch.randn(3, 4).npu()
    assert t.device.type == "npu"
    test_pass(f"tensor on NPU: {t.device}")
except Exception as e:
    test_fail("tensor to NPU", e)

try:
    a = torch.randn(3, 4).npu()
    b = torch.randn(3, 4).npu()
    c = a + b
    result = c.cpu()
    assert result.shape == (3, 4)
    test_pass("NPU add operation")
except Exception as e:
    test_fail("NPU add", e)

try:
    a = torch.randn(4, 3).npu()
    b = torch.randn(3, 5).npu()
    c = torch.matmul(a, b)
    result = c.cpu()
    assert result.shape == (4, 5)
    test_pass("NPU matmul operation")
except Exception as e:
    test_fail("NPU matmul", e)

# ============================================================
# Level 3: Monarch Rust bindings
# ============================================================
print_section("Level 3: Monarch Rust Bindings")

try:
    from monarch._rust_bindings import has_tensor_engine
    assert has_tensor_engine(), "tensor engine not enabled"
    test_pass("has_tensor_engine() = True")
except Exception as e:
    test_fail("has_tensor_engine", e)
    sys.exit(1)

try:
    from monarch._rust_bindings.monarch_extension import client
    test_pass("client module")
except Exception as e:
    test_fail("client module", e)

try:
    from monarch._rust_bindings.monarch_extension import tensor_worker
    test_pass("tensor_worker module")
except Exception as e:
    test_fail("tensor_worker module", e)

try:
    from monarch._rust_bindings.monarch_extension import mesh_controller
    test_pass("mesh_controller module")
except Exception as e:
    test_fail("mesh_controller module", e)

try:
    from monarch._rust_bindings.monarch_extension import convert
    test_pass("convert module")
except Exception as e:
    test_fail("convert module", e)

# ============================================================
# Level 4: Monarch Actor / ProcMesh (no device needed)
# ============================================================
print_section("Level 4: ProcMesh Creation")

try:
    from monarch._src.actor.host_mesh import this_host
    host = this_host()
    test_pass("this_host() created")
except Exception as e:
    test_fail("this_host()", e)
    traceback.print_exc()
    sys.exit(1)

try:
    pm = host.spawn_procs()
    pm.initialized.get()
    test_pass("spawn_procs() (no device) initialized")
except Exception as e:
    test_fail("spawn_procs()", e)
    traceback.print_exc()

# ============================================================
# Level 5: ProcMesh with GPU/NPU dimension
# ============================================================
print_section("Level 5: ProcMesh with Device Dimension")

try:
    pm = host.spawn_procs(per_host={"gpus": 2})
    test_pass("spawn_procs(per_host={'gpus': 2}) created")
except Exception as e:
    test_fail("spawn_procs with gpus", e)
    traceback.print_exc()

# ============================================================
# Level 6: Tensor Engine on NPU
# ============================================================
print_section("Level 6: Tensor Engine (spawn_tensor_engine)")

try:
    import monarch
    from monarch.mesh_controller import spawn_tensor_engine

    pm = host.spawn_procs(per_host={"gpus": 2})
    dm = spawn_tensor_engine(pm)
    test_pass("spawn_tensor_engine() created DeviceMesh")

    with dm.activate():
        r = monarch.inspect(torch.zeros(3, 4))
    test_pass(f"monarch.inspect basic: shape={r.shape}")

    with dm.activate():
        r = monarch.inspect(2 * torch.ones(3, 4))
    expected = 2 * torch.ones(3, 4)
    assert torch.allclose(r, expected), f"expected {expected}, got {r}"
    test_pass("monarch.inspect with computation")

    dm.exit()
    test_pass("dm.exit() clean shutdown")
except Exception as e:
    test_fail("tensor engine", e)
    traceback.print_exc()

# ============================================================
# Summary
# ============================================================
print_section("Done")
print("NPU backend validation complete.")

# Do not leave multiple proc meshes for the one-second atexit fallback to reap.
from monarch._src.actor.actor_mesh import shutdown_context

shutdown_context().get(timeout=75.0)
