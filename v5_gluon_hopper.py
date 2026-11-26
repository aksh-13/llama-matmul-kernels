"""v5: the Gluon kernel using the two H100-only features, TMA and wgmma.

v4 used the instructions every recent NVIDIA card has (cp.async, mma.sync).
The H100 ("Hopper") adds two better ones, and Triton already uses the second
one for v3 on this card. v5 switches the Gluon kernel over to both:

  TMA    "tensor memory accelerator". One thread says "copy the tile at
         (row, col) into shared memory" and hardware does the rest. With
         cp.async every thread computed its own addresses and masks.
         Rows/columns outside the matrix come back as zeros automatically,
         so the kernel needs no masks at all.

  wgmma  "warp-group MMA". The newer tensor-core instruction. Four warps act
         as one group and the instruction reads its inputs straight from
         shared memory, so the operands never pass through registers.

What changes against v4:
  - pointers and masks      ->  tensor descriptors (tma.make_tensor_descriptor)
  - cp.async + wait_group   ->  tma.async_load + one mbarrier per buffer
  - load to registers, mma  ->  warpgroup_mma reading shared memory directly
  - convert layout + store  ->  store to shared memory, TMA copies it out

One new limit: a warp group covers 4 x 16 = 64 rows, so BLOCK_M is at least 64.
That hurts small M (M=1 becomes 63 rows of padding), where v4 stays the better kernel.

Only runs on Hopper (H100/H200).

    python v5_gluon_hopper.py   # smoke test fixed configs, then tune every M
"""
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    mbarrier,
    tma,
    warpgroup_mma,
)

GROUP_M = 8
MAX_BLOCK_K = 128


def _scratch_alloc(size: int, alignment: int, stream):
    """Descriptors built inside the kernel live in a little GPU scratch buffer.

    Triton asks for that buffer at every launch and requires us to say where
    it comes from.
    """
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(_scratch_alloc)


# ----------------------------------------------------------------------------
# The wgmma layout. These two rules are from the Gluon wgmma tutorial.
# ----------------------------------------------------------------------------
@gluon.constexpr_function
def wgmma_layout(block_m, block_n, num_warps):
    # The smallest unit is one warp group: 4 warps stacked along M.
    # Add warps along M while the tile is tall enough, otherwise along N.
    warps_per_cta = [4, 1]
    while warps_per_cta[0] * warps_per_cta[1] != num_warps:
        if block_m > 16 * warps_per_cta[0]:
            warps_per_cta[0] *= 2
        else:
            warps_per_cta[1] *= 2
    # One instruction makes a 16 x n piece. Use the widest n that fits.
    m_reps = triton.cdiv(block_m, 16)
    n_reps = triton.cdiv(num_warps, m_reps)
    max_n = max(block_n // n_reps, 8)
    n = 256
    while n > max_n or block_n % n != 0:
        n -= 8
    # k = 16: one instruction consumes 16 bf16 values along K (256 bits)
    return gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=warps_per_cta, instr_shape=[16, n, 16])


@gluon.jit
def issue_loads(i, x_desc, w_desc, x_smem, w_smem, bars, off_m, off_n,
                BLOCK_K: gl.constexpr, NUM_BUFFERS: gl.constexpr):
    """Start TMA copies of K-chunk i into buffer slot i % NUM_BUFFERS. Does not wait."""
    k0 = i * BLOCK_K
    bar = bars.index(i % NUM_BUFFERS)
    # An mbarrier is a counter in shared memory. We tell it how many bytes to
    # expect; TMA counts them down as they arrive; a waiter wakes at zero.
    mbarrier.expect(bar, x_desc.block_type.nbytes + w_desc.block_type.nbytes)
    tma.async_load(x_desc, [off_m, k0], bar, x_smem.index(i % NUM_BUFFERS))
    tma.async_load(w_desc, [k0, off_n], bar, w_smem.index(i % NUM_BUFFERS))
    return i + 1


@gluon.jit
def multiply_chunk(i, acc, x_smem, w_smem, bars, NUM_BUFFERS: gl.constexpr):
    """Wait for K-chunk i to land, then acc += x_chunk @ w_chunk with wgmma."""
    # Each use of a barrier flips its phase bit, so slot reuse number
    # (i // NUM_BUFFERS) waits on phase 0, 1, 0, 1, ...
    mbarrier.wait(bars.index(i % NUM_BUFFERS), (i // NUM_BUFFERS) & 1)
    return warpgroup_mma(x_smem.index(i % NUM_BUFFERS), w_smem.index(i % NUM_BUFFERS), acc)


@gluon.jit
def matmul_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_wk, stride_om,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
    GROUP_M: gl.constexpr, NUM_BUFFERS: gl.constexpr,
):
    # ---- which tile am I? Identical to v2/v3/v4. ----
    pid = gl.program_id(0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + pid_in_group % group_size_m
    pid_n = pid_in_group // group_size_m
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    # ---- shared-memory layouts: TMA and wgmma both need this "NVMMA" kind ----
    x_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.bfloat16)
    w_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], gl.bfloat16)
    o_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.bfloat16)

    # ---- descriptors: "this pointer is a [rows, cols] matrix, hand me tiles of this size" ----
    x_desc = tma.make_tensor_descriptor(x_ptr, [M, K], [stride_xm, 1], [BLOCK_M, BLOCK_K], x_layout)
    w_desc = tma.make_tensor_descriptor(w_ptr, [K, N], [stride_wk, 1], [BLOCK_K, BLOCK_N], w_layout)
    o_desc = tma.make_tensor_descriptor(out_ptr, [M, N], [stride_om, 1], [BLOCK_M, BLOCK_N], o_layout)

    # ---- buffers, and one barrier per buffer slot ----
    x_smem = gl.allocate_shared_memory(gl.bfloat16, [NUM_BUFFERS, BLOCK_M, BLOCK_K], x_layout)
    w_smem = gl.allocate_shared_memory(gl.bfloat16, [NUM_BUFFERS, BLOCK_K, BLOCK_N], w_layout)
    bars = gl.allocate_shared_memory(gl.int64, [NUM_BUFFERS, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(NUM_BUFFERS):
        mbarrier.init(bars.index(i), count=1)

    # ---- the pipeline: same fill / steady / drain shape as v4 ----
    acc = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, wgmma_layout(BLOCK_M, BLOCK_N, gl.num_warps()))
    num_chunks = K // BLOCK_K
    load_i = 0
    mma_i = 0

    for _ in gl.static_range(NUM_BUFFERS - 1):
        load_i = issue_loads(load_i, x_desc, w_desc, x_smem, w_smem, bars, off_m, off_n, BLOCK_K, NUM_BUFFERS)

    for _ in range(num_chunks - (NUM_BUFFERS - 1)):
        load_i = issue_loads(load_i, x_desc, w_desc, x_smem, w_smem, bars, off_m, off_n, BLOCK_K, NUM_BUFFERS)
        acc = multiply_chunk(mma_i, acc, x_smem, w_smem, bars, NUM_BUFFERS)
        mma_i += 1

    for _ in gl.static_range(NUM_BUFFERS - 1):
        acc = multiply_chunk(mma_i, acc, x_smem, w_smem, bars, NUM_BUFFERS)
        mma_i += 1

    for i in gl.static_range(NUM_BUFFERS):
        mbarrier.invalidate(bars.index(i))

    # ---- round to bf16, park the tile in shared memory, let TMA write it out ----
    # TMA skips rows/columns outside the matrix, so there is no store mask either.
    o_smem = gl.allocate_shared_memory(gl.bfloat16, [BLOCK_M, BLOCK_N], o_layout)
    o_smem.store(acc.to(gl.bfloat16))
    fence_async_shared()  # make the plain store above visible to the async copy below
    tma.async_copy_shared_to_global(o_desc, [off_m, off_n], o_smem)
    tma.store_wait(pendings=0)


# ----------------------------------------------------------------------------
# Autotuning. BLOCK_M starts at 64 (the wgmma minimum).
# ----------------------------------------------------------------------------
_SHAPES = [
    (64, 64, 64), (64, 128, 64), (64, 256, 64), (64, 128, 128), (64, 256, 128), (64, 256, 32),
    (128, 64, 64), (128, 128, 32), (128, 128, 64), (128, 128, 128),
    (128, 256, 32), (128, 256, 64), (128, 256, 128),
    (256, 128, 32), (256, 128, 64),
]
# (num_warps, NUM_BUFFERS). Configs that don't fit in shared memory are skipped by the tuner.
_SCHEDULES = [(4, 2), (4, 3), (4, 4), (8, 2), (8, 3), (8, 4)]

CONFIGS = [
    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "NUM_BUFFERS": buffers}, num_warps=warps)
    for bm, bn, bk in _SHAPES
    for warps, buffers in _SCHEDULES
]


def _prune(configs, named_args, **kwargs):
    """Same rule as v3/v4: skip tiles much taller than the matrix."""
    m = named_args["M"]
    sizes = sorted({c.kwargs["BLOCK_M"] for c in configs})
    limit = next((s for s in sizes if s >= m), sizes[-1])
    return [c for c in configs if c.kwargs["BLOCK_M"] <= limit]


matmul_kernel_tuned = triton.autotune(
    configs=CONFIGS,
    key=["M", "N", "K"],
    prune_configs_by={"early_config_prune": _prune},
    cache_results=True,
)(matmul_kernel)


def _check(x: torch.Tensor, w: torch.Tensor):
    assert x.dtype == w.dtype == torch.bfloat16 and x.shape[1] == w.shape[0]
    assert x.is_contiguous() and w.is_contiguous(), "TMA needs the last dimension to have stride 1"
    assert x.shape[1] % MAX_BLOCK_K == 0, "K must be a multiple of every BLOCK_K"
    return x.shape[0], w.shape[1], x.shape[1]


def launch(x: torch.Tensor, w: torch.Tensor):
    """Run the tuned kernel; return (out, compiled kernel)."""
    M, N, K = _check(x, w)
    out = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    compiled = matmul_kernel_tuned[grid](
        x, w, out, M, N, K, x.stride(0), w.stride(0), out.stride(0), GROUP_M=GROUP_M,
    )
    return out, compiled


def matmul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return launch(x, w)[0]


def matmul_fixed(x, w, BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, NUM_BUFFERS=2, num_warps=8):
    """One hand-picked config, no tuner. Use this first: errors are much easier to read."""
    M, N, K = _check(x, w)
    out = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    matmul_kernel[grid](
        x, w, out, M, N, K, x.stride(0), w.stride(0), out.stride(0),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M, NUM_BUFFERS=NUM_BUFFERS, num_warps=num_warps,
    )
    return out


if __name__ == "__main__":
    import csv
    import os

    from gpu_specs import detect, slug
    from roofline import K, M_VALUES, N
    from test_matmul import ATOL, RTOL

    name, _ = detect()
    w = (torch.randn(K, N, device="cuda") * 0.02).to(torch.bfloat16)

    print("smoke test (fixed configs)")
    for m, kwargs in [(512, {}), (512, {"NUM_BUFFERS": 3, "num_warps": 4, "BLOCK_M": 64}),
                      (4096, {"BLOCK_N": 256}), (16, {"BLOCK_M": 64}), (1, {"BLOCK_M": 64})]:
        x = torch.randn(m, K, device="cuda").to(torch.bfloat16)
        out, ref = matmul_fixed(x, w, **kwargs), torch.matmul(x, w)
        err = (out.float() - ref.float()).abs().max().item()
        torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)
        print(f"  M={m:<5} {kwargs or 'defaults'}: ok, max abs diff {err:.4f}")

    rows = []
    for m in M_VALUES:
        x = torch.randn(m, K, device="cuda").to(torch.bfloat16)
        matmul(x, w)  # first call for this M runs the tuner
        best = matmul_kernel_tuned.best_config
        tried = len(_prune(CONFIGS, {"M": m}))
        rows.append({"gpu": name, "M": m, **best.kwargs, "num_warps": best.num_warps, "configs_tried": tried})
        print(rows[-1])

    os.makedirs("results", exist_ok=True)
    path = f"results/v5_configs_{slug(name)}.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {path}")
