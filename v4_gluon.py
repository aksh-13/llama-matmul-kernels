"""v4: the v3 kernel rewritten in Gluon, with the compiler's choices made by hand.

Gluon is Triton's lower-level sibling. Same Python-like syntax, same idea of
one program per tile, but the compiler no longer decides three things for you:

  1. LAYOUTS    which thread holds which elements of every tensor
  2. SHARED MEMORY   where loaded data waits before the tensor cores use it
  3. PIPELINING      when each load is started and when we wait for it

In v3, Triton made all three choices. dump_ir.py saved them (ir/<gpu>/*.ttgir).
This file writes the same choices out explicitly, so the first goal is "same
speed as v3", not "faster". Once it matches, each choice can be changed.

This version uses the instructions every NVIDIA card since the 30-series has:
  cp.async  async copy from GPU memory into shared memory   (cp.async_load)
  mma.sync  the tensor-core multiply                        (mma_v2)
An H100 version would swap these for TMA and wgmma.

    python v4_gluon.py   # smoke test one fixed config, then tune every M
"""
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp
from triton.experimental.gluon.language.nvidia.ampere import mma_v2

GROUP_M = 8
MAX_BLOCK_K = 128


# ----------------------------------------------------------------------------
# Choice 1: layouts. These run at compile time and return a layout object.
# Each one reproduces the rule Triton used for v3 (compare with the #blocked,
# #mma and #shared lines at the top of ir/<gpu>/v3_M*.ttgir).
# ----------------------------------------------------------------------------
@gluon.constexpr_function
def blocked_layout(cols, num_warps):
    """Layout for a [rows, cols] tensor that is loaded from or stored to GPU memory.

    Each thread owns 8 neighbouring bf16 values in a row (16 bytes, one memory
    request). Threads of a warp spread across the columns first, then down the
    rows. Warps stack down the rows.
    """
    threads_across = min(32, cols // 8)
    return gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[32 // threads_across, threads_across],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )


@gluon.constexpr_function
def mma_layout(block_m, block_n, num_warps):
    """Layout the tensor cores need for the [BLOCK_M, BLOCK_N] accumulator.

    One mma.sync instruction makes a 16 x 8 piece of the output per warp, so
    the warps are arranged in a grid over the tile. Keep doubling whichever
    direction has more pieces left to cover.
    """
    warps_m, warps_n = 1, 1
    while warps_m * warps_n < num_warps:
        if block_m // 16 // warps_m >= block_n // 8 // warps_n:
            warps_m *= 2
        else:
            warps_n *= 2
    return gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[warps_m, warps_n], instr_shape=[16, 8])


@gluon.constexpr_function
def shared_layout(cols):
    """Shared-memory layout for a [rows, cols] bf16 tile.

    "Swizzled" means each row is stored slightly shuffled, so that the threads
    of a warp reading a column don't all hit the same memory bank at once.
    """
    per_phase = max(128 // (cols * 2), 1)
    return gl.SwizzledSharedLayout(vec=8, per_phase=per_phase, max_phase=8 // per_phase, order=[1, 0])


# ----------------------------------------------------------------------------
# Choices 2 and 3: shared memory and pipelining.
# ----------------------------------------------------------------------------
@gluon.jit
def issue_loads(i, x_smem, w_smem, x_ptrs, w_ptrs, x_mask, w_mask, stride_xk, stride_wk,
                BLOCK_K: gl.constexpr, NUM_BUFFERS: gl.constexpr):
    """Start copying K-chunk number i of x and W into shared-memory slot i % NUM_BUFFERS.

    This does not wait. The copy runs in the background.
    """
    k0 = i * BLOCK_K
    cp.async_load(x_smem.index(i % NUM_BUFFERS), x_ptrs + k0 * stride_xk, mask=x_mask)
    cp.async_load(w_smem.index(i % NUM_BUFFERS), w_ptrs + k0 * stride_wk, mask=w_mask)
    cp.commit_group()  # the two copies above form one group we can wait on
    return i + 1


@gluon.jit
def multiply_chunk(i, acc, x_smem, w_smem, a_layout: gl.constexpr, b_layout: gl.constexpr,
                   NUM_BUFFERS: gl.constexpr):
    """acc += x_chunk @ w_chunk for K-chunk i, which must already be in shared memory."""
    a = x_smem.index(i % NUM_BUFFERS).load(a_layout)  # shared memory -> registers
    b = w_smem.index(i % NUM_BUFFERS).load(b_layout)
    return mma_v2(a, b, acc, input_precision="tf32")  # tensor cores, same setting v3 used


@gluon.jit
def matmul_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
    GROUP_M: gl.constexpr, NUM_BUFFERS: gl.constexpr,
):
    # ---- which tile am I? Identical to v2/v3 (grouped ordering). ----
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

    # ---- choice 1: pick every layout ----
    x_layout: gl.constexpr = blocked_layout(BLOCK_K, gl.num_warps())  # x tile   [BLOCK_M, BLOCK_K]
    w_layout: gl.constexpr = blocked_layout(BLOCK_N, gl.num_warps())  # W tile   [BLOCK_K, BLOCK_N]
    #                                                                   out tile [BLOCK_M, BLOCK_N] reuses w_layout
    acc_layout: gl.constexpr = mma_layout(BLOCK_M, BLOCK_N, gl.num_warps())
    # k_width=2: two bf16 values fit in one 32-bit register
    a_layout: gl.constexpr = gl.DotOperandLayout(parent=acc_layout, operand_index=0, k_width=2)
    b_layout: gl.constexpr = gl.DotOperandLayout(parent=acc_layout, operand_index=1, k_width=2)

    # ---- addresses of the first K-chunk of my x tile and my W tile ----
    # In Gluon even an index range needs a layout. SliceLayout(1, L) means
    # "the row index of a tensor laid out as L"; SliceLayout(0, L) the column index.
    x_m = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, x_layout))
    x_k = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, x_layout))
    w_k = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(1, w_layout))
    w_n = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, w_layout))

    x_ptrs = x_ptr + x_m[:, None] * stride_xm + x_k[None, :] * stride_xk
    w_ptrs = w_ptr + w_k[:, None] * stride_wk + w_n[None, :] * stride_wn

    # Rows past M and columns past N must not be read. Whatever sits in their
    # slots only reaches output rows/columns that the final store masks out.
    # K needs no mask: the launcher checks K is a multiple of BLOCK_K.
    x_mask = (x_m < M)[:, None]  # one yes/no per row, stretched across the tile by the load
    w_mask = (w_n < N)[None, :]  # one yes/no per column

    # ---- choice 2: shared memory, NUM_BUFFERS slots for each operand ----
    x_smem = gl.allocate_shared_memory(gl.bfloat16, [NUM_BUFFERS, BLOCK_M, BLOCK_K], layout=shared_layout(BLOCK_K))
    w_smem = gl.allocate_shared_memory(gl.bfloat16, [NUM_BUFFERS, BLOCK_K, BLOCK_N], layout=shared_layout(BLOCK_N))

    # ---- choice 3: the pipeline ----
    # fp32 running total for the whole tile, living in registers.
    acc = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, acc_layout)
    num_chunks = K // BLOCK_K
    load_i = 0  # next K-chunk to start loading
    mma_i = 0   # next K-chunk to multiply

    # Fill: start NUM_BUFFERS - 1 loads before doing any arithmetic.
    for _ in gl.static_range(NUM_BUFFERS - 1):
        load_i = issue_loads(load_i, x_smem, w_smem, x_ptrs, w_ptrs, x_mask, w_mask, stride_xk, stride_wk,
                             BLOCK_K, NUM_BUFFERS)

    # Steady state: start one more load, then multiply the oldest chunk while
    # the newer ones are still arriving.
    for _ in range(num_chunks - (NUM_BUFFERS - 1)):
        load_i = issue_loads(load_i, x_smem, w_smem, x_ptrs, w_ptrs, x_mask, w_mask, stride_xk, stride_wk,
                             BLOCK_K, NUM_BUFFERS)
        cp.wait_group(NUM_BUFFERS - 1)  # block until at most this many loads are still in flight
        acc = multiply_chunk(mma_i, acc, x_smem, w_smem, a_layout, b_layout, NUM_BUFFERS)
        mma_i += 1

    # Drain: nothing left to load, finish the chunks already on their way.
    for i in gl.static_range(NUM_BUFFERS - 1):
        cp.wait_group(NUM_BUFFERS - 2 - i)
        acc = multiply_chunk(mma_i, acc, x_smem, w_smem, a_layout, b_layout, NUM_BUFFERS)
        mma_i += 1

    # ---- round to bf16, move from the tensor-core layout to a store-friendly one, write ----
    out = gl.convert_layout(acc.to(gl.bfloat16), w_layout)
    o_m = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, w_layout))
    out_ptrs = out_ptr + o_m[:, None] * stride_om + w_n[None, :] * stride_on
    gl.store(out_ptrs, out, mask=(o_m < M)[:, None] & (w_n < N)[None, :])


# ----------------------------------------------------------------------------
# Autotuning: same tile shapes as v3. NUM_BUFFERS replaces num_stages, because
# pipelining is now our code, not a compiler option.
# ----------------------------------------------------------------------------
_SHAPES = [
    (16, 64, 64), (16, 128, 64), (16, 256, 64),
    (16, 64, 128), (16, 128, 128), (16, 256, 128), (16, 256, 32),
    (32, 64, 64), (32, 128, 64), (32, 256, 64), (32, 128, 32),
    (64, 64, 32), (64, 128, 32), (64, 128, 64), (64, 256, 32), (64, 256, 64),
    (128, 64, 32), (128, 128, 32), (128, 128, 64), (128, 256, 32), (128, 256, 64),
    (256, 64, 32), (256, 128, 32),
]
# (num_warps, NUM_BUFFERS). Configs whose buffers don't fit in shared memory
# fail to launch and the tuner skips them.
_SCHEDULES = [(4, 2), (4, 3), (4, 4), (8, 2), (8, 3), (8, 4)]

CONFIGS = [
    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "NUM_BUFFERS": buffers}, num_warps=warps)
    for bm, bn, bk in _SHAPES
    for warps, buffers in _SCHEDULES
]


def _prune(configs, named_args, **kwargs):
    """Same rule as v3: skip tiles much taller than the matrix."""
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
    assert x.shape[1] % MAX_BLOCK_K == 0, "the kernel has no K mask: K must be a multiple of every BLOCK_K"
    return x.shape[0], w.shape[1], x.shape[1]


def launch(x: torch.Tensor, w: torch.Tensor):
    """Run the tuned kernel; return (out, compiled kernel)."""
    M, N, K = _check(x, w)
    out = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    compiled = matmul_kernel_tuned[grid](
        x, w, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        GROUP_M=GROUP_M,
    )
    return out, compiled


def matmul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return launch(x, w)[0]


def matmul_fixed(x, w, BLOCK_M=64, BLOCK_N=128, BLOCK_K=32, NUM_BUFFERS=2, num_warps=4):
    """One hand-picked config, no tuner. Use this first: errors are much easier to read."""
    M, N, K = _check(x, w)
    out = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    matmul_kernel[grid](
        x, w, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
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

    # 1. Smoke test: one fixed config per case, compared with torch.
    print("smoke test (fixed configs)")
    for m, kwargs in [(512, {}), (512, {"NUM_BUFFERS": 3, "num_warps": 8}),
                      (16, {"BLOCK_M": 16, "BLOCK_K": 64}), (1, {"BLOCK_M": 16, "BLOCK_K": 64})]:
        x = torch.randn(m, K, device="cuda").to(torch.bfloat16)
        out, ref = matmul_fixed(x, w, **kwargs), torch.matmul(x, w)
        err = (out.float() - ref.float()).abs().max().item()
        torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)
        print(f"  M={m:<5} {kwargs or 'defaults'}: ok, max abs diff {err:.4f}")

    # 2. Tune every M and save the winners, as v3 does.
    rows = []
    for m in M_VALUES:
        x = torch.randn(m, K, device="cuda").to(torch.bfloat16)
        matmul(x, w)  # first call for this M runs the tuner
        best = matmul_kernel_tuned.best_config
        tried = len(_prune(CONFIGS, {"M": m}))
        rows.append({"gpu": name, "M": m, **best.kwargs, "num_warps": best.num_warps, "configs_tried": tried})
        print(rows[-1])

    os.makedirs("results", exist_ok=True)
    path = f"results/v4_configs_{slug(name)}.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {path}")
