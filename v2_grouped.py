"""v2: v1 plus grouped tile ordering, so neighbouring programs share data in L2.

Programs with nearby ids run at about the same time, and whatever they load
sits in the GPU's L2 cache for a while. So the ORDER in which program ids map
to tiles decides how often a load is an L2 hit instead of a trip to memory.

Every tile needs BLOCK_M rows of x (all of K) and BLOCK_N columns of W (all
of K). Tiles in the same row share their x data; tiles in the same column
share their W data. v2 uses a 1D grid and maps the id to a tile itself:
walk GROUP_M rows at a time, and inside that band go column by column.

    GROUP_M = 2, 4 x 3 tiles, numbers are program ids:
        0  2  4
        1  3  5
        6  8 10
        7  9 11

Consecutive programs then share a W column block (reused GROUP_M times in a
row), and the whole band keeps reusing the same GROUP_M row blocks of x.
Block sizes are the same as v1, so any change in speed is from ordering alone.
"""
import torch
import triton
import triton.language as tl

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32
GROUP_M = 8


@triton.jit
def matmul_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # --- new in v2: turn a 1D program id into (pid_m, pid_n), band by band ---
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n            # tiles in one full band
    group_id = pid // num_pid_in_group                # which band
    first_pid_m = group_id * GROUP_M                  # first tile row of the band
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)  # last band may be shorter
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + pid_in_group % group_size_m  # go down the band first...
    pid_n = pid_in_group // group_size_m               # ...then move one column right

    # --- everything below is identical to v1 ---
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    m_mask = offs_m[:, None] < M
    n_mask = offs_n[None, :] < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_left = K - k * BLOCK_K
        x = tl.load(x_ptrs, mask=m_mask & (offs_k[None, :] < k_left), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_k[:, None] < k_left) & n_mask, other=0.0)
        acc = tl.dot(x, w, acc)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=m_mask & n_mask)


def matmul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    assert x.dtype == w.dtype == torch.bfloat16 and x.shape[1] == w.shape[0]
    M, K = x.shape
    N = w.shape[1]
    out = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)  # 1D: one id per tile
    matmul_kernel[grid](
        x, w, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
    )
    return out
