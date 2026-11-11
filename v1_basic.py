"""v1: basic tiled matmul. One program per output tile, fixed block sizes.

The output [M, N] is cut into BLOCK_M x BLOCK_N tiles. The GPU runs one copy
of the kernel (a "program") per tile; program (pid_m, pid_n) owns one tile.
Each program walks along K in steps of BLOCK_K: load a BLOCK_M x BLOCK_K
piece of x and a BLOCK_K x BLOCK_N piece of W, multiply them (tl.dot, which
runs on the tensor cores) and add into an fp32 accumulator. At the end it
rounds the accumulator to bf16 and writes its tile.
"""
import torch
import triton
import triton.language as tl

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32


@triton.jit
def matmul_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,   # element steps between rows / columns of x
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Which tile is this program responsible for?
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row and column indices of the tile, and the first K block.
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to the first [BLOCK_M, BLOCK_K] piece of x and [BLOCK_K, BLOCK_N] piece of W.
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    # Rows/columns past the edge of the matrix are masked: loaded as 0, never stored.
    m_mask = offs_m[:, None] < M
    n_mask = offs_n[None, :] < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_left = K - k * BLOCK_K
        x = tl.load(x_ptrs, mask=m_mask & (offs_k[None, :] < k_left), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_k[:, None] < k_left) & n_mask, other=0.0)
        acc = tl.dot(x, w, acc)          # acc += x @ w, bf16 inputs, fp32 sum
        x_ptrs += BLOCK_K * stride_xk    # step to the next K block
        w_ptrs += BLOCK_K * stride_wk

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=m_mask & n_mask)


def matmul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    assert x.dtype == w.dtype == torch.bfloat16 and x.shape[1] == w.shape[0]
    M, K = x.shape
    N = w.shape[1]
    out = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))  # one program per tile
    matmul_kernel[grid](
        x, w, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out
