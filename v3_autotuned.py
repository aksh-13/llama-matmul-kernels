"""v3: v2 plus @triton.autotune, so each M gets its own block sizes.

v1/v2 used 64x64x32 tiles for every M. That is a poor fit at both ends:
  - M=1 or 16: most of a 64-row tile is padding, and the job is really
    "stream W through the GPU", which wants wide N and long K blocks.
  - M=4096: bigger tiles reuse each loaded value in more products.
The best shape also depends on num_warps (threads per program = 32 x warps)
and num_stages (how many K-loop iterations of loads are kept in flight, so
memory latency overlaps with the tensor-core work).

@triton.autotune compiles the kernel once per config, times each one the
first time it sees a new key (here: M, N, K), and from then on uses the
winner. The kernel body is identical to v2.

    python v3_autotuned.py   # tune every M, print and save the chosen configs
"""
import torch
import triton
import triton.language as tl

GROUP_M = 8

# (BLOCK_M, BLOCK_N, BLOCK_K). tl.dot needs every dimension >= 16, so
# BLOCK_M=16 is the smallest tile available for M=1 and M=16.
_SHAPES = [
    # small M: thin tiles, wide/long blocks to stream W
    (16, 64, 64), (16, 128, 64), (16, 256, 64),
    (16, 64, 128), (16, 128, 128), (16, 256, 128), (16, 256, 32),
    (32, 64, 64), (32, 128, 64), (32, 256, 64), (32, 128, 32),
    # medium and large M
    (64, 64, 32), (64, 128, 32), (64, 128, 64), (64, 256, 32), (64, 256, 64),
    (128, 64, 32), (128, 128, 32), (128, 128, 64), (128, 256, 32), (128, 256, 64),
    (256, 64, 32), (256, 128, 32),
]
# (num_warps, num_stages)
_SCHEDULES = [(4, 2), (4, 3), (4, 4), (8, 2), (8, 3), (8, 4)]

CONFIGS = [
    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
                  num_warps=warps, num_stages=stages)
    for bm, bn, bk in _SHAPES
    for warps, stages in _SCHEDULES
]


def _prune(configs, named_args, **kwargs):
    """Skip tiles much taller than the matrix: they are mostly padding.

    Keeps BLOCK_M up to the smallest listed size that covers M, so M=1 and
    M=16 only try BLOCK_M=16, while M=4096 tries everything.
    """
    m = named_args["M"]
    sizes = sorted({c.kwargs["BLOCK_M"] for c in configs})
    limit = next((s for s in sizes if s >= m), sizes[-1])
    return [c for c in configs if c.kwargs["BLOCK_M"] <= limit]


@triton.autotune(
    configs=CONFIGS,
    key=["M", "N", "K"],                       # retune whenever the shape changes
    prune_configs_by={"early_config_prune": _prune},
    cache_results=True,                        # remember winners on disk between runs
)
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
    # grouped tile ordering, as in v2
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + pid_in_group % group_size_m
    pid_n = pid_in_group // group_size_m

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
    # The grid depends on the block sizes the tuner picks, so it is a function of them.
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    matmul_kernel[grid](
        x, w, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        GROUP_M=GROUP_M,
    )
    return out


if __name__ == "__main__":
    import csv
    import os

    from gpu_specs import detect, slug
    from roofline import K, M_VALUES, N

    name, _ = detect()
    w = (torch.randn(K, N, device="cuda") * 0.02).to(torch.bfloat16)
    rows = []
    for m in M_VALUES:
        x = torch.randn(m, K, device="cuda").to(torch.bfloat16)
        matmul(x, w)  # first call for this M runs the tuner
        best = matmul_kernel.best_config
        tried = len(_prune(CONFIGS, {"M": m}))
        rows.append({"gpu": name, "M": m, **best.kwargs, "num_warps": best.num_warps,
                     "num_stages": best.num_stages, "configs_tried": tried})
        print(rows[-1])

    os.makedirs("results", exist_ok=True)
    path = f"results/v3_configs_{slug(name)}.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {path}")
