"""Benchmark torch.matmul (cuBLAS) and every kernel version at every M.

    python bench.py   # prints the table, saves results/bench_<gpu>.csv

Timing uses triton.testing.do_bench, which warms up, flushes the GPU's L2
cache before each run and times with CUDA events. We report the median.

Columns:
  TFLOP/s    FLOPs / time                     (compare to the GPU's peak)
  GB/s       minimum bytes moved / time       (compare to peak bandwidth)
  % roof     roofline best time / time        (100% = as fast as physics allows)
  vs torch   torch time / time                (>1 means faster than cuBLAS)
"""
import csv
import importlib
import os

import torch
import triton.testing

from gpu_specs import detect, slug
from roofline import K, M_VALUES, N, predict
from test_matmul import KERNELS


def load_impls() -> dict:
    impls = {"torch": torch.matmul}
    for kernel in KERNELS:
        try:
            impls[kernel] = importlib.import_module(kernel).matmul
        except ModuleNotFoundError as e:
            if e.name != kernel:
                raise
    return impls


def main():
    name, spec = detect()
    impls = load_impls()
    w = (torch.randn(K, N, device="cuda") * 0.02).to(torch.bfloat16)

    rows = []
    for m in M_VALUES:
        x = torch.randn(m, K, device="cuda").to(torch.bfloat16)
        roof = predict(m, spec)
        torch_ms = None
        for impl, fn in impls.items():
            fn(x, w)  # first call compiles (and autotunes) outside the timed region
            ms = triton.testing.do_bench(lambda: fn(x, w), warmup=50, rep=200,
                                         return_mode="median")
            if impl == "torch":
                torch_ms = ms
            rows.append({
                "gpu": name,
                "kernel": impl,
                "M": m,
                "time_ms": ms,
                "tflops": roof["flops"] / (ms * 1e-3) / 1e12,
                "gbs": roof["bytes"] / (ms * 1e-3) / 1e9,
                "roofline_ms": roof["best_ms"],
                "bound": roof["bound"],
                "pct_of_roofline": 100 * roof["best_ms"] / ms,
                "speedup_vs_torch": torch_ms / ms,
            })

    print(f"{name}: peak {spec.bf16_tflops:.1f} TFLOP/s, {spec.mem_bw_gbs:.0f} GB/s\n")
    print(f"{'kernel':<13} {'M':>5} {'time ms':>9} {'TFLOP/s':>8} {'GB/s':>7} "
          f"{'bound':>8} {'% roof':>7} {'vs torch':>9}")
    for r in rows:
        print(f"{r['kernel']:<13} {r['M']:>5} {r['time_ms']:>9.4f} {r['tflops']:>8.2f} "
              f"{r['gbs']:>7.1f} {r['bound']:>8} {r['pct_of_roofline']:>6.1f}% "
              f"{r['speedup_vs_torch']:>8.2f}x")

    os.makedirs("results", exist_ok=True)
    path = f"results/bench_{slug(name)}.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
