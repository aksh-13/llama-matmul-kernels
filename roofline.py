"""Roofline predictions for out = x @ W_gate, x:[M,4096], W:[4096,14336], bf16.

The roofline says a kernel can't beat either of two limits:
  compute: time >= FLOPs / peak FLOP/s
  memory:  time >= bytes moved / peak memory bandwidth
The best possible time is the larger of the two, and whichever is larger
names the bound. Arithmetic intensity (FLOPs per byte) decides it: below the
GPU's ridge point (peak FLOP/s / peak bytes/s) the kernel is memory-bound.

    python roofline.py   # prints the table, saves results/roofline_<gpu>.csv
"""
import csv
import os

from gpu_specs import GpuSpec, detect, slug

K = 4096          # Llama-3.1-8B hidden size
N = 14336         # MLP intermediate size
M_VALUES = [1, 16, 128, 512, 4096]
BYTES_PER_ELEM = 2  # bf16


def flops(m: int) -> int:
    # one multiply + one add per (row, k, column)
    return 2 * m * K * N


def bytes_moved(m: int) -> int:
    # Minimum traffic: read x once, read W once, write out once.
    return BYTES_PER_ELEM * (m * K + K * N + m * N)


def predict(m: int, spec: GpuSpec) -> dict:
    f, b = flops(m), bytes_moved(m)
    t_compute = f / (spec.bf16_tflops * 1e12)
    t_memory = b / (spec.mem_bw_gbs * 1e9)
    return {
        "M": m,
        "flops": f,
        "bytes": b,
        "intensity_flop_per_byte": f / b,
        "t_compute_ms": t_compute * 1e3,
        "t_memory_ms": t_memory * 1e3,
        "bound": "compute" if t_compute > t_memory else "memory",
        "best_ms": max(t_compute, t_memory) * 1e3,
    }


if __name__ == "__main__":
    name, spec = detect()
    ridge = spec.bf16_tflops * 1e12 / (spec.mem_bw_gbs * 1e9)
    print(f"{name}: {spec.bf16_tflops:.1f} TFLOP/s, {spec.mem_bw_gbs:.0f} GB/s, "
          f"ridge point {ridge:.0f} FLOP/byte\n")
    rows = [predict(m, spec) for m in M_VALUES]
    print(f"{'M':>5} {'GFLOP':>9} {'MB moved':>9} {'FLOP/byte':>10} "
          f"{'t_comp ms':>10} {'t_mem ms':>9} {'bound':>8} {'best ms':>8}")
    for r in rows:
        print(f"{r['M']:>5} {r['flops'] / 1e9:>9.2f} {r['bytes'] / 1e6:>9.1f} "
              f"{r['intensity_flop_per_byte']:>10.1f} {r['t_compute_ms']:>10.4f} "
              f"{r['t_memory_ms']:>9.4f} {r['bound']:>8} {r['best_ms']:>8.4f}")

    os.makedirs("results", exist_ok=True)
    path = f"results/roofline_{slug(name)}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved {path}")
