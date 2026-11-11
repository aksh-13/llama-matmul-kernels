# Results: Triton matmul for the Llama-3.1-8B gate projection

`out = x @ W_gate`, with `x` of shape `[M, 4096]` and `W_gate` of shape `[4096, 14336]`, bf16 inputs and output, fp32 accumulation. The baseline is `torch.matmul`, which calls cuBLAS.

All numbers below are from one GPU: an NVIDIA GeForce RTX 4090 (a 48 GB variant, 128 SMs) with torch 2.14.1+cu132 and triton 3.8.0. Raw data is in `results/`.

## Summary

- v3 (autotuned) is faster than torch at M = 128, 512 and 4096, level at M = 16, and 12% slower at M = 1.
- v1 (fixed 64x64x32 tiles) is within 4% of torch for M = 16 to 512 and about 15% slower at M = 1 and M = 4096.
- v2 (grouped ordering) is not faster than v1 on this GPU. v1's 2D grid already ran tiles in a cache-friendly order.
- Memory-bound sizes (M <= 128) all sit near 67-69% of the roofline. Compute-bound sizes reach at most 53%, and that figure depends on an estimated peak.

## Roofline spec values used

| GPU | Peak memory bandwidth | Peak dense bf16 | Source |
|---|---|---|---|
| RTX 4090 | 1008 GB/s | 330 TFLOP/s | 384-bit GDDR6X at 21 Gbps. The bf16 figure is an estimate (4 x the 82.6 TFLOP/s fp32 spec); NVIDIA publishes no tensor-core number for this card. |

Both torch and v3 top out near 170-176 TFLOP/s, about half of 330. The estimate may be too high for this card, so read "% of roofline" at M = 512 and 4096 with that in mind.

## Roofline predictions

| M | GFLOP | MB moved | FLOP/byte | Bound | Best time (ms) |
|---|---|---|---|---|---|
| 1 | 0.12 | 117.5 | 1.0 | memory | 0.1165 |
| 16 | 1.88 | 118.0 | 15.9 | memory | 0.1171 |
| 128 | 15.03 | 122.2 | 123.1 | memory | 0.1212 |
| 512 | 60.13 | 136.3 | 441.1 | compute | 0.1822 |
| 4096 | 481.04 | 268.4 | 1792.0 | compute | 1.4577 |

The ridge point is 327 FLOP/byte. W alone is 117.4 MB, so for M <= 128 the job is mostly reading W once.

## Benchmark: every version against torch

Median time from `triton.testing.do_bench`, which flushes the GPU's L2 cache before each run. "% of roofline" is the roofline best time divided by the measured time. "vs torch" is torch time divided by kernel time, so above 1 means faster than cuBLAS.

| M | Bound | Kernel | Time (ms) | TFLOP/s | GB/s | % of roofline | vs torch |
|---|---|---|---|---|---|---|---|
| 1 | memory | torch | 0.1505 | 0.78 | 780.4 | 77.4% | 1.00x |
| 1 | memory | v1_basic | 0.1731 | 0.68 | 678.8 | 67.3% | 0.87x |
| 1 | memory | v2_grouped | 0.1731 | 0.68 | 678.8 | 67.3% | 0.87x |
| 1 | memory | v3_autotuned | 0.1720 | 0.68 | 682.9 | 67.7% | 0.88x |
| 16 | memory | torch | 0.1741 | 10.79 | 678.0 | 67.3% | 1.00x |
| 16 | memory | v1_basic | 0.1731 | 10.86 | 682.0 | 67.7% | 1.01x |
| 16 | memory | v2_grouped | 0.1741 | 10.79 | 678.0 | 67.3% | 1.00x |
| 16 | memory | v3_autotuned | 0.1720 | 10.92 | 686.1 | 68.1% | 1.01x |
| 128 | memory | torch | 0.1842 | 81.60 | 663.1 | 65.8% | 1.00x |
| 128 | memory | v1_basic | 0.1782 | 84.37 | 685.6 | 68.0% | 1.03x |
| 128 | memory | v2_grouped | 0.1802 | 83.41 | 677.8 | 67.2% | 1.02x |
| 128 | memory | v3_autotuned | 0.1751 | 85.85 | 697.6 | 69.2% | 1.05x |
| 512 | compute | torch | 0.4207 | 142.92 | 324.0 | 43.3% | 1.00x |
| 512 | compute | v1_basic | 0.4362 | 137.85 | 312.5 | 41.8% | 0.96x |
| 512 | compute | v2_grouped | 0.4434 | 135.61 | 307.4 | 41.1% | 0.95x |
| 512 | compute | v3_autotuned | 0.3645 | 164.94 | 373.9 | 50.0% | 1.15x |
| 4096 | compute | torch | 2.8344 | 169.71 | 94.7 | 51.4% | 1.00x |
| 4096 | compute | v1_basic | 3.3331 | 144.32 | 80.5 | 43.7% | 0.85x |
| 4096 | compute | v2_grouped | 3.4796 | 138.25 | 77.1 | 41.9% | 0.81x |
| 4096 | compute | v3_autotuned | 2.7399 | 175.57 | 98.0 | 53.2% | 1.03x |

Run-to-run variation is about 1-3%, so differences of that size are noise.

## What changed in each version

### v1: basic tiled matmul (`v1_basic.py`)

The output is cut into 64 x 64 tiles and one program computes each tile, on a 2D grid. Each program loops over K in blocks of 32, loading a piece of `x` and a piece of `W`, multiplying them with `tl.dot` on the tensor cores and adding into an fp32 accumulator, then rounds to bf16 and stores. With untuned block sizes it already matches cuBLAS at M = 16 and 128, because those sizes are memory-bound and the work is reading W, which v1 does at about 685 GB/s (68% of the roofline). It is 13% slower at M = 1, where a 64-row tile is almost all padding, and 15% slower at M = 4096, where it reaches 144 TFLOP/s (44% of the roofline).

### v2: grouped tile ordering (`v2_grouped.py`)

v2 switches to a 1D grid and maps each program id to a tile itself, walking `GROUP_M = 8` tile rows at a time and going column by column inside that band, so programs that run close together share a block of W in the L2 cache. Block sizes are the same as v1. It is not faster: level with v1 up to M = 512 and about 4-6% slower at M = 4096. A sweep over `GROUP_M` showed why. Against plain row-by-row order (`GROUP_M = 1`) grouping is 13-15% faster, so the technique works, but v1's 2D grid already performs like the large-group end of that sweep. That fits the 2D grid being launched with the tile-row index changing fastest, which is an inference from timings; CUDA does not promise a launch order. For M <= 64 there is only one tile row, so ordering cannot matter.

### v3: autotuned (`v3_autotuned.py`)

v3 keeps v2's kernel body and adds `@triton.autotune`, keyed on (M, N, K), over 138 configs of `BLOCK_M`, `BLOCK_N`, `BLOCK_K`, `num_warps` and `num_stages`. Tiles much taller than M are pruned, so M = 1 and 16 try only the 42 configs with `BLOCK_M = 16`, the smallest `tl.dot` allows. The tuner picked a different config for every M:

| M | BLOCK_M | BLOCK_N | BLOCK_K | num_warps | num_stages |
|---|---|---|---|---|---|
| 1 | 16 | 64 | 64 | 4 | 4 |
| 16 | 16 | 128 | 64 | 4 | 4 |
| 128 | 64 | 256 | 32 | 8 | 3 |
| 512 | 64 | 128 | 32 | 4 | 3 |
| 4096 | 256 | 128 | 32 | 8 | 3 |

The compute-bound sizes gained the most: v3 is 22% faster than v2 at M = 512 and 27% faster at M = 4096, reaching 165-176 TFLOP/s and passing cuBLAS (1.15x and 1.03x). Bigger tiles do more arithmetic per byte loaded. The memory-bound sizes barely moved and stay at 68-69% of the roofline. M = 1 is the one size still behind cuBLAS (683 against 780 GB/s); even a 16-row tile is 15 rows of padding for one real row.

## Compiler view of v3 (`ir/`)

`dump_ir.py` saves three forms of the compiled v3 kernel for each M under `ir/<gpu>/`. For M = 4096 they are 211, 303 and 1713 lines.

- **TTIR (`.ttir`)** is the kernel as written, on whole tensors and with no hardware detail. The K loop holds two `tt.load`, one `tt.dot` (`tensor<256x32xbf16> * tensor<32x128xbf16> -> tensor<256x128xf32>`) and the pointer updates; one `tt.store` follows the loop.
- **TTIR to TTGIR (`.ttgir`)**: the compiler decides where every tensor lives and which thread holds which elements. Each tensor gains a layout (`#blocked` for plain per-thread data, `#nvidia_mma` for tensor-core operands and results). The loads become asynchronous copies from global memory into shared memory (`ttg.async_copy_global_to_local`, `ttg.local_alloc`), and the loop is pipelined: `num_stages = 3` shows up as copies for later iterations being issued before the current `tt.dot` runs. The module header records the target (`cuda:89`) and `num-warps = 8`.
- **TTGIR to PTX (`.ptx`)**: tensors disappear and everything becomes per-thread scalar instructions for one GPU architecture (`.target sm_89`). The dot is 64 `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` instructions, which is the tensor-core op with bf16 inputs and fp32 accumulation. The async copies are `cp.async.cg.shared.global`, operands are read from shared memory with `ldmatrix`, and the final rounding is `cvt.rn.bf16x2.f32`.

## Reproducing on another GPU

```
pip install -r requirements.txt
python gpu_specs.py        # GPU name and spec row in use
python roofline.py         # results/roofline_<gpu>.csv
pytest -v test_matmul.py
python v3_autotuned.py     # tune, results/v3_configs_<gpu>.csv
python bench.py            # results/bench_<gpu>.csv
python dump_ir.py          # ir/<gpu>/
```

Every output file is named after the GPU, so runs on different GPUs sit side by side.
