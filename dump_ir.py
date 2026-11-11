"""Save the compiler's intermediate forms of the v3 kernel, for every M.

Triton lowers the Python kernel in stages; the compiled kernel keeps each one:
  ttir   Triton IR: the kernel as written, on whole tensors, hardware-agnostic
  ttgir  Triton GPU IR: same ops, now with layouts, shared memory, pipelining
  ptx    NVIDIA assembly text: what the driver turns into machine code

    python dump_ir.py   # writes ir/<gpu>/v3_M<m>.{ttir,ttgir,ptx}

Each M has its own tuned config, so each M gets its own set of files.
"""
import os

import torch

import v3_autotuned
from gpu_specs import detect, slug
from roofline import K, M_VALUES, N

STAGES = ["ttir", "ttgir", "ptx"]

if __name__ == "__main__":
    name, _ = detect()
    out_dir = os.path.join("ir", slug(name))
    os.makedirs(out_dir, exist_ok=True)
    w = (torch.randn(K, N, device="cuda") * 0.02).to(torch.bfloat16)
    for m in M_VALUES:
        x = torch.randn(m, K, device="cuda").to(torch.bfloat16)
        _, compiled = v3_autotuned.launch(x, w)
        sizes = []
        for stage in STAGES:
            text = compiled.asm[stage]
            with open(os.path.join(out_dir, f"v3_M{m}.{stage}"), "w") as fh:
                fh.write(text)
            sizes.append(f"{stage} {len(text.splitlines())} lines")
        print(f"M={m}: {v3_autotuned.matmul_kernel.best_config} -> {', '.join(sizes)}")
    print(f"saved to {out_dir}/")
