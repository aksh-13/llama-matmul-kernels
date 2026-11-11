"""GPU spec table used by the roofline model, plus GPU detection.

Run directly to print the detected GPU and the spec row in use:
    python gpu_specs.py
"""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuSpec:
    key: str
    mem_bw_gbs: float   # peak memory bandwidth, GB/s (1 GB = 1e9 bytes)
    bf16_tflops: float  # peak dense bf16 tensor-core throughput, TFLOP/s
    source: str         # where each number comes from


SPECS = {
    "rtx3090": GpuSpec(
        "rtx3090", 936.0, 142.0,
        "bandwidth: 384-bit GDDR6X @ 19.5 Gbps/pin = 936 GB/s. "
        "bf16: ESTIMATE, NVIDIA publishes no tensor-core figure for this card; "
        "4 x the 35.6 TFLOP/s fp32 spec (10496 cores x 2 x 1.695 GHz).",
    ),
    "rtx4090": GpuSpec(
        "rtx4090", 1008.0, 330.0,
        "bandwidth: 384-bit GDDR6X @ 21 Gbps/pin = 1008 GB/s. "
        "bf16: ESTIMATE, NVIDIA publishes no tensor-core figure for this card; "
        "4 x the 82.6 TFLOP/s fp32 spec (16384 cores x 2 x 2.52 GHz).",
    ),
    "h100_sxm": GpuSpec(
        "h100_sxm", 3350.0, 989.5,
        "H100 datasheet: 3.35 TB/s; bf16 listed as 1,979 TFLOP/s *with sparsity*, "
        "dense is half = 989.5.",
    ),
    "h100_pcie": GpuSpec(
        "h100_pcie", 2000.0, 756.5,
        "H100 datasheet: 2 TB/s; bf16 listed as 1,513 TFLOP/s *with sparsity*, "
        "dense is half = 756.5.",
    ),
}

# Substrings of torch.cuda.get_device_name(), checked in order.
_NAME_PATTERNS = [
    ("RTX 3090 Ti", None),      # different card (1008 GB/s, faster clock): not in table
    ("RTX 3090", "rtx3090"),
    ("RTX 4090 D", None),       # export variant with fewer cores: not in table
    ("RTX 4090", "rtx4090"),
    ("H100 PCIe", "h100_pcie"),
    ("H100 80GB HBM3", "h100_sxm"),
]


def spec_for_name(name: str) -> GpuSpec:
    """Map a device name to its spec row. GPU_SPEC env var overrides the match."""
    override = os.environ.get("GPU_SPEC")
    if override:
        return SPECS[override]
    for pattern, key in _NAME_PATTERNS:
        if pattern in name:
            if key is None:
                break
            return SPECS[key]
    raise KeyError(
        f"No spec row for GPU {name!r}. Add one to SPECS in gpu_specs.py, "
        f"or set GPU_SPEC to one of {sorted(SPECS)}."
    )


def detect() -> tuple[str, GpuSpec]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU visible to torch.")
    name = torch.cuda.get_device_name()
    return name, spec_for_name(name)


if __name__ == "__main__":
    import torch
    import triton

    name, spec = detect()
    print(f"GPU:            {name}")
    print(f"spec row:       {spec.key}")
    print(f"mem bandwidth:  {spec.mem_bw_gbs:.0f} GB/s")
    print(f"peak bf16:      {spec.bf16_tflops:.1f} TFLOP/s (dense)")
    print(f"source:         {spec.source}")
    print(f"torch {torch.__version__} (CUDA {torch.version.cuda}), triton {triton.__version__}")
