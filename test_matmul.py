"""Correctness: every kernel version must match torch.matmul at every M.

Each kernel file (v1_basic.py, ...) exposes `matmul(x, w) -> out`. Versions
that don't exist yet are skipped, so this file works from step 3 onward.

    pytest -v test_matmul.py

Tolerance: both sides accumulate in fp32 and round the result to bf16, but
they add the 4096 terms in different orders, so results can land one bf16
step apart. bf16 keeps 8 significant bits, so one step is up to 2**-7 ~ 0.8%
of the value; rtol allows two steps.

atol is set by the baseline, not by our kernels: at M=128 cuBLAS picks a less
accurate path (measured against a float64 result it is off by up to ~0.007
beyond rtol, where every other M is within 0.00002). 1e-2 clears that and is
still tight enough to catch a single dropped k-term at M >= 16.
"""
import importlib

import pytest
import torch

from roofline import K, M_VALUES, N

KERNELS = ["v1_basic", "v2_grouped", "v3_autotuned"]
RTOL = 2 * 2**-7
ATOL = 1e-2


@pytest.fixture(scope="module")
def w():
    g = torch.Generator(device="cuda").manual_seed(0)
    # 0.02 std is roughly the scale of real Llama weights
    return (torch.randn(K, N, device="cuda", generator=g) * 0.02).to(torch.bfloat16)


@pytest.mark.parametrize("m", M_VALUES)
@pytest.mark.parametrize("kernel", KERNELS)
def test_matches_torch(kernel, m, w):
    try:
        mod = importlib.import_module(kernel)
    except ModuleNotFoundError as e:
        if e.name != kernel:
            raise
        pytest.skip(f"{kernel}.py not written yet")
    g = torch.Generator(device="cuda").manual_seed(m)
    x = torch.randn(m, K, device="cuda", generator=g).to(torch.bfloat16)

    out = mod.matmul(x, w)
    ref = torch.matmul(x, w)

    assert out.shape == (m, N) and out.dtype == torch.bfloat16
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)
