# Mixed-IQ native down rows (experimental)

The native grouped CUDA/HIP format list includes IQ3_S and IQ2_S down projections.
This adds format registrations, not new device arithmetic: the existing `Fmt` and
`Split` implementations supply the dot products. It does not add SYCL or Metal
support, auxiliary-weight loading, or a new prefill algorithm.

The CPU dispatcher can reuse the existing reference-order `iq256x_rows` API for
IQ2_S gate/up with IQ2_S or IQ3_S down. Set `STRATA_IQ_DOWN_EXACT=1` **before process
startup** to enable it. Unset, `0`, or any other value keeps the existing CPU path.
`STRATA_NO_IQ256` disables it and `STRATA_IQ_MT_MIN` still sets the lower group-size
threshold. No speedup is claimed without a same-machine before/after measurement.

The CPU path is deliberately narrow: 2048 hidden values, 512 intermediate values,
2–8 tokens, Q8_K activations, the original contiguous blob geometry, no scale tail
and no Hadamard rotations. It requires both runtime AVX2 without AVX-512 and a
GGML reference compiled for AVX2 without AVX-512. A runtime ISA cap cannot replace
the compiled-reference check. Other layouts keep the existing fallback. Upstream's
legacy multi-token rounding is intentional; this is not an arithmetic bugfix.

## CPU regression checks

On Linux with GCC/G++, an existing llama.cpp checkout, and AVX2/FMA/F16C:

```sh
python3 tools/test_iq_down_dispatch.py --llama-dir /path/to/llama.cpp --output /fresh/dispatch
python3 tools/test_iq_down_exact.py --llama-dir /path/to/llama.cpp --output /fresh/exact
python3 tools/test_iq_down_registration.py --output /fresh/registration
```

Use a fresh output directory for each invocation. The runners do not download
dependencies or access a GPU. They write compiler commands and results to
`receipt.json`. `CC` and `CXX` can select compatible GCC drivers.

- Dispatch: the real baseline-ISA `native_expert.cpp` with instrumented row
  backends. Checks selection, argument forwarding, switches, compiled/runtime ISA
  mismatch, geometry, rotation and existing fallbacks. This is not arithmetic.
- Exact: separately compiles the real IQ and native translation units and GGML's
  AVX2 dots. Compares public `iq256x_rows` and `native_down_rows` bitwise against
  the reference for both formats, scalar/gather decode, widths 1–8, split and empty
  row ranges, padding and unchanged inputs/inactive outputs. CPU/reference probes
  and unrelated backends are stand-ins; this is not full-engine integration.
- Registration: compiles the real format-list block and `d_qk` switch with geometry
  stand-ins, with Q6_K enabled and disabled. This is not device compilation.

The initial CPU run used GCC/G++ 15.2.0 and llama.cpp
`3cf03257f219afbe7334045ff7c6a06ac68c627d`: 10,368 public exact-row comparisons and
1,179,648 native comparisons each with the switch off and on, all bitwise equal,
with no input/output-preservation failures. The routing check passed 64 cases in
each of six fresh-process modes; both 16-case registration checks passed.
These are synthetic CPU results, not model accuracy or performance measurements.

## Required device and model qualification

This port has **not** been CUDA/HIP compiled or GPU/model qualified. Keep it in
draft until those gates have real evidence. The device parity fixture now includes
both down formats across its existing gate/up matrix and the 2048/512 layout.
After configuring each backend as described in the build documentation, build
`native_grouped_parity` and run its checks without `--bench`. It compares grouped
launch variants, not an independent floating-point oracle. Also compare each new
device down format against a dequantized/reference dot at the target layout;
a passing grouped-launch comparison alone cannot qualify new format arithmetic.

Before claiming model support, run a controlled mixed-format model path at the
same source revision, with settings and weight provenance recorded. Compare the
CPU switch off/on with fixed routing and no unrelated experimental switches.
Check decode, multi-token prefill and rotated-layout fallback separately. CUDA and
HIP builds and device parity are separate gates; no SYCL/Metal compatibility or
end-to-end token identity is implied by these CPU tests.

## Origin

Developed as part of [MoEmentum low-VRAM inference experiments](https://github.com/CrashxZ/MoEmentum).
The original down-row work is in
[`bc7a84d1`](https://github.com/CrashxZ/MoEmentum/commit/bc7a84d14a663d8fe63a56fd2695038f8018b36c);
the later compiled-reference/rotation/geometry guards are in
[`67bb9c51`](https://github.com/CrashxZ/MoEmentum/commit/67bb9c5157b53b8b1b9ff8075bcf939223ae9f1e).
This is a focused port of that dispatch/support work to the current upstream API,
not a claim to have introduced upstream's existing exact arithmetic. Historical
benchmark and full-engine qualification claims are not transferred to this port.
