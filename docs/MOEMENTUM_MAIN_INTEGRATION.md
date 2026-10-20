# Main-source integration status

This branch integrates exact fork-main commit `e88ee39d1d6be6e3364183af3b39b62a7ed6f211`
into MoEmentum `91bd33549debaa6ca956042922754a1f8398b499`. It is a source candidate,
not a deployed or performance-qualified engine. Earlier benchmark and serving documents
remain historical evidence for their pinned source and binaries.

## Decisions

- Preserve the MoEmentum README, graphics, build guides and web branding. Refresh
  `README.upstream.md` with the integrated revision's original README; preserve attribution,
  licenses and upstream documentation. Inherit upstream's web sampling fixes.
- Combine both ignore lists. Link the parity executable with both upstream's `strata_engine`
  (its new load-check mode) and `strata_mmq` (our Q36 fixture). Retain CUDA-only Q36 test registration.
- Retain IQ2_S, IQ3_S and Q4_K down-format instantiations alongside upstream's new trellis
  formats; do not replace upstream's role lists or whole-block geometry checks.
- Keep the reference-order IQ down reduction alongside upstream's new exact-singleton
  gate/up routines. The latter do not supersede the down-row optimization: upstream still
  uses `row_dot_any` for these public down rows and GGML for this native down dispatch.
- Keep every original geometry/stride/activation/token/ISA/override condition. Additionally
  exclude newly introduced Hadamard-folded formats and require GGML itself to have AVX2,
  not AVX-512. Runtime ISA caps alone do not describe GGML's compiled dot arithmetic.
  Rejected cases continue into the existing dispatcher.
- Retain the Q36 host GGML decoder for quantized auxiliary weights, its controlled size/type
  rejection, and upstream's newer vision, batching and streaming interfaces unchanged.
- Keep `STRATA_MMQ_Q4_K_QSUM` default-off and enabled only by the exact value `1`.
  Upstream added a scatter quantizer that bypasses our repair. Opted-in Q4_K scatter calls
  now use upstream's existing gather fallback through `quantize`; all other calls keep
  upstream's routing. This is control-flow compatibility, not renewed GPU numerical qualification.

## Cheap local gates

No remote host, GPU, checkpoint or production service is used by these tests.

- `tools/test_moementum_iq_down.py`: actual IQ translation unit versus pinned GGML AVX2
  reference, **10,368 outputs, zero bit mismatches, zero preservation failures**. Plain/gather,
  both formats and widths 1–8. No VNNI, full native pool or GPU qualification is implied.
- `tools/test_moementum_dispatch.py`: actual native-expert translation unit linked with
  instrumented probes/backends, **60 cases in each of three fresh processes, all passing**.
  Includes compiled-GGML ISA mismatch, newly folded formats, widths, geometry and overrides.
  Before the added guards, eight default-mode cases failed. This is dispatch validation,
  not numerical evidence; the IQ numerical test above is separate.
- `tools/test_moementum_qsum_routing.py`: executes the actual extracted host scatter function
  with instrumented backends, **20 routing checks, all passing**. Before the routing fix,
  two opted-in Q4_K checks failed. No CUDA arithmetic is executed.
- Existing IQ pack tests: **30 passed** with an existing `STRATA_GGUF_PY` checkout.
- Existing pool-plan tests: **9 passed**.
- Existing Q36 config, family-name and web-sampling tests: **11 passed**.
- Broader selected serving tests were blocked by missing Python `regex`: the run reported
  **45 tests, 26 errors** (including a class-setup error). Do not treat that suite as passing.
- CPU-only CMake configure could not start because `cmake` is absent. CUDA/HIP configuration,
  linking and backend builds have not been attempted.

Both IQ runners accept `--llama-dir` pointing to an existing llama.cpp checkout and
`--output` pointing to a fresh directory. The QSUM runner needs only `--output`.
The GGML revision used for the numerical gate is the project's unchanged pin
`3cf03257f219afbe7334045ff7c6a06ac68c627d`. These tools do not download dependencies.

## Review and execution boundary

Specification review, then independent code-quality review, must precede isolated backend
build/model work. Outstanding qualification includes CUDA/HIP compilation, the mixed-expert
and upload GPU fixtures, updated quantizer/DS4 layout compatibility, real native-pool and
model token parity, and controlled before/after timing. Upstream changed much more than
this optimization; old throughput numbers cannot be transferred to the merged source.
