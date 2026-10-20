# MoEmentum: measured scope and reproducibility

The complete sanitized sample record is [moementum/benchmark.json](moementum/benchmark.json). It contains measured durations and source/model/placement/binary SHA256 pins, not checkpoint bytes, prompts, hostnames or credentials.

## Source lineage

The tested parent Strata source is local commit `a43dbc9480480df8ef193045fc0d0a34aa84bec6`, based on upstream plus two local commits for mixed expert compatibility and default-off Q4 diagnostics. MoEmentum retains that history rather than pretending to be an unrelated engine. The new optimization changes two files:

- `src/kernels/cpu/native_expert.cpp`: narrow IQ2_S/IQ3_S multi-token down dispatch.
- `src/kernels/cpu/iq_avx2_rows.inl`: GGML reference-order down reduction.

The unchanged `iq_avx2.cpp` includes the changed inline file, so the measured full-engine candidate replaced **two CPU archive members**, `native_expert.cpp.o` and `iq_avx2.cpp.o`, in the original 13-member CPU archive. All other members and their order were verified unchanged; the caller object was unchanged between arms. A full clean build via CMake also rebuilds that dependency closure. This is not an unchanged one-translation-unit recipe.

Candidate source SHA256:

- `iq_avx2.cpp`: `58e9c7e49b8158f698cda59aff06c245f62aaecd33556b1e88e234140890fded`
- `iq_avx2_rows.inl`: `81cea87e725eace203a3fbe0beba1ae3164b57fa447b79fb7a652c843f6094dc`
- `native_expert.cpp`: `992734526b95a6d8b5bb471626738182f71e99e8d4512c863a1ded4aef741049`

Measured baseline executable SHA256: `3480700c61af79aa35ff09185046676af8a158b6a2caee3b6e2368c56e76fa22`.
Measured optimized executable SHA256: `b89134e1b7c35336e5fe36da80792ec4fc52bacfa0daebe9697351da32ccb3b6`.

A fresh toolchain/link/build need not produce identical executable bytes. Those pins identify the actual measured artifacts, not a claim of reproducible builds across systems.

## Full-engine observations

Linux guest, six vCPUs,32GiB RAM, RTX3050 6GB; five expert threads, context4096, chunk128, expert budget3GiB, hybrid rows128, MTP0. Qwen3.8-35B-A3B-Distill IQ2_M, same model, settings and calibrated placement in both arms. No forced tokens. Prefill and decode are separate measurements.

### Four-task trial

A,C,C,A,A,C order, three fresh lifetimes per arm, four natural requests per lifetime. Four distinct input lengths:100,190,1767,2514. Max generation256, temperature0, seed42, no KV reuse. All emitted token sequences, counts, stop reasons and prompt/read/reuse metadata matched across24 requests and918 token occurrences, including EOS.

All samples and medians are retained in the JSON. Prefill median time reductions:24.35%,26.94%,30.78%,30.31%. Decode approximately unchanged. Whole-process medians include startup, all four requests and teardown:81.397s baseline vs61.119s optimized, about24.91% less time. Whole maintenance window490.610s also included checks and production restoration; it is not an inference latency.

**Task correctness is separate:** both arms consistently passed the record lookup and failed the arithmetic, strict-key sorting and filtering tasks. One of four unique tasks passed, not an absolute-quality success. The model's wrong answers are preserved. No stock llama.cpp same-task comparison has yet established their cause.

### Earlier two-prompt trial

A,C,C,A order; two frozen prompts of871 and1097 tokens,64-token generation cap; eight requests,272 emitted token occurrences, all full-token matches. Median prefill times12.663→8.297s and15.523→11.116s. Those prompts/repetitions are separate from the four-task trial. Do not pool the two experiments as if every prompt had the same generation budget or repetition count.

## Kernel verification

Authentic captured activations were replayed through the real public IQ API against GGML, with zero bit differences in90,112 repeated comparisons. A separate native numerical adapter proved the dispatch was exercised; it was not a complete real-pool CPU fixture. Full model executions provide the downstream integration evidence described above.

The newly publishable synthetic test requires no captures or model: `tools/test_moementum_iq_down.py`. It compiles the actual IQ translation unit, using pinned GGML source dependencies, and exercises both formats, plain/gather variants, widths1..8, four fixed seeds, padded strides, empty and reverse-ordered partitions, output sentinels and input preservation. Before the kernel change it returned3644 mismatches. With the exact candidate source it checked10,368 outputs with zero bit mismatches or preservation failures. It is an AVX2/GCC/Linux numerical test, not full native dispatcher, sanitizer, VNNI execution, Windows or model-quality qualification.

## Failed evidence is not erased

An earlier absolute-quality trial stopped after baseline A's four requests, before C ran. The first scorer mistakenly passed terminal `<|im_end|>` text to JSON parsing; baseline also had genuine wrong answers. The failure was preserved. A new, explicitly separated parity experiment followed a regression-tested terminal-ID decoding correction. It strips only the final declared EOS for visible-text scoring while retaining every generated ID for parity. Wrong answers remain wrong. The failed trial is not retroactively scored as a pass.

An initial build also failed closed on additional IQ-TU header dependencies. Those exact dependencies were pinned and checked before a fresh private build completed. No failing namespace was overwritten and no deadline was silently extended.

## Limits and reproduction boundary

- Four held-out tasks plus two earlier prompts are finite coverage, not proof of universal parity or model quality.
- Three processes per arm and a fixed order are observational measurements, not formal significance or control of all cache/thermal effects. No sample was dropped.
- No new speedup is claimed for decode, AMD, Windows, Mac, VNNI, AVX-512, other models, quantizations or layouts.
- The published basic launch example deliberately omits machine-specific calibrated placement. It cannot reproduce exact benchmark timings on its own.
- Full raw service/lease ownership receipts and real activation data remain private. Public summaries and hashes are audit aids, not a self-contained third-party reproduction of the full model experiment.
- No packaged binary release, clean one-click installer qualification or production deployment has been completed.

Upstream attribution and license notices remain intact. This is an experimental Strata fork, not an endorsed upstream performance claim.
