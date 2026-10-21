# Experiment: 32K context in VRAM

This branch, **`exp-32k-vram`**, preserves the integrated MoEmentum engine and the tested 32,768-token launch profile. It is an experimental snapshot, not the production/default branch or a binary release. Attention KV history and its FP16 staging remain in GPU memory; routed expert weights fall back to system RAM under this configuration.

## Result on our hardware

RTX 3050 6GB (6,144MiB), Intel Core i7-8700 as reported by the Linux/KVM guest, six vCPUs, 32GiB guest RAM. Model: Qwen3.8-35B-A3B-Distill IQ2_M, 12,558,245,760 bytes. Same model as the earlier experiments, not a smaller replacement.

- Engine context: **32,768 cells**; `READY 32768 stop` observed.
- Meaningful document: 440 distinct maintenance records with retrieval markers at beginning, middle and end.
- **30,460 actual prompt tokens read**, no prompt reuse, max-new reservation 128; 30,588 total reserved cells fits inside the configured 32,768.
- **36 emitted IDs including natural EOS**, 35 decode transitions.
- **Retrieval: all three markers correct.** The 73-token short smoke also passed.
- No OOM, timeout, retry or fallback attempt. One long request, not a repeatability or boundary-saturation qualification.

| Long request metric | Observed |
| --- | ---: |
| Prompt processing | 91.852 tokens/s |
| Prefill time | 331.621s |
| Answer generation | 8.825 decode transitions/s |
| Decode time | 3.966s |
| Full request wall, submission through DONE | 335.615s |
| Sampled device-wide GPU memory maximum | 5,777 / 6,144MiB |
| Sampled aggregate owned-process RSS maximum | 23.894GiB |
| Minimum sampled host MemAvailable | 17.703GiB |

GPU sampling was nominally every 2s; process RSS/MemAvailable every 1s. These are observed samples, not true instantaneous peaks. RSS includes model mappings/pinned arenas and is not an additive physical-RAM allocation or directly subtractable from MemAvailable. See [sanitized sample data](moementum/32k-vram.json) for actual counters, timing and sampling scope.

Launch through READY took 6.956s with uncontrolled OS page cache. It is not a cold-storage load benchmark. Prefill and decode TPS measure separate engine stages; full request wall excludes startup and process teardown.

## The critical placement trade-off

Requested `--expert-vram 2` does **not** mean the engine retained 2GiB of resident experts. After allocating context and other buffers, upstream's VRAM clamping selected **zero resident expert slots**. About 10.51GiB of expert data stayed in pinned host RAM and 1.59GiB of GPU staging remained. All routed expert entries used the CPU, while other model operations stayed GPU-backed.

Aggregate routing: **0 GPU + 9,784,640 CPU = 9,784,640 entries**, no slot replacements. Five requested CPU pool workers plus the participating caller are logged as six CPU threads.

The original frozen test controller incorrectly required `GPU expert entries > 0` for this capacity experiment. Its overall verdict therefore remains **FAIL** after both requests completed. That receipt was not rewritten. Independent raw-protocol validation passed exact IDs, actual-read counts, reuse, chunks, EOS, routing totals and retrieval scoring. **Functional fit/retrieval PASS is separate from that failed placement-assumption gate.**

This is not evidence that the prior 3GiB, 23-slot calibrated resident-expert layout fits 32K. It is not a 4K-versus-32K speedup comparison. The separate inherited streamed-Q4 absolute FP screen remains FAIL and is not waived here.

## Reproduce the launch profile

Build this branch's source explicitly using the [source build guide](MOEMENTUM_INSTALL.md). Do not rely on inherited installers that may fetch upstream binaries. The measured source revision was `b91d795466e6242474bee4423a5e788525c4fc22`; this branch adds only documentation and a launch example on top of it. The measured executable SHA256 was `fb0df4dcd2dc586a48a55e45bf8a0a55d833e827543f474769ad26007f640079`; a fresh build need not reproduce its binary hash.

```bash
git clone --branch exp-32k-vram https://github.com/CrashxZ/MoEmentum.git
cd MoEmentum
# Follow docs/MOEMENTUM_INSTALL.md to build and prepare your exact model/tokenizer.
```

The [example server configuration](../configs/experiments/32k-vram.example.json) contains placeholders. Copy it outside version control and replace every absolute path. **Unset the prior calibrated-placement controls**: its frozen slot map does not match this lower-budget layout. Do not pass empty strings—the controls use environment-variable presence semantics.

```bash
unset Q36_INITIAL_PLACEMENT_MAP_V4 Q36_STATELESS_ADAPTIVE_V2 Q36_EXPECTED_MODEL_SHA256
python serve/server.py --engine strata --config /absolute/path/to/your-32k-config.json \
  --host 127.0.0.1 --port 8080
```

Equivalent native engine arguments (the server wrapper adds `--serve` itself):

```text
--native /absolute/path/to/model.gguf --max-context 32768 --chunk 128
--expert-vram 2 --cpu-threads 5 --hybrid-rows 128 --mtp 0 --no-warmup
```

Keep `STRATA_MMQ_Q4_K_QSUM=0`. RAM-backed KV is **not implemented by this profile**: it deliberately keeps context in VRAM. A separately named RAM-cache experiment must not be confused with these measured results.

Only a private native-protocol run of this profile was exercised. The JSON is a launch example for the existing server schema, not a separately qualified 32K web installer/deployment. Check effective slots, actual memory and long-prompt correctness on your own hardware. Use loopback unless authentication/TLS is configured.

## Provenance and boundaries

[Sanitized data](moementum/32k-vram.json) preserves both request timings, counts and input/output hashes. Model SHA256: `9c095175f7af0c4acc18e552f0dd7ac8180d81f9ac4bf38d62a03c76a0d6e084`. Private raw native streams, generated-token arrays, controller/watchdog receipts and machine topology remain outside Git; no credentials, addresses, model weights or binary blobs are published.

The prior service installation/configuration was not changed. Inference was intentionally left stopped after this experiment, pending a separate deployment decision. This is not a claim about its current live status on another machine or at a future time.

Strata/GGML attribution and original license notices are preserved. No upstream endorsement, universal memory minimum, general model-quality pass, concurrency qualification or production promotion is implied.
