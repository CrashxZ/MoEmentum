# MoEmentum: served throughput on our hardware

[Project README](../README.md) · [Sanitized per-request data](moementum/serving.json)

These results are a separate deployment experiment, not a new baseline-versus-candidate comparison. They do not replace the [controlled prefill trial](MOEMENTUM_BENCHMARK.md).

## Hardware and configuration

Linux/KVM guest reporting Intel Core i7-8700 @ 3.20GHz; six vCPUs, 32GiB RAM; NVIDIA GeForce RTX 3050 with 6,144MiB VRAM. Five CPU expert threads. Qwen3.8-35B-A3B-Distill IQ2_M, about 12.56GB GGUF. Hardware names were read back from the inference guest when preparing this documentation.

Context 4,096, chunk 128, expert VRAM budget 3GiB, hybrid rows 128, MTP 0, reasoning effort `none`; frozen calibrated placement. Exact model, placement and deployed executable hashes are retained in the JSON. That executable is the same candidate used for the controlled trial.

## Measurement

Three serial nonstreaming requests per workload, over an authenticated TLS API from a LAN client, after model-load/completion/streaming smoke checks. No concurrency or reused prompt tokens. No sample dropped. Figures are medians of engine-reported per-stage rates and separately timed full HTTP request durations.

| Workload | Prompt tokens | Completion tokens | Prefill TPS | Decode TPS | HTTP seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short generation | 71 | 128 | 101.4 | 38.4 | 4.169 |
| Long prefill | 1,763 | 23 | 102.4 | 39.8 | 17.951 |

Short generation reached its 128-token cap; long prefill produced 23 tokens and stopped naturally. Both returned nonempty content. These throughput requests were not a model-quality evaluation.

**TPS definitions:** prefill is the engine's new-prompt processing rate; decode is its answer-generation rate. Whole HTTP duration includes both stages and transport/server overhead. Dividing completion tokens by HTTP time gives a different, lower end-to-end metric. Stage medians must not be combined into a synthetic request latency.

The separate post-restart completion took 7.892s including cold load, prompt processing, generation and HTTP overhead. It is not one of these six warm samples. Idle unload was configured at 60s but its cycle was not separately measured; reboot was not tested.

## Deployment acceptance versus release qualification

Correct-key model enumeration, rejected missing/wrong credentials, nonempty completion, SSE deltas and termination, and a completion after service restart were verified. The backend was loopback-only behind an authenticated TLS proxy. The web branding was later updated without changing kernel arithmetic or model weights.

This validates one private deployment, not a publicly released installer or a general minimum-hardware claim. There is no served baseline arm, no saturation test, no independent third-party replication and no formal confidence interval. The controlled optimization's unchanged model-quality limitations still apply.

## Data provenance and privacy

`moementum/serving.json` preserves all six timing/usage rows and the recomputed medians from the private deployment receipts. Only hardware, configuration, hashes, token counts and timing metadata are published; no prompts, completion text, addresses, credentials, certificates or model files are included. The private original response captures remain outside Git.
