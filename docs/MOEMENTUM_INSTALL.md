# Build and run MoEmentum

[Back to the project README](../README.md)

## Install from source: Linux + NVIDIA

### 1. Prerequisites

- Linux x86-64; **AVX2, FMA and F16C** for the optimized CPU kernel. Other CPUs can use upstream paths, but this optimization is not qualified there.
- An NVIDIA driver and a **CUDA development toolkit containing `nvcc`**, not only the CUDA runtime shipped in Python wheels. Use the CUDA/toolchain compatibility guidance in [upstream installation docs](../docs/INSTALL.md).
- Git, C/C++ compilers supporting C++20, Python3.10+, venv/pip, CMake3.24+ and Ninja.
- Sufficient RAM, GPU memory and disk for your chosen checkpoint and build. Our tested GGUF is about12.56GB; leave additional space for dependencies, build artifacts and runtime allocations. Other models need different resources.

On Debian/Ubuntu, the basic CPU build prerequisites are:

```bash
sudo apt-get update
sudo apt-get install git build-essential python3 python3-venv python3-pip
```

Install the appropriate NVIDIA driver/CUDA toolkit separately. Check them before building:

```bash
nvidia-smi
nvcc --version
```

### 2. Get MoEmentum and Python dependencies

Clone the published optimization branch:

```bash
git clone --branch moementum/iq-down-prefill https://github.com/CrashxZ/MoEmentum.git
cd MoEmentum
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

The upstream [requirements file](../requirements.txt) pins the Python tools. There is no separate package named `moementum` to install with pip.

### 3. Compile the actual fork

Example for the tested Ampere RTX3050 (`CUDA architecture86`):

```bash
cmake -S . -B build-moementum -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_NATIVE_EXPERTS=ON \
  -DSTRATA_MMQ_KQUANTS=ON \
  -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared
cmake --build build-moementum --target strata-q36 -j 2
```

Use the architecture appropriate to your GPU, not86 universally. The target and native-expert dependency linkage are defined in [CMakeLists.txt](../CMakeLists.txt); GGML is fetched at the revision pinned there. Limit parallel jobs on RAM-constrained machines.

**Do not assume `./setup.sh` or `START-HERE.bat` installed MoEmentum.** The inherited installer normally downloads an upstream Strata binary. Building this checkout and explicitly selecting its `build-moementum/strata-q36` executable avoids that silent substitution. The inherited setup's `qwen36` family is marked download-only in this snapshot, so it is not the installation route for this tested smaller-model configuration.

The measured C binary was built and fully linked from these kernel source bytes in a controlled private build. The generic clean-checkout installation above is derived from the actual CMake targets; it has **not been rerun end-to-end as a fresh installer qualification**. Windows/AMD/Mac builds are inherited upstream capabilities, not newly validated MoEmentum platforms.

### 4. Supply a compatible model

Obtain your model from a trusted distributor and obey its license. MoEmentum does not redistribute weights. Our tested checkpoint is the **IQ2_M quantization of Empero's Qwen3.8-35B-A3B-Distill**, not the larger model advertised in the upstream homepage examples. Its checksum is in the benchmark record.

Set the local absolute path, without duplicating a model you already have:

```bash
export MODEL=/absolute/path/to/Qwen3.8-35B-A3B-IQ2_M.gguf
sha256sum "$MODEL"
```

The fast-path guards validate the expert layout. A similarly named GGUF with different tensor geometry may use fallback paths and must not be advertised as having the measured speedup.

### 5. Optional browser UI / API

Use Strata's existing Python server with an explicit engine config. First extract the tokenizer from **your exact GGUF**:

```bash
python tools/strata_tokenizer.py --gguf "$MODEL" --out local-pack --check
```

Save a local `moementum.local.json` based on this example, replacing **every absolute path** with yours:

```json
{
  "exe": "/absolute/path/to/MoEmentum/build-moementum/strata-q36",
  "cwd": "/absolute/path/to/MoEmentum",
  "tokenizer": "/absolute/path/to/MoEmentum/local-pack/tokenizer",
  "model_name": "moementum-local",
  "args": [
    "--native", "/absolute/path/to/Qwen3.8-35B-A3B-IQ2_M.gguf",
    "--max-context", "4096", "--chunk", "128",
    "--expert-vram", "3", "--cpu-threads", "5",
    "--hybrid-rows", "128", "--mtp", "0", "--no-warmup"
  ],
  "env": {
    "STRATA_MMQ_KQ": "1",
    "STRATA_MMQ_Q4_K_QSUM": "0"
  }
}
```

```bash
python serve/server.py --engine strata --config moementum.local.json --host 127.0.0.1 --port 8080
```

The server wrapper adds `--serve` to the engine invocation. Do not add it a second time to the config above. It provides the inherited browser interface and compatible API; see [server options](../serve/server.py) and [API docs](../docs/DETAILS.md). Start with loopback only. Do not expose an unauthenticated server to your LAN or the internet; review upstream authentication/TLS options first.

This example does **not** embed our machine-specific calibrated placement map. It is a starting configuration, **not an exact reproduction of the benchmark placement or a guarantee of its timings**. Start separately from any production service; change thread/context/GPU budgets only after checking available resources.

## CPU-only regression test

No GPU, checkpoint or private activation captures are needed. Supply a llama.cpp source tree at the tested dependency revision:

```bash
git clone https://github.com/ggml-org/llama.cpp.git /path/to/llama.cpp
git -C /path/to/llama.cpp checkout 3cf03257f219afbe7334045ff7c6a06ac68c627d
python tools/test_moementum_iq_down.py \
  --llama-dir /path/to/llama.cpp \
  --output ./moementum-test-run1
```

Use a **fresh output directory** each time. GCC/G++ on an AVX2/FMA/F16C Linux host are required. The runner compiles GGML reference objects plus the real IQ TU, then checks IQ3_S/IQ2_S, plain/gather variants, widths1..8, multiple seeds, padded strides, partitioned writes and input preservation. This tests the numerical kernel, **not the full native dispatcher or model accuracy**; native dispatch/full-engine evidence is documented separately.

Fresh verification: the unchanged kernel failed with3644 bit mismatches; the submitted kernel passed **10,368 comparisons with zero bit mismatches and zero preservation failures**. The test returns nonzero on failure. Unsupported CPUs are not qualified by this AVX2-targeted standalone harness. This runner is GCC/Linux-oriented; it is not a Windows or sanitizer qualification.

## Limitations and fallback

- Optimized dispatch is narrow: matching IQ2_S/IQ3_S expert geometry, multi-token groups and an eligible AVX2 CPU. No blanket claim for other models, quantizations or CPUs.
- `STRATA_NO_IQ256=1` selects existing fallbacks, but disables other IQ256 paths too; it is **not** an exact reproduction of benchmark baseline A.
- Keep a separate known-good executable/config. Do not overwrite production automatically.
- Reference-order reduction is verified on exercised synthetic and captured activations, not every possible input or hardware ISA.
- Model correctness and stock llama.cpp attribution are separate from optimization parity.
- No published binary, unattended installer, universal quality gate or cross-platform release qualification is claimed. A private Linux/NVIDIA service deployment was separately validated; see [served measurements](MOEMENTUM_SERVING.md).

