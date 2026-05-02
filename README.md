# Microcontroller_device
Voice microcontroler

# TSHM — streaming tests & ONNX hosting

This repository contains tools to export a tiny TSHM streaming model to ONNX and run **streaming inference** with both Python and a standalone C++ ONNX Runtime host. It also includes example host/test code and diagnostic logs so you can reproduce the results I obtained locally.

> Files of interest:

* `host_test_cpp_stream.cpp` — C++ streaming host (top-level example)
* `tshm/tshm_to_onnx.py` — export/checkpoint → ONNX helpers (streaming export)
* `tshm/export_streaming_onnx.py` — alternate/utility exporter (if present)
* `tshm/host_test_cpp_stream.cpp` — C++ host inside `tshm/` (same/variant of top-level)
* `host_test_stream_only_nojson.py` — Python streaming host/test (utility used in dev, included elsewhere)

---

## What this repo does

1. Loads a trained TSHM checkpoint (PyTorch).
2. Runs **streaming** inference frame-by-frame with a PyTorch checkpoint (simulating step-wise streaming).
3. Exports streaming encoder/head to ONNX (encoder step model + head model) for low-latency deployment.
4. Runs the ONNX encoder step repeatedly and accumulates a pooled representation, then runs the ONNX head for classification — tested with both Python (onnxruntime) and a C++ onnxruntime host.
5. Measures runtime (encoder per-frame, head), compares ONNX vs checkpoint logits, and prints small numeric diagnostic logs.

---

## Requirements

### Python

* Python 3.8+
* `torch` (same major version used for training)
* `onnx` (optional)
* `onnxruntime` (for Python ONNX inference)
* `numpy`
* (optional) `torchaudio` if using audio preprocessing scripts included elsewhere

Install typical deps:

```bash
pip install torch numpy onnx onnxruntime
# (optionally) pip install torchaudio
```

### C++ (ONNX Runtime)

To compile the C++ host you need ONNX Runtime C++ dev files (headers + library). Download the ONNX Runtime Linux package (or build from source) and point the compiler to include /lib paths.

Common dev package variables used in examples:

* `ORT_DIR` — path to onnxruntime package root (contains `include/` and `lib/`)

---

## Build (C++ host)

If you downloaded an ONNX Runtime release bundle and set `ORT_DIR`, compile like this:

```bash
# example: set ORT_DIR to your extracted onnxruntime package
export ORT_DIR=$PWD/onnxruntime-linux-x64-<version>

g++ host_test_cpp_stream.cpp -o host_test_cpp_stream \
    -I${ORT_DIR}/include -L${ORT_DIR}/lib -lonnxruntime \
    -O2 -std=c++17 -pthread
```

If linking errors about `onnxruntime_cxx_api.h` or `GetInputName` appear, ensure:

* You used the C++ API header (`include/onnxruntime_cxx_api.h`) from the dev package.
* You used the matching library version (header + lib from same package).
* Use `-pthread` and correct `-L` path.

If you have to set runtime loader path:

```bash
export LD_LIBRARY_PATH=${ORT_DIR}/lib:$LD_LIBRARY_PATH
./host_test_cpp_stream <encoder.onnx> <head.onnx> <feat.bin>
```

---

## Export ONNX (PyTorch checkpoint → encoder_step.onnx + head.onnx)

Use the included helper (example):

```bash
python3 tshm/tshm_to_onnx.py \
  --ckpt best_tshm_mfcc.pth \
  --encoder_out tshm_stream_encoder_step.onnx \
  --head_out tshm_stream_head.onnx \
  --input_dim 40 --d_model 48 --n_classes 10 \
  --n_layers 1 --r 16 --K 8 --ff_hidden 32
```

Notes / tips:

* `torch.load` may produce a `FutureWarning` about `weights_only`. For untrusted checkpoints prefer `torch.load(..., weights_only=True)` if supported by your PyTorch binary; otherwise keep using `map_location="cpu"`.
* If `model.head(...).numpy()` throws `RuntimeError: Can't call numpy() on Tensor that requires grad`, wrap inference with `torch.no_grad()` or call `.detach().cpu().numpy()` on outputs.

---

## Run streaming tests — Python

The Python streaming host runs the encoder step on each frame, accumulates a pooled vector and runs the head.

Example usage (python host_test_stream_only_nojson.py included earlier):

```bash
# ONNX backend
python3 host_test_stream_only_nojson.py \
  --feat feats/yes_1.npy \
  --encoder tshm_stream_encoder_step.onnx \
  --head tshm_stream_head.onnx \
  --backend onnx

# Checkpoint backend (PyTorch)
python3 host_test_stream_only_nojson.py \
  --feat feats/yes_1.npy \
  --ckpt best_tshm_mfcc.pth \
  --backend checkpoint
```

Output (example):

```
[ckpt] top1: 2 prob=0.825064
[compare] max abs diff between ONNX and checkpoint streaming logits: 2.384186e-06
[compare] top1 onnx=2 top1_pt=2
```

This shows ONNX and PyTorch checkpoint produce essentially identical logits (tiny numerical diff).

---

## Run streaming tests — C++ host

After compiling:

```bash
./host_test_cpp_stream tshm_stream_encoder_step.onnx tshm_stream_head.onnx feat.bin
```

Example output (observed during testing):

```
[info] Loaded features: T=161 F=40 total=6440
[stream] Top1: 4 prob=0.527625
[timing] encoder total ms: 8.68514  calls: 161  avg ms/frame: 0.053945
[timing] head ms: 0.012755
[timing] total ms (encoder+head): 8.69789
```

Notes:

* `feat.bin` is the binary feature file format used by the project: 2×int32 header (T, F), then `T*F` float32 values.
* The C++ host runs onnx runtime encoder step per frame and returns a pooled head prediction. You can time and print per-frame latency and head latency like shown.

---

## Observed results (summary)

These are the logs I recorded while testing — include them in the README so users reproduce exactly:

* ONNX vs checkpoint: `max_abs_diff ≈ 2.384186e-06` (negligible).
* Python checkpoint top1: `2` (prob ~0.825064).
* C++ ONNX stream run (example): `Top1: 4 prob=0.527625`
* C++ timing example:

  * encoder total ms: `8.68514`
  * calls (frames): `161`
  * avg ms/frame: `0.053945`
  * head ms: `0.012755`
  * total encoder+head: `8.69789`

> Interpretation: on a typical desktop/server CPU, the streamed encoder step is extremely cheap (~0.054 ms / frame in the example). Actual speed varies widely by CPU, ONNX Runtime build, and optimization flags.

---

## Troubleshooting & common fixes

* **`onnxruntime_cxx_api.h: No such file or directory`**
  Make sure you installed/downloaded the onnxruntime C++ package and point `-I` to the `include` directory.

* **`Session::GetInputName` / `GetOutputName` not found**
  Ensure you are using the C++ API compatible with your ONNX Runtime version. Call signatures changed between some ORT releases — use the cxx_api header shipped with your ORT binary.

* **`Can't call numpy() on Tensor that requires grad`**
  Wrap inference sections in:

  ```py
  with torch.no_grad():
      logits = model.head(torch.from_numpy(pooled)).detach().cpu().numpy()
  ```

  Or call `.detach().cpu().numpy()`.

* **`FutureWarning: torch.load weights_only`**
  If you load third-party checkpoints on untrusted machines, prefer `weights_only=True` once your PyTorch supports that flag. For now `torch.load(ckpt, map_location="cpu")` is commonly used.

---

## Reproducibility & tips

* Use the same PyTorch & ONNX Runtime major versions used for export and inference.
* When comparing ONNX vs checkpoint logits, make sure:

  * Inputs are identical (same pre-processing, dtype, padding).
  * ONNX export used consistent ops and `opset_version` (recommend opset 12–14).
* If you plan to deploy to low-power MCUs (ESP32), convert ONNX → SavedModel → TFLite and use quantization (post-training int8).
---

---

## Example README quick-start (copy-paste)

```bash
# export ONNX (python)
python3 tshm/tshm_to_onnx.py --ckpt best_tshm_mfcc.pth \
  --encoder_out tshm_stream_encoder_step.onnx --head_out tshm_stream_head.onnx \
  --input_dim 40 --d_model 48 --n_classes 10

# run python onnx test
python3 host_test_stream_only_nojson.py --feat feats/yes_1.npy \
  --encoder tshm_stream_encoder_step.onnx --head tshm_stream_head.onnx --backend onnx

# compile C++ host (must set ORT_DIR)
export ORT_DIR=/path/to/onnxruntime-linux-x64-<version>
g++ host_test_cpp_stream.cpp -o host_test_cpp_stream \
    -I${ORT_DIR}/include -L${ORT_DIR}/lib -lonnxruntime -O2 -std=c++17 -pthread

# run C++ host
./host_test_cpp_stream tshm_stream_encoder_step.onnx tshm_stream_head.onnx /path/to/feats.bin
```

---
