#!/usr/bin/env python3
"""
Streaming-only host test without external JSON.

Usage examples:
  python host_test_stream_only_nojson.py --feat feats/yes_1.npy --encoder tshm_stream_encoder_step.onnx --head tshm_stream_head.onnx --backend onnx
  python host_test_stream_only_nojson.py --feat feats/yes_1.npy --ckpt best_tshm_mfcc.pth --backend checkpoint
  python host_test_stream_only_nojson.py --feat feats/yes_1.npy --encoder tshm_stream_encoder_step.onnx --head tshm_stream_head.onnx --ckpt best_tshm_mfcc.pth --backend both
"""
import argparse
from pathlib import Path
import numpy as np
import torch

try:
    import onnxruntime as ort
except Exception:
    ort = None

from tshm_audio_classification_code_reg import TSHMClassifier

def load_bin_features(bin_path):
    with open(bin_path, "rb") as f:
        t_bytes = f.read(4)
        f_bytes = f.read(4)
        if len(t_bytes) < 4 or len(f_bytes) < 4:
            raise RuntimeError("Invalid .bin feature file (header too short)")
        T = int(np.frombuffer(t_bytes, dtype=np.int32)[0])
        Fv = int(np.frombuffer(f_bytes, dtype=np.int32)[0])
        data = np.frombuffer(f.read(), dtype=np.float32)
        expected = T * Fv
        if data.size != expected:
            raise RuntimeError(f"mismatch data size {data.size} expected {expected}")
        arr = data.reshape((T, Fv))
        return arr
def load_npy_or_bin(path):
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(str(p)).astype(np.float32)
    elif p.suffix == ".bin":
        return load_bin_features(str(p))
    else:
        raise RuntimeError("unsupported feature file; use .npy or .bin")

def softmax_np(x):
    x = np.asarray(x)
    e = np.exp(x - np.max(x))
    return e / e.sum()

# PyTorch checkpoint streaming
def run_pytorch_streaming(ckpt, feat, cfg):
    device = torch.device("cpu")
    model = TSHMClassifier(input_dim=cfg["input_dim"], n_classes=cfg["n_classes"],
                           d_model=cfg["d_model"], n_layers=cfg["n_layers"],
                           r=cfg["r"], K=cfg["K"], ff_hidden=cfg["ff_hidden"],
                           use_pos=False, dropout=0.0, causal=True)
    model.eval()
    sd = torch.load(ckpt, map_location="cpu")
    # accept wrapped checkpoint
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    try:
        model.load_state_dict(sd)
    except Exception:
        model.load_state_dict(sd, strict=False)

T, F = feat.shape
    states = model.encoder.init_stream_state(batch_size=1, device=torch.device("cpu"))
    pooled_sum = np.zeros((cfg["d_model"],), dtype=np.float32)
    count = 0
    with torch.no_grad():
        for t in range(T):
            x_t = torch.from_numpy(feat[t]).unsqueeze(0)  # (1, F)
            out_t, states = model.encoder.forward_step(x_t, states)  # (1, d)
            pooled_sum += out_t[0].cpu().numpy()
            count += 1
    pooled = (pooled_sum / max(1, count)).reshape(1, cfg["d_model"]).astype(np.float32)

    # --- FIX: run head under no_grad and detach before converting to numpy ---
    with torch.no_grad():
        head_in = torch.from_numpy(pooled)
        logits_t = model.head(head_in)
        logits = logits_t.detach().cpu().numpy()
    return logits
# ONNX streaming using input shapes from the ONNX model
def run_onnx_streaming(encoder_onnx, head_onnx, feat):
    if ort is None:
        raise RuntimeError("onnxruntime not installed")
    sess_enc = ort.InferenceSession(str(encoder_onnx), providers=["CPUExecutionProvider"])
    sess_head = ort.InferenceSession(str(head_onnx), providers=["CPUExecutionProvider"])

    enc_inputs = sess_enc.get_inputs()
    enc_input_names = [i.name for i in enc_inputs]
    enc_outputs = sess_enc.get_outputs()
    enc_output_names = [o.name for o in enc_outputs]

 # first input expected to be x_t
    if len(enc_input_names) < 2:
        raise RuntimeError("Unexpected encoder.onnx input layout (need x_t + states)")

    # Build zero states by reading input shapes from ONNX session
    # Inputs shapes may be like [1, r] or [1, d, pad]; some dims can be None — we try to infer sizes from concrete dims
    state_values = []
    # skip input 0 (x_t)
    for inp in enc_inputs[1:]:
        shape = inp.shape  # may be like [1, 16] or [1, 48, 2]
        # replace None with 1 (conservative)
        concrete_shape = [1 if (d is None) else int(d) for d in shape]
        state_values.append(np.zeros(tuple(concrete_shape), dtype=np.float32))
# streaming loop
    T, F = feat.shape
    pooled_sum = None
    count = 0
    for t in range(T):
        x_t = feat[t:t+1, :].astype(np.float32)
        feed = {enc_input_names[0]: x_t}
        for i, nm in enumerate(enc_input_names[1:]):
            feed[nm] = state_values[i]
        outs = sess_enc.run(enc_output_names, feed)
        # outs[0] is h_out, rest are new states in same order
        h_out = outs[0]  # shape (1, d)
        new_states = outs[1:]
        for i in range(len(state_values)):
            state_values[i] = new_states[i].astype(np.float32)
        if pooled_sum is None:
            pooled_sum = np.zeros_like(h_out[0])
        pooled_sum += h_out[0]
        count += 1
 pooled = (pooled_sum / max(1, count)).reshape(1, h_out.shape[1]).astype(np.float32)
    head_in_name = sess_head.get_inputs()[0].name
    head_out_name = sess_head.get_outputs()[0].name
    head_out = sess_head.run([head_out_name], {head_in_name: pooled})[0]
    return head_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat", required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--encoder", default=None)
    parser.add_argument("--head", default=None)
    parser.add_argument("--backend", default="both", choices=["onnx", "checkpoint", "both"])
    parser.add_argument("--n_mfcc", type=int, default=40)
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--d_model", type=int, default=48)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--ff_hidden", type=int, default=32)
    args = parser.parse_args()

    feat = load_npy_or_bin(args.feat)
    cfg = dict(input_dim=args.n_mfcc, n_classes=args.n_classes, d_model=args.d_model,
               n_layers=args.n_layers, r=args.r, K=args.K, ff_hidden=args.ff_hidden)

    do_onnx = args.backend in ("onnx", "both")
    do_ckpt = args.backend in ("checkpoint", "both")

    onnx_logits = None
    pt_logits = None
if do_onnx:
        if args.encoder is None or args.head is None:
            raise RuntimeError("ONNX backend requested but encoder/head not provided")
        print("[main] Running ONNX streaming...")
        onnx_out = run_onnx_streaming(args.encoder, args.head, feat)
        onnx_logits = np.asarray(onnx_out).reshape(1, -1)
        probs = softmax_np(onnx_logits.reshape(-1))
        print(f"[onnx] top1: {int(np.argmax(probs))} prob={float(np.max(probs)):.6f}")

    if do_ckpt:
        if args.ckpt is None:
            raise RuntimeError("Checkpoint backend requested but --ckpt not provided")
        print("[main] Running PyTorch checkpoint streaming...")
        pt_out = run_pytorch_streaming(args.ckpt, feat, cfg)
        pt_logits = np.asarray(pt_out).reshape(1, -1)
        probs = softmax_np(pt_logits.reshape(-1))
        print(f"[ckpt] top1: {int(np.argmax(probs))} prob={float(np.max(probs)):.6f}")

 if onnx_logits is not None and pt_logits is not None:
        diff = np.max(np.abs(onnx_logits - pt_logits))
        print(f"[compare] max abs diff between ONNX and checkpoint streaming logits: {diff:.6e}")
        print(f"[compare] top1 onnx={int(np.argmax(onnx_logits))} top1_pt={int(np.argmax(pt_logits))}")

if __name__ == "__main__":
    main()
#python3 /workspace/tshm_device/main/tshm_to_onnx.py  --feat /workspace/speech_command/yes/0a5636ca_nohash_0.npy --encoder tshm_stream_encoder_step.onnx --head tshm_stream_head.onnx --ckpt best_tshm_mfcc.pth --backend both
