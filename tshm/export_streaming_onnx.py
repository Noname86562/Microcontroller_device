#!/usr/bin/env python3
"""
export_streaming_onnx.py

Defensive/robust exporter that:
 - canonicalizes init_stream_state -> list of 4-tuples (S, M_pref, N_pref, gate_buf)
 - runs per-layer forward_step tests using tuple states (to match runtime expectations)
 - exports encoder-step ONNX and head ONNX
 - optional smoke test via onnxruntime
"""
import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import torch

try:
    import onnxruntime as ort
except Exception:
    ort = None

from tshm_audio_classification_code_reg import TSHMClassifier
# ---------------- helpers ----------------
def _extract_state_dict(sd):
    if isinstance(sd, dict):
        for k in ("state_dict", "model_state_dict", "model", "model_state"):
            if k in sd and isinstance(sd[k], dict):
                return sd[k]
        vals = list(sd.values())
        if vals and all(isinstance(v, (torch.Tensor, np.ndarray)) for v in vals):
            return sd
    return sd
def _load_checkpoint_file(ckpt_path, map_location="cpu"):
    try:
        sd = torch.load(ckpt_path, map_location=map_location, weights_only=True)  # type: ignore
        if isinstance(sd, dict):
            return _extract_state_dict(sd)
        return sd
    except TypeError:
        pass
    except Exception as e:
        print("[ckpt] weights_only warning:", e)
    sd_all = torch.load(ckpt_path, map_location=map_location)
    return _extract_state_dict(sd_all)
def _to_torch_tensor(x, device):
    """Convert x (Tensor/ndarray/list/scalar) -> torch.Tensor on device float32."""
    if isinstance(x, torch.Tensor):
        return x.clone().detach().to(dtype=torch.float32, device=device)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x.astype(np.float32)).to(device=device)
    if isinstance(x, (list, tuple)):
        try:
            arr = np.asarray(x, dtype=np.float32)
            return torch.from_numpy(arr).to(device=device)
        except Exception:
            raise RuntimeError(f"Cannot convert list/tuple to tensor (len={len(x)})")
    if isinstance(x, (int, float)):
        return torch.tensor(x, dtype=torch.float32, device=device)
    raise RuntimeError(f"Unsupported state element type for conversion: {type(x)}")
  def derive_state_shapes_from_encoder(encoder, batch_size=1):
    shapes = []
    try:
        d_model = int(encoder.embed.out_features)
    except Exception:
        d_model = None
    for layer in encoder.layers:
        r = getattr(layer, "r", None)
        gate_k = getattr(layer, "gate_kernel", None)
        if d_model is None:
            d_model = getattr(layer, "d", d_model)
        if r is None or gate_k is None:
            raise RuntimeError("Layer missing 'r' or 'gate_kernel'; cannot derive shapes")
        pad = max(0, int(gate_k) - 1)
        shapes.append({
            "S": (batch_size, int(r)),
            "M_pref": (batch_size, int(r)),
            "N_pref": (batch_size, int(r)),
            "gate_buf": (batch_size, int(d_model), pad)
        })
    return shapes
def canonicalize_init_states_to_tuples(raw_states, encoder, device):
    """
    Convert raw init_stream_state output into a canonical list of 4-tuples:
      [(S, M_pref, N_pref, gate_buf), ...]
    Accepts:
     - list/tuple of dicts with keys S/M_pref/N_pref/gate_buf
     - list/tuple of tuples/lists (len==4)
     - flat tuple/list that is multiple-of-4 (will chunk)
    Fallback: derive shapes and return zeros shaped tensors.
    """
    if raw_states is None:
        raise RuntimeError("init_stream_state returned None")

    # If it's a single tuple-of-4 returned inside a list, keep it; if it's dict/list handle accordingly.
    items = []
    if isinstance(raw_states, dict):
        # wrap single dict -> list
        raw_states = [raw_states]
    if isinstance(raw_states, (list, tuple)):
        # if raw_states itself is a flat tuple-of-length 4*L, e.g., returned as tuple(S,M,N,gate,S2,...)
        if not raw_states:
             raise RuntimeError("init_stream_state returned empty sequence")
        # Heuristic: if first element is Tensor or ndarray or list/tuple -> treat as list of elements
        # If first element is a dict -> per-layer dicts
        first = raw_states[0]
        if isinstance(first, dict):
            for st in raw_states:
                S = st.get("S", st.get("s", None))
                M = st.get("M_pref", st.get("M", None))
                N = st.get("N_pref", st.get("N", None))
                G = st.get("gate_buf", st.get("gate_buffer", st.get("gate", None)))
                items.append((S, M, N, G))
        elif isinstance(first, (list, tuple)) and len(first) == 4:
            for st in raw_states:
                items.append(tuple(st))
        else:
            # Could be flat sequence; attempt chunk by 4
            flat = list(raw_states)
            if len(flat) % 4 == 0 and all(not isinstance(x, dict) for x in flat):
               for i in range(0, len(flat), 4):
                    items.append((flat[i], flat[i+1], flat[i+2], flat[i+3]))
            else:
                # last resort: if raw_states is single tuple containing tuple-of-4 as element (like [ (S,M,N,G) ])
                if isinstance(first, (list, tuple)) and len(first) != 4 and len(raw_states) == 1:
                    inner = first
                    if len(inner) % 4 == 0:
                        for i in range(0, len(inner), 4):
                            items.append((inner[i], inner[i+1], inner[i+2], inner[i+3]))
                    else:
                        raise RuntimeError("Could not canonicalize init_stream_state (unexpected structure)")
                else:
                    raise RuntimeError("Could not canonicalize init_stream_state (unexpected structure)")

    else:
        raise RuntimeError("init_stream_state returned unsupported type: " + str(type(raw_states)))
 # Convert elements to tensors and fill missing with zeros using derived shapes if necessary
    # Convert elements that are not None into tensors
    converted = []
    for i, (S, M, N, G) in enumerate(items):
        try:
            tS = _to_torch_tensor(S, device) if S is not None else None
        except Exception:
            tS = None
        try:
            tM = _to_torch_tensor(M, device) if M is not None else None
        except Exception:
            tM = None
        try:
            tN = _to_torch_tensor(N, device) if N is not None else None
        except Exception:
            tN = None
        try:
            tG = _to_torch_tensor(G, device) if G is not None else None
        except Exception:
            tG = None
        converted.append((tS, tM, tN, tG))

 # If any None, fill by shapes derived from encoder
    any_none = any(any(x is None for x in tup) for tup in converted)
    if any_none:
        shapes = derive_state_shapes_from_encoder(encoder, batch_size=1)
        Lshape = len(shapes)
        Litems = len(converted)
        # for each item fill using shapes[i] or last shape
        for i in range(Litems):
            shape_index = i if i < Lshape else (Lshape - 1)
            shp = shapes[shape_index]
            Sshp = shp["S"]; Mshp = shp["M_pref"]; Nshp = shp["N_pref"]; Gshp = shp["gate_buf"]
            S, M, N, G = converted[i]

if S is None:
                converted[i] = (torch.zeros(Sshp, dtype=torch.float32, device=device),) + converted[i][1:]
            if converted[i][1] is None:
                converted[i] = (converted[i][0], torch.zeros(Mshp, dtype=torch.float32, device=device), converted[i][2], converted[i][3])
            if converted[i][2] is None:
                converted[i] = (converted[i][0], converted[i][1], torch.zeros(Nshp, dtype=torch.float32, device=device), converted[i][3])
            if converted[i][3] is None:
                converted[i] = (converted[i][0], converted[i][1], converted[i][2], torch.zeros(Gshp, dtype=torch.float32, device=device))

# Final check
    for i, tup in enumerate(converted):
        if not (isinstance(tup, tuple) and len(tup) == 4 and all(isinstance(x, torch.Tensor) for x in tup)):
            raise RuntimeError(f"Canonicalization failed for layer {i}: got {tup}")

    return converted

# ---------------- wrapper ----------------
class StreamingEncoderWrapper(torch.nn.Module):
    """
    Wraps encoder.forward_step into ONNX-exportable single-step module.

    Inputs order: x_t, s0_S, s0_M, s0_N, s0_gate, s1_S, ...
    Outputs order: h_out, s0_S_out, s0_M_out, s0_N_out, s0_gate_out, s1_S_out, ...
    """
    def __init__(self, encoder, n_layers, d_model_hint=None):
        super().__init__()
        self.encoder = encoder
        self.n_layers = int(n_layers)
        self.d_model_hint = int(d_model_hint) if d_model_hint is not None else None

def forward(self, x_t, *state_inputs):
        # Reconstruct list-of-tuples states from flattened state_inputs
        states = []
        idx = 0
        for li in range(self.n_layers):
            S = state_inputs[idx]; idx += 1
            M_pref = state_inputs[idx]; idx += 1
            N_pref = state_inputs[idx]; idx += 1
            gate_buf = state_inputs[idx]; idx += 1

            # Heuristic: if gate_buf has swapped dims (1, d_model, pad) vs (1, pad, d_model), try permute
            try:
                if self.d_model_hint is not None and gate_buf.dim() == 3:
                    a, b, c = gate_buf.shape
                    # if second dim not hint but third equals hint -> permute
                    if b != self.d_model_hint and c == self.d_model_hint:
                        gate_buf = gate_buf.permute(0, 2, 1).contiguous()
            except Exception:
                pass
 states.append((S, M_pref, N_pref, gate_buf))

        # Call encoder.forward_step with tuple-states (matching training runtime)
        h_t, new_states = self.encoder.forward_step(x_t, states)

        # new_states may be list-of-tuples or list-of-dicts; normalize to tuple order
        outputs = [h_t]
        for st in new_states:
            if isinstance(st, dict):
                outputs.append(st["S"])
                outputs.append(st.get("M_pref", st.get("M", None)))
                outputs.append(st.get("N_pref", st.get("N", None)))
                outputs.append(st.get("gate_buf", st.get("gate_buf", st.get("gate", None))))
            elif isinstance(st, (list, tuple)):
                # assume len == 4
                outputs.append(st[0])
                outputs.append(st[1])
                outputs.append(st[2])
                outputs.append(st[3])
            else:
                # unknown format -- attempt to raise clearly
                 raise RuntimeError(f"Encoder returned unexpected state format: {type(st)}")
        return tuple(outputs)
def build_head_module_from_classifier(classifier):
    ln = torch.nn.LayerNorm(classifier.head[0].normalized_shape)
    linear = torch.nn.Linear(classifier.head[2].in_features, classifier.head[2].out_features, bias=True)
    with torch.no_grad():
        ln.weight.copy_(classifier.head[0].weight)
        ln.bias.copy_(classifier.head[0].bias)
        linear.weight.copy_(classifier.head[2].weight)
        linear.bias.copy_(classifier.head[2].bias)
    return torch.nn.Sequential(ln, linear)
# ---------------- tests ----------------
def test_layers_forward_step_with_tuples(encoder, init_states_tuples, device):
    """
    Call each layer.forward_step with the tuple-form layer_state to catch errors early.
    Returns (True, "") on success else (False, diagnostic_msg).
    """
    print("[test] Running per-layer forward_step with tuple states (diagnostics):")
    # create a dummy h with the encoder embed dim if available else fallback
    if hasattr(encoder, "embed") and hasattr(encoder.embed, "out_features"):
        h = torch.zeros((1, int(encoder.embed.out_features)), dtype=torch.float32, device=device)
    else:
        # fallback small tensor
        h = torch.zeros((1, 1), dtype=torch.float32, device=device)

    states = list(init_states_tuples)
 for i, layer in enumerate(encoder.layers):
        s_tuple = states[i]
        print(f"[test] Layer {i}: state tuple shapes:", tuple((tuple(x.shape) for x in s_tuple)))
        try:
            # pass tuple-form into layer.forward_step
            h, updated = layer.forward_step(h, s_tuple)
        except Exception as e:
            tb = traceback.format_exc()
            return False, f"[test] Layer {i} forward_step FAILED: {e}\n{tb}"

# accept updated either as tuple/list of 4 or dict
        if isinstance(updated, dict):
            # convert to tuple for next layer use if needed
            try:
                u = (updated["S"], updated.get("M_pref", updated.get("M")), updated.get("N_pref", updated.get("N")), updated.get("gate_buf", updated.get("gate")))
                states[i] = u
            except Exception:
                return False, f"[test] Layer {i} returned dict but couldn't extract S/M/N/gate"
        elif isinstance(updated, (list, tuple)):
            if len(updated) == 4:
                states[i] = tuple(updated)
            else:
                return False, f"[test] Layer {i} returned tuple/list length {len(updated)} (expected 4)"
        else:
            return False, f"[test] Layer {i} returned unexpected type {type(updated)}"
    return True, ""
# ---------------- exporter ----------------
def export_streaming_onnx(ckpt, out_prefix, params, device="cpu", smoke_test=False):
    device = torch.device(device)
    classifier = TSHMClassifier(input_dim=params["n_mfcc"], n_classes=params["n_classes"],
                                d_model=params["d_model"], n_layers=params["n_layers"],
                                r=params["r"], K=params["K"], ff_hidden=params["ff_hidden"],
                                use_pos=False, dropout=0.0, causal=True)
    classifier.to(device)
    classifier.eval()

sd = _load_checkpoint_file(ckpt, map_location=device)
    try:
        classifier.load_state_dict(sd)
        print("[export] loaded checkpoint (strict)")
    except Exception:
        res = classifier.load_state_dict(sd, strict=False)
        print("[export] loaded checkpoint (non-strict). missing_keys:", getattr(res, "missing_keys", None))

 # call init_stream_state and canonicalize to tuples
    try:
        try:
            raw_states = classifier.encoder.init_stream_state(batch_size=1, device=device)
        except TypeError:
            # some implementations don't accept device kw
            raw_states = classifier.encoder.init_stream_state(batch_size=1)
        print("[export] raw init_stream_state repr (first 1500 chars):")
        print(repr(raw_states)[:1500])
        init_states = canonicalize_init_states_to_tuples(raw_states, classifier.encoder, device)
        print("[export] canonicalized init states to tuple form successfully.")

except Exception as e:
        print("[export] canonicalization FAILED:", e)
        traceback.print_exc()
        # fallback: derive shapes and zero init
        shapes = derive_state_shapes_from_encoder(classifier.encoder, batch_size=1)
        init_states = []
        for st in shapes:
            S = torch.zeros(st["S"], dtype=torch.float32, device=device)
            M = torch.zeros(st["M_pref"], dtype=torch.float32, device=device)
            N = torch.zeros(st["N_pref"], dtype=torch.float32, device=device)
            G = torch.zeros(st["gate_buf"], dtype=torch.float32, device=device)
            init_states.append((S, M, N, G))
        print("[export] Built zero init states from derived shapes.")

 # run per-layer tuple-based tests BEFORE exporting
    ok, msg = test_layers_forward_step_with_tuples(classifier.encoder, init_states, device)
    if not ok:
        print("[export] Per-layer test failed. Diagnostics follow:")
        print(msg)
        raise RuntimeError("Per-layer forward_step check failed. See diagnostics above.")
# prepare dummy flattened state tensors for ONNX export (flat list in same ordering)
    dummy_states = []
    for (S, M, N, G) in init_states:
        dummy_states.append(S.clone().detach().to(dtype=torch.float32, device=device))
        dummy_states.append(M.clone().detach().to(dtype=torch.float32, device=device))
        dummy_states.append(N.clone().detach().to(dtype=torch.float32, device=device))
        dummy_states.append(G.clone().detach().to(dtype=torch.float32, device=device))

    # build wrapper and export
    wrapper = StreamingEncoderWrapper(classifier.encoder, n_layers=params["n_layers"], d_model_hint=params.get("d_model", None))
    wrapper.eval()
# deterministic names
    input_names = ["x_t"]
    for li in range(params["n_layers"]):
        input_names += [f"s{li}_S", f"s{li}_M", f"s{li}_N", f"s{li}_gate"]
    output_names = ["h_out"]
    for li in range(params["n_layers"]):
        output_names += [f"s{li}_S_out", f"s{li}_M_out", f"s{li}_N_out", f"s{li}_gate_out"]

    x_dummy = torch.zeros((1, params["n_mfcc"]), dtype=torch.float32, device=device)
    encoder_path = Path(f"{out_prefix}_encoder_step.onnx")
    try:
        torch.onnx.export(wrapper, tuple([x_dummy] + dummy_states), str(encoder_path),
                          opset_version=14, input_names=input_names, output_names=output_names, dynamic_axes=None)
        print("[export] Written encoder ONNX:", encoder_path)
    except Exception as e:
        print("[export] Failed to export encoder ONNX:", e)
        traceback.print_exc()
        raise

# export head
    head_mod = build_head_module_from_classifier(classifier)
    head_mod.eval()
    head_in = torch.zeros((1, params["d_model"]), dtype=torch.float32, device=device)
    head_path = Path(f"{out_prefix}_head.onnx")
    try:
        torch.onnx.export(head_mod, head_in, str(head_path), opset_version=14,
                          input_names=["pooled"], output_names=["logits"])
        print("[export] Written head ONNX:", head_path)
    except Exception as e:
        print("[export] Failed to export head ONNX:", e)
        traceback.print_exc()
        raise
desc = {
        "encoder": str(encoder_path.name),
        "head": str(head_path.name),
        "n_layers": params["n_layers"],
        "d_model": params["d_model"],
        "r": params["r"],
        "n_mfcc": params["n_mfcc"],
        "input_names": input_names,
        "output_names": output_names
    }
    Path(f"{out_prefix}_stream_desc.json").write_text(json.dumps(desc, indent=2))
    print("[export] Written descriptor:", f"{out_prefix}_stream_desc.json")

    # optional smoke test with onnxruntime (few frames)
    if smoke_test:
        if ort is None:
            print("[smoke] onnxruntime not installed; skipping smoke test.")
            return
        try:
            print("[smoke] Running small PyTorch vs ONNX smoke test (4 frames)...")
            sess = ort.InferenceSession(str(encoder_path), providers=["CPUExecutionProvider"])
            enc_in_names = [i.name for i in sess.get_inputs()]
            enc_out_names = [o.name for o in sess.get_outputs()]

            # build state numpy arrays for onnx runtime from init_states
            onnx_states = []
            for (S, M, N, G) in init_states:
                onnx_states += [S.cpu().numpy(), M.cpu().numpy(), N.cpu().numpy(), G.cpu().numpy()]
                torch.manual_seed(0)
            test_frames = [torch.randn((1, params["n_mfcc"]), dtype=torch.float32) for _ in range(4)]

            # PyTorch streaming outputs
            pt_states = list(init_states)
            pt_outs = []
            for f in test_frames:
                out_t, pt_states = classifier.encoder.forward_step(f.to(device), pt_states)
                # detach before converting to numpy to avoid gradient-related error
                pt_outs.append(out_t.detach().cpu().numpy())
              
            # ONNX streaming
            or_states = [s.copy() for s in onnx_states]
            onnx_outs = []
            for f in test_frames:
                feed = {enc_in_names[0]: f.cpu().numpy()}
                for i, nm in enumerate(enc_in_names[1:]):
                    feed[nm] = or_states[i]
                outs = sess.run(enc_out_names, feed)
                h_out = outs[0]
                onnx_outs.append(h_out)
                for i in range(len(or_states)):
                    or_states[i] = outs[1 + i].astype(np.float32)
            diffs = [float(np.max(np.abs(p - o))) for p, o in zip(pt_outs, onnx_outs)]
            print("[smoke] per-frame max abs diffs (PyTorch vs ONNX):", diffs)
        except Exception as e:
            print("[smoke] smoke test failed:", e)
            traceback.print_exc()

print("[export] Completed exports.")


def main():
    ap = argparse.ArgumentParser(description="Export streaming ONNX from checkpoint (robust/tuple-state variant)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_prefix", default="tshm_stream")
    ap.add_argument("--n_mfcc", type=int, default=40)
    ap.add_argument("--n_classes", type=int, default=10)
    ap.add_argument("--d_model", type=int, default=48)
    ap.add_argument("--n_layers", type=int, default=1)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--ff_hidden", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--smoke_test", action="store_true")
    args = ap.parse_args()

 params = {
        "n_mfcc": args.n_mfcc,
        "n_classes": args.n_classes,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "r": args.r,
        "K": args.K,
        "ff_hidden": args.ff_hidden
    }
    export_streaming_onnx(args.ckpt, args.out_prefix, params, device=args.device, smoke_test=args.smoke_test)
if __name__ == "__main__":
    main()

#python3 export_streaming_onnx.py --ckpt best_tshm_mfcc.pth --n_mfcc 40 --n_classes 10 --d_model 48 --n_layers 1 --r 16 --K 8 --ff_hidden 32

#python3 export_streaming_onnx.py --ckpt best_tshm_mfcc.pth --out_prefix tshm_stream --smoke_test
