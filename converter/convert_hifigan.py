"""NSF-HiFiGAN (mini_nsf) checkpoint -> GGUF converter.

Architecture (modules/nsf_hifigan/models.py Generator, mini_nsf=True):
  source = fastsinegen(f0)  [libmininsf host-side; not stored in GGUF]
  x = conv_pre(Conv1d 128->512 k7 p3)
  for i in 5 ups (rates [8,8,2,2,2], kernels [16,16,4,4,4]):
      x = LReLU(x); x = ups[i](ConvTranspose1d); if i==1: x += source_conv(source)
      x = mean(resblocks[i*3..i*3+2](x))          # ResBlock1: 3x dilated Conv1d pairs
  x = LReLU(x); x = conv_post(Conv1d 16->1 k7 p3); x = tanh(x)

Tensor layout matches ggml conv ops (gguf-py reverses the numpy shape list
into ggml ne order, so ne0 = K = the last numpy dim):
  convs   Conv1d [OC,IC,K]        -> ggml kernel [K,IC,OC]      (ne0=K)
  ups     stored as sub-pixel conv1d `hifigan.upsub.N` (phase-major, tiled bias)
          -> ggml raw [Cout*s, Cin, M]; engine does graph interleave
  source  Conv1d 1->256 k1        -> [1,1,256]

Quantization: this component is NOT quantized (per decision).  The converter
offers F32 (exact golden; weights and compute in fp32) and, reserved for
future fp16/bf16-trained checkpoints, an F16 line (kernels fp16, bias stays
F32).  See the engine's Precision class for how each line is executed.

Only third-party deps are torch + gguf (Python package) + numpy.
"""
import argparse
import json
import os
from typing import Optional

import numpy as np
import torch

from gguf import GGUFWriter
from gguf.constants import GGMLQuantizationType as QT


# Official OpenVPI SingingVocoders v1.0.0 preset.  The release archive ships
# the checkpoint and notices but no config.json, so keeping the published
# architecture here makes that asset directly convertible.  Unknown
# checkpoints still require an explicit --config; audio rates are not safely
# inferable from tensor shapes alone.
OFFICIAL_PRESETS = {
    "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02": {
        "sampling_rate": 44100,
        "hop_size": 512,
        "num_mels": 128,
        "upsample_initial_channel": 512,
        "upsample_rates": [8, 8, 2, 2, 2],
        "upsample_kernel_sizes": [16, 16, 4, 4, 4],
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "mini_nsf": True,
        "noise_sigma": 0.01,
    },
}


def materialize_weight_norm(sd: dict):
    """hifigan stores weight_norm as (weight_g, weight_v); synthesize weight = g * v/||v||."""
    keys = list(sd.keys())
    for k in keys:
        if k.endswith(".weight_g"):
            base = k[: -len("weight_g")]
            vk = base + "weight_v"
            if vk not in sd:
                continue
            g = sd[k].numpy().astype(np.float32)
            v = sd[vk].numpy().astype(np.float32)
            axes = tuple(range(1, v.ndim))
            norm = np.sqrt((v ** 2).sum(axis=axes, keepdims=True)) + 1e-12
            sd[base + "weight"] = torch.from_numpy((g * (v / norm)).astype(np.float32))
            del sd[k]
            del sd[vk]
    return True


def load_generator_state(obj: object) -> dict:
    """Accept native generator checkpoints and Lightning-style state_dicts."""
    if not isinstance(obj, dict):
        raise RuntimeError("checkpoint root must be a dict")
    if "generator" in obj:
        return dict(obj["generator"])
    if "state_dict" in obj:
        prefix = "generator."
        sd = {
            k[len(prefix):]: v
            for k, v in obj["state_dict"].items()
            if k.startswith(prefix)
        }
        if sd:
            return sd
    raise RuntimeError("checkpoint has neither 'generator' nor generator.* entries in 'state_dict'")


def load_config(ckpt_path: str, config_path: Optional[str]) -> dict:
    """Read explicit architecture metadata, or a recognized official preset."""
    if config_path:
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)
    stem = os.path.splitext(os.path.basename(ckpt_path))[0]
    if stem in OFFICIAL_PRESETS:
        print("using built-in config preset", stem)
        return dict(OFFICIAL_PRESETS[stem])
    raise RuntimeError(
        "--config is required for this checkpoint; automatic config is only "
        "available for: " + ", ".join(sorted(OFFICIAL_PRESETS))
    )


def conv1d_to_ggml(w: np.ndarray) -> np.ndarray:
    """PyTorch Conv1d [OC, IC, K] stored RAW (file ne0 = K last, matching
    ggml_conv_1d's [K, IC, OC] kernel layout)."""
    return np.ascontiguousarray(w, dtype=np.float32)


def convt1d_to_ggml(w: np.ndarray) -> np.ndarray:
    """PyTorch ConvTranspose1d [Cin, Cout, K] stored RAW (file ne0 = K,
    matching ggml_conv_transpose_1d's [K, Cout, Cin] layout)."""
    return np.ascontiguousarray(w, dtype=np.float32)


def deconv1d_as_subpixel(weight: np.ndarray, bias: np.ndarray, stride: int):
    """Split ConvTranspose1d into per-phase conv1d kernels (EXACT, numpy).

    Phase-major channel c = r*Cout + o (r = phase, o = output channel).  Bias
    must be TILED (bias.repeat(s), concatenated) so channel r*Cout+o gets
    b[o] -- np.repeat (interleave) is wrong and silently mismatches torch.
    """
    Cin, Cout, K = weight.shape
    s, M = stride, (K + stride - 1) // stride
    W_sub = np.zeros((Cin, Cout * s, M), dtype=np.float32)
    for r in range(s):
        sl = slice(r * Cout, (r + 1) * Cout)
        for m in range(M):
            k = r + s * m
            if k < K:
                W_sub[:, sl, (M - 1) - m] = weight[:, :, k]
    b_sub = np.tile(bias, s).astype(np.float32)
    return W_sub, b_sub


def subpixel_to_ggml(ws: np.ndarray) -> np.ndarray:
    """Phase-major [Cin, r*Cout+o, M] -> GGUF raw [Cout*s, Cin, M]
    (file ne0 = M taps, ne1 = Cin, ne2 = Cout*s)."""
    return np.ascontiguousarray(ws.transpose(1, 0, 2), dtype=np.float32)


def write_gguf(path: str, arch: str, tensors: dict, meta: dict, dtype: str):
    """Write GGUF metadata and kernels, retaining F32 biases in either precision."""
    writer = GGUFWriter(path, arch)
    for k, v in meta.items():
        if isinstance(v, str):
            writer.add_string(k, v)
        elif isinstance(v, bool):
            writer.add_bool(k, v)
        elif isinstance(v, int):
            writer.add_int32(k, v)
        elif isinstance(v, float):
            writer.add_float32(k, v)
        elif isinstance(v, (list, tuple)):
            writer.add_array(k, [float(x) for x in v])
    for name in sorted(tensors.keys()):
        w = tensors[name]
        # Bias stays F32 in every mode (engine ggml_add requires F32 bias;
        # making it F16 breaks binary_op).  Only kernels are quantized to F16.
        if name.endswith(".bias"):
            writer.add_tensor(name, w.astype(np.float32), raw_dtype=QT.F32)
        elif dtype == "F16":
            writer.add_tensor(name, w.astype(np.float16), raw_dtype=QT.F16)
        else:
            writer.add_tensor(name, w.astype(np.float32), raw_dtype=QT.F32)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main():
    """Convert a tensor checkpoint to the requested F32/F16 GGUF file."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="model.ckpt path")
    ap.add_argument("--config", help="config.json path (optional for a known official release)")
    ap.add_argument("--out", required=True, help="output .gguf path")
    ap.add_argument("--dtype", default="F32", choices=["F32", "F16"],
                    help="kernel dtype. F32 = exact golden default (weights AND compute in fp32); F16 = reserved for future fp16/bf16-trained checkpoints (bias always stays F32)")
    args = ap.parse_args()

    ckpt_dir = os.path.dirname(args.ckpt)
    cfg = load_config(args.ckpt, args.config)
    obj = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    sd = load_generator_state(obj)
    print("loading", ckpt_dir)

    materialize_weight_norm(sd)
    sd = {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in sd.items()}

    tensors = {}
    for k in sorted(sd.keys()):
        w = sd[k]
        if k.endswith("num_batches_tracked"):
            continue
        if k.endswith(".weight") and k.startswith("ups."):
            # Sub-pixel (phase-major) upsample instead of ConvTranspose1d:
            # a regular conv1d + graph interleave, which rides the fast F32
            # im2col+mul_mat path.  Exact vs torch (see docs/verification_*).
            idx = int(k.split(".")[1])
            s = int(cfg["upsample_rates"][idx])
            base = k[: -len(".weight")]
            bias = sd[base + ".bias"]
            Ws, bs = deconv1d_as_subpixel(w, bias, s)
            tensors[f"hifigan.upsub.{idx}.weight"] = subpixel_to_ggml(Ws)   # [CS,Cin,M]
            tensors[f"hifigan.upsub.{idx}.bias"]   = bs                     # [CS] (tiled)
        elif k.endswith(".weight"):
            tensors["hifigan." + k] = conv1d_to_ggml(w)    # [K,IC,OC]
        elif k.endswith(".bias"):
            tensors["hifigan." + k] = np.ascontiguousarray(w, dtype=np.float32)
        else:
            raise RuntimeError(f"unhandled key {k} {w.shape}")

    meta = {
        "audio.type": "hifigan",
        "audio.sample_rate": cfg["sampling_rate"],
        "audio.hop": cfg["hop_size"],
        "audio.n_mels": cfg["num_mels"],
        "hifigan.num_upsamples": len(cfg["upsample_rates"]),
        "hifigan.num_resblocks": 3 * len(cfg["upsample_rates"]),
        "hifigan.upsample_initial_channel": cfg["upsample_initial_channel"],
        "hifigan.upsample_rates": [float(x) for x in cfg["upsample_rates"]],
        "hifigan.upsample_kernels": [float(x) for x in cfg["upsample_kernel_sizes"]],
        "hifigan.resblock_kernels": [float(x) for x in cfg["resblock_kernel_sizes"]],
        "hifigan.resblock_dilations": [float(x) for x in cfg["resblock_dilation_sizes"][0]],
        "hifigan.mini_nsf": bool(cfg.get("mini_nsf", True)),
        "hifigan.subpixel": True,
        "hifigan.noise_sigma": float(cfg.get("noise_sigma", 0.0)),
        "hifigan.lrelu_slope": 0.1,
        "hifigan.source_sr": float(cfg["sampling_rate"] / np.prod(cfg["upsample_rates"][2:])),
        "hifigan.upp": int(np.prod(cfg["upsample_rates"][:2])),
    }
    write_gguf(args.out, "hifigan", tensors, meta, args.dtype)
    size = os.path.getsize(args.out)
    print(f"wrote {args.out}: {size/1e6:.2f} MB ({args.dtype})")


if __name__ == "__main__":
    main()
