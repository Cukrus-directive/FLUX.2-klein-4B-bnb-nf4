# flux.2-klein-4b bf16 transformer -> bitsandbytes nf4
#
# input:  ~/models/_klein-src  (transformer/, vae/, scheduler/, model_index.json from the bfl repo)
# output: ~/models/flux2-klein-4b-bnb4  (~2.5 gb)
#
# the original transformer is one 7.75gb file and diffusers loads each shard whole,
# so it gets re-sharded into 1gb pieces first. peak ram ~2.2gb.
#
# python klein-quantize.py [--keep-shards]

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = Path.home() / "models" / "_klein-src"
DST = Path.home() / "models" / "flux2-klein-4b-bnb4"
SHARD_BYTES = 1 << 30
# left in bf16: embedders, time embed, shared modulation, output layers
SKIP = ["x_embedder", "context_embedder", "time_guidance_embed",
        "double_stream_modulation_img", "double_stream_modulation_txt",
        "single_stream_modulation", "norm_out", "proj_out"]


def mem(tag):
    with open("/proc/meminfo") as f:
        info = dict(line.split(":", 1) for line in f)
    avail = int(info["MemAvailable"].split()[0]) / 2**20
    print(f"[{tag}] MemAvailable {avail:.2f} GB, torch alloc "
          f"{torch.cuda.memory_allocated() / 2**30:.2f} GB, peak "
          f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)


def reshard(src_file, out_dir):
    index_path = out_dir / "diffusion_pytorch_model.safetensors.index.json"
    if index_path.exists():
        print("shards exist, skipping re-shard")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_file.parent / "config.json", out_dir / "config.json")
    parts, cur, cur_bytes = [], {}, 0

    def flush():
        name = f"part-{len(parts) + 1:05d}.safetensors"
        save_file(cur, out_dir / name)
        parts.append((name, list(cur)))
        print(f"  wrote {name} ({cur_bytes / 1e9:.2f} GB)", flush=True)

    # one tensor at a time so the whole file is never in ram
    with safe_open(src_file, framework="pt") as f:
        for key in f.keys():
            t = f.get_tensor(key)
            nbytes = t.numel() * t.element_size()
            if cur and cur_bytes + nbytes > SHARD_BYTES:
                flush()
                cur, cur_bytes = {}, 0
            cur[key] = t
            cur_bytes += nbytes
        if cur:
            flush()
    cur = None
    weight_map = {}
    for i, (name, keys) in enumerate(parts, 1):
        final = f"diffusion_pytorch_model-{i:05d}-of-{len(parts):05d}.safetensors"
        (out_dir / name).rename(out_dir / final)
        weight_map.update({k: final for k in keys})
    # index written last so a half finished run gets redone
    index_path.write_text(json.dumps({"metadata": {}, "weight_map": weight_map}, indent=2))
    print(f"re-sharded into {len(parts)} files")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-shards", action="store_true")
    args = ap.parse_args()

    from diffusers import BitsAndBytesConfig, Flux2Transformer2DModel

    sharded = SRC / "transformer-sharded"
    t0 = time.time()
    reshard(SRC / "transformer" / "diffusion_pytorch_model.safetensors", sharded)
    print(f"re-shard {time.time() - t0:.0f} s")
    mem("before load")

    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              llm_int8_skip_modules=SKIP)
    t0 = time.time()
    tr = Flux2Transformer2DModel.from_pretrained(sharded, quantization_config=qcfg,
                                                 torch_dtype=torch.bfloat16)
    print(f"load+quantize {time.time() - t0:.0f} s")
    mem("quantized")

    n4 = sum(1 for m in tr.modules() if m.__class__.__name__ == "Linear4bit")
    nl = sum(1 for m in tr.modules() if isinstance(m, torch.nn.Linear)
             and m.__class__.__name__ != "Linear4bit")
    print(f"Linear4bit: {n4}, bf16 Linear kept: {nl}")

    DST.mkdir(parents=True, exist_ok=True)
    tr.save_pretrained(DST / "transformer")
    for sub in ("vae", "scheduler"):
        shutil.copytree(SRC / sub, DST / sub, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("*.ok"))
    # no text encoder in the output
    mi = json.loads((SRC / "model_index.json").read_text())
    mi["text_encoder"] = [None, None]
    mi["tokenizer"] = [None, None]
    (DST / "model_index.json").write_text(json.dumps(mi, indent=2))
    size = sum(p.stat().st_size for p in DST.rglob("*") if p.is_file()) / 1e9
    print(f"saved {DST} ({size:.2f} GB)")

    if not args.keep_shards:
        shutil.rmtree(sharded)
        print("removed re-shard copy")


if __name__ == "__main__":
    main()
