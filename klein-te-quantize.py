# qwen3-4b text encoder (klein / z-image) bf16 -> bitsandbytes nf4
#
# input:  ~/models/_qwen3-src/{text_encoder,tokenizer}
# output: ~/models/klein-te-nf4/{text_encoder,tokenizer}
#
# source shards are ~4gb, re-shard to 1gb first so ram stays low on the 8gb board
#
# python klein-te-quantize.py [--keep-shards]

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = Path.home() / "models" / "_qwen3-src"
DST = Path.home() / "models" / "klein-te-nf4"
SHARD_BYTES = 1 << 30


def mem(tag):
    with open("/proc/meminfo") as f:
        info = dict(line.split(":", 1) for line in f)
    avail = int(info["MemAvailable"].split()[0]) / 2**20
    print(f"[{tag}] MemAvailable {avail:.2f} GB, torch alloc "
          f"{torch.cuda.memory_allocated() / 2**30:.2f} GB, peak "
          f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)


def reshard(src_dir, out_dir):
    index_path = out_dir / "model.safetensors.index.json"
    if index_path.exists():
        print("shards exist, skipping re-shard")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "generation_config.json"):
        shutil.copy(src_dir / name, out_dir / name)
    src_index = json.loads((src_dir / "model.safetensors.index.json").read_text())
    files = sorted(set(src_index["weight_map"].values()))

    parts, cur, cur_bytes = [], {}, 0

    def flush():
        name = f"part-{len(parts) + 1:05d}.safetensors"
        save_file(cur, out_dir / name, metadata={"format": "pt"})
        parts.append((name, list(cur)))
        print(f"  wrote {name} ({cur_bytes / 1e9:.2f} GB)", flush=True)

    # one tensor at a time
    for fname in files:
        with safe_open(src_dir / fname, framework="pt") as f:
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
        final = f"model-{i:05d}-of-{len(parts):05d}.safetensors"
        (out_dir / name).rename(out_dir / final)
        weight_map.update({k: final for k in keys})
    index_path.write_text(json.dumps({"metadata": {}, "weight_map": weight_map}, indent=2))
    print(f"re-sharded into {len(parts)} files")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-shards", action="store_true")
    args = ap.parse_args()

    from transformers import BitsAndBytesConfig, Qwen3ForCausalLM

    sharded = SRC / "text_encoder-sharded"
    t0 = time.time()
    reshard(SRC / "text_encoder", sharded)
    print(f"re-shard {time.time() - t0:.0f} s")
    mem("before load")

    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.bfloat16)
    t0 = time.time()
    te = Qwen3ForCausalLM.from_pretrained(sharded, quantization_config=qcfg,
                                          dtype=torch.bfloat16, device_map={"": 0})
    print(f"load+quantize {time.time() - t0:.0f} s")
    mem("quantized")

    n4 = sum(1 for m in te.modules() if m.__class__.__name__ == "Linear4bit")
    nl = sum(1 for m in te.modules() if isinstance(m, torch.nn.Linear)
             and m.__class__.__name__ != "Linear4bit")
    print(f"Linear4bit: {n4}, bf16 Linear kept: {nl}")

    DST.mkdir(parents=True, exist_ok=True)
    te.save_pretrained(DST / "text_encoder")
    shutil.copytree(SRC / "tokenizer", DST / "tokenizer", dirs_exist_ok=True)
    size = sum(p.stat().st_size for p in DST.rglob("*") if p.is_file()) / 1e9
    print(f"saved {DST} ({size:.2f} GB)")

    if not args.keep_shards:
        shutil.rmtree(sharded)
        print("removed re-shard copy")


if __name__ == "__main__":
    main()
