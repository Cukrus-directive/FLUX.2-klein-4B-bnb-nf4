---
license: apache-2.0
base_model: black-forest-labs/FLUX.2-klein-4B
base_model_relation: quantized
library_name: diffusers
pipeline_tag: image-to-image
tags:
- flux2
- flux2-klein
- image-editing
- text-to-image
- bitsandbytes
- nf4
- 4-bit
- jetson
---

# FLUX.2-klein-4B NF4

[FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) (commit
`e7b7dc27f91d`) with the transformer quantized to bitsandbytes NF4. Loads with
`Flux2KleinPipeline.from_pretrained`. Image editing with 1-3 reference images, and
text-to-image.

| | bf16 | NF4 |
|---|---|---|
| transformer | 7.75 GB | 2.29 GB |
| VAE | 168 MB | 168 MB (unchanged) |

Text encoder (Qwen3-4B) not included. Load it from the base repo, or pass `prompt_embeds`.

![NF4 vs bf16](assets/nf4-vs-bf16.png)

## Quantization

- NF4, double quant, bf16 compute
- kept in bf16 (~0.37 GB): `x_embedder`, `context_embedder`, `time_guidance_embed`,
  `double_stream_modulation_img`, `double_stream_modulation_txt`, `single_stream_modulation`,
  `norm_out`, `proj_out`
- VAE and scheduler unchanged

`klein-quantize.py` re-shards the single 7.75 GB file into 1 GB shards before quantizing, so
peak memory stays around 2.2 GB. Runs on an 8 GB Jetson.

## Sanity check vs bf16

Rendered with the same seed, initial noise and prompt embeddings in NF4 and bf16.

| | PSNR | mean abs diff |
|---|---|---|
| edit, car to red | 24.5 dB | 9.2 / 255 |
| text-to-image, fox | 21.0 dB | 15.3 / 255 |

## Examples

4 steps, generated on a Jetson Orin Nano 8GB.

| source | "change the car's paint to glossy bright red, keep everything else the same" | "add a golden retriever sitting on the sidewalk next to the car" |
|---|---|---|
| ![](assets/source-car.png) | ![](assets/car-red-ref1024.png) | ![](assets/car-dog-ref768.png) |

| source | "turn this photo into a loose watercolor painting" | text-to-image: "a red fox standing in fresh snow, golden hour, photorealistic" |
|---|---|---|
| ![](assets/source-fox.png) | ![](assets/fox-watercolor-ref768.png) | ![](assets/t2i-fox.png) |

## Usage

Requires diffusers 0.37+ and bitsandbytes with kernels for your GPU (on Jetson, build it from
source with `-DCOMPUTE_CAPABILITY=87`).

```python
import torch
from diffusers import Flux2KleinPipeline
from diffusers.utils import load_image
from transformers import AutoTokenizer, Qwen3ForCausalLM

base = "black-forest-labs/FLUX.2-klein-4B"
text_encoder = Qwen3ForCausalLM.from_pretrained(base, subfolder="text_encoder", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(base, subfolder="tokenizer")

pipe = Flux2KleinPipeline.from_pretrained(
    "cukrus/FLUX.2-klein-4B-bnb-nf4",
    text_encoder=text_encoder, tokenizer=tokenizer, torch_dtype=torch.bfloat16,
).to("cuda")
pipe.vae.enable_tiling()

out = pipe(prompt="change the car's paint to glossy bright red, keep everything else the same",
           image=load_image("car.png"),  # or a list of up to 3
           num_inference_steps=4, guidance_scale=1.0,
           generator=torch.Generator("cuda").manual_seed(4200)).images[0]
out.save("edit.png")
```

Use `num_inference_steps=4, guidance_scale=1.0`. The pipeline defaults (50 / 4.0) are for the
base model.

### Without the text encoder

Pass `prompt_embeds` of shape (1, 512, 7680): Qwen3-4B hidden states 9, 18, 27 stacked,
thinking off, right-padded to 512. Keep the pad positions, there is no text mask.
Remote encoder service:
[qwen3-4b-remote-text-encoder](https://huggingface.co/cukrus/qwen3-4b-remote-text-encoder).

## Jetson Orin Nano 8GB

JetPack 7.2, CUDA 13.2, torch 2.12.1+cu132, diffusers 0.39. 1024x1024, 4 steps, VAE tiled at
512px, text encoder on a separate machine.

| | reference | denoise | total | torch peak | system RAM free (min) |
|---|---|---|---|---|---|
| text-to-image | - | 24.9 s | 37.2 s | 2.95 GB | 2.54 GB |
| edit | 512px | 31.1 s | 43.5 s | 3.09 GB | 2.37 GB |
| edit | 768px | 39.6 s | 44-52 s | 3.25 GB | 1.83-2.26 GB |
| edit | 1024px | 52.2 s | 57.9 s | 3.46 GB | 1.85 GB |

Total includes ~8 s encoding for a new prompt. Each reference adds as many tokens as the
output (1024px = 4096). References below 1 MP are faster but the framing drifts slightly.

## License

Apache-2.0, same as [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
by Black Forest Labs.
