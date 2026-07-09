#!/usr/bin/env python
"""
Encode DreamZero-style text prompts through the city96 UMT5-xxl ENCODER GGUF and
save the resulting text embeddings (the exact tensor the DreamZero DiT consumes).

The encoding pipeline is copied 1:1 from vllm_omni DreamZeroPipeline so the output
is what the DiT would actually see:
  tokenizer(prompt, max_length=512, padding="max_length", truncation=True)
  -> UMT5EncoderModel(input_ids, attention_mask).last_hidden_state
  -> .clone().to(bfloat16)
  -> zero out padded positions ([:, seq_len:] = 0)

*** dtype matches REAL SERVING: bf16 weights + bf16 compute on XPU. ***
Verified (3 independent sources) that vLLM-Omni serving runs the UMT5 text encoder
in bfloat16 — OmniDiffusionConfig.dtype default = bf16, every dreamzero*.yaml sets
dtype: bfloat16, the checkpoint text_encoder tensors are BF16, and the module is
built inside set_default_torch_dtype(bf16) so params are bf16 with no autocast. So
we cast the dequantized GGUF weights to bf16 and run compute in bf16, to make the
ONLY variable between the F16 and Q8_0 runs the weight quantization.

Weights are dequantized from GGUF by transformers' native gguf_file= loader, then
cast to bf16.

Usage:
  python umt5_gguf_encode.py --gguf <path-to.gguf> --prompts prompts.json \
      --out out.pt --label bf16 --device xpu
"""
import argparse, json, time, os, sys
import torch


def log(m):
    print(f"[umt5-enc] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="path to umt5-xxl-encoder-*.gguf")
    ap.add_argument("--prompts", required=True, help="json list of {id, prompt}")
    ap.add_argument("--out", required=True, help="output .pt file")
    ap.add_argument("--label", default="", help="tag stored in metadata (e.g. bf16 / q8_0)")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--device", default="xpu", choices=["xpu", "cpu"],
                    help="compute device; serving uses xpu")
    args = ap.parse_args()

    from transformers import UMT5EncoderModel, AutoTokenizer

    dev = args.device
    if dev == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        log("WARNING: xpu requested but not available; falling back to cpu")
        dev = "cpu"
    device = torch.device(dev, 0) if dev == "xpu" else torch.device("cpu")

    gguf_dir = os.path.dirname(args.gguf)
    gguf_file = os.path.basename(args.gguf)

    # ---- tokenizer: load from the GGUF itself (google/umt5-xxl repo was deleted;
    #      the GGUF embeds the identical t5 sentencepiece vocab) ----
    log(f"loading tokenizer from GGUF {gguf_file} ...")
    tokenizer = AutoTokenizer.from_pretrained(gguf_dir, gguf_file=gguf_file)

    # ---- encoder weights: transformers dequantizes GGUF -> fp32, then cast bf16 ----
    log(f"loading UMT5EncoderModel weights from GGUF (dequant -> bf16) ...")
    t0 = time.perf_counter()
    model = UMT5EncoderModel.from_pretrained(gguf_dir, gguf_file=gguf_file,
                                             torch_dtype=torch.float32)
    # Match serving: bf16 params (weight rounding happens HERE), bf16 compute.
    model = model.to(dtype=torch.bfloat16, device=device)
    model.eval()
    log(f"weights loaded+cast in {time.perf_counter()-t0:.1f}s "
        f"(dtype={next(model.parameters()).dtype}, device={next(model.parameters()).device}, "
        f"n_layers={model.config.num_layers}, d_model={model.config.d_model})")

    prompts = json.load(open(args.prompts))
    log(f"encoding {len(prompts)} prompts (max_length={args.max_length}) on {dev} bf16 ...")

    records = []
    with torch.no_grad():
        for p in prompts:
            text = p["prompt"]
            enc = tokenizer(text, max_length=args.max_length,
                            padding="max_length", truncation=True,
                            return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            seq_len = int(attn.gt(0).sum(dim=1)[0])

            # --- exact DreamZero _encode_text math (bf16 in, bf16 out) ---
            out = model(input_ids, attention_mask=attn).last_hidden_state  # bf16
            emb = out.clone().to(dtype=torch.bfloat16)
            emb[:, seq_len:] = 0

            records.append({
                "id": p["id"],
                "prompt": text,
                "n_real_tokens": seq_len,
                "input_ids": input_ids[0].to(torch.int32).cpu(),
                # the DiT-facing bf16 embedding (this is exactly what serving feeds the DiT)
                "embedding_bf16": emb[0].float().cpu().to(torch.bfloat16),
            })
            log(f"  id={p['id']:>2}  '{text[:40]}'  real_tokens={seq_len}  "
                f"emb_shape={tuple(emb[0].shape)}")

    payload = {
        "label": args.label,
        "gguf": gguf_file,
        "max_length": args.max_length,
        "device": dev,
        "compute_dtype": "bfloat16",
        "store_dtype": "bfloat16(embedding_bf16)",
        "d_model": model.config.d_model,
        "records": records,
    }
    torch.save(payload, args.out)
    log(f"SAVED {len(records)} embeddings -> {args.out}")
    # tiny stat so the log is self-describing
    m0 = records[0]["embedding_bf16"][:records[0]["n_real_tokens"]].float()
    log(f"sanity: id=1 bf16 emb mean={m0.mean():.5f} std={m0.std():.5f} "
        f"absmax={m0.abs().max():.5f}")


if __name__ == "__main__":
    main()
