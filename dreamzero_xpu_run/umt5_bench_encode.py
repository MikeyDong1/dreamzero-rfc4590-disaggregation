#!/usr/bin/env python
"""
Measure per-prompt UMT5 encode latency for a GGUF encoder, matching DreamZero
serving (bf16 weights + bf16 compute on XPU). Excludes weight-load time and warms
up the device (first call JITs kernels) before timing. Each call is XPU-synced so
we measure real device time, not async dispatch.

Reports per-prompt mean/median/p95 over N timed prompts (default: the whole file).
"""
import argparse, json, time, os
import torch


def log(m):
    print(f"[umt5-bench] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--device", default="xpu", choices=["xpu", "cpu"])
    ap.add_argument("--warmup", type=int, default=5, help="untimed warmup encodes")
    ap.add_argument("--repeat", type=int, default=1,
                    help="times to loop the whole prompt set (more = tighter stats)")
    ap.add_argument("--out", default="", help="optional json to save timings")
    args = ap.parse_args()

    from transformers import UMT5EncoderModel, AutoTokenizer

    dev = args.device
    if dev == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        log("WARNING: xpu unavailable; using cpu"); dev = "cpu"
    device = torch.device(dev, 0) if dev == "xpu" else torch.device("cpu")

    def sync():
        if dev == "xpu":
            torch.xpu.synchronize()

    gdir, gfile = os.path.dirname(args.gguf), os.path.basename(args.gguf)
    log(f"loading tokenizer + weights ({gfile}) -> bf16 on {dev} ...")
    tok = AutoTokenizer.from_pretrained(gdir, gguf_file=gfile)
    model = UMT5EncoderModel.from_pretrained(gdir, gguf_file=gfile, torch_dtype=torch.float32)
    model = model.to(dtype=torch.bfloat16, device=device)
    model.eval()

    prompts = json.load(open(args.prompts))
    log(f"{len(prompts)} prompts | warmup={args.warmup} | repeat={args.repeat}")

    # Pre-tokenize (tokenization is CPU and not what we're timing; serving tokenizes
    # too, but we isolate the ENCODE step which is the device-bound cost).
    toks = []
    for p in prompts:
        enc = tok(p["prompt"], max_length=args.max_length, padding="max_length",
                  truncation=True, return_tensors="pt")
        toks.append((enc["input_ids"].to(device), enc["attention_mask"].to(device)))

    def encode(ids, attn):
        out = model(ids, attention_mask=attn).last_hidden_state
        emb = out.clone().to(dtype=torch.bfloat16)
        return emb

    # ---- warmup (untimed): first calls JIT kernels / allocate caches ----
    with torch.no_grad():
        for i in range(args.warmup):
            ids, attn = toks[i % len(toks)]
            encode(ids, attn)
        sync()

    # ---- timed ----
    per_prompt_ms = []
    with torch.no_grad():
        for _ in range(args.repeat):
            for ids, attn in toks:
                sync()
                t0 = time.perf_counter()
                encode(ids, attn)
                sync()
                per_prompt_ms.append((time.perf_counter() - t0) * 1000.0)

    import statistics as st
    def pct(v, p):
        s = sorted(v); k = (len(s)-1)*p/100.0; lo=int(k); hi=min(lo+1,len(s)-1)
        return s[lo] + (s[hi]-s[lo])*(k-lo)
    res = {
        "label": args.label, "gguf": gfile, "device": dev, "compute_dtype": "bfloat16",
        "max_length": args.max_length, "n_timed": len(per_prompt_ms),
        "warmup": args.warmup, "repeat": args.repeat,
        "per_prompt_ms": {
            "mean": round(st.mean(per_prompt_ms), 3),
            "median": round(pct(per_prompt_ms, 50), 3),
            "p95": round(pct(per_prompt_ms, 95), 3),
            "min": round(min(per_prompt_ms), 3),
            "max": round(max(per_prompt_ms), 3),
            "stdev": round(st.pstdev(per_prompt_ms), 3),
        },
    }
    log("==== RESULT ====")
    log(f"label={args.label} gguf={gfile} device={dev} bf16  n_timed={res['n_timed']}")
    pp = res["per_prompt_ms"]
    log(f"per-prompt encode: mean {pp['mean']:.2f} ms | median {pp['median']:.2f} ms | "
        f"p95 {pp['p95']:.2f} ms | min {pp['min']:.2f} | max {pp['max']:.2f} | sd {pp['stdev']:.2f}")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)
        log(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
