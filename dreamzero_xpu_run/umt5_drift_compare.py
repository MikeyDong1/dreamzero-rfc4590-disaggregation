#!/usr/bin/env python
"""
Compare two saved UMT5 embedding sets (bf16 baseline vs Q8_0) and report how much
int8 weight quantization drifts the DiT-facing text embedding.

Metrics per prompt, computed over the REAL-TOKEN span only (padded positions are
zero in both and would inflate similarity):
  - cosine similarity (flattened real-token embedding), higher = closer
  - mean per-token cosine similarity
  - relative L2 error  ||q - b|| / ||b||
  - max abs elementwise diff
"""
import argparse, json
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="bf16 .pt")
    ap.add_argument("--test", required=True, help="q8_0 .pt")
    ap.add_argument("--out", required=True, help="json report")
    args = ap.parse_args()

    b = torch.load(args.baseline, weights_only=False)
    q = torch.load(args.test, weights_only=False)
    byid_b = {r["id"]: r for r in b["records"]}
    byid_q = {r["id"]: r for r in q["records"]}

    rows = []
    for i in sorted(byid_b):
        rb, rq = byid_b[i], byid_q[i]
        n = int(rb["n_real_tokens"])
        eb = rb["embedding_bf16"][:n].float()   # [n, 4096]
        eq = rq["embedding_bf16"][:n].float()
        vb, vq = eb.reshape(-1), eq.reshape(-1)
        cos_flat = torch.nn.functional.cosine_similarity(vb, vq, dim=0).item()
        cos_tok = torch.nn.functional.cosine_similarity(eb, eq, dim=1).mean().item()
        rel_l2 = (torch.linalg.vector_norm(vq - vb) /
                  torch.linalg.vector_norm(vb)).item()
        max_abs = (eq - eb).abs().max().item()
        rows.append({
            "id": i, "prompt": rb["prompt"], "n_real_tokens": n,
            "cosine_flat": round(cos_flat, 6),
            "mean_token_cosine": round(cos_tok, 6),
            "rel_l2_error": round(rel_l2, 6),
            "max_abs_diff": round(max_abs, 6),
        })

    import statistics as st

    def pct(vals, p):
        s = sorted(vals)
        if not s:
            return float("nan")
        k = (len(s) - 1) * (p / 100.0)
        lo = int(k); hi = min(lo + 1, len(s) - 1)
        return s[lo] + (s[hi] - s[lo]) * (k - lo)

    cos = [r["cosine_flat"] for r in rows]
    tcos = [r["mean_token_cosine"] for r in rows]
    l2 = [r["rel_l2_error"] for r in rows]
    agg = {
        "n_prompts": len(rows),
        "baseline": {"label": b["label"], "gguf": b["gguf"], "device": b.get("device"),
                     "compute_dtype": b.get("compute_dtype")},
        "test": {"label": q["label"], "gguf": q["gguf"], "device": q.get("device"),
                 "compute_dtype": q.get("compute_dtype")},
        "cosine_flat":   {"mean": round(st.mean(cos), 6), "min": round(min(cos), 6),
                          "p50": round(pct(cos, 50), 6), "p05": round(pct(cos, 5), 6)},
        "token_cosine":  {"mean": round(st.mean(tcos), 6), "min": round(min(tcos), 6)},
        "rel_l2_error":  {"mean": round(st.mean(l2), 6), "median": round(pct(l2, 50), 6),
                          "p95": round(pct(l2, 95), 6), "max": round(max(l2), 6),
                          "stdev": round(st.pstdev(l2), 6)},
        "max_abs_diff_any": round(max(r["max_abs_diff"] for r in rows), 6),
    }
    worst = sorted(rows, key=lambda r: r["cosine_flat"])[:10]
    best = sorted(rows, key=lambda r: -r["cosine_flat"])[:5]
    report = {"summary": agg,
              "worst10_by_cosine": [{k: r[k] for k in ("id", "prompt", "cosine_flat", "rel_l2_error")} for r in worst],
              "per_prompt": rows}
    json.dump(report, open(args.out, "w"), indent=2)

    print("=== UMT5 int8 (Q8_0) drift vs bf16 baseline — DiT text embedding ===")
    print(f"baseline: {agg['baseline']}")
    print(f"test:     {agg['test']}")
    print(f"n_prompts: {agg['n_prompts']}")
    print("-" * 72)
    print("cosine(flat)   mean {mean:.6f}  median {p50:.6f}  p05 {p05:.6f}  min {min:.6f}".format(**agg["cosine_flat"]))
    print("token cosine   mean {mean:.6f}  min {min:.6f}".format(**agg["token_cosine"]))
    print("rel-L2 error   mean {mean:.5f}  median {median:.5f}  p95 {p95:.5f}  max {max:.5f}  sd {stdev:.5f}".format(**agg["rel_l2_error"]))
    print(f"max |elem diff| (any prompt): {agg['max_abs_diff_any']:.6f}")
    print("-" * 72)
    print("worst 10 prompts by cosine(flat):")
    print(f"  {'id':>3} {'cos_flat':>10} {'rel_L2':>8}  prompt")
    for r in worst:
        print(f"  {r['id']:>3} {r['cosine_flat']:>10.6f} {r['rel_l2_error']:>8.5f}  {r['prompt']}")
    print("best 5 (for reference):")
    for r in best:
        print(f"  {r['id']:>3} {r['cosine_flat']:>10.6f} {r['rel_l2_error']:>8.5f}  {r['prompt']}")
    print("-" * 72)
    print(f"SAVED report -> {args.out}")


if __name__ == "__main__":
    main()
