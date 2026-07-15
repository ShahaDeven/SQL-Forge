"""
SQLForge Phase 5 — cost per query: self-hosted fine-tune vs the Claude API.

"54x cheaper" is the headline every self-hosting writeup reaches for, and on its own
it's misleading: it only holds when the GPU is saturated. A rented GPU bills by the
hour whether or not it's serving anything, so cost/query is a function of VOLUME.
The API is the opposite — pure per-token, zero idle cost. So the honest artifact
isn't a ratio, it's a BREAK-EVEN: above N queries/day self-hosting wins; below it,
the API does, and the GPU is burning money on idle.

Every input here is measured, not assumed:
  - throughput  : `generation_meta` (wall_s, gen_tokens) recorded by phase0/run_baseline.py
  - Claude tokens: Anthropic's count_tokens endpoint on the SAME 55 prompts, cached to
                   token_counts.json so this runs without an API key. Claude's tokenizer
                   is NOT Qwen's — Sonnet 5 counts 1394 input tokens where Qwen counts
                   744 (~1.9x), so substituting Qwen counts would understate API cost by
                   nearly half. Re-measure with --refresh-tokens if the prompt changes.

Run (offline, uses the cached counts):
    python sqlforge/eval/cost_analysis.py
    python sqlforge/eval/cost_analysis.py --gpu-rate 0.44   # community-cloud A40
"""

import os
import json
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "phase0", "results")
CACHE_PATH = os.path.join(BASE_DIR, "token_counts.json")

# --- API list prices, $/million tokens (docs.claude.com, 2026-07). Sonnet 5 also has an
#     introductory $2/$10 rate through 2026-08-31; both are reported below. ---
API_PRICING = {
    "claude-sonnet-5": {"in": 3.00, "out": 15.00, "label": "Claude Sonnet 5"},
    "claude-sonnet-5-intro": {"in": 2.00, "out": 10.00, "label": "Claude Sonnet 5 (intro)"},
    "claude-haiku-4-5": {"in": 1.00, "out": 5.00, "label": "Claude Haiku 4.5"},
}

# RunPod A40 (48GB) Secure Cloud list price at time of writing. This is the single
# biggest lever in the whole analysis — override it for your own provider/rate.
DEFAULT_GPU_RATE = 0.79

LOCAL_MODELS = {
    "SFT-7B (74.5%)": "sft_qwen7b_zeroshot.json",
    "SFT-1.5B (70.9%)": "eval_1p5b_sft.json",
}
N_QUERIES = 55


def local_throughput(fname):
    """Queries/hour from the measured wall-clock of the 55-query suite (vLLM, batched)."""
    with open(os.path.join(RESULTS_DIR, fname), encoding="utf-8") as f:
        meta = json.load(f)["summary"]["generation_meta"]
    qps = N_QUERIES / meta["wall_s"]
    return qps * 3600, meta


def api_cost_per_query(counts, pricing):
    return counts["in"] / 1e6 * pricing["in"] + counts["out"] / 1e6 * pricing["out"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-rate", type=float, default=DEFAULT_GPU_RATE,
                    help=f"GPU $/hour (default {DEFAULT_GPU_RATE}, RunPod A40 Secure Cloud).")
    args = ap.parse_args()

    with open(CACHE_PATH, encoding="utf-8") as f:
        counts = json.load(f)["models"]

    # --- self-hosted: fixed $/hour spread over however many queries you actually serve ---
    local = {}
    for name, fname in LOCAL_MODELS.items():
        qph, meta = local_throughput(fname)
        local[name] = {
            "queries_per_hour": qph,
            "cost_per_1k_saturated": args.gpu_rate / qph * 1000,
            "out_tokens": meta["gen_tokens"] / N_QUERIES,
        }

    # --- API: pure per-token, no idle cost ---
    api = {}
    for key, pricing in API_PRICING.items():
        c = counts[key if key in counts else "claude-sonnet-5"]
        api[key] = {
            "label": pricing["label"],
            "cost_per_query": api_cost_per_query(c, pricing),
            "in": c["in"], "out": c["out"],
        }

    w = 78
    print("=" * w)
    print(f"COST PER QUERY - self-hosted vs API   (GPU ${args.gpu_rate:.2f}/hr)")
    print("=" * w)
    print(f"{'Option':28s}{'tokens in/out':>18s}{'$/1k queries':>16s}{'vs Sonnet 5':>14s}")
    print("-" * w)

    sonnet = api["claude-sonnet-5"]["cost_per_query"] * 1000
    for name, d in local.items():
        ratio = sonnet / d["cost_per_1k_saturated"]
        print(f"{name:28s}{'- / ' + str(round(d['out_tokens'])):>18s}"
              f"{'$' + format(d['cost_per_1k_saturated'], '.3f'):>16s}{format(ratio, '.0f') + 'x cheaper':>14s}")
    for key in ("claude-haiku-4-5", "claude-sonnet-5-intro", "claude-sonnet-5"):
        d = api[key]
        c1k = d["cost_per_query"] * 1000
        rel = "-" if key == "claude-sonnet-5" else f"{sonnet / c1k:.1f}x cheaper"
        print(f"{d['label']:28s}{str(round(d['in'])) + ' / ' + str(round(d['out'])):>18s}"
              f"{'$' + format(c1k, '.2f'):>16s}{rel:>14s}")
    print("-" * w)
    print("  Self-hosted $/1k assumes a SATURATED GPU - the best case, not the usual one.")

    # --- the honest number: at what volume does the GPU stop being wasted? ---
    # Break-even depends ONLY on (GPU $/hr) / (API $/query) - NOT on local throughput.
    # Throughput sets CAPACITY (can the GPU physically serve that volume?), not the
    # crossover point. Reporting it per local-model pair would imply a relationship
    # that doesn't exist, so it's reported per API model, with capacity separately.
    print("\n" + "=" * w)
    print("BREAK-EVEN - self-hosting only wins above this volume")
    print("=" * w)
    print(f"{'vs API model':24s}{'queries/hour':>16s}{'queries/day':>16s}")
    print("-" * w)
    breakeven = {}
    for key in ("claude-sonnet-5", "claude-haiku-4-5"):
        qph = args.gpu_rate / api[key]["cost_per_query"]
        breakeven[key] = {"label": api[key]["label"],
                          "queries_per_hour": qph, "queries_per_day": qph * 24}
        print(f"{api[key]['label']:24s}{format(qph, ',.0f'):>16s}{format(qph * 24, ',.0f'):>16s}")
    print("-" * w)
    print("  Below these volumes the API is cheaper - a rented GPU bills while idle.")

    print("\n" + "=" * w)
    print("CAPACITY - can the GPU clear its own break-even?")
    print("=" * w)
    print(f"{'Self-hosted model':24s}{'max queries/day':>18s}{'util @ Sonnet-5 B/E':>22s}")
    print("-" * w)
    be_qph = breakeven["claude-sonnet-5"]["queries_per_hour"]
    for name, d in local.items():
        cap = d["queries_per_hour"] * 24
        util = be_qph / d["queries_per_hour"] * 100
        d["max_queries_per_day"] = cap
        d["util_at_breakeven_pct"] = util
        print(f"{name:24s}{format(cap, ',.0f'):>18s}{format(util, '.1f') + '%':>22s}")
    print("-" * w)
    print("  Break-even needs only a sliver of the GPU, so any real production volume")
    print("  clears it easily - the headroom above break-even is where the savings live.")
    print("=" * w)

    out = {
        "gpu_rate_per_hour": args.gpu_rate,
        "self_hosted": local,
        "api": api,
        "break_even": breakeven,
    }
    dest = os.path.join(BASE_DIR, "figures", "cost_analysis.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n  detail -> {dest}")


if __name__ == "__main__":
    main()
