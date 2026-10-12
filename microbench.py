#!/usr/bin/env python3
"""Controlled requests for calibrating alpha (per context token) and beta (per output token).

Each request has an exact context length (random token IDs, so no accidental prefix-cache hits)
and an exact output length (ignore_eos + min_tokens). Requests run one at a time with an idle
gap, in shuffled order, so per-call energy can be read from sampler.py's CSV.

With --cache-repeat each prompt is sent a second time right away (kind="hit") to measure
the energy of a fully cached prefix (input for the KV-cache discussion).

Usage:
  python microbench.py -o calls.jsonl
  python microbench.py -o calls.jsonl --ctx 1024,4096 --out 64,256 --reps 2   # quick smoke test
"""
import argparse
import json
import random
import time
import urllib.request


def post(url, payload, timeout=1800):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def ints(s):
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out-file", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default=None, help="default: first model from /v1/models")
    ap.add_argument("--ctx", type=ints, default=ints("1024,2048,4096,8192,16384"))
    ap.add_argument("--out", type=ints, default=ints("64,256,1024"))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--gap", type=float, default=3.0, help="idle seconds between requests")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--cache-repeat", action="store_true")
    ap.add_argument("--token-range", type=ints, default=ints("1000,100000"),
                    help="random token-id range; must avoid special tokens")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    model = args.model or get(base + "/models")["data"][0]["id"]
    rng = random.Random(args.seed)
    lo, hi = args.token_range

    def request(prompt, n_out):
        payload = {"model": model, "prompt": prompt, "max_tokens": n_out, "min_tokens": n_out,
                   "ignore_eos": True, "temperature": 0.0}
        t0m, t0w = time.monotonic(), time.time()
        r = post(base + "/completions", payload)
        t1m, t1w = time.monotonic(), time.time()
        return r["usage"], t0m, t1m, t0w, t1w

    print(f"model={model}; warmup x{args.warmup}", flush=True)
    for _ in range(args.warmup):
        request([rng.randrange(lo, hi) for _ in range(512)], 32)

    grid = [(c, o, r) for c in args.ctx for o in args.out for r in range(args.reps)]
    rng.shuffle(grid)
    t_begin = time.monotonic()
    with open(args.out_file, "w") as f:
        for i, (ctx, n_out, rep) in enumerate(grid):
            prompt = [rng.randrange(lo, hi) for _ in range(ctx)]
            for kind in (["miss", "hit"] if args.cache_repeat else ["miss"]):
                usage, t0m, t1m, t0w, t1w = request(prompt, n_out)
                f.write(json.dumps({
                    "kind": kind, "ctx_target": ctx, "out_target": n_out, "rep": rep,
                    "t_start_mono": t0m, "t_end_mono": t1m, "t_start_wall": t0w, "t_end_wall": t1w,
                    "prompt_tokens": usage["prompt_tokens"],
                    "completion_tokens": usage["completion_tokens"],
                    "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                }) + "\n")
                f.flush()
                time.sleep(args.gap)
            done = i + 1
            eta = (time.monotonic() - t_begin) / done * (len(grid) - done)
            print(f"[{done}/{len(grid)}] ctx={ctx} out={n_out} {t1m - t0m:.1f}s  eta {eta / 60:.1f} min",
                  flush=True)


if __name__ == "__main__":
    main()
