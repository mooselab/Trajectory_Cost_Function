#!/usr/bin/env python3
"""Check 2-GPU communication outside vLLM; every step times out after 60 s instead of 30 min.

Steps (each rank prints "ok" when a step finishes):
  1. gloo barrier        CPU-side sync (what vLLM hung on)
  2. nccl all_reduce     GPU collective
  3. p2p access          whether CUDA reports direct GPU<->GPU access
  4. p2p copy            an actual 4 MB GPU->GPU copy

Usage:
  .venv/bin/python dist_test.py
  NCCL_P2P_DISABLE=1 .venv/bin/python dist_test.py
"""
import datetime
import os
import sys
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

STEP_TIMEOUT = 60


def step(rank, name, fn):
    print(f"[rank{rank}] {name} ...", flush=True)
    t = time.time()
    fn()
    print(f"[rank{rank}] {name}: ok ({time.time() - t:.1f}s)", flush=True)


def worker(rank, world, init_file):
    torch.cuda.set_device(rank)
    timeout = datetime.timedelta(seconds=STEP_TIMEOUT)
    dist.init_process_group("nccl", init_method=f"file://{init_file}", rank=rank,
                            world_size=world, timeout=timeout)
    gloo = dist.new_group(backend="gloo", timeout=timeout)
    peer = 1 - rank

    step(rank, "1 gloo barrier", lambda: dist.barrier(group=gloo))

    x = torch.ones(1, device="cuda")

    def allreduce():
        dist.all_reduce(x)
        torch.cuda.synchronize()

    step(rank, "2 nccl all_reduce", allreduce)
    print(f"[rank{rank}] 3 p2p access to GPU{peer}: {torch.cuda.can_device_access_peer(rank, peer)}", flush=True)

    y = torch.ones(1 << 20, device=f"cuda:{rank}")

    def p2p_copy():
        y.to(f"cuda:{peer}")
        torch.cuda.synchronize(peer)

    step(rank, "4 p2p copy", p2p_copy)
    dist.barrier(group=gloo)
    dist.destroy_process_group()


def main():
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")  # same as 02_serve_vllm.sh
    print(f"GPUs={torch.cuda.device_count()} torch={torch.__version__} "
          f"NCCL_P2P_DISABLE={os.environ.get('NCCL_P2P_DISABLE', '-')} "
          f"GLOO_SOCKET_IFNAME={os.environ['GLOO_SOCKET_IFNAME']}", flush=True)
    init_file = os.path.join(tempfile.mkdtemp(), "init")
    ctx = mp.start_processes(worker, args=(2, init_file), nprocs=2, join=False, start_method="spawn")
    deadline = time.time() + 5 * STEP_TIMEOUT
    while not ctx.join(timeout=5):
        if time.time() > deadline:
            print("TIMEOUT: a step is hanging; the last step without 'ok' is the culprit", flush=True)
            for p in ctx.processes:
                p.kill()
            sys.exit(1)
    print("ALL OK", flush=True)


if __name__ == "__main__":
    main()
