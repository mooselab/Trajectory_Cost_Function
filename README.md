
## Requirements

- Linux machine with NVIDIA GPUs (NVML total-energy counter) and RAPL exposed through powercap
- Docker usable without sudo, for Terminal-Bench tasks
- [uv](https://docs.astral.sh/uv/) (installed by `01_setup.sh` if missing)
- Harbor 0.24.0 on `PATH`, with the Terminus-2 agent

Tested setup: 2x NVIDIA L4, AMD EPYC 7313P, Ubuntu 24.04, vLLM 0.30.0,
Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 with tensor parallelism over both GPUs. Package versions are in
`env_freeze.txt`.

## Files

| File | Purpose |
|---|---|
| `config.sh` | Shared settings: model, tensor parallelism, maximum context, ports, data directory |
| `00_env_check.sh` | Read-only probe of Docker, RAPL, GPUs and disk |
| `sudo_setup.sh` | One-time setup that needs sudo: RAPL read access through udev, docker group |
| `01_setup.sh` | Creates `.venv` with vLLM and the measurement dependencies, downloads the model |
| `02_serve_vllm.sh` | Starts vLLM with prefix caching and cached-token reporting |
| `03_calibrate.sh` | Loaded-idle baseline, microbenchmark under the sampler, then `calibrate.py` |
| `04_run_trial.sh` | Runs one trajectory: clears the prefix cache, starts proxy and sampler, runs Harbor |
| `05_oracle_check.sh` | Runs the reference solutions to find tasks whose environment works |
| `06_run_batch.sh` | Loaded-idle baseline, then every task x policy x repetition through `04_run_trial.sh` |
| `sampler.py` | Reads NVML and RAPL cumulative energy counters (10 Hz by default) into a CSV |
| `llm_proxy.py` | Logging proxy between the agent and vLLM; one JSON line per model call |
| `microbench.py` | Calibration requests with exact context and output lengths |
| `calibrate.py` | Energy per call from the counters, fit of the coefficients, figures |
| `align_trial.py` | Splits a trajectory into model, command and waiting time and applies the cost function |
| `energylib.py` | Energy between two timestamps from cumulative counters |
| `dist_test.py` | Two-GPU communication check with short timeouts |

## Running the experiment

All scripts are run from the repository folder.

1. System setup, once: `sudo bash sudo_setup.sh`, then log in again so the docker group applies.
2. `bash 00_env_check.sh` and `bash 01_setup.sh`.
3. Bare-idle baseline, before vLLM starts:
   `.venv/bin/python sampler.py -o data/calib/idle_bare.csv --duration 300`
4. Start the model server in its own terminal: `bash 02_serve_vllm.sh`.
5. Calibration: `bash 03_calibrate.sh` (5 context lengths x 3 output lengths x 5 repetitions, each prompt sent twice
   for a cache miss and a cache hit). Coefficients are written to `results/calib/params.json`.
6. Task check: `bash 05_oracle_check.sh tasks/candidates.txt`.
7. Trajectories: `bash 06_run_batch.sh tasks/final.txt "summarize-off summarize-8k" 3`.
   Raw data goes to `data/traj/<task>__<policy>__r<rep>/`.
8. Alignment and estimates:
   `.venv/bin/python align_trial.py data/traj/<trajectory dirs> --calib results/calib --out results/trials_formal.csv`

Context policies of Terminus-2 in `04_run_trial.sh`: `summarize-off` never summarizes, `summarize-8k` summarizes once
the context exceeds about 8K tokens, and `summarize-on` is the agent default. Sampling uses the model defaults unless
`TEMPERATURE` is set.

Server-specific settings (GPU peer-to-peer transfers and vLLM custom all-reduce disabled, PyTorch sampler instead of
FlashInfer, 32K maximum context) are set and explained in `02_serve_vllm.sh` and `config.sh`.

## Data in this repository

| Path | Content |
|---|---|
| `tasks/tb2_tasks.csv` | Terminal-Bench 2.0 task metadata used for selection |
| `tasks/candidates.txt`, `tasks/oracle_ok.txt`, `tasks/oracle_times.csv` | Selection steps |
| `tasks/pilot.txt`, `tasks/pilot2.txt` | Pilot task lists |
| `tasks/final.txt` | The 20 tasks of the formal run |
| `data/calib/idle_bare.csv`, `data/calib/idle_loaded.csv` | Idle baselines without and with the model loaded (300 s each) |
| `data/calib/samples.csv`, `data/calib/calls.jsonl` | Energy samples and call log of the microbenchmark (150 calls) |
| `results/calib/params.json` | Idle powers, fitted coefficients, repeatability |
| `results/calib/calls_energy.csv` | Measured GPU and CPU energy of every microbenchmark call |
| `results/trials_formal.csv` | One row per trajectory of the formal run (120): times, tokens, measured energy per component, estimates |

Raw sampler, proxy and terminal-recording files are not included because of their size.

## License

MIT License, see `LICENSE`.
