# Autonomous Loop Program

## Task
Run a full AMD MI210 (`gfx90a`) Relax E2E training job overnight with W&B logging enabled, using the local helper launchers instead of modifying Relax core code.

## Success Criteria
Command-based criteria:

- `./amd_qwen3_4b_overnight_loop.sh`
- A successful run is an attempt whose launcher exits with code `0`.
- A run that stays alive for at least 600 seconds is considered healthy and should keep running; if it later exits non-zero, the loop retries automatically.

## Files In Scope

- `amd_qwen3_4b_2gpu_e2e.sh`
- `amd_qwen3_4b_overnight_loop.sh`
- `loop-program.md`
- `results.tsv`
- `log/`

## The Experiment Loop

LOOP FOREVER:

1. Re-read this file and `results.tsv`.
2. Launch `./amd_qwen3_4b_overnight_loop.sh`.
3. If an attempt fails before the 600 second health window, treat it as a startup failure and retry.
4. If an attempt becomes healthy and later exits non-zero, treat it as a runtime failure and retry.
5. Keep all attempt logs under `log/`.
6. Record every attempt outcome in `results.tsv`.

## Error Handling

- Proxy failures against `127.0.0.1` are handled by bypassing proxy variables for local Ray control-plane traffic.
- W&B credentials are sourced from `./.env`.
- If the launcher exits immediately, preserve the attempt log and retry after a short delay.

## Rules

- Do not modify Relax core code or argument parsing during the overnight loop.
- Do not delete prior logs.
- Leave the supervising tmux session running so the user can inspect output later.
