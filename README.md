# gpu_assassin (gpu-guard)

A root-managed systemd service that reclaims idle GPU memory and blocks known
GPU auto-claim services on shared multi-GPU servers.

## Policy

- A job holding more than `memory_threshold_mib` (1024 MiB) of GPU memory, whose
  GPUs all stay below `utilization_threshold` (5%) on both compute and memory
  activity for `idle_seconds` (1800 s), receives SIGTERM; if it still holds the
  GPU after `term_grace_seconds` (60 s), SIGKILL. A warning is logged
  `warning_seconds` before termination.
- Sampled every second via `nvidia-smi`. Telemetry failures, unknown values or
  sample gaps over `max_sample_gap_seconds` reset all idle timers; they never
  cause kills.
- Processes are grouped into jobs by systemd unit, container cgroup (Docker,
  Kubernetes) or process group. A job stays alive if any of its GPUs is active.
- Processes matching `blocked_units` / `blocked_commands` are stopped on sight.
- Signals use pidfds plus start-time checks to avoid PID reuse races.

Full policy text (Chinese): `POLICY.zh-CN.txt`. Configuration: `config.json`.

## Deploy

```sh
python3 -m unittest test_gpu_guard
python3 deploy_guard.py --host "$(hostname)" --gpu-count 10 --output install-guard.py
sudo /usr/bin/python3 -I - < install-guard.py
```

`deploy_guard.py` builds a self-contained installer bound to one hostname. It
backs up overwritten files to `/var/lib/gpu-guard/backups/` and checks that the
service reports fresh, enforcing telemetry with the expected GPU count.
Pass `--skip-polkit` on hosts with polkit 0.105 (no JS rules support); the
systemd default `auth_admin` still blocks ordinary users from stopping it.

## Operate

```sh
cat /run/gpu-guard/status.json
sudo journalctl -u gpu-guard.service
sudo tail /var/log/gpu-guard/events.jsonl
sudo systemctl stop gpu-guard.service     # maintenance
sudo systemctl restart gpu-guard.service  # after editing /etc/gpu-guard/config.json
```

Set `"enforce": false` in the config for dry-run mode (logs `would_signal`).

`gpu_quarantine.py` / `gpu-quarantine.service` unbind one specific failed GPU
by UUID; edit the UUID before using them on another machine.
