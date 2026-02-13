# Khoury Discovery Cluster — Slurm Cheat Sheet

---

## Interactive GPU Session

Request an interactive shell with a GPU:

```bash
srun --partition=sharing --nodes=1 --pty --gres=gpu:h100:1 --ntasks=1 --mem=80GB --time=1:00:00 /bin/bash
```

| Flag | Meaning |
|---|---|
| `--partition=sharing` | Target the sharing partition |
| `--nodes=1` | Single node |
| `--pty` | Allocate a pseudo-terminal (interactive) |
| `--gres=gpu:h100:1` | Request 1x H100 GPU (change model/count as needed) |
| `--ntasks=1` | One task |
| `--time=30:00:00` | Wall-time limit of 30 hours |

> **Tip:** Replace `h100` with `a100`, `v100-sxm2`, `l40s`, etc. to request a different GPU type.

---

## Job Management

```bash
# List your running/pending jobs
squeue -u $USER

# Cancel all your jobs
scancel -u $USER

# Release a held job
scontrol release <job_id>
```
