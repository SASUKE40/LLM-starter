# Khoury Discovery Cluster — GPU Inventory

> Source: [GPU Monitor](https://rc-docs.northeastern.edu/) — *snapshot, not live*  
> Contact: rchelp@northeastern.edu

---

## Summary

| GPU Model | Nodes | GPUs/Node | Total GPUs | Partitions |
|---|---|---|---|---|
| **H200** | 12 | 8 | 96 | gpu, gpu-short, gpu-interactive |
| **H100** | 1 | 4 | 4 | sharing |
| **A100** (8-gpu) | 4 | 8 | 32 | sharing |
| **A100** (7-gpu) | 1 | 7 | 7 | sharing |
| **A100** (4-gpu) | 6 | 4 | 24 | gpu, gpu-short, gpu-interactive |
| **A100** (3-gpu) | 4 | 3 | 12 | sharing, gpu, gpu-short, gpu-interactive |
| **A6000** (8-gpu) | 2 | 8 | 16 | sharing |
| **A6000** (2-gpu) | 1 | 2 | 2 | sharing |
| **A5000** | 5 | 8 | 40 | sharing |
| **A30** | 1 | 6 | 6 | sharing |
| **L40s** (8-gpu) | 3 | 8 | 24 | sharing |
| **L40s** (4-gpu) | 5 | 4 | 20 | sharing |
| **L40** | 2 | 10 | 20 | sharing |
| **V100-SXM2** (4-gpu) | 27 | 4 | 108 | gpu, gpu-short, sharing, courses-gpu, gpu-interactive |
| **V100-SXM2** (3-gpu) | 3 | 3 | 9 | gpu, gpu-short, gpu-interactive |
| **V100-PCIe** | 12 | 2 | 24 | gpu, gpu-short, gpu-interactive |
| **V100** | 8 | 4 | 32 | sharing |
| **T4** | 3 | 4 | 12 | gpu, gpu-short, gpu-interactive |
| **P100** (4-gpu) | 12 | 4 | 48 | courses-gpu, sharing |
| **P100** (3-gpu) | 4 | 3 | 12 | courses-gpu, sharing |
| **Quadro** | 2 | 3 | 6 | sharing |
| | **118 nodes** | | **554 GPUs** | |

---

## Totals by GPU Generation

| Generation | Total GPUs | Notes |
|---|---|---|
| H-series (H100, H200) | 100 | Newest / highest throughput |
| A-series (A100, A6000, A5000, A30) | 139 | Strong for large-model training |
| L-series (L40, L40s) | 64 | Ada Lovelace |
| V-series (V100 variants) | 173 | Largest pool, older Volta |
| Other (T4, P100, Quadro) | 78 | Budget / courses |

---

## Pending Jobs (at snapshot time)

| GPU | Jobs in Queue |
|---|---|
| H200 | 20 |
| V100-SXM2 | 13 |
| K80 | 2 |
| T4 | 1 |

---

## Useful Commands

```bash
# Check GPU utilization on a specific node
sinfo -n <nodename> --Format="Gres:30,GresUsed:30"

# Full node details
scontrol show node <nodename>
```

> **Note:** "Mixed" nodes have some GPUs allocated but may still have free slots.  
> Actual wait times depend on fairshare score in addition to resource availability.
