# Explorer-02 — System Specification

> **Rocky Linux 9.3 (Blue Onyx)** on dual-socket Intel Xeon Gold 5318Y  
> Kernel `5.14.0-362.13.1.el9_3.x86_64` — Support ends **2032-05-31**

---

## Quick Reference

| Category | Detail | Value |
|---|---|---|
| **Host** | Hostname | `explorer-02` |
| | OS | Rocky Linux 9.3 (Blue Onyx) |
| | Kernel | 5.14.0-362.13.1.el9_3.x86_64 |
| | Architecture | x86_64 |
| **CPU** | Model | Intel Xeon Gold 5318Y @ 2.10 GHz |
| | Sockets / Cores / Threads | 2 / 24 / 2 (96 logical CPUs) |
| | Max Frequency | 3.4 GHz |
| | NUMA Nodes | 2 |
| **Cache** | L1d / L1i | 2.3 MiB / 1.5 MiB |
| | L2 / L3 | 60 MiB / 72 MiB |
| **Memory** | Total / Available | 250 GiB / 220 GiB |
| | Swap Total / Used | 19 GiB / 12 GiB |
| **Users** | Active user mounts | ~150+ |

---

## CPU Details

```
Architecture:         x86_64
CPU op-mode(s):       32-bit, 64-bit
Address sizes:        46 bits physical, 57 bits virtual
Byte Order:           Little Endian

Vendor ID:            GenuineIntel
Model name:           Intel(R) Xeon(R) Gold 5318Y CPU @ 2.10GHz
CPU family:           6
Model:                106
Stepping:             6

Socket(s):            2
Core(s) per socket:   24
Thread(s) per core:   2
Total logical CPUs:   96

CPU max MHz:          3400.0000
CPU min MHz:          800.0000
BogoMIPS:             4200.00

Virtualization:       VT-x
```

### Key CPU Flags

`avx512f` `avx512dq` `avx512cd` `avx512bw` `avx512vl` `avx512ifma` `avx512vbmi` `avx512_vbmi2` `avx512_vnni` `avx512_bitalg` `avx512_vpopcntdq` `avx2` `avx` `sse4_2` `sse4_1` `ssse3` `sse2` `sse` `fma` `aes` `sha_ni` `vmx` `la57`

### NUMA Topology

| Node | CPUs |
|---|---|
| Node 0 | 0, 2, 4, 6, ... 92, 94 (even) |
| Node 1 | 1, 3, 5, 7, ... 93, 95 (odd) |

### Cache Hierarchy

| Level | Size | Instances |
|---|---|---|
| L1d | 2.3 MiB | 48 |
| L1i | 1.5 MiB | 48 |
| L2 | 60 MiB | 48 |
| L3 | 72 MiB | 2 |

---

## Memory

```
              total        used        free      shared  buff/cache   available
Mem:          250Gi        30Gi       8.8Gi       4.0Gi       217Gi       220Gi
Swap:          19Gi        12Gi       6.9Gi
```

---

## Storage

### Local Disk (`/dev/sda` — 447.1 GiB)

| Partition | Size | Used | Mount Point |
|---|---|---|---|
| sda1 | 1G | <1% | `/boot/efi` |
| sda2 | 2G | 16% | `/boot` |
| sda3 | 29.3G | 48% | `/var` |
| sda4 | 19.5G | 21% | `/` |
| sda5 | 19.5G | — | `[SWAP]` |
| sda6 | 9.8G | 2% | `/srv` |
| sda7 | 9.8G | 8% | `/var/log` |
| sda8 | 9.8G | 2% | `/var/log/audit` |
| sda9 | 9.8G | 64% | `/var/tmp` |
| sda11 | 336.6G | 8% | `/tmp` |

### Network Storage (VAST)

| Mount | Source | Size | Used | Alert |
|---|---|---|---|---|
| `/home` | `vast1-mghpcc-eth.neu.edu:/discovery/home` | 155T | **95%** | :warning: Critical |
| `/scratch` | `vast1-mghpcc-eth.neu.edu:/discovery/scratch` | 2.2P | 60% | |
| `/projects` | `vast1-mghpcc-eth.neu.edu:/work_project` | 3.7P | 79% | |
| `/shared` | `vast1-mghpcc-eth.neu.edu:/vast_shared` | 30T | 73% | |
| `/datasets` | `vast1-mghpcc-eth.neu.edu:/datasets` | 40T | 31% | |
| `/courses` | `vast1-mghpcc-eth.neu.edu:/courses` | 36T | 32% | |

---

## Network Interfaces

| # | Name | State | Address | MTU | Notes |
|---|---|---|---|---|---|
| 1 | `lo` | UP | 127.0.0.1/8 | 65536 | Loopback |
| 4 | `internal` | **UP** | 10.99.200.107/16 | 9000 | Primary internal (Jumbo frames) |
| 5 | `eno12409np1` | **UP** | 129.10.0.146/24 | 1500 | External |
| 2 | `eno8303` | DOWN | — | 1500 | |
| 3 | `eno8403` | DOWN | — | 1500 | |
| 6 | `ens7f0np0` | DOWN | — | 1500 | |
| 7 | `ens7f1np1` | DOWN | — | 1500 | |
| 8 | `ibp23s0` | DOWN | — | 4092 | InfiniBand |

---

## Security — CPU Vulnerability Mitigations

| Vulnerability | Status |
|---|---|
| Gather data sampling | Mitigated (Microcode) |
| Itlb multihit | Not affected |
| L1tf | Not affected |
| Mds | Not affected |
| Meltdown | Not affected |
| MMIO stale data | Mitigated (Clear CPU buffers; SMT vulnerable) |
| Retbleed | Not affected |
| Spec rstack overflow | Not affected |
| Spec store bypass | Mitigated (disabled via prctl) |
| Spectre v1 | Mitigated (usercopy/swapgs barriers) |
| Spectre v2 | Mitigated (Enhanced IBRS, IBPB, RSB filling) |
| Srbds | Not affected |
| Tsx async abort | Not affected |

---

## OS Release Info

| Field | Value |
|---|---|
| Name | Rocky Linux |
| Version | 9.3 (Blue Onyx) |
| ID | `rocky` |
| ID Like | `rhel centos fedora` |
| Platform ID | `platform:el9` |
| Support End | 2032-05-31 |
| Home URL | https://rockylinux.org/ |
| Bug Report URL | https://bugs.rockylinux.org/ |
