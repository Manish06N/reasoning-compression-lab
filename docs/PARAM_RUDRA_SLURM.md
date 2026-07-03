# PARAM Rudra SLURM — GPU quota and `--exclusive` trap

Last updated: 2026-07-03

## User GPU quota

| Limit | Value |
|-------|-------|
| `MaxTRESPerUser` (gpu QOS) | **`gres/gpu=2`** |
| ragpu node layout | **2× A100 80GB** per node (`Gres=gpu:2`) |

You can run **two parallel 1-GPU inference jobs** (e.g. b01 Qwen + Llama split cells).

## The `--exclusive` trap (do not repeat)

Each inference cell loads **one model on one GPU** (`--gres=gpu:1`). That is correct.

**Problem:** `sbatch --exclusive` on a 2-GPU ragpu node makes SLURM count **both GPUs on the node** toward your QOS limit, even when `TresPerNode=gres/gpu:1`.

| Job | Flags | Counted toward your 2-GPU quota |
|-----|-------|----------------------------------|
| Qwen (running) | `gres/gpu:1`, shared (`OverSubscribe=OK`) | **1** |
| Llama (pending) | `gres/gpu:1`, **`--exclusive`** (`OverSubscribe=NO`) | **2** |
| **Total** | | **3 → `QOSMaxGRESPerUser`** |

Symptom: second cell stays `PENDING (QOSMaxGRESPerUser)` while only one job is running.

Observed on job **86748** (2026-07-03): manual `sbatch --exclusive` for Llama while **86743** Qwen used 1 GPU.

## Correct submit policy

| Mode | `--exclusive`? | Why |
|------|----------------|-----|
| **Split 1-GPU cells** (default `QREASON_SUBMIT_2GPU_MODE=split`) | **Never** | Two cells = 1+1 GPUs within quota |
| **Single 1-GPU cell** (`submit_hpc_blocks.sh cell …`) | **Never** | Same QOS trap |
| **2-GPU block job** (`exclusive_block`) | Optional (`QREASON_SLURM_EXCLUSIVE=1`) | One job owns `--gres=gpu:2`; exclusive is a different pattern |
| **Smoke test exclusive** (`smoke_test_quick_exclusive.slurm`) | OK when **no other GPU jobs** | Validation only, not parallel b01 |

## Dirty GPU without exclusive

Instead of `--exclusive`, use:

1. **`metadata/dirty_nodes.txt`** — auto-appended when VRAM preflight fails (`record_dirty_node` in launcher)
2. **`submit_hpc_blocks.sh`** — merges dirty nodes into `sbatch --exclude=…`
3. **`QREASON_MIN_FREE_GPU_MB`** (default 40000) — refuse to start vLLM on a busy assigned GPU
4. **Requeue** — up to `QREASON_GPU_PREFLIGHT_REQUEUE_MAX` (default 12)

## Commands

```bash
# b01 both cells — correct (non-exclusive split)
bash scripts/hpc/submit_hpc_blocks.sh b01

# Resubmit one cell only — correct
bash scripts/hpc/submit_hpc_blocks.sh cell configs/cells/level_c_bf16_seed0.json b01_parallel_bf16_anchors

# WRONG for parallel work — do not use manual sbatch --exclusive with gres/gpu:1
# sbatch --exclusive --gres=gpu:1 ...   # counts as 2 GPUs on ragpu nodes
```

## Verify

```bash
squeue -u $USER -o "%.10i %.2t %b %R"
scontrol show job <JOBID> | grep -E "OverSubscribe|ReqTRES|Reason"
```

- `OverSubscribe=OK` → shared 1-GPU job (good for split)
- `OverSubscribe=NO` + `gres/gpu:1` on ragpu → **QOS trap** if you have another GPU job