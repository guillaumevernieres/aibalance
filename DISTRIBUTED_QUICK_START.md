# Distributed Training - Quick Start

## What Changed

I've fixed the distributed training implementation in `ufsemulator/training.py`:

1. **Fixed `setup_distributed()`**: Now properly detects and handles SLURM environments
2. **Better error handling**: Improved debugging output for distributed setup
3. **Proper rank/world_size detection**: Correctly parses SLURM environment variables
4. **Gradient synchronization**: DDP automatically averages gradients across all processes

## How to Use

### Option 1: Single-Node Threading (Current Default)
```bash
cd hpc
sbatch train.sh
```
- Best for: Single node with many CPUs
- Uses: 40 threads in one process
- No communication overhead

### Option 2: Test Distributed Setup (Recommended First)
```bash
cd hpc
sbatch test_distributed.sh
```
- Verifies distributed communication works
- Tests all-reduce and broadcast operations
- Takes ~5 minutes

### Option 3: Distributed Training (2 Nodes)
```bash
cd hpc
sbatch train_distributed.sh
```
- Uses: 2 nodes, 1 process per node, 40 threads per process
- Effective parallelism: 80 cores total
- Gradients averaged across 2 processes

### Option 4: Distributed Training (4 Nodes)
```bash
cd hpc
sbatch train_distributed_multinode.sh
```
- Uses: 4 nodes, 1 process per node, 40 threads per process
- Effective parallelism: 160 cores total
- Gradients averaged across 4 processes
- Effective batch size: batch_size × 4

## How Gradient Averaging Works

When you use distributed training:

1. **Data is split**: Each process gets different batches
   - Process 0: samples 0-999 (if batch_size=1000)
   - Process 1: samples 1000-1999
   - Process 2: samples 2000-2999
   - Process 3: samples 3000-3999

2. **Each process computes gradients independently**:
   - Process 0: gradient based on samples 0-999
   - Process 1: gradient based on samples 1000-1999
   - etc.

3. **DDP averages gradients automatically**:
   ```python
   # After loss.backward(), DDP does:
   gradient_final = (grad_p0 + grad_p1 + grad_p2 + grad_p3) / 4
   ```

4. **All processes update identically**:
   - Same averaged gradients
   - Same model weights after optimizer.step()

This is equivalent to training with batch_size=4000 on a single machine!

## Expected Speedup

**Theoretical**: N× speedup with N processes

**Realistic**:
- 2 processes: 1.7-1.9× speedup (10-15% communication overhead)
- 4 processes: 2.5-3.5× speedup (15-30% communication overhead)

Communication overhead depends on:
- Model size (more parameters = more to sync)
- Network speed between nodes
- Batch size (larger batches = less frequent sync)

## Monitoring

Check the output file for:

```
Distributed training initialized: gloo backend, 4 processes
Master: node01:29500
[Rank 0] Training samples: 100000
[Rank 1] Training samples: 100000
[Rank 2] Training samples: 100000
[Rank 3] Training samples: 100000
Epoch 10/100 - Train: 0.001234, Val: 0.001456
```

Each process will show its own logs, but only Rank 0 saves checkpoints and plots.

## Troubleshooting

### Processes hang at initialization
- Check: `test_distributed.sh` works first
- Solution: May need to adjust network settings or firewall

### "Address already in use"
- Problem: Port 29500 is in use
- Solution: Change `MASTER_PORT=29501` in SLURM script

### Slower than single-node
- Cause: Communication overhead too high for your model/batch size
- Solutions:
  - Increase batch size to reduce sync frequency
  - Use GPUs instead (NCCL is much faster than Gloo)
  - Stick with single-node threading

## When to Use What

| Scenario | Recommended Mode | Script |
|----------|-----------------|--------|
| Testing, development | Single-node threading | `train.sh` |
| Dataset fits on 1 node | Single-node threading | `train.sh` |
| Very large dataset | Distributed 2-4 nodes | `train_distributed*.sh` |
| Multiple GPUs available | Distributed GPU | (need to modify for GPU) |
| Production inference | C++ LibTorch | `ufs_emulator.cpp` |

## Why C++ Was Faster

Your C++ LibTorch code is faster because:
1. **No GIL (Global Interpreter Lock)**: True parallelism
2. **Lower overhead**: No Python interpreter
3. **Better threading**: OpenMP more efficient than Python threading
4. **Memory locality**: Better cache usage

**Recommendation**:
- Use Python for training and experimentation (faster development)
- Use C++ for production inference and DA (faster execution)
- You already have the export pipeline: `export_to_torchscript.py`

## Next Steps

1. **Test basic distributed**:
   ```bash
   sbatch test_distributed.sh
   ```

2. **If test passes, try 2-node training**:
   ```bash
   sbatch train_distributed.sh
   ```

3. **Compare results**:
   - Check training time
   - Verify loss curves are similar
   - Look for speedup in logs

4. **If slower than single-node**:
   - Communication overhead too high
   - Stick with `train.sh` (single-node threading)
   - Consider GPUs if available

See `DISTRIBUTED_TRAINING.md` for detailed explanation of how it all works!
