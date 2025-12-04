# Distributed Training Guide

## How Distributed Training Works in PyTorch

### Overview

Distributed training in PyTorch allows you to train models across multiple processes, either on:
- Multiple GPUs on a single node
- Multiple nodes with CPUs
- Multiple nodes with multiple GPUs each

The key concept is **Data Parallel Training** using DistributedDataParallel (DDP).

### Data Parallel Training Process

1. **Model Replication**: Each process gets an identical copy of the model
2. **Data Sharding**: The dataset is split across processes using `DistributedSampler`
   - Each process gets different batches
   - No data duplication between processes
3. **Forward Pass**: Each process computes forward pass on its local batch independently
4. **Backward Pass**: Each process computes gradients on its local batch
5. **Gradient Synchronization**:
   - **All-Reduce operation**: Gradients are averaged across all processes
   - This happens automatically via DDP hooks after backward()
   - Uses efficient collective communication (NCCL for GPU, Gloo for CPU)
   - All processes end up with identical averaged gradients
6. **Optimizer Step**: Each process updates its model identically (since gradients are synced)

### Communication Pattern

```python
# What happens in each training iteration:
for batch in local_dataloader:  # Each process: different batches
    outputs = model(batch)       # Independent forward pass per process
    loss = criterion(outputs, targets)
    loss.backward()              # Compute local gradients

    # DDP automatically does ALL-REDUCE:
    # gradient[i] = mean(gradient_process0[i], gradient_process1[i], ..., gradient_processN[i])
    # After this, all processes have identical gradients

    optimizer.step()             # All processes update model identically
```

### Gradient Averaging Example

If you have 2 processes with batch size 32 each:

**Process 0**: Computes gradients on samples 0-31
**Process 1**: Computes gradients on samples 32-63

After backward():
- Process 0 has gradient G0 (based on samples 0-31)
- Process 1 has gradient G1 (based on samples 32-63)

After DDP All-Reduce:
- Both processes have gradient = (G0 + G1) / 2

This is equivalent to computing gradient on a batch of size 64!

## Training Modes in This Project

### 1. Single-Process Threading (Default)

**Best for**: Single-node CPU training

```bash
# Uses threading within a single process
sbatch hpc/train.sh
```

- Uses `torch.set_num_threads()` for CPU parallelism
- Efficient for single-node with many cores
- No communication overhead
- Simpler code, easier to debug

### 2. Distributed Multi-Node CPU Training

**Best for**: Very large datasets that don't fit on one node, or when you want to scale across multiple nodes

```bash
# 2 nodes, 1 process per node, 40 threads per process
sbatch hpc/train_distributed.sh

# 4 nodes, 1 process per node, 40 threads per process
sbatch hpc/train_distributed_multinode.sh
```

**How it works**:
- SLURM launches N processes (one per node typically)
- Each process uses threading (OMP_NUM_THREADS=40)
- Processes communicate via Gloo backend (CPU)
- Gradients are averaged across processes using All-Reduce
- Effective batch size = batch_size × num_processes

**Example**: 4 nodes, batch_size=1000
- Each process sees batches of 1000 samples
- Gradients computed on 4000 samples total per step
- Training is 4× faster (ideally) due to parallel processing

### 3. Distributed Multi-GPU Training

**Best for**: When you have access to multiple GPUs

```bash
# Single node with 4 GPUs
torchrun --nproc_per_node=4 scripts/train.py --config config.yaml

# Multi-node with GPUs (on HPC)
sbatch hpc/train_distributed_gpu.sh
```

**How it works**:
- Each GPU gets its own process
- Uses NCCL backend (optimized for GPU communication)
- Much faster gradient synchronization than CPU

## Configuration Requirements

For distributed training to work efficiently:

1. **Batch size**: Consider the effective batch size
   - Effective batch = batch_size × world_size
   - May need to adjust learning rate accordingly
   - Rule of thumb: learning_rate ∝ sqrt(effective_batch_size)

2. **Data loading**:
   - Uses `DistributedSampler` to shard data
   - Each process loads different data
   - No shuffle in DataLoader (handled by sampler)

3. **Checkpointing**:
   - Only rank 0 saves checkpoints to avoid conflicts
   - All processes save identical models (gradients are synced)

4. **Validation**:
   - Each process validates on different data subset
   - Losses can be averaged across processes for global metric

## Why Single-Node Threading vs Distributed?

### Use Single-Node Threading When:
- Training on single node with many CPUs (like your current setup)
- Dataset fits comfortably in memory
- Want simpler code and debugging
- Communication overhead would dominate computation

### Use Distributed When:
- Dataset is too large for single node memory
- Have access to multiple nodes or GPUs
- Computation time dominates communication time
- Need to scale to very large models or datasets

## Performance Comparison

**Your Current Setup** (Single node, 40 CPUs):
- Mode 1 (Threading): 1 process × 40 threads = 40 cores utilized
- Mode 2 (4 nodes distributed): 4 processes × 40 threads = 160 cores utilized
  - Theoretical speedup: 4×
  - Actual speedup: 2-3× (due to communication overhead)

**Communication Overhead**:
- CPU Gloo backend: ~10-30% overhead for gradient synchronization
- GPU NCCL backend: ~1-5% overhead (much faster)

## Testing Distributed Training

1. **Start small**: Test with 2 nodes first
```bash
sbatch hpc/train_distributed.sh
```

2. **Check output**: Look for:
```
Distributed training initialized: gloo backend, 2 processes
Master: node01:29500
[Rank 0] ...
[Rank 1] ...
```

3. **Monitor convergence**: Loss should be similar to single-node training
   - May converge faster (more data per step)
   - May need learning rate adjustment

4. **Scale up**: Once working, try 4 nodes
```bash
sbatch hpc/train_distributed_multinode.sh
```

## Troubleshooting

### "Address already in use"
- Master port conflict: Change `MASTER_PORT` in SLURM script
- Default is 29500, try 29501, 29502, etc.

### Processes hanging at initialization
- Check SLURM_NODELIST is parsed correctly
- Verify nodes can communicate (firewall issues)
- Check timeout setting in `setup_distributed()`

### Different losses on different ranks
- Bug: Gradients not syncing properly
- Check DDP is wrapping model correctly
- Verify All-Reduce is happening

### Slower than single-node
- Communication overhead too high
- Try larger batch sizes to reduce communication frequency
- May need GPU instead of CPU for better scaling

## When C++ LibTorch is Better

Your observation about C++ being faster is correct. Consider C++ when:
- Inference in production (no training)
- Jacobian computation for DA
- Integration with existing C++ forecast models
- Don't need frequent model architecture changes

Python is better for:
- Experimentation and rapid development
- Training (more flexible, better debugging)
- Leveraging PyTorch ecosystem (pretrained models, tools)

**Recommended workflow**:
1. Develop and train in Python
2. Export to TorchScript (.ts file)
3. Deploy in C++ for production inference and DA
