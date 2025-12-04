#!/usr/bin/env bash
#SBATCH --account=da-cpu
#SBATCH --qos=debug
#SBATCH --output=test_distributed.out
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00

export PYTHONUNBUFFERED=1
export MASTER_PORT=29500

source /scratch3/NCEPDEV/da/Guillaume.Vernieres/venvs/mlvb/bin/activate

echo "Testing distributed setup..."
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks: $SLURM_NTASKS"
echo "Nodelist: $SLURM_NODELIST"
echo ""

# Create a simple test script
cat > test_dist.py << 'EOF'
import os
import torch
import torch.distributed as dist
from datetime import timedelta

def setup_distributed():
    """Test distributed setup."""
    rank = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ['SLURM_NPROCS'])
    local_rank = int(os.environ.get('SLURM_LOCALID', rank))

    # Parse SLURM node list
    node_list = os.environ.get('SLURM_NODELIST', 'localhost')
    if '[' in node_list:
        base = node_list.split('[')[0]
        ranges = node_list.split('[')[1].split(']')[0]
        first_range = ranges.split(',')[0]
        if '-' in first_range:
            first_num = first_range.split('-')[0]
        else:
            first_num = first_range
        master_node = base + first_num
    else:
        master_node = node_list.split(',')[0]

    os.environ['MASTER_ADDR'] = master_node
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')

    print(f"[Rank {rank}/{world_size}] Node: {os.environ.get('SLURMD_NODENAME', 'unknown')}")
    print(f"[Rank {rank}] Master: {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")

    # Initialize process group
    backend = "gloo"  # CPU backend
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=5)
    )

    print(f"[Rank {rank}] Process group initialized successfully!")

    # Test all-reduce
    tensor = torch.tensor([rank + 1.0])
    print(f"[Rank {rank}] Before all-reduce: {tensor.item()}")

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"[Rank {rank}] After all-reduce: {tensor.item()} (should be {sum(range(1, world_size+1))})")

    # Test broadcast
    if rank == 0:
        broadcast_tensor = torch.tensor([42.0])
    else:
        broadcast_tensor = torch.tensor([0.0])

    print(f"[Rank {rank}] Before broadcast: {broadcast_tensor.item()}")
    dist.broadcast(broadcast_tensor, src=0)
    print(f"[Rank {rank}] After broadcast: {broadcast_tensor.item()} (should be 42.0)")

    # Barrier to sync
    dist.barrier()
    if rank == 0:
        print("\n✓ All distributed operations successful!")

    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    setup_distributed()
EOF

# Run the test with srun
stdbuf -oL -eL srun python test_dist.py

echo ""
echo "Test completed!"
