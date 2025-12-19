#!/usr/bin/env bash
#SBATCH --account=da-cpu
#SBATCH --qos=batch
#SBATCH --output=train_distributed.out
#SBATCH --nodes=2              # Use 2 nodes for distributed training
#SBATCH --ntasks-per-node=1             # 1 task (1 per node)
#SBATCH --cpus-per-task=80     # 80 CPUs per task for threading
#SBATCH --time=01:00:00

# Set threading environment variables (per process)
export OMP_NUM_THREADS=80
export MKL_NUM_THREADS=80
export NUMEXPR_NUM_THREADS=80
export PYTHONUNBUFFERED=1

# Distributed training settings
export MASTER_PORT=29500

source /scratch3/NCEPDEV/da/Guillaume.Vernieres/venvs/mlvb/bin/activate

echo "============================================"
echo "Distributed Training Setup"
echo "============================================"
echo "Start time: $(date)"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Nodelist: $SLURM_NODELIST"
echo "Master port: $MASTER_PORT"
echo "Config: config_aice.yaml"
echo ""

# SLURM will automatically set:
# - SLURM_PROCID (rank of this process)
# - SLURM_NPROCS (total number of processes)
# - SLURM_LOCALID (local rank on this node)
# - SLURM_NODELIST (list of allocated nodes)

# Run distributed training
# srun will launch one process per task, SLURM env vars will be set automatically
stdbuf -oL -eL srun python ../scripts/train.py --config config_aice.yaml

echo ""
echo "Training finished at $(date)"
