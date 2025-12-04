#!/usr/bin/env bash
#SBATCH --account=da-cpu
#SBATCH --qos=debug
#SBATCH --output=train_distributed_multinode.out
#SBATCH --nodes=4              # Use 4 nodes for distributed training
#SBATCH --ntasks-per-node=1    # 1 process per node
#SBATCH --cpus-per-task=40     # 40 CPUs per task for threading
#SBATCH --time=02:00:00

# Set threading environment variables (per process)
export OMP_NUM_THREADS=40
export MKL_NUM_THREADS=40
export NUMEXPR_NUM_THREADS=40
export PYTHONUNBUFFERED=1

# Distributed training settings
export MASTER_PORT=29500

source /scratch3/NCEPDEV/da/Guillaume.Vernieres/venvs/mlvb/bin/activate

echo "============================================"
echo "Multi-Node Distributed Training Setup"
echo "============================================"
echo "Start time: $(date)"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Total tasks: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Nodelist: $SLURM_NODELIST"
echo "Master port: $MASTER_PORT"
echo "Config: config_aice.yaml"
echo ""
echo "Each of the $SLURM_NTASKS processes will use $OMP_NUM_THREADS threads"
echo "Total effective parallelism: $SLURM_NTASKS processes × $OMP_NUM_THREADS threads"
echo ""

# Run distributed training
# srun will launch SLURM_NTASKS processes (one per node in this config)
# Each process will get SLURM_PROCID, SLURM_NPROCS, SLURM_LOCALID set automatically
stdbuf -oL -eL srun python ../scripts/train.py --config config_aice.yaml

echo ""
echo "Training finished at $(date)"
