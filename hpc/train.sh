#!/usr/bin/env bash
#SBATCH --account=da-cpu
#SBATCH --qos=debug
#SBATCH --output=train.out
#SBATCH --nodes=1
#SBATCH --ntasks=1          # Single task
#SBATCH --cpus-per-task=40  # 40 CPUs for threading
#SBATCH --time=00:30:00     # Increased time for actual training

# Set threading environment variables
export OMP_NUM_THREADS=40
export MKL_NUM_THREADS=40
export NUMEXPR_NUM_THREADS=40
export PYTHONUNBUFFERED=1

source /scratch3/NCEPDEV/da/Guillaume.Vernieres/venvs/mlvb/bin/activate

echo "Starting training at $(date)"
echo "Config: config_aice.yaml"
echo "Threads: $OMP_NUM_THREADS"
echo ""

# Run simplified training (no distributed code!)
stdbuf -oL -eL python ../scripts/train.py --config config_aice.yaml

echo ""
echo "Training finished at $(date)"