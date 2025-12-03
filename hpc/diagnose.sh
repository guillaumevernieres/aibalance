#!/usr/bin/env bash
#SBATCH --account=da-cpu
#SBATCH --qos=debug
#SBATCH --output=diagnose.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=00:05:00

# Set threading environment variables
export OMP_NUM_THREADS=40
export MKL_NUM_THREADS=40
export NUMEXPR_NUM_THREADS=40
export PYTHONUNBUFFERED=1

source /scratch3/NCEPDEV/da/Guillaume.Vernieres/venvs/mlvb/bin/activate

python diagnose.py
