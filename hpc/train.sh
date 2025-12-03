#!/usr/bin/env bash
#SBATCH --account=da-cpu
#SBATCH --qos=debug
#SBATCH --output=train.out
#SBATCH --nodes=1
#SBATCH --ntasks=1          # Single task (no distributed training)
#SBATCH --cpus-per-task=40  # 40 CPUs for threading
#SBATCH --time=00:05:00

# Set threading environment variables
export OMP_NUM_THREADS=40
export MKL_NUM_THREADS=40
export NUMEXPR_NUM_THREADS=40

# Disable Python output buffering to see real-time logs
export PYTHONUNBUFFERED=1

source /scratch3/NCEPDEV/da/Guillaume.Vernieres/venvs/mlvb/bin/activate

# Pre-process NetCDF to .npz if it doesn't exist (much faster for repeated runs)
if [ ! -f gdas.ice.t00z.inst.f009.npz ]; then
    echo "Converting NetCDF to .npz for faster loading..."
    python -c "
from pathlib import Path
import sys
sys.path.insert(0, str(Path('..').resolve()))
from ufsemulator.data import create_training_data_from_netcdf
from ufsemulator.training import load_config
config = load_config('config_aice.yaml')
config['use_cuda'] = False  # Force CPU mode
create_training_data_from_netcdf('gdas.ice.t00z.inst.f009.nc', config, 'gdas.ice.t00z.inst.f009.npz')
"
fi

# Run training with pre-processed data
stdbuf -oL -eL python ../scripts/train.py \
    --config config_aice.yaml \
    --no-distributed \
    --data-path gdas.ice.t00z.inst.f009.npz