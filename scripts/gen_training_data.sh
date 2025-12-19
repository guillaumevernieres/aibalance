#!/usr/bin/env bash
#SBATCH --account=da-cpu
#SBATCH --qos=debug
#SBATCH --output=gen_training_data.out
#SBATCH --nodes=2
#SBATCH --ntasks=36
##SBATCH --nodes=2
##SBATCH --ntasks-per-node=16
#SBATCH --time=00:10:00


# This script generates training data from the output of the UFS by interpolating all fields to a common Gaussian grid.

# User-defined variables
export OOPS_DEBUG=0
GDASApp_ROOT=${GDASApp_ROOT:-"/scratch3/NCEPDEV/da/Guillaume.Vernieres/sandboxes/GDASApp"}
JEDI_SCRATCH_DIR=${JEDI_SCRATCH_DIR:-"/scratch3/NCEPDEV/da/Guillaume.Vernieres/coupled_test/"}
JEDI_BIN_DIR=${JEDI_BIN_DIR:-"/scratch3/NCEPDEV/da/Guillaume.Vernieres/sandboxes/jedi-bundle/build/bin"}
WRKDIR="/scratch3/NCEPDEV/da/Guillaume.Vernieres/runs/gen_training_data_workdir"

# Load necessary modules
module use $GDASApp_ROOT/modulefiles
module load GDAS/ursa.intel

# Create working directory
cp -r $JEDI_SCRATCH_DIR $WRKDIR
cd $WRKDIR

# Interp fv3 to Gaussian grid
#cp /scratch3/NCEPDEV/da/Guillaume.Vernieres/runs/aibalance/configs/fv3_to_gauss.yaml .
srun -n 36 $JEDI_BIN_DIR/fv3jedi_converttostructuredgrid.x fv3_to_gauss.yaml

# Interp mom6_cice6 to Gaussian grid
cp /scratch3/NCEPDEV/da/Guillaume.Vernieres/runs/aibalance/configs/mom6_cice6_to_gauss.yaml .
srun -n 36 $JEDI_BIN_DIR/soca_converttostructuredgrid.x mom6_cice6_to_gauss.yaml

