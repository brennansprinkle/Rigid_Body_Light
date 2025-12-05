#!/bin/bash
#SBATCH --output=slurm/build%j-rigiddynamics.out
#SBATCH --job-name=build-rigiddynamics
#SBATCH --mail-user=adam.carter@sorbonne-universite.fr
#SBATCH --mail-type=NONE
#SBATCH --partition=thin
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00

set -e # stop on error

module load eigen/3.4.0
module load suitesparse/7.7.0
cd ~/Rigid_Body_Light
# cmake .. # pip install builds it I think
# make
conda run --name rigidlibm --live-stream pip install -e .

exit 0
