#!/bin/sh -l
#SBATCH -A lpl
#SBATCH -t 336:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=24
#SBATCH --mail-type=TIME_LIMIT
#SBATCH --mail-user=yoop@purdue.edu
#SBATCH --job-name=Iterations_Nitramines

module load intel/17.0.1.132 impi/2017.1.132

cd $SLURM_SUBMIT_DIR

python run.py
