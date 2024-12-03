#!/bin/bash
#SBATCH -A lpl
#SBATCH -N 1
#SBATCH -n 24
#SBATCH -t 24:00:00
#SBATCH --job-name=Training
module load intel/17.0.1.132
module load impi/2017.1.132
cd $SLURM_SUBMIT_DIR

mpiexec -np 24 /workspace/CabanaMD-Setup/n2p2/bin/nnp-norm

cp -f output.nn input.nn

mpiexec -np 24 /workspace/CabanaMD-Setup/n2p2/bin/nnp-scaling 100

mpiexec -np 24 /workspace/CabanaMD-Setup/n2p2/bin/nnp-train

