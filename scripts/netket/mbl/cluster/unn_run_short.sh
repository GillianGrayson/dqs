#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --oversubscribe
#SBATCH --mem=4000
#SBATCH --output=/common/home/yusipov_i/source/qs/scripts/netket/mbl/cluster/output/%j.txt

code_dir=/common/home/yusipov_i/source/qs/scripts/netket/mbl/cluster

srun python $code_dir/ndm_vs_exact.py
