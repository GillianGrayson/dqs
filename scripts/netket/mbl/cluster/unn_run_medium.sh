#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --output=/common/home/yusipov_i/source/qs/scripts/netket/mbl/cluster/output/%j.txt

code_dir=/common/home/yusipov_i/source/qs/scripts/netket/mbl/cluster

srun python $code_dir/rhos.py
