#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --output=/home/denysov/yusipov/qs/scripts/netket/mbl/cluster/output/%j.out
#SBATCH --mem=4000

scratch=/scratch/denysov/yusipov/qs/netket/$1
code_base=/home/denysov/yusipov/qs/scripts/netket/mbl/cluster
mkdir -p $scratch
mkdir -p $1
cd $scratch
cp $1/config.xlsx .

srun python $code_base/rhos.py

cp -r $scratch/* $1
rm -r $scratch/*
