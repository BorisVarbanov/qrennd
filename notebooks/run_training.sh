#!/bin/bash -l
#
#SBATCH --job-name="qrennd-decoder-training"
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --output=/scratch/bmvarbanov/.log/surf17/s17_%j.out
#SBATCH --error=/scratch/bmvarbanov/.log/surf17/s17_%j.err
#SBATCH --mail-user=b.m.varbanov@tudelft.nl
#SBATCH --mail-type=END,FAIL


module load 2022r2
module load python
module load openmpi
module load py-tensorflow

srun python train_qrennd.py