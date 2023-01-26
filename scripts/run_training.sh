#!/bin/bash -l
#
#SBATCH --job-name="qrennd-train-test"
#SBATCH --partition=gpu
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --mail-user=b.m.varbanov@tudelft.nl
#SBATCH --mail-type=END,FAIL

module load 2022r2
module load openmpi
module load cuda/11.1.1
module load cudnn/8.0.5.39-11.1
module load miniconda3

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')

conda activate qrennd-gpu-env
srun python train_qrennd.py

/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"