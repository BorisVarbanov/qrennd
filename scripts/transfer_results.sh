#!/bin/sh
#
#SBATCH --job-name="transfer_output"
#SBATCH --partition=trans
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --account=research-qutech-qcd

exp_name = "20230117-d3_rot-surf_circ-level_test-train"

source="/scratch/${USER}/output/"
destination='/tudelft.net/staff-umbrella/qrennd/output/'

rsync -av --no-perms "${source}/${exp_name}" "${destination}/${exp_name}"
