#!/bin/sh
#
#SBATCH --job-name="transfer_train_result"
#SBATCH --partition=trans
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

results=(
 'sim-output'
 'parameter-files'
 'output.txt'
 'err.txt'
)

source="/scratch/${USER}/MySimulation"
destination='/tudelft.net/staff-umbrella/MyProject/DelftBlueResults/'

for result in "${results[@]}"
do
  rsync -av --no-perms "${source}/${result}" "${destination}"
done
