#!/bin/bash -l
#SBATCH -o ./logs/tjob.out.%A_%a
#SBATCH -e ./logs/tjob.err.%A_%a
#SBATCH --job-name=exploration
#SBATCH --mail-type=ALL
#SBATCH --mail-user=marcel.binz@tuebingen.mpg.de
#SBATCH --time=100:00:00
#SBATCH --cpus-per-task=8

cd ~/Einstellung/

module purge
module load gcc/10
module load anaconda/3/2020.02
conda activate pytorch-gpu

python3 simulate_cluster.py --id ${SLURM_ARRAY_TASK_ID}
