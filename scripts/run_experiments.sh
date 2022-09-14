#!/bin/bash
#SBATCH -A m3058
#SBATCH -C gpu
#SBATCH -q regular 
#SBATCH -t 04:00:00
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 10
#SBATCH --gpus-per-task=1
#SBATCH --output=test1.txt
#SBATCH --error=error_test1.txt

module load cgpu
export SLURM_CPU_BIND="cores"
module load python
source activate pytorch

srun python -u VB_running_Malikas_Code.py
