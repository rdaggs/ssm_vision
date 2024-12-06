#!/bin/bash
#SBATCH --nodes=1                        # number of nodes
#SBATCH --ntasks-per-node=1              # number of tasks per node
#SBATCH --cpus-per-task=2                # number of CPU cores per task
#SBATCH --time=10:00:00                  # max wall time (10 hours)
#SBATCH --mem=16GB                       # memory per node
#SBATCH --gres=gpu:1                     # request 1 GPU
#SBATCH --job-name=train-vim-test        # job name
#SBATCH --output=train-vim-model.out     # standard output file
#SBATCH --error=train-vim-model.err      # standard error file

module purge
module load python/3.8.6
conda activate /scratch/rpd4362/env39
cd /scratch/rpd4362/ssm_vision
pip install torch torchvision torchinfo zetascale swarms einops
python train_vim.py
