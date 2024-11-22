#!/bin/bash
#SBATCH --nodes=1                        # number of nodes
#SBATCH --ntasks-per-node=1              # number of tasks per node
#SBATCH --cpus-per-task=1                # number of CPU cores per task
#SBATCH --time=10:00:00                  # max wall time (10 hours)
#SBATCH --mem=16GB                       # memory per node
#SBATCH --gres=gpu:1                     # request 1 GPU
#SBATCH --job-name=train-vim-model       # job name
#SBATCH --output=train-vim-model.out     # standard output file
#SBATCH --error=train-vim-model.err      # standard error file

module purge
module load python/3.11.7
conda activate /scratch/rpd4362/env39
pip install torch torchvision torchinfo zetascale swarms einops
python train_vim.py
