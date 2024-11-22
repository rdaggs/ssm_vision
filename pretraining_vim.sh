#!/bin/bash
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks-per-node=1              # Number of tasks per node
#SBATCH --cpus-per-task=4                # Number of CPU cores per task
#SBATCH --time=12:00:00                  # Max wall time (12 hours)
#SBATCH --mem=16GB                       # Memory per node
#SBATCH --gres=gpu:1                     # Request 1 GPU
#SBATCH --job-name=train-vim-model       # Job name
#SBATCH --output=train-vim-model.out     # Standard output file
#SBATCH --error=train-vim-model.err      # Standard error file

module purge
module load python/3.8.6

source /scratch/rpd4362/env39/bin/activate

pip install torch torchvision torchinfo zetascale swarms einops

# Run the Python script
python train_vim.py
