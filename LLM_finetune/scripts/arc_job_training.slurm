#!/bin/bash
#SBATCH --job-name=arc_training_job
#SBATCH --output=/work/zwe996/arc_training_job_%j.out
#SBATCH --error=/work/zwe996/arc_training_job_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=10:00:00
#SBATCH --partition=gpu1v100
#SBATCH --ntasks=1

# Source the ~/.bashrc file to apply your configurations
source ~/.bashrc

# Activate your conda environment
conda activate /work/zwe996/env

# Run the training script
bash run_training.sh
