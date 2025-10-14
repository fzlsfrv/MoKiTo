#!/bin/bash
#SBATCH --job-name=my_job              # Name of the job
#SBATCH --output=isokann_job_output_1.txt     # Standard output and error log
#SBATCH --ntasks=1                     # Run on a single task
#SBATCH --mem=128G                      # Total memory
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1                   # replace with number of GPUs required
#SBATCH --time=168:00:00               # Infinite time (0 days, 0 hours, 0 minutes)

# Initialize Conda
eval "$(conda shell.bash hook)"

conda activate /scratch/htc/fsafarov/openmm_ff
python isokann_random_search_1.py
