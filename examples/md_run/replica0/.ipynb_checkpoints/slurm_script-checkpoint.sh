#!/bin/bash
#SBATCH --job-name=my_job              # Name of the job
#SBATCH --output=my_job_output.txt     # Standard output and error log
#SBATCH --ntasks=1                     # Run on a single task
#SBATCH --cpus-per-task=32             # Number of CPU cores per task
#SBATCH --mem=64G                      # Total memory
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1                   # replace with number of GPUs required
#SBATCH --time=168:00:00               # Infinite time (0 days, 0 hours, 0 minutes)

# Initialize Conda
eval "$(conda shell.bash hook)"

conda activate /scratch/htc/fsafarov/openmm_ff
python calculate_PWDs.py
