#!/bin/bash
#SBATCH --job-name=my_job_pwds              # Name of the job
#SBATCH --output=pwd_outputs/my_job.%j_output_half.txt     # Standard output and error log
#SBATCH --error=pwd_outputs/%x.%j.err
#SBATCH --ntasks=1                     # Run on a single task
#SBATCH --cpus-per-task=16             # Number of CPU cores per task
#SBATCH --mem=80G                      # Total memory
#SBATCH --nodelist=htc-gpu022
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1                   # replace with number of GPUs required
#SBATCH --time=168:00:00               # Infinite time (0 days, 0 hours, 0 minutes)

# Initialize Conda
eval "$(conda shell.bash hook)"

conda activate /scratch/htc/fsafarov/openmm_ff
python calculate_PWDs.py
