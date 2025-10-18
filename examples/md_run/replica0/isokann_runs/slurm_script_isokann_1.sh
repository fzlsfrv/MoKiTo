#!/bin/bash
#SBATCH --job-name=my_job              # Name of the job
#SBATCH --output=../isokann_outputs/full_isokann_%j_out.txt     # Standard output and error log
#SBATCH --error=../isokann_outputs/%x.%j.err
#SBATCH --ntasks=1                     # Run on a single task
#SBATCH --mem=80G                      # Total memory
#SBATCH --nodelist=htc-gpu020          # request an 80GB gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1                   # replace with number of GPUs required
#SBATCH --time=168:00:00               # Infinite time (0 days, 0 hours, 0 minutes)

# Initialize Conda
eval "$(conda shell.bash hook)"

conda activate /scratch/htc/fsafarov/openmm_ff
python isokann_final_version_1.py
