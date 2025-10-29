#!/bin/bash
#SBATCH --job-name=_job_full             # Name of the job
#SBATCH --output=../isokann_outputs/isokann_%x.%j_output_full.txt     # Standard output and error log
#SBATCH --error=../isokann_outputs/full_%x.%j.err
#SBATCH --ntasks=1                     # Run on a single task
#SBATCH --cpus-per-task=32             # Number of CPU cores per task
#SBATCH --mem=128GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1                   # replace with number of GPUs required
#SBATCH --time=168:00:00               # Infinite time (0 days, 0 hours, 0 minutes)

# Initialize Conda
eval "$(conda shell.bash hook)"

conda activate /scratch/htc/fsafarov/openmm_ff
python isokann_1.py
