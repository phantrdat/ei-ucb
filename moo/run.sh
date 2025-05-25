#!/bin/bash
#SBATCH --output=results/log/%x.out
#SBATCH --error=results/log/%x.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=24:00:00 
#SBATCH --mem=32G
#SBATCH --qos=batch-short

# Load necessary modules
module load Anaconda3
source activate
conda activate nose1

# Run the Python script
python run.py --problem $problem