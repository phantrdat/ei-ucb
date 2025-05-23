#!/bin/bash
#SBATCH --job-name=ei_vanishing_grad
#SBATCH --output=logdir/%x_%j.out   # %x = job name, %j = job ID
#SBATCH --error=logdir/%x_%j.err    # optional: separate error log
#SBATCH --mem=8G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
/home/pdat/.conda/envs/BO_env/bin/python vanishing_grad_ei.py
