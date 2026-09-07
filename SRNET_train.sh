#!/bin/bash
#SBATCH -c 1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH -J srnet_steganalysis
#SBATCH -o slurm-%j.out
#SBATCH -p gpu
#SBATCH --gres="gpu:a100:1"

python3 train.py
python3 test.py