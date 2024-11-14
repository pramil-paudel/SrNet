#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres="gpu:titanxp:1"
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH -J SRNet
#SBATCH -o slurm-%j.out

python3 train.py --cover_path /scratch/p522p287/DATA/STEN_DATA/IMAGE_NET_OUT/SRNET/cover_train --stego_path /scratch/p522p287/DATA/STEN_DATA/IMAGE_NET_OUT/SRNET/container_train/ --valid_cover_path /scratch/p522p287/DATA/STEN_DATA/IMAGE_NET_OUT/SRNET/cover_validation --valid_stego_path /scratch/p522p287/DATA/STEN_DATA/IMAGE_NET_OUT/SRNET/container_validation