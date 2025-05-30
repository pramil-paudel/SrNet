#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres="gpu:titanxp:1"
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH -J SRNet
#SBATCH -o slurm-%j.out

#python3 train.py --cover_path /scratch/p522p287/DATA/STEN_DATA/COCO_OUT/cover_train --stego_path /scratch/p522p287/DATA/STEN_DATA/COCO_OUT/container_train/ --valid_cover_path /scratch/p522p287/DATA/STEN_DATA/COCO_OUT/cover_validation --valid_stego_path /scratch/p522p287/DATA/STEN_DATA/COCO_OUT/container_validation
python3 test.py