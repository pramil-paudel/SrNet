#!/bin/bash
#SBATCH -p intel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH -t 00:20:00
#SBATCH --time=48:00:00
#SBATCH -J SRNET
#SBATCH -o slurm-%j.out

python3 main.py --cover_path /scratch/p522p287/DATA/STEN_DATA/IMAGE_NET_OUT/SRNET/cover_train --stego_path /scratch/p522p287/DATA/STEN_DATA/IMAGE_NET_OUT/SRNET/container_train/ --valid_cover_path /scratch/p522p287/DATA/STEN_DATA/IMAGE_NET_OUT/SRNET/cover_validation --valid_stego_path /scratch/p522p287/DATA/STEN_DATA/IMAGE_NET_OUT/SRNET/container_validation

