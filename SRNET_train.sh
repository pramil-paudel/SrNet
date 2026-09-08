#!/bin/bash
#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres="gpu:a100:1"
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH -J SrNEtTesting
#SBATCH -o slurm-%j.out
#python3 train.py
python3 test.py
#python3 sanity_check_srnet.py