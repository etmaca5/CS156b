#!/bin/bash

#SBATCH --job-name=test_resnet_lung
#SBATCH -A CS156b
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres gpu:1
#SBATCH --mail-user=imantrip@caltech.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --partition=gpu
#SBATCH --output=/groups/CS156b/2024/butters/%j.out
#SBATCH --error=/groups/CS156b/2024/butters/%j.err

cd /groups/CS156b/2024/butters
python test_resnet_lung.py