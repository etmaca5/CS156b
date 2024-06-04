#!/bin/bash

#SBATCH --job-name=resnet_support_devices
#SBATCH -A CS156b
#SBATCH --time=13:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres gpu:3
#SBATCH --mail-user=jkwon@caltech.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --partition=gpu
#SBATCH --mem=64G
#SBATCH --output=/groups/CS156b/2024/butters/%j.out
#SBATCH --error=/groups/CS156b/2024/butters/%j.err

source /Users/anush/CS156b-1/apaul_butters/bin/activate
# /home/ecasanov/CS156b/myenv/bin/activate
cd /groups/CS156b/2024/butters
python resnet_support_devices.py
