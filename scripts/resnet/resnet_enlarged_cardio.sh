#!/bin/bash

#SBATCH --job-name=resnet_enlarged_cardio
#SBATCH -A CS156b
#SBATCH --time=18:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres gpu:3
#SBATCH --mail-user=jmyles@caltech.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --partition=gpu
#SBATCH --mem=64G
#SBATCH --output=/groups/CS156b/2024/butters/%j.out
#SBATCH --error=/groups/CS156b/2024/butters/%j.err

source /home/ecasanov/CS156b/myenv/bin/activate
cd /groups/CS156b/2024/butters
python resnet_enlarged_cardio.py
